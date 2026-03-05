"""Run this script with 'torchrun'."""

import argparse
import gzip
import logging
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('./scripts/train_llama.py'), "..")))
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from datetime import timedelta
from pathlib import Path
from typing import Optional, TextIO
from galore_utils.modeling_llama import LlamaForCausalLM
from galore_utils import training_utils

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import swanlab, transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from olmo.data import build_train_dataloader, build_parquet_dataloader
from olmo.eval import build_evaluators, Evaluator
from olmo.eval.evaluator import EvaluatorType
from torchmetrics import MeanMetric
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_optimizer_LLR, build_scheduler
from olmo.torch_util import (
    SingleAccelerator,
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")
transformers.logging.set_verbosity_error()

def main(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # Ensure run name set.
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    barrier()

    # Set CUDA device.
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        torch.cuda.empty_cache()
        device = torch.device("cuda")

    # Fill some configuration options.
    if cfg.model_type == "olmo":
        cfg.model.precision = cfg.precision
    elif cfg.model_type == "llama":
        cfg.llama_model.precision = cfg.precision

    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        log.warning(
            "You set the deprecated config option `no_decay_norm_and_bias`. For compatibility, this"
            "setting will take precedence over all other weight decay configurations. Please change"
            "your config to use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None  # So nobody uses this by accident.

    # Display and save configuration.
    if get_global_rank() == 0:
        if cfg.data.paths is not None and len(cfg.data.paths) < 50:
            log.info("Configuration:")
            log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    barrier()

    # Maybe start W&B run.
    if cfg.swanlab is not None and (get_global_rank() == 0 or not cfg.swanlab.rank_zero_only):
        swanlab_dir = Path(cfg.save_folder) / "swanlab"
        swanlab_dir.mkdir(parents=True, exist_ok=True)
        swanlab.init(
            dir=str(swanlab_dir),
            project=cfg.swanlab.project,
            entity=cfg.swanlab.entity,
            group=cfg.swanlab.group,
            name=cfg.swanlab.name,
            tags=cfg.swanlab.tags,
            config=cfg.asdict(exclude=["swanlab"]),
        )

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_parquet_dataloader(cfg)

    barrier()

    # Initialize the model.
    log.info("Building model...")
    if cfg.model_type == "olmo":
        model = OLMo(cfg.model).cuda()
    elif cfg.model_type == "llama":
        llama_cfg = cfg.llama_model.asdict()
        llama_cfg["max_position_embeddings"] = llama_cfg.pop("max_sequence_length")
        model = LlamaForCausalLM(LlamaConfig(**llama_cfg)).cuda()

    log.info(f"\n{model}\n")
    log.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    log.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")
    log.info(f"Saving model to {cfg.save_folder} every {cfg.save_interval_unsharded} update steps")

    if cfg.compile is not None:
        if cfg.model_type == "olmo":
            for block in model.transformer.blocks:
                block.compile(**cfg.compile.asdict())
        elif cfg.model_type == "llama":
            # For Llama models, compile each decoder layer
            for layer in model.model.layers:
                layer.compile(**cfg.compile.asdict())

    if cfg.distributed_strategy == DistributedStrategy.ddp:
        log.info("Wrapping model with DDP...")
        assert cfg.ddp is not None, "DistributedStrategy ddp needs cfg.ddp to be set!"

        if cfg.ddp.find_unused_params is True and cfg.ddp.grad_sync_mode != DDPGradSyncMode.micro_batch:
            raise OLMoConfigurationError(
                "`find_unused_params` is set to True. DDP needs to synchronize gradients for every micro-batch to avoid errors. Set `grad_sync_mode` to `micro_batch`."
            )

        param_init_fn = None

        # move to cuda before calling ddp
        dist_model = DDP(model.to(device), 
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False, 
            find_unused_parameters=cfg.ddp.find_unused_params)
    elif cfg.distributed_strategy == DistributedStrategy.fsdp: 
        # Wrap the model in FSDP.
        log.info("Wrapping model with FSDP...")
        assert cfg.fsdp is not None, "DistributedStrategy fsdp needs cfg.fsdp to be set!"
        wrap_policy = model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            # This prevents any parameters from being initialized twice
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())

            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None

        # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
        device_mesh = None
        hybrid_sharding_fsdp_kwargs = {}
        if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            if version.parse(torch.__version__) < version.parse("2.2.0"):
                # Device mesh was not added to PyTorch until v2.2.0
                raise OLMoConfigurationError(
                    "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
                )

            from torch.distributed.device_mesh import init_device_mesh

            num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )

            if num_model_replicas <= 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

            if get_world_size() % num_model_replicas != 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")

            device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
            hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

        dist_model = FSDP(
            model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=cfg.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
            limit_all_gathers=True,
            device_id=get_local_rank(),
            param_init_fn=param_init_fn,
            **hybrid_sharding_fsdp_kwargs,
        )

    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if cfg.model_type == "olmo":
        if param_init_fn is not None or cfg.distributed_strategy == DistributedStrategy.ddp:
            model.reset_parameters()

    log.info(f"Peak GPU Memory (MB) after {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(dist_model)

    # Construct optimizer and learning rate scheduler.
    lr_scheduler = None
    layer_count = None
    if cfg.LLR.use_modulewise_lr:
        optim, lr_scheduler, layer_count = build_optimizer_LLR(cfg, dist_model)
    else:
        optim = build_optimizer_LLR(cfg, dist_model)
    print('*********************************')
    print(optim)
    print('*********************************')

    scheduler = training_utils.get_scheculer(
        optimizer=optim,
        scheduler_type=cfg.scheduler.name,
        num_training_steps=cfg.stop_at,
        warmup_steps=cfg.scheduler.t_warmup,
        min_lr_ratio=cfg.scheduler.alpha_f,
        last_epoch=-1)

    # Data indices file.
    indices_file: Optional[TextIO] = None
    if cfg.save_data_indices:
        indices_file_path = Path(cfg.save_folder) / f"data-indices/rank{get_global_rank()}.tsv.gz"
        if indices_file_path.exists() and not cfg.save_overwrite:
            raise OLMoConfigurationError(f"{indices_file_path} already exists, use --save_overwrite to overwrite")
        indices_file_path.parent.mkdir(exist_ok=True, parents=True)
        indices_file = gzip.open(indices_file_path, "wt")

    # Consolidate components into `Trainer` object.
    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=model,
        dist_model=dist_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=None,
        indices_file=indices_file,
        LLR=lr_scheduler,
        layer_count=layer_count,
    ) as trainer:
        if cfg.load_path is not None:
            log.info(f"Loading checkpoint from {cfg.load_path}...")
            if cfg.model_type == "olmo":
                trainer.restore_checkpoint(cfg.load_path,
                    load_optimizer_state=not cfg.reset_optimizer_state,
                    load_trainer_state=not cfg.reset_trainer_state,
                    sharded_checkpointer=cfg.load_path_sharded_checkpointer)
                # If we have to, set a new scheduler:
                if cfg.reset_optimizer_state and not cfg.reset_trainer_state:
                    trainer.scheduler = BoltOnWarmupScheduler.wrap(
                        trainer.scheduler,
                        trainer.global_step,
                        int(trainer.global_step + cfg.scheduler.t_warmup),
                    )
            elif cfg.model_type == "llama":
                trainer.restore_checkpoint_LLR(log=log)
                
            log.info("Checkpoint successfully loaded")

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    if not "LOCAL_RANK" in os.environ:
        os.environ['RANK'] = '0'
        os.environ["LOCAL_RANK"] = '0'
        os.environ["WORLD_SIZE"] = '1'
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = '26000'
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    log.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30), rank=global_rank, world_size=world_size)
    log.info("Process group initialized")
    device = f"cuda:{local_rank}"

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    # Parse arguments using argparse
    parser = argparse.ArgumentParser(description="Train OLMo model")
    parser.add_argument("--config", type=str, default="configs/llama_1024/llama-60M-test.yaml")
    args, remaining_args = parser.parse_known_args()

    yaml_path = args.config
    
    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in remaining_args])
    main(cfg)
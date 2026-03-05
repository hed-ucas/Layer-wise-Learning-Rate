from __future__ import annotations

import cProfile
import functools
import gc
import logging
import math
import os
import json
import random
import shutil
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple, Union
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils
import torch.utils.hooks
import swanlab
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig,
)
from .data import IterableDataset
from .eval import Evaluator
from .exceptions import OLMoConfigurationError
from .model import OLMo
from .optim import Optimizer, Scheduler
from .torch_util import (
    SingleAccelerator,
    barrier,
    gc_cuda,
    get_default_device,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    is_distributed,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value,
)
from .util import upload

from olmo.torch_util import get_global_rank
from olmo.optim import clip_gradients_simple
from galore_utils import training_utils
import openpyxl
from openpyxl import Workbook
from olmo.LRUnbalance import layerTempbalance
import datasets
import datasets.distributed
from tqdm import tqdm

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)

def record_lrs_to_excel(optimizer, excel_path='learning_rates.xlsx', global_step=0):
    lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
    row = [global_step] + lr_list

    if global_step==0:
        if os.path.exists(excel_path):
            os.remove(excel_path)

        wb = Workbook()
        ws = wb.active

        header = ['global_step'] + [f'param_group_{i}_lr' for i in range(len(lr_list))]
        ws.append(header)
        ws.append(row)
        wb.save(excel_path)
    else:
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active
        ws.append(row)
        wb.save(excel_path)

@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    total_training_Gflops: float = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(
        self,
        global_total_tokens: int,
        device_batch_num_tokens: int,
        num_fwd_flops: int,
        num_bck_flops: int,
        record: bool = True,
    ) -> None:
        self.global_total_tokens = global_total_tokens
        # num_fwd_flops and num_bck_flops from the OLMo model computes flops per token
        # converting to GFLOPs here prevents numerical issues while logging
        self.total_training_Gflops = (num_fwd_flops + num_bck_flops) * global_total_tokens / 1e9

        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}

        # plot flops related metrics
        metrics["throughput/total_training_Gflops"] = self.total_training_Gflops
        metrics["throughput/total_training_log_Gflops"] = math.log(self.total_training_Gflops)

        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}

def cleanup_old_checkpoints(save_dir, keep_last_n=2, log=None):
    checkpoints = []
    for item in os.listdir(save_dir):
        if item.startswith("model_") and os.path.isdir(os.path.join(save_dir, item)):
            try:
                step = int(item.split("_")[1])
                checkpoints.append((step, os.path.join(save_dir, item)))
            except:
                continue
    
    if len(checkpoints) <= keep_last_n:
        return
    
    checkpoints.sort()
    for step, path in checkpoints[:-keep_last_n]:
        shutil.rmtree(path)
        log.info(f"Deleted old checkpoint: model_{step}")

def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


fused_loss_fn: Optional[Callable]

try:
    import flash_attn
    from flash_attn.ops.triton.cross_entropy import (
        cross_entropy_loss as flash_cross_entropy_loss,  # type: ignore
    )

    def fused_loss_fn(
        logits,
        labels,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
    ):
        # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
        ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

        if ce_loss_use_ignore_index_param:
            ignore_index_kwarg = {"ignore_index": ignore_index}
        else:
            ignore_index_kwarg = {"ignored_index": ignore_index}

        loss, z_loss = flash_cross_entropy_loss(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_multiplier,
            inplace_backward=False,
            process_group=None,
            **ignore_index_kwarg,
        )

        mask = labels != ignore_index

        if reduction == "mean":
            loss = loss.sum() / mask.sum()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss

        if not compute_z_loss:
            return loss, None

        if reduction == "mean":
            z_loss = z_loss.sum() / mask.sum()
        elif reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss

        return loss, z_loss

except ImportError:
    fused_loss_fn = None


@dataclass
class Trainer:
    cfg: TrainConfig
    model: OLMo
    dist_model: Union[DDP, FSDP, SingleAccelerator]
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[Evaluator]
    epoch: Optional[int] = None
    global_step: int = 0
    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""
    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    indices_file: Optional[TextIO] = None
    _start_time: float = 0.0
    _gc_init_state: bool = True
    init_log: bool = True
    continue_training: bool = False
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    LLR: Optional[layerTempbalance] = None
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    linear_scheduler: Optional[Scheduler] = None
    layer_count: Optional[int] = None
    opt_params_groups: Optional[List[float]] = None
    LLR_performed: bool = False

    def __post_init__(self):
        if self.cfg.fused_loss:
            if fused_loss_fn is not None:
                self.loss_fn = fused_loss_fn
            else:
                raise NameError("`fused_loss_fn` is not defined. Please ensure that `flash_attn` is installed.")

    @property
    def dataset(self) -> IterableDataset:
        assert isinstance(self.train_loader.dataset, IterableDataset)
        return self.train_loader.dataset

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        return math.ceil(self.max_steps / self.batches_per_epoch)

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch or 0,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch") or 0
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_step = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        assert self.epoch is not None
        # Reshuffle dataset if needed.
        if self.dataset.epoch != self.epoch:
            log.info(f"Reshuffling data loader for epoch {self.epoch}...")
            self.dataset.reshuffle(self.epoch)

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset, IterableDataset)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        log.info("Resetting learning rate...")
        new_learning_rate = self.scheduler.get_lr()
        for num, group in enumerate(self.optim.param_groups):
            group["lr"] = new_learning_rate[num]
            group["initial_lr"] = self.cfg.optimizer.learning_rate
            if "weight_decay" in group and group["weight_decay"] > 0.0:
                group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available():
            if rng_state["cuda"] is not None:
                torch.cuda.set_rng_state(rng_state["cuda"])
            else:
                log.warning("CUDA is available, but no RNG state was provided.")
        if torch.backends.mps.is_available():
            if rng_state["mps"] is not None:
                torch.mps.set_rng_state(rng_state["mps"])
            else:
                log.warning("MPS is available, but no RNG state was provided.")

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        # Flush data indices file.
        # TODO: upload the indices files?
        if self.indices_file is not None:
            self.indices_file.flush()

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        current_checkpoints.append(checkpoint_dir)

        # Save the checkpoint.
        try:
            checkpointer.save_checkpoint(
                checkpoint_dir,
                self.dist_model,
                self.optim,
                self.trainer_state_dict(),
            )
        except FileExistsError:
            raise OLMoConfigurationError(
                f"Checkpoint for step {self.global_step} already exists, use --save_overwrite to overwrite it"
            )

        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise

        # Remove old checkpoints.
        # For DDP, checkpoint_type being passed to remove_checkpoint is always `unsharded`.
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self.remove_checkpoint(0, checkpoint_type)

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def save_checkpoint_LLR(
        self, log: logging.Logger, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        current_model_directory = f"{self.cfg.save_folder}/model_{self.global_step}"
        log.info(f"Saving model and optimizer to {current_model_directory}, update step {self.global_step}")
        os.makedirs(self.cfg.save_folder, exist_ok=True)

        tmp = self.dist_model.module.generation_config.pad_token_id
        self.dist_model.module.generation_config.pad_token_id = 1
        self.dist_model.module.save_pretrained(current_model_directory, max_shard_size='100GB', safe_serialization=False)
        self.dist_model.module.generation_config.pad_token_id = tmp

        optimizer_checkpoint = {
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.cfg.asdict(exclude=["swanlab"]),
            "dtype": self.cfg.precision,
            "random_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            }
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": self.global_step,
            "tokens_seen": self.global_train_tokens_seen
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

        cleanup_old_checkpoints(self.cfg.save_folder, keep_last_n=self.cfg.save_num_unsharded_checkpoints_to_keep, log=log)

        gc_cuda()
        return current_model_directory

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()

    def restore_checkpoint_LLR(
        self,
        log: logging.Logger,
    ): 
        # load 1: model state
        log.info("*" * 40)
        log.info(f"Loading model from {self.cfg.load_path}")
        checkpoint_path = os.path.join(self.cfg.load_path,'pytorch_model.bin')
        self.dist_model.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        log.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(self.cfg.load_path, "training_state.json")):
            log.info(f"Loading training state from {self.cfg.load_path}")
            with open(os.path.join(self.cfg.load_path, "training_state.json")) as f:
                _old_state = json.load(f)
            self.global_step = _old_state["global_step"]
            self.global_train_tokens_seen = _old_state["tokens_seen"]
            self.cfg.stop_after = self.cfg.stop_at - self.global_step
            self.init_log = False
            self.continue_training = True

            log.info(f"global_step       : {self.global_step}")
            log.info(f"Will train for {self.cfg.stop_after} update steps")
        else:
            log.warning(f"Did not find training state in {self.cfg.load_path}")
        
        log.info("*" * 40)

        # load 2: optimizer and scheduler state
        if self.cfg.load_path is not None:
            optimizer_checkpoint_path = os.path.join(self.cfg.load_path, "optimizer.pt")
            if os.path.exists(optimizer_checkpoint_path):
                log.info(f"Loading optimizer and scheduler state from {optimizer_checkpoint_path}")
                optimizer_checkpoint = torch.load(optimizer_checkpoint_path, map_location="cpu")
                
                self.optim.load_state_dict(optimizer_checkpoint["optimizer"])
                self.scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
                if "random_states" in optimizer_checkpoint:
                    random.setstate(optimizer_checkpoint["random_states"]["python"])
                    np.random.set_state(optimizer_checkpoint["random_states"]["numpy"])
                    torch.set_rng_state(optimizer_checkpoint["random_states"]["torch"])
                    torch.cuda.set_rng_state_all(optimizer_checkpoint["random_states"]["torch_cuda"])
                    log.info("Random states restored")
                
                log.info("Optimizer and scheduler states loaded successfully")
            else:
                log.warning(f"Optimizer checkpoint not found at {optimizer_checkpoint_path}")
        
        # Precisely adjust data loader start index to continue from checkpoint
        # Calculate exactly how many examples and tokens have been trained
        if self.global_step > 0:
            # Get max_sequence_length based on model type
            if self.cfg.model_type == "llama":
                max_seq_len = self.cfg.llama_model.max_sequence_length
            elif self.cfg.model_type == "olmo":
                max_seq_len = self.cfg.model.max_sequence_length
            else:
                raise ValueError(f"Unknown model_type: {self.cfg.model_type}")
            
            # Precisely calculate examples and tokens seen
            examples_seen = self.global_step * self.cfg.global_train_batch_size
            tokens_seen_calculated = examples_seen * max_seq_len
            
            # Set start_index to exactly continue from where we left off
            # Add a random offset between 50-100 to introduce some variation
            random_offset = random.randint(50, 100)
            
            if get_global_rank() == 0:
                log.info(f"(examples_seen: {examples_seen:,d}, random_offset: {random_offset}, "
                        f"tokens_seen_calculated: {tokens_seen_calculated:,d}, "
                        f"tokens_seen_from_checkpoint: {self.global_train_tokens_seen:,d})")
        gc_cuda()

        
    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def _setup_module_output_save_hooks(self, micro_batch_idx: int) -> List[torch.utils.hooks.RemovableHandle]:
        if (
            self.cfg.module_outputs_save_steps is None
            or self.global_step not in self.cfg.module_outputs_save_steps
        ):
            return []

        if micro_batch_idx != 0 or get_global_rank() != 0:
            # Hook is currently only used on the first microbatch of rank 0
            return []

        trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
        if trace_save_folder.exists():
            if self.cfg.save_overwrite:
                shutil.rmtree(trace_save_folder)
            else:
                raise OLMoConfigurationError(
                    f"Attempting to overwrite traces at step {self.global_step} without --save_overwrite"
                )
        trace_save_folder.mkdir(parents=True)

        def trace_outputs_hook(
            module_name: str, _: torch.nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            if len(args) == 0:
                log.info("No input args for module %s, output %s", module_name, output)

            module_input = args[0] if len(args) > 0 else torch.tensor(())
            trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
            trace_save_folder.mkdir(parents=True, exist_ok=True)

            module_occurence_num = 0
            while (
                module_input_filepath := trace_save_folder / f"{module_name}_{module_occurence_num}_input.pt"
            ).exists():
                module_occurence_num += 1
            torch.save(module_input, module_input_filepath)

            module_output_filepath = trace_save_folder / f"{module_name}_{module_occurence_num}_output.pt"
            torch.save(output, module_output_filepath)

        output_hooks = []
        for module_name, module in self.model.named_modules(prefix="model"):
            output_hooks.append(module.register_forward_hook(functools.partial(trace_outputs_hook, module_name)))

        return output_hooks

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        if self.cfg.model_type == "llama":
            logits = self.dist_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            ).logits
        else:
            logits = self.dist_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                attention_bias=batch.get("attention_bias"),
                doc_lens=batch.get("doc_lens"),
                max_doc_lens=batch.get("max_doc_lens"),
            ).logits
        
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits

    def train_micro_batch(
        self, micro_batch: Dict[str, Any], num_micro_batches: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # ce_loss, z_loss, logits = self.model_forward(micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="sum")
        z_loss = None
        labels = micro_batch["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        ce_loss = self.dist_model(**micro_batch, labels=labels).loss

        ce_loss = ce_loss / num_micro_batches

        # In case this helps with memory utilization.
        del micro_batch

        loss = ce_loss

        # del logits

        return loss, ce_loss, z_loss

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        batch_size_in_tokens = batch["input_ids"].numel()

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            # setup sync context for DDP for all micro-batches except the last
            grad_sync_context = nullcontext
            if (
                self.cfg.distributed_strategy == DistributedStrategy.ddp
                and self.cfg.ddp is not None
                and self.cfg.ddp.grad_sync_mode == DDPGradSyncMode.batch
            ):
                if micro_batch_idx != num_micro_batches - 1:
                    grad_sync_context = self.dist_model.no_sync

            # Register output hooks
            output_hooks: List[torch.utils.hooks.RemovableHandle] = []
            output_hooks += self._setup_module_output_save_hooks(micro_batch_idx)

            with grad_sync_context():
                autocast_device = "mps" if self.device.type == "mps" else "cuda"
                with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                    # Run forward pass.
                    loss, ce_loss, z_loss = self.train_micro_batch(micro_batch, num_micro_batches)
                    
                    # Check for CUDA errors immediately after forward pass
                    # CUDA errors are asynchronous, so we need to synchronize to catch them early
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError as e:
                            error_msg = str(e)
                            if "CUDA" in error_msg or "device-side assert" in error_msg:
                                raise RuntimeError(f"CUDA error detected after forward pass: {error_msg}")

                    # Update overall CE batch loss.
                    ce_batch_loss += ce_loss.detach()

                    # Update overall Z batch loss.
                    if z_loss is not None:
                        assert z_batch_loss is not None
                        z_batch_loss += z_loss.detach()

                # Run backward pass.
                loss.backward()
                
                # Check for CUDA errors after backward pass as well
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "CUDA" in error_msg or "device-side assert" in error_msg:
                            raise RuntimeError(f"CUDA error detected after backward pass: {error_msg}")

            # Remove output hooks
            for hook in output_hooks:
                hook.remove()

        return ce_batch_loss, z_batch_loss

    def train_step(self, init_log, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Write data-indices to file.
        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass with error handling
        ce_batch_loss, z_batch_loss = self.train_batch(batch)
        
        # Check for CUDA errors before distributed operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # This will raise if there's a pending CUDA error
        
        # Collect loss, potentially reducing over all ranks.
        # Check for CUDA errors before NCCL operations
        if reduce_global_loss:
            # Synchronize before NCCL operations to catch any pending CUDA errors
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()

        # LLR assign
        lr = self.cfg.optimizer.learning_rate
        if self.cfg.optimizer.name.lower() == "muon":
            lr = self.cfg.optimizer.muon_lr
        else:
            lr = self.cfg.optimizer.learning_rate

        if self.global_step <= self.cfg.stop_at*self.cfg.LLR_ratio:
            if self.cfg.LLR.use_modulewise_lr:
                if ((self.global_step < (self.cfg.stop_at * self.cfg.LLR.num_grad_steps)) and self.continue_training) or ((self.global_step < (self.cfg.stop_at * self.cfg.LLR.num_grad_steps)) and (((self.global_step % (self.cfg.LLR.unbalancedlr_every * self.cfg.LLR.grad_unbalancedlr_every) == 0) and self.global_step>0)  or (self.global_step==1 and init_log == True))):
                    print(f'\n one step of modulewsie lr----grad-alpha at step: {self.global_step}')
                    self.opt_params_groups, layer_count, self.linear_scheduler = self.LLR.step(self.cfg, self.optim, lr, int(self.cfg.LLR.linear_steps*self.cfg.LLR.grad_unbalancedlr_every), self.global_step, rank0=get_global_rank()==0, alpha_metric=self.cfg.LLR.grad_alpha_metric)         
                    init_log = False
                    self.LLR_performed = True
                    self.continue_training = False
        
                elif ((self.global_step >= (self.cfg.stop_at * self.cfg.LLR.num_grad_steps)) and self.continue_training) or (((self.global_step >= (self.cfg.stop_at * self.cfg.LLR.num_grad_steps)) and ((self.global_step % self.cfg.LLR.unbalancedlr_every  == 0) or init_log == True))):
                    print(f'one step of modulewsie lr----weight-alpha at step: {self.global_step}')
                    self.opt_params_groups, layer_count, self.linear_scheduler = self.LLR.step(self.cfg, self.optim, lr, self.cfg.LLR.linear_steps, self.global_step, rank0=get_global_rank()==0, alpha_metric='weight')
                    init_log = False
                    self.LLR_performed = True
                    self.continue_training = False
                
                if self.LLR_performed:
                    flag1 = (self.global_step <= (self.cfg.stop_at * self.cfg.LLR.num_grad_steps)) and (self.global_step % (self.cfg.LLR.unbalancedlr_every * self.cfg.LLR.grad_unbalancedlr_every)) > min(self.cfg.LLR.linear_steps, (self.cfg.LLR.unbalancedlr_every * self.cfg.LLR.grad_unbalancedlr_every))
                    gs2 = self.global_step-(self.cfg.stop_at * self.cfg.LLR.num_grad_steps)
                    flag2 = (self.global_step >= (self.cfg.stop_at * self.cfg.LLR.num_grad_steps)) and (gs2 % self.cfg.LLR.unbalancedlr_every) > min(self.cfg.LLR.linear_steps, self.cfg.LLR.unbalancedlr_every)
                    if flag1 or flag2:
                        self.linear_scheduler = None
                        for index, param_group in enumerate(self.optim.param_groups):
                            if index <= self.layer_count - 1:
                                param_group['initial_lr'] = self.opt_params_groups[index]
                            else:
                                param_group['initial_lr'] = self.cfg.optimizer.learning_rate          
                            
                        self.scheduler = training_utils.get_scheculer(
                            optimizer=self.optim,
                            scheduler_type=self.cfg.scheduler.name,
                            num_training_steps=self.cfg.stop_at,
                            warmup_steps=self.cfg.scheduler.t_warmup,
                            min_lr_ratio=self.cfg.scheduler.alpha_f,
                            last_epoch=self.global_step-2)     
                        self.LLR_performed = False
        else:
            self.linear_scheduler = None
            if self.cfg.LLR.use_modulewise_lr and self.continue_training:
                self.opt_params_groups, layer_count, self.linear_scheduler = self.LLR.step(self.cfg, self.optim, lr, self.cfg.LLR.linear_steps, self.global_step, rank0=get_global_rank()==0, alpha_metric='weight')
                for index, param_group in enumerate(self.optim.param_groups):
                        if index <= self.layer_count - 1:
                            param_group['initial_lr'] = self.opt_params_groups[index]
                        else:
                            param_group['initial_lr'] = self.cfg.optimizer.learning_rate          
                        
                self.scheduler = training_utils.get_scheculer(
                    optimizer=self.optim,
                    scheduler_type=self.cfg.scheduler.name,
                    num_training_steps=self.cfg.stop_at,
                    warmup_steps=self.cfg.scheduler.t_warmup,
                    min_lr_ratio=self.cfg.scheduler.alpha_f,
                    last_epoch=self.global_step-2)     
                self.LLR_performed = False
                self.continue_training = False

        if get_global_rank() == 0 and self.global_step-1 % self.cfg.LLR.log_LR_every ==0:
            print('out 2 excel')
            record_lrs_to_excel(self.optim, './LR_log/' + self.cfg.swanlab.name + '.xlsx', self.global_step-1)

        # Adjust the learning rate.
        if self.linear_scheduler is not None:
            if get_global_rank() == 0:
                print(f'step linear scheduler at step: {self.global_step}')
            self.linear_scheduler.step()
        else:
            if get_global_rank() == 0:
                print(f'step scheduler at step: {self.global_step}')
            self.scheduler.step()

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()

        return metrics, init_log

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, _, logits = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"
        
        def format_time(seconds: float) -> str:
            h, m = int(seconds // 3600), int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_time(value) if name == 'time/estimated_seconds_remaining' else format_float(value)}"
                    for name, value in metrics.items()
                    if name == "optim/total_grad_norm"
                    or not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.swanlab is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.swanlab.log_interval
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.swanlab.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.swanlab is not None and self.global_step % self.cfg.swanlab.log_interval == 0:
            return True
        else:
            return False

    @torch.no_grad()
    def eval(self, target_eval_tokens: int = 10_000_000) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.dist_model.eval()

        world_size = get_world_size()
        global_rank = get_global_rank()
        
        # Load validation dataset directly from parquet files
        _time = time.time()
        val_data = datasets.load_dataset("parquet",
            data_files={"train": self.cfg.dataset_path, 
                "validation": self.cfg.eval_dataset_path},
            split="validation", streaming=True,)
        log.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

        # Split dataset by node for distributed training
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, 
            rank=global_rank, 
            world_size=world_size
        )

        # Get tokenizer and max_length
        max_length = self.cfg.llama_model.max_sequence_length
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)

        tokenizer.pad_token_id = 0
        
        pad_idx = self.pad_token_id if hasattr(self, 'pad_token_id') else (tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0)
        
        # Create tokenization function for batched processing
        def preprocess_batched(examples):
            tokenized = tokenizer(
                examples["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            # Convert to list format for map function
            return tokenized
        
        # Map dataset to tokenized format
        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            batch_size=1000,  # Process in batches for efficiency
            remove_columns=["text"] if hasattr(val_data, 'features') and "text" in val_data.features else [],
        )
        
        # Create collate function for batching
        def collate_fn(batch_list):
            """Collate a list of examples into a batch."""
            batch = {
                "input_ids": torch.stack([torch.tensor(example["input_ids"]).long() for example in batch_list]),
                "attention_mask": torch.stack([torch.tensor(example["attention_mask"]).long() for example in batch_list]),
            }
            return batch
        
        # Create batch function for streaming dataset
        def batch_fn(dataset, batch_size):
            """Create batches from tokenized dataset."""
            batch = []
            for example in dataset:
                batch.append(example)
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch)
        
        val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)
        
        log.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

        # Track tokens and loss
        evaluated_on_tokens = 0
        total_loss = torch.tensor(0.0).to(self.device)
        total_batches = 1
        
        # Get batch size for evaluation
        batch_size = self.cfg.device_eval_batch_size if self.cfg.device_eval_batch_size is not None else self.cfg.device_train_batch_size
        
        # Create progress bar (only show on rank 0 to avoid clutter)
        if global_rank == 0:
            pbar = tqdm(
                desc="Evaluating",
                unit="tokens",
                total=target_eval_tokens,
                initial=0,
                dynamic_ncols=True,
            )
        else:
            pbar = None
        
        # Run model over batches until target tokens reached
        for batch in val_data_mapped.batch(batch_size=batch_size):
            # Check if we've already reached the target token count
            if evaluated_on_tokens > target_eval_tokens:
                break
            
            total_batches += 1

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Prepare labels (similar to training)
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100

            # Calculate loss
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                outputs = self.dist_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=labels,
                )
                loss = outputs.loss
                
            total_loss += loss.detach()

            # Count tokens (non-padding tokens)
            # Following reference code: multiply by world_size to account for all ranks processing data in parallel
            batch_tokens = (batch["input_ids"] != pad_idx).sum().item() * world_size
            evaluated_on_tokens += batch_tokens
            
            # Update progress bar
            if pbar is not None:
                current_loss = total_loss.item() / total_batches
                pbar.update(batch_tokens)
                pbar.set_postfix({
                    "tokens": f"{evaluated_on_tokens:,}/{target_eval_tokens:,}",
                    "loss": f"{current_loss:.4f}",
                    "batches": total_batches,
                })
        if pbar is not None:
            pbar.close()

        # Calculate average loss
        total_loss = total_loss / total_batches

        # Gather losses across all ranks
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
        dist.all_gather(gathered_losses, total_loss)
        avg_loss = sum([t.item() for t in gathered_losses]) / world_size

        # Store metrics
        eval_metrics = {
            "eval/validation/CrossEntropyLoss": avg_loss,
            "eval/validation/Perplexity": math.exp(avg_loss),
            "eval/validation/tokens_evaluated": evaluated_on_tokens,
        }
        
        log.info(f"Evaluation completed: loss={avg_loss:.4f}, perplexity={math.exp(avg_loss):.2f}, tokens={evaluated_on_tokens:,}")

        # Eval compiles a bunch more versions, and the result is terrible. This way we get back to zero.
        if self.cfg.compile is not None:
            torch.compiler.reset()

        return eval_metrics

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif (
                self.cfg.early_stopping_factor is not None
                and self.global_step > self.cfg.scheduler.t_warmup
                and self.cur_train_loss > self.cfg.early_stopping_factor * self.min_train_loss
            ):
                # Next check if early stopping loss criteria is met.
                should_cancel = True
                cancel_reason = "early stopping from loss increase"
            elif swanlab.run is not None and (api_key := os.environ.get("swanlab_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                from requests.exceptions import RequestException
                from swanlab.errors import CommError

                try:
                    api = swanlab.Api(api_key=api_key)
                    run = api.run(swanlab.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {"cancel", "canceled", "cancelled"}:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except (RequestException, CommError):
                    log.info("Failed to check if W&B run is cancelled, continuing run.")

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=self.cfg.llama_model.max_sequence_length)
        self.tokenizer.pad_token_id = 0

        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()
        
        # eval_metrics = self.eval()
        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if get_global_rank() == 0 and swanlab.run is not None:
                swanlab.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.dist_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if get_global_rank() == 0 and swanlab.run is not None:
                swanlab.log(sys_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: int = self.cfg.stop_at
        save_checkpoints: bool = True
        init_log = self.init_log

        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.cfg.max_epochs):
                for batch in self.train_loader: # 16
                    # Validate token IDs before any state updates to avoid rollback complexity
                    labels = batch["input_ids"].squeeze().clone() 
                    labels[labels == self.pad_token_id] = -100 

                    batch["input_ids"] = batch["input_ids"].squeeze()
                    batch["attention_mask"] = batch["attention_mask"].squeeze()

                    batch_size, seq_len = batch["input_ids"].size()
                    if self.cfg.model_type == "llama":
                        assert seq_len == self.cfg.llama_model.max_sequence_length
                    elif self.cfg.model_type == "olmo":
                        assert seq_len == self.cfg.model.max_sequence_length
                    assert batch_size == self.cfg.device_train_batch_size
                    global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                    self.global_step += 1
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len
                    speed_monitor.batch_start(
                        global_total_tokens=self.global_train_tokens_seen,
                        device_batch_num_tokens=batch_size * seq_len,  # num tokens in batch for this device
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        num_fwd_flops=self.model.num_fwd_flops,  # this is per token
                        num_bck_flops=self.model.num_bck_flops,  # this is per token
                        record=not first_batch,
                    )

                    should_log_this_step = self.should_log_this_step()

                    # Run train step on batch.
                    # If CUDA error occurs (e.g., invalid token IDs), skip this batch and continue with next one
                    metrics, init_log = self.train_step(init_log, batch, reduce_global_loss=should_log_this_step)

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())
                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            # Calculate estimated time remaining
                            if "throughput/device/batches_per_second" in metrics:
                                remaining_steps = stop_at - self.global_step
                                batches_per_second = metrics["throughput/device/batches_per_second"]
                                if remaining_steps > 0 and batches_per_second > 0:
                                    metrics["time/estimated_seconds_remaining"] = remaining_steps / batches_per_second
                            self.log_metrics_to_console(
                                f"[step={self.global_step}/{self.cfg.stop_at},epoch={epoch}]",
                                metrics,
                            )
                        else:
                            log.info(f"[step={self.global_step}/{self.cfg.stop_at},epoch={epoch}]")

                    # Log metrics to W&B.
                    if (
                        swanlab.run is not None
                        and self.cfg.swanlab is not None
                        and self.global_step % self.cfg.swanlab.log_interval == 0
                        and get_global_rank() == 0
                    ):
                        swanlab.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = min(stop_at, self.global_step + extra_steps)

                    # Maybe save sharded checkpoint.
                    if self.cfg.distributed_strategy == DistributedStrategy.fsdp:
                        if save_checkpoints and (
                            cancel_initiated
                            or (
                                self.cfg.save_interval is not None
                                and self.global_step % self.cfg.save_interval == 0
                                and self.cfg.save_num_checkpoints_to_keep != 0
                            )
                        ):
                            log.info("Saving checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Remove any ephemeral checkpoints.
                            while self.ephemeral_checkpoints:
                                self.remove_ephemeral_checkpoint()

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                            # If the run was just canceled this will be the final checkpoint.
                            if cancel_initiated:
                                save_checkpoints = False
                        elif (
                            self.cfg.save_interval_ephemeral is not None
                            and self.global_step % self.cfg.save_interval_ephemeral == 0
                        ):
                            log.info("Saving ephemeral checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                    # Maybe save unsharded checkpoint.
                    # This code snippet should always execute when running DDP.
                    if (
                        save_checkpoints
                        and self.cfg.save_interval_unsharded is not None
                        and self.global_step % self.cfg.save_interval_unsharded == 0
                        and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                    ):
                        log.info("Saving unsharded checkpoint...")
                        if self.cfg.model_type == "olmo":
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                        elif self.cfg.model_type == "llama":
                            if get_global_rank() == 0:
                                checkpoint_path = self.save_checkpoint_LLR(log, CheckpointType.unsharded)
                                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    if not cancel_initiated and (
                        self.global_step % self.cfg.eval_interval == 0 or self.global_step >= stop_at
                    ):
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if swanlab.run is not None and get_global_rank() == 0:
                            swanlab.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.dist_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    self.dataset.start_index = 0
                    if self.epoch < self.cfg.max_epochs:
                        log.info(f"Reshuffling data loader for epoch {self.epoch}...")
                        self.dataset.reshuffle(self.epoch)
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                if self.cfg.model_type == "olmo":
                    checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                elif self.cfg.model_type == "llama":
                    if get_global_rank() == 0:
                        checkpoint_path = self.save_checkpoint_LLR(log, CheckpointType.unsharded)
                        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
                and self.cfg.distributed_strategy == DistributedStrategy.fsdp
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self.indices_file is not None:
            self.indices_file.flush()
            self.indices_file.close()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if swanlab.run is not None and get_global_rank() == 0:
            swanlab.finish()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)

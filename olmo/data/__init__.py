import importlib
import itertools, random
import logging, os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, cast

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset as PyTorchIterableDataset
from torch.utils.data import IterableDataset, get_worker_info
import datasets
import datasets.distributed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig
from ..exceptions import OLMoConfigurationError
from ..torch_util import barrier, get_global_rank, get_world_size
from .collator import CustomDatasetDataCollator, DataCollator
from .custom_datasets import build_custom_dataset, extract_module_and_class
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader", "ParquetTextDataset", "build_parquet_dataloader", "StreamingParquetIterableDataset"]

LOGGER = logging.getLogger(__name__)

class StreamingParquetIterableDataset(PyTorchIterableDataset[Dict[str, Any]]):
    """
    A PyTorch IterableDataset for streaming parquet files.
    This class handles DDP data splitting, worker-level data distribution, and tokenization.
    """
    
    def __init__(
        self,
        hf_dataset: Any,  # datasets.IterableDataset
        text_column: str = "text",
        tokenizer: Optional[Any] = None,  # Tokenizer object (should support __call__ with text)
        batch_size: int = 1,
        max_length: int = 1024,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: int = 0,
        epoch: int = 0,
    ):
        super().__init__()
        self._hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.batch_size = batch_size
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the streaming dataset with tokenization and batching.
        Handles worker-level splitting if using DataLoader with multiple workers.
        """        
        # Handle worker-level data splitting (for DataLoader workers)
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self._hf_dataset)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self._hf_dataset, worker_id, None, num_workers)
        
        # With tokenizer: collect examples into batches and tokenize
        batch = []
        for example in iter_data:            
            # Tokenize the example
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)
            
            # Yield batch when it reaches batch_size
            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []
            
            # Yield remaining examples if any
        if batch:
            yield self._format_batch(batch)
    
    def _format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format a batch of tokenized examples into a single batch tensor."""
        input_ids = torch.stack([item["input_ids"].squeeze() for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze() for item in batch])
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ParquetTextDataset(Dataset[Dict[str, Any]]):
    """
    A PyTorch Dataset that reads text samples from parquet files using datasets library.
    Each sample is expected to contain a text field.
    
    :param parquet_paths: List of paths to parquet files
    :param text_column: Name of the column containing the text (default: 'text')
    :param use_streaming: If True, use streaming mode (faster but requires different handling)
    """
    
    def __init__(
        self,
        parquet_paths: List[PathOrStr],
        text_column: str = "text",
        use_streaming: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        split_by_node: bool = True,
    ):
        self.use_streaming = use_streaming
        self.parquet_paths = [Path(p) for p in parquet_paths]
        self.text_column = text_column
        self.rank = rank
        self.world_size = world_size
        self.split_by_node = split_by_node
        
        self._hf_dataset = None

        self._texts: List[str] = []  # Will remain empty in streaming mode
        self._load_with_streaming()
    
    def _load_with_streaming(self):
        """Load parquet files using datasets library with streaming mode."""
        LOGGER.info(f"Loading {len(self.parquet_paths)} parquet files using streaming mode...")
        
        # Convert paths to strings and check existence
        parquet_path_strs = []
        for parquet_path in self.parquet_paths:
            parquet_path = Path(parquet_path)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            parquet_path_strs.append(str(parquet_path))
        
        # Use datasets library with streaming
        if len(parquet_path_strs) == 1:
            data_files = parquet_path_strs[0]
        else:
            data_files = parquet_path_strs
        
        self._hf_dataset = datasets.load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=True,
        )
        
        self._hf_dataset = datasets.distributed.split_dataset_by_node(
            self._hf_dataset,
            rank=self.rank,
            world_size=self.world_size,
        )
        LOGGER.info(f"Dataset split completed for rank {self.rank}")
        
        LOGGER.info("Streaming dataset created. Data will be loaded on-demand.")
        
    def _load_all_parquet_files(self):
        """Load all parquet files using datasets library for better performance."""
        LOGGER.info(f"Loading {len(self.parquet_paths)} parquet files using datasets library...")
        
        # Convert paths to strings and check existence
        parquet_path_strs = []
        for parquet_path in self.parquet_paths:
            parquet_path = Path(parquet_path)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            parquet_path_strs.append(str(parquet_path))
        
        # Use datasets library for efficient loading (non-streaming for indexing support)
        try:
            if len(parquet_path_strs) == 1:
                data_files = parquet_path_strs[0]
            else:
                data_files = parquet_path_strs
            
            # Load without streaming to enable indexing
            hf_dataset = datasets.load_dataset(
                "parquet",
                data_files=data_files,
                split="train",
                streaming=False,
            )
            
            # Validate text column exists
            column_names = hf_dataset.column_names if hasattr(hf_dataset, 'column_names') else list(hf_dataset[0].keys()) if len(hf_dataset) > 0 else []
            if column_names and self.text_column not in column_names:
                raise ValueError(
                    f"Column '{self.text_column}' not found in parquet files. "
                    f"Available columns: {column_names}"
                )
            
            # Extract texts - datasets library handles this efficiently
            LOGGER.info("Extracting text column from dataset...")
            texts = hf_dataset[self.text_column]
            
            # Convert to list of strings and filter out None/NaN
            self._texts = [str(t) if t is not None else "" for t in texts]
            LOGGER.info(f"Total samples loaded: {len(self._texts)}")
            
        except Exception as e:
            LOGGER.warning(f"Failed to load with datasets library: {e}. Falling back to pandas/pyarrow...")
            # Fallback to original method
            self._use_pandas = False
            self._pd = None
            self._pq = None
            
            try:
                import pandas as pd
                self._use_pandas = True
                self._pd = pd
            except ImportError:
                try:
                    import pyarrow.parquet as pq
                    self._pq = pq
                except ImportError:
                    raise OLMoConfigurationError(
                        "Either 'pandas' or 'pyarrow' must be installed to read parquet files"
                    )
            
            if self._use_pandas and self._pd is not None:
                # Using pandas
                for parquet_path in self.parquet_paths:
                    if not parquet_path.exists():
                        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
                    LOGGER.info(f"Loading {parquet_path}...")
                    df = self._pd.read_parquet(parquet_path)
                    
                    if self.text_column not in df.columns:
                        available_cols = list(df.columns)
                        raise ValueError(
                            f"Column '{self.text_column}' not found in {parquet_path}. "
                            f"Available columns: {available_cols}"
                        )
                    
                    texts = df[self.text_column].tolist()
                    # Convert to strings and filter out None/NaN
                    texts = [str(t) if t is not None else "" for t in texts]
                    self._texts.extend(texts)
                    LOGGER.info(f"Loaded {len(texts)} samples from {parquet_path}")
            elif self._pq is not None:
                # Using pyarrow
                for parquet_path in self.parquet_paths:
                    if not parquet_path.exists():
                        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
                    LOGGER.info(f"Loading {parquet_path}...")
                    table = self._pq.read_table(str(parquet_path))
                    
                    if self.text_column not in table.column_names:
                        available_cols = table.column_names
                        raise ValueError(
                            f"Column '{self.text_column}' not found in {parquet_path}. "
                            f"Available columns: {available_cols}"
                        )
                    
                    texts = table[self.text_column].to_pylist()
                    # Convert to strings and filter out None
                    texts = [str(t) if t is not None else "" for t in texts]
                    self._texts.extend(texts)
                    LOGGER.info(f"Loaded {len(texts)} samples from {parquet_path}")
            else:
                raise RuntimeError("Neither pandas nor pyarrow is available")
            
            LOGGER.info(f"Total samples loaded: {len(self._texts)}")
    
    def __len__(self) -> int:
        if self.use_streaming:
            # Streaming mode doesn't support len()
            # Return a large number or raise error
            raise TypeError("Streaming dataset does not support len(). Use iter() instead.")
        return len(self._texts)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a dictionary with the text sample."""
        if self.use_streaming:
            raise TypeError("Streaming dataset does not support indexing. Use iter() instead.")
        
        if index < 0:
            index = len(self._texts) + index
        if index >= len(self._texts):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._texts)}")
        
        return {
            "text": self._texts[index],
        }
    
    def __iter__(self):
        """Iterator for streaming mode."""
        if self.use_streaming and self._hf_dataset is not None:
            for example in self._hf_dataset:
                # Handle both dict and object access
                if isinstance(example, dict):
                    text = example.get(self.text_column, "")
                else:
                    text = getattr(example, self.text_column, "")
                yield {"text": str(text) if text is not None else ""}
        else:
            # Non-streaming mode: iterate over cached texts
            for text in self._texts:
                yield {"text": text}


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    if train_config.model_type == "llama":
        return MemMapDataset(
            *paths,
            chunk_size=train_config.llama_model.max_sequence_length,
            memmap_dtype=data_config.effective_memmap_dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            pad_token_id=train_config.llama_model.pad_token_id,
            eos_token_id=train_config.llama_model.eos_token_id,
            generate_attention_mask=data_config.generate_attention_mask,
            generate_doc_lengths=data_config.generate_doc_lengths,
            label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
            instance_filter_config=data_config.instance_filter,
        )
    elif train_config.model_type == "olmo":
        return MemMapDataset(
            *paths,
            chunk_size=train_config.model.max_sequence_length,
            memmap_dtype=data_config.effective_memmap_dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            pad_token_id=train_config.model.pad_token_id,
            eos_token_id=train_config.model.eos_token_id,
            generate_attention_mask=data_config.generate_attention_mask,
            generate_doc_lengths=data_config.generate_doc_lengths,
            label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
            instance_filter_config=data_config.instance_filter,
        )


def build_collator(train_config: TrainConfig) -> DataCollator:
    """Returns a collator for the train dataloader. Either returns the default
    collator or a custom collator specified in the train config.

    :param train_config: OLMo train config
    :raises OLMoConfigurationError: Raises an error if the collate function is not found
    :return: Collator for the train dataloader
    """
    if train_config.model_type == "llama":
        pad_token_id = train_config.llama_model.pad_token_id
    elif train_config.model_type == "olmo":
        pad_token_id = train_config.model.pad_token_id

    if train_config.data.custom_dataset:
        if train_config.data.custom_dataset.collate_fn:
            module, function = extract_module_and_class(train_config.data.custom_dataset.collate_fn)
            if module is None:
                if train_config.data.custom_dataset.module is None:
                    module, _ = extract_module_and_class(train_config.data.custom_dataset.name)
                else:
                    module = train_config.data.custom_dataset.module
            try:
                assert module is not None
                collator = getattr(importlib.import_module(module), function)
            except AttributeError:
                raise OLMoConfigurationError(
                    f"collate_fn {train_config.data.custom_dataset.collate_fn} not found in {module}. Please specify the full module path of the function."
                )
            return collator

        return CustomDatasetDataCollator(
            pad_direction=train_config.data.pad_direction,
            pad_token_id=pad_token_id,
            **train_config.data.custom_dataset.collate_config.asdict(),  # type: ignore
        )
    else:
        return DataCollator(
            pad_direction=train_config.data.pad_direction, pad_token_id=pad_token_id
        )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(
    train_config: TrainConfig,
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    collator = build_collator(train_config)
    if train_config.data.custom_dataset:
        if train_config.data.paths is not None or train_config.data.datasets is not None:
            raise OLMoConfigurationError(
                "custom_dataset_class is mutually exclusive with DataConfig.paths and DataConfig.datasets"
            )
        dataset = build_custom_dataset(train_config)
    else:
        dataset = build_memmap_dataset(
            train_config, train_config.data, include_instance_metadata=include_instance_metadata
        )
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up offsets cache to avoid recomputing in each worker process
    offsets_cache_file = work_dir / "dataset_offsets_cache.npy"
    if hasattr(dataset, 'set_offsets_cache_file'):
        dataset.set_offsets_cache_file(offsets_cache_file)
    
    dataset = IterableDataset(
        dataset,  # type: ignore
        train_config.global_train_batch_size,
        seed=seed,
        epoch=train_config.epoch or 0,
        shuffle=True,
        drop_last=train_config.data.drop_last,
        world_size=world_size,
        rank=rank,
        fs_local_rank=fs_local_rank,
        work_dir=work_dir,
    )
    barrier()
    # Pre-compute dataset offsets in the main process to avoid each worker
    # process computing them independently, which causes significant delay
    # when starting the first iteration. The offsets will be cached to disk
    # so worker processes can load them quickly.
    if hasattr(dataset.dataset, 'offsets'):
        _ = dataset.dataset.offsets
    barrier()
    out = DataLoader(
        dataset,
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
    return out


def build_parquet_dataloader(
    train_config: TrainConfig,
    tokenizer: Optional[Any] = None,
    parquet_paths: Optional[List[PathOrStr]] = None,
    text_column: str = "text",
    use_streaming: bool = True,
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    split_by_node: bool = True,
) -> DataLoader:

    assert train_config.device_train_batch_size is not None
    
    # Get rank and world_size for DDP data splitting
    actual_rank = rank if rank is not None else get_global_rank()
    actual_world_size = world_size if world_size is not None else get_world_size()
    
    # Build dataset from parquet files using datasets library (faster than pandas/pyarrow)
    dataset = datasets.load_dataset("parquet",
            data_files={"train": train_config.dataset_path, 
                "validation": train_config.eval_dataset_path},
            split="train", streaming=True,)

    dataset = datasets.distributed.split_dataset_by_node(
        dataset,
        rank=int(os.environ['RANK']),
        world_size=int(os.environ["WORLD_SIZE"]))

    if train_config.load_path is not None:
        dataset = dataset.shuffle(random.randint(0, 2**4 - 1))

    # Handle streaming vs non-streaming modes differently
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    
    # Get max_length from train_config
    if train_config.model_type == "llama":
        max_length = train_config.llama_model.max_sequence_length
    elif train_config.model_type == "olmo":
        max_length = train_config.model.max_sequence_length
    else:
        max_length = 1024  # Default fallback
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    
    tokenizer.pad_token_id = 0

    # Use StreamingParquetIterableDataset (with or without tokenization)
    iterable_dataset = StreamingParquetIterableDataset(
        hf_dataset=dataset,
        text_column=text_column,
        tokenizer=tokenizer,
        batch_size=train_config.device_train_batch_size if tokenizer is not None else 1,
        max_length=max_length,
        rank=actual_rank,
        world_size=actual_world_size,
        seed=seed,
        epoch=train_config.epoch or 0,
    )

    out = DataLoader(
        iterable_dataset,
        batch_size=1,
        drop_last=train_config.data.drop_last,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
    dataloader_iterator = iter(out)
    first_dataloader_batch = next(dataloader_iterator)
    return out

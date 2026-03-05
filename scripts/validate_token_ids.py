#!/usr/bin/env python3
"""
Script to validate token IDs in data files to check if they exceed vocabulary size.
This helps diagnose CUDA "device-side assert triggered" errors.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def validate_token_ids(data_path: str, vocab_size: int, max_samples: int = 1000):
    """Validate that all token IDs in the data file are within [0, vocab_size)."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"Error: File {data_path} does not exist")
        return False
    
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path, mmap_mode='r')
    except Exception as e:
        print(f"Error loading file: {e}")
        return False
    
    print(f"Data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Valid token ID range: [0, {vocab_size})")
    print()
    
    # Check for invalid token IDs
    print("Checking for invalid token IDs...")
    invalid_mask = (data >= vocab_size) | (data < 0)
    num_invalid = invalid_mask.sum()
    
    if num_invalid > 0:
        print(f"❌ ERROR: Found {num_invalid} invalid token IDs!")
        invalid_indices = np.where(invalid_mask)
        invalid_values = data[invalid_mask]
        
        print(f"\nFirst 20 invalid token IDs:")
        for i, (idx, val) in enumerate(zip(zip(*invalid_indices)[:20], invalid_values[:20])):
            print(f"  Position {idx}: value {val}")
        
        print(f"\nInvalid token ID statistics:")
        print(f"  Min invalid value: {invalid_values.min()}")
        print(f"  Max invalid value: {invalid_values.max()}")
        print(f"  Values >= vocab_size: {(invalid_values >= vocab_size).sum()}")
        print(f"  Values < 0: {(invalid_values < 0).sum()}")
        
        return False
    else:
        print(f"✅ All token IDs are valid (within [0, {vocab_size}))")
        
        # Additional statistics
        print(f"\nToken ID statistics:")
        print(f"  Min: {data.min()}")
        print(f"  Max: {data.max()}")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Unique values: {len(np.unique(data))}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Validate token IDs in data files")
    parser.add_argument("data_path", type=str, help="Path to .npy data file")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size (default: 32000)")
    parser.add_argument("--max-samples", type=int, default=None, help="Number of samples to check (default: all)")
    
    args = parser.parse_args()
    
    is_valid = validate_token_ids(args.data_path, args.vocab_size, args.max_samples)
    
    if not is_valid:
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        print("The data file contains invalid token IDs that exceed the vocabulary size.")
        print("This will cause CUDA 'device-side assert triggered' errors during training.")
        print("\nPossible solutions:")
        print("1. Re-tokenize the data with the correct tokenizer")
        print("2. Check if the vocabulary size in config matches the tokenizer")
        print("3. Verify the data files were created with the same tokenizer as used in training")
        print("="*60)
        exit(1)
    else:
        print("\n✅ Data file validation passed!")
        exit(0)

if __name__ == "__main__":
    main()


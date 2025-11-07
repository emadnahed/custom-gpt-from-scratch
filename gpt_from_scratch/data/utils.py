"""
Data utility functions for loading prepared datasets and creating batches.
"""

import os
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import torch


def load_prepared_dataset(data_dir: str = 'data', split: str = 'train') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a previously prepared dataset

    Args:
        data_dir: Directory containing processed data
        split: Dataset split to load ('train' or 'val')
    """
    split_path = os.path.join(data_dir, f'{split}.bin')
    vocab_path = os.path.join(data_dir, 'vocab.pkl')

    data = np.fromfile(split_path, dtype=np.uint16)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    return data, vocab


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random batch from the dataset

    Args:
        data: Numpy array of token indices
        block_size: Context length
        batch_size: Number of sequences in batch
        device: Device to put tensors on
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

    if 'cuda' in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

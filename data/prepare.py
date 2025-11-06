"""
Data preparation and loading utilities for GPT training

This script provides:
1. DataLoader for streaming data during training
2. Functions to prepare datasets from Hugging Face
3. Character-level or BPE tokenization
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Simple text dataset for character-level or token-level training"""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of text
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def prepare_dataset_char_level(dataset_name='wikitext', split='train', data_dir='data'):
    """
    Prepare a dataset for character-level modeling

    Args:
        dataset_name: Name of the dataset (from Hugging Face)
        split: Dataset split ('train', 'validation', 'test')
        data_dir: Directory to save processed data
    """
    print(f"Preparing {dataset_name} dataset ({split} split)...")

    # Load dataset
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    elif dataset_name == 'openwebtext':
        dataset = load_dataset('openwebtext', split=split[:5])  # Use first 5% for memory
    elif dataset_name == 'shakespeare':
        # Download Shakespeare from a simple source
        with open('shakespeare.txt', 'r') as f:
            text = f.read()
        dataset = [{'text': text}]
    else:
        dataset = load_dataset(dataset_name, split=split)

    # Concatenate all text
    print("Concatenating text...")
    text = '\n'.join(dataset['text']) if hasattr(dataset, '__getitem__') else dataset

    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} characters")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the entire dataset
    print("Encoding text...")
    data = np.array([stoi[ch] for ch in text], dtype=np.uint16)

    # Save to disk
    os.makedirs(data_dir, exist_ok=True)
    split_path = os.path.join(data_dir, f'{split}.bin')
    vocab_path = os.path.join(data_dir, 'vocab.pkl')

    data.tofile(split_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

    print(f"Saved {len(data):,} tokens to {split_path}")
    print(f"Saved vocabulary to {vocab_path}")

    return data, stoi, itos, vocab_size


def load_prepared_dataset(data_dir='data', split='train'):
    """
    Load a previously prepared dataset

    Args:
        data_dir: Directory containing processed data
        split: Dataset split to load
    """
    split_path = os.path.join(data_dir, f'{split}.bin')
    vocab_path = os.path.join(data_dir, 'vocab.pkl')

    # Load data
    data = np.fromfile(split_path, dtype=np.uint16)

    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    return data, vocab


def get_batch(data, block_size, batch_size, device='cpu'):
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


class StreamingDataLoader:
    """
    Memory-efficient data loader that streams batches on-the-fly
    Useful for very large datasets
    """

    def __init__(self, data_path, block_size, batch_size, device='cpu'):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.current_pos = 0

    def next_batch(self):
        """Get the next batch of data"""
        return get_batch(self.data, self.block_size, self.batch_size, self.device)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


# Example: Prepare a small Shakespeare dataset for quick testing
def prepare_shakespeare(data_dir='data'):
    """Prepare a small Shakespeare dataset for testing"""

    # Download Shakespeare text
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    import urllib.request
    print("Downloading Shakespeare dataset...")

    os.makedirs(data_dir, exist_ok=True)
    shakespeare_path = os.path.join(data_dir, 'shakespeare.txt')

    if not os.path.exists(shakespeare_path):
        urllib.request.urlretrieve(url, shakespeare_path)

    with open(shakespeare_path, 'r') as f:
        text = f.read()

    # Split into train and validation
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text = text[int(n*0.9):]

    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} characters")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode train and val
    train_data = np.array([stoi[ch] for ch in train_text], dtype=np.uint16)
    val_data = np.array([stoi[ch] for ch in val_text], dtype=np.uint16)

    # Save to disk
    train_data.tofile(os.path.join(data_dir, 'train.bin'))
    val_data.tofile(os.path.join(data_dir, 'val.bin'))

    with open(os.path.join(data_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

    print(f"Train: {len(train_data):,} tokens")
    print(f"Val: {len(val_data):,} tokens")
    print("Dataset prepared successfully!")


if __name__ == '__main__':
    # Prepare Shakespeare dataset for quick testing
    prepare_shakespeare('data')

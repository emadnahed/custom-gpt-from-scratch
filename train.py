"""
Training script for GPT model

Usage:
    python train.py --config config/train_default.py

This script handles:
- Model initialization
- Training loop with gradient accumulation
- Evaluation and logging
- Checkpoint saving
- Learning rate scheduling
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model.transformer import GPT, GPTConfig, create_model
from data.prepare import get_batch, load_prepared_dataset


# Configuration
# =============================================================================
# Default configuration - can be overridden by config file or command line
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True

# Model
model_preset = 'small'  # or set individual parameters below
# block_size = 256
# vocab_size = 8192
# n_layer = 6
# n_head = 6
# n_kv_head = 3
# n_embd = 384

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 256

# Optimizer
learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 6e-5

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False

# =============================================================================

# Load configuration from file if specified
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('config/train_default.py').read())  # overrides from config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# Setup
# =============================================================================
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Auto-detect device
if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

print(f"Using device: {device}")

# Data loading
# =============================================================================
data_dir = os.path.join('data')

# Load data
train_data, vocab = load_prepared_dataset(data_dir, split='train')
val_data, _ = load_prepared_dataset(data_dir, split='val')

print(f"Train dataset: {len(train_data):,} tokens")
print(f"Val dataset: {len(val_data):,} tokens")

# Update vocab size from data
vocab_size = vocab['vocab_size']
print(f"Vocabulary size: {vocab_size}")


def get_train_batch():
    return get_batch(train_data, block_size, batch_size, device)


def get_val_batch():
    return get_batch(val_data, block_size, batch_size, device)


# Model initialization
# =============================================================================
print("\nInitializing model...")

# Create model config
if model_preset:
    model = create_model(model_preset)
    # Update vocab size to match data
    model.config.vocab_size = vocab_size
    # Reinitialize embedding and output layers with correct vocab size
    model.transformer.wte = torch.nn.Embedding(vocab_size, model.config.n_embd)
    model.lm_head = torch.nn.Linear(model.config.n_embd, vocab_size, bias=False)
    model.transformer.wte.weight = model.lm_head.weight
    model.apply(model._init_weights)
else:
    # Create model from individual config parameters
    model_config = GPTConfig(**{k: v for k, v in config.items() if k in GPTConfig.__annotations__})
    model_config.vocab_size = vocab_size
    model = GPT(model_config)

model.to(device)

# Compile model if requested
if compile_model:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Mixed precision training
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype))
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device_type == 'cuda'))


# Training utilities
# =============================================================================

def get_lr(it):
    """Learning rate schedule with warmup and cosine decay"""
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    """Evaluate the model on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_train_batch() if split == 'train' else get_val_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Training loop
# =============================================================================
print("\nStarting training...")
print(f"Total iterations: {max_iters}")
print(f"Batch size: {batch_size}")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
print(f"Device: {device}, dtype: {dtype}")
print("=" * 80)

X, Y = get_train_batch()  # Fetch first batch
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
best_val_loss = 1e9

for iter_num in range(max_iters):

    # Determine learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and log
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': model.config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'vocab': vocab,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # Training step
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # Get next batch
        X, Y = get_train_batch()

        # Backward pass
        scaler.scale(loss).backward()

    # Clip gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # Let first few iters stabilize
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    local_iter_num += 1

print("\nTraining complete!")

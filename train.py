"""
Training script for GPT model

Usage:
    python train.py --config config/train_default.py
    python train.py --interactive  # Interactive hardware selection
    python train.py --show-hardware  # Show available hardware and exit

This script handles:
- Model initialization
- Training loop with gradient accumulation
- Evaluation and logging
- Checkpoint saving
- Learning rate scheduling
- Automatic hardware detection (CUDA, ROCm, MPS, Intel XPU, CPU)
"""

import os
import sys
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model.transformer import GPT, GPTConfig, create_model
from data.prepare import get_batch, load_prepared_dataset
from utils.hardware_detector import (
    HardwareDetector,
    auto_detect_device,
    interactive_device_selection
)


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
device = 'auto'  # Set to 'auto' for automatic detection, or specify 'cuda', 'mps', 'cpu', etc.
dtype = 'auto'  # Set to 'auto' for automatic selection, or specify 'bfloat16', 'float16', 'float32'
compile_model = False
interactive_hardware = False  # Set to True to interactively select hardware

# =============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GPT model')
parser.add_argument('--config', type=str, default='config/train_default.py', help='Path to config file')
parser.add_argument('--interactive', action='store_true', help='Interactively select hardware')
parser.add_argument('--show-hardware', action='store_true', help='Show available hardware and exit')
args = parser.parse_args()

# Show hardware and exit if requested
if args.show_hardware:
    detector = HardwareDetector()
    detector.print_hardware_summary()
    sys.exit(0)

# Load configuration from file if specified
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# Load default config first
if os.path.exists('config/train_default.py'):
    import importlib.util
    spec = importlib.util.spec_from_file_location("default_config", 'config/train_default.py')
    default_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(default_config)
    # Update globals with default config values
    for key in dir(default_config):
        if not key.startswith('_') and isinstance(getattr(default_config, key), (int, float, bool, str)):
            globals()[key] = getattr(default_config, key)

# Override with custom config if specified
if os.path.exists(args.config) and args.config != 'config/train_default.py':
    import importlib.util
    import sys
    import os
    
    module_name = os.path.basename(args.config).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, args.config)
    custom_config = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = custom_config
    spec.loader.exec_module(custom_config)
    # Update globals with custom config values
    for key in dir(custom_config):
        if not key.startswith('_') and isinstance(getattr(custom_config, key), (int, float, bool, str)):
            globals()[key] = getattr(custom_config, key)

# Create config dictionary for logging
config = {k: globals()[k] for k in config_keys if k in globals()}

# Setup
# =============================================================================
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

# Hardware detection and selection
# =============================================================================
print("\n" + "="*80)
print("HARDWARE SETUP")
print("="*80)

if args.interactive or interactive_hardware:
    # Interactive hardware selection
    device, dtype = interactive_device_selection()
else:
    # Automatic hardware detection
    if device == 'auto':
        device, dtype_detected = auto_detect_device()
        print(f"Auto-detected device: {device}")

        # Use detected dtype if dtype is also set to auto
        if dtype == 'auto':
            dtype = dtype_detected
            print(f"Auto-selected dtype: {dtype}")
    else:
        print(f"Using configured device: {device}")

        # Auto-detect dtype if set to auto
        if dtype == 'auto':
            detector = HardwareDetector()
            for hw_device in detector.get_available_devices():
                if detector.get_device_string(hw_device) == device:
                    dtype = detector.get_optimal_dtype(hw_device)
                    print(f"Auto-selected dtype: {dtype}")
                    break

# Determine device type for mixed precision
device_type = 'cuda' if 'cuda' in device else ('cpu' if device == 'cpu' else device)

# Enable TF32 for CUDA devices (improves performance on Ampere+ GPUs)
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"\nFinal configuration:")
print(f"  Device: {device}")
print(f"  Device Type: {device_type}")
print(f"  Dtype: {dtype}")
print("="*80 + "\n")

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

# Mixed precision training setup
# Note: MPS doesn't support autocast yet (as of PyTorch 2.0), so we use nullcontext for MPS and CPU
# See: https://github.com/pytorch/pytorch/issues/77764
if device_type == 'cuda':
    # CUDA: Use autocast for automatic mixed precision (AMP)
    ctx = torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype))
    # Enable gradient scaling for float16 to prevent underflow
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
else:
    # MPS/CPU: Use default precision since autocast isn't supported
    ctx = nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # Disable for non-CUDA


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

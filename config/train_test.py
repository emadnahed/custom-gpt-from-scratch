"""
Quick test configuration - Just 5 iterations to verify everything works
"""

# Model configuration
model_preset = 'tiny'  # Tiny model trains fastest

# Data
dataset = 'shakespeare'
batch_size = 8  # Small batch for quick test
block_size = 128  # Match tiny model's block size

# Training - Very short test
max_iters = 5  # Just 5 iterations for testing
eval_interval = 5  # Evaluate at the end
eval_iters = 2  # Quick evaluation
log_interval = 1  # Log every iteration
always_save_checkpoint = False  # Don't save for test

# Optimizer
learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = False
warmup_iters = 0
lr_decay_iters = 5
min_lr = 3e-5

# System
device = 'cpu'  # Using CPU for stable test
dtype = 'float32'
compile_model = False

# Output
out_dir = 'out_test'

# Advanced
gradient_accumulation_steps = 1
gradient_checkpointing = False

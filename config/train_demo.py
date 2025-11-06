"""
Demo training configuration - Quick training to show the system working
"""

# Model configuration
model_preset = 'tiny'  # Tiny model trains fastest

# Data
dataset = 'shakespeare'
batch_size = 16  # Larger batch for faster training
block_size = 128  # Match tiny model's block size

# Training - Short demo
max_iters = 200  # Train for 200 iterations (quick demo)
eval_interval = 50  # Evaluate every 50 iterations
eval_iters = 10  # Quick evaluation
log_interval = 5  # Log frequently to see progress
always_save_checkpoint = True

# Optimizer
learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 50
lr_decay_iters = 500
min_lr = 3e-5

# System
device = 'cpu'  # Using CPU for stable demo (MPS has some compatibility issues)
dtype = 'float32'
compile_model = False  # Don't compile for demo

# Output
out_dir = 'out'

# Advanced
gradient_accumulation_steps = 1
gradient_checkpointing = False

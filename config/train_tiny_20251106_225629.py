"""
Training configuration generated on 2025-11-06 22:56:29
"""

# Model preset
model_preset = 'tiny'

# Dataset
dataset = 'shakespeare (current)'
batch_size = 16

# Training
max_iters = 200
eval_interval = 50
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

# Optimizer
learning_rate = 0.0003
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 20
lr_decay_iters = 200
min_lr = 2.9999999999999997e-05

# Hardware
device = 'cpu'
dtype = 'float32'
compile_model = False

# Advanced
gradient_accumulation_steps = 1
gradient_checkpointing = False

# Output
out_dir = 'out'

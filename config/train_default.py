"""
Default training configuration for GPT model
Modify these parameters based on your hardware and data
"""

# Model configuration
model_preset = 'small'  # Options: tiny, small, medium, large
# Or manually configure:
# block_size = 256
# vocab_size = 8192
# n_layer = 6
# n_head = 6
# n_kv_head = 3
# n_embd = 384

# Data
dataset = 'openwebtext'  # Dataset to use (from Hugging Face)
batch_size = 12  # Training batch size
block_size = 256  # Context length (must match model)

# Training
max_iters = 100  # Total training iterations
eval_interval = 50  # Evaluate every N iterations
eval_iters = 10  # Number of iterations for evaluation
log_interval = 10  # Log every N iterations
save_interval = 1000  # Save checkpoint every N iterations

# Optimizer
learning_rate = 3e-4  # Peak learning rate
weight_decay = 1e-1  # Weight decay
beta1 = 0.9  # Adam beta1
beta2 = 0.95  # Adam beta2
grad_clip = 1.0  # Gradient clipping

# Learning rate schedule
warmup_iters = 100  # Warmup iterations
lr_decay_iters = 5000  # Iterations to decay over
min_lr = 3e-5  # Minimum learning rate

# System
device = 'cpu'  # auto, cuda, cpu, mps (for Mac)
dtype = 'float32'  # float32, bfloat16, float16
compile_model = False  # Use torch.compile (requires PyTorch 2.0+)

# Output
out_dir = 'out'  # Output directory for checkpoints
wandb_log = False  # Log to Weights & Biases
wandb_project = 'custom-gpt'
wandb_run_name = 'run-' + str(max_iters)

# Advanced
gradient_accumulation_steps = 1  # Gradient accumulation
gradient_checkpointing = False  # Enable gradient checkpointing (saves memory)

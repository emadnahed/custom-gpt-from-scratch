"""
Interactive Configuration Builder for GPT Training

This makes it easy to create custom training configurations
"""

import os
from datetime import datetime
from pathlib import Path


def get_user_choice(prompt, options, default=None):
    """Get user choice from a list of options"""
    print(f"\n{prompt}")
    for i, (key, desc) in enumerate(options.items(), 1):
        default_marker = " (default)" if default and key == default else ""
        print(f"  [{i}] {key}: {desc}{default_marker}")

    while True:
        choice = input(f"\nEnter choice (1-{len(options)}) [default: {default or 1}]: ").strip()

        if not choice and default:
            return default

        if not choice:
            return list(options.keys())[0]

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return list(options.keys())[idx]
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def get_user_input(prompt, default=None, input_type=str):
    """Get user input with validation"""
    default_text = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{default_text}: ").strip()

        if not value and default is not None:
            return default

        if not value:
            print("Please enter a value")
            continue

        try:
            return input_type(value)
        except ValueError:
            print(f"Please enter a valid {input_type.__name__}")


def create_training_config(device=None, dtype=None, dataset=None):
    """
    Create a training configuration interactively
    Returns: path to created config file
    """
    print("\n" + "="*70)
    print("INTERACTIVE TRAINING CONFIGURATION BUILDER")
    print("="*70)

    config = {}

    # Model Architecture
    print("\n📦 MODEL ARCHITECTURE")
    print("-" * 70)

    use_preset = input("Use model preset? (y/n) [y]: ").strip().lower() != 'n'

    if use_preset:
        presets = {
            'tiny': 'Fast, ~6M params (good for testing)',
            'small': 'Balanced, ~25M params (good for laptops)',
            'medium': 'Powerful, ~80M params (needs good GPU)',
            'large': 'Maximum, ~350M params (needs lots of memory)',
        }
        config['model_preset'] = get_user_choice("Select model preset:", presets, 'tiny')
    else:
        print("\nCustom model architecture:")
        config['model_preset'] = None
        config['n_layer'] = get_user_input("Number of transformer layers", 6, int)
        config['n_head'] = get_user_input("Number of attention heads", 6, int)
        config['n_kv_head'] = get_user_input("Number of KV heads (for GQA)", 3, int)
        config['n_embd'] = get_user_input("Embedding dimension", 384, int)
        config['block_size'] = get_user_input("Context length (block size)", 256, int)
        config['mlp_ratio'] = get_user_input("MLP expansion ratio", 4.0, float)
        config['dropout'] = get_user_input("Dropout rate", 0.1, float)

    # Dataset
    print("\n📚 DATASET")
    print("-" * 70)
    config['dataset'] = dataset or 'shakespeare'
    config['batch_size'] = get_user_input("Batch size", 16, int)
    if 'block_size' not in config:
        config['block_size'] = get_user_input("Context length", 256, int)

    # Training Duration
    print("\n⏱️  TRAINING DURATION")
    print("-" * 70)

    duration_options = {
        'quick': 'Quick test (200 iterations, ~5 min)',
        'short': 'Short training (1000 iterations, ~20 min)',
        'medium': 'Medium training (5000 iterations, ~1-2 hours)',
        'long': 'Full training (20000+ iterations, several hours)',
        'custom': 'Custom (specify iterations)',
    }

    duration = get_user_choice("Select training duration:", duration_options, 'quick')

    if duration == 'quick':
        config['max_iters'] = 200
        config['eval_interval'] = 50
    elif duration == 'short':
        config['max_iters'] = 1000
        config['eval_interval'] = 200
    elif duration == 'medium':
        config['max_iters'] = 5000
        config['eval_interval'] = 500
    elif duration == 'long':
        config['max_iters'] = 20000
        config['eval_interval'] = 1000
    else:  # custom
        config['max_iters'] = get_user_input("Number of iterations", 500, int)
        config['eval_interval'] = get_user_input("Evaluate every N iterations", 100, int)

    config['eval_iters'] = get_user_input("Iterations for evaluation", 20, int)
    config['log_interval'] = get_user_input("Log every N iterations", 10, int)

    # Optimizer
    print("\n🎯 OPTIMIZER SETTINGS")
    print("-" * 70)

    use_default_optimizer = input("Use default optimizer settings? (y/n) [y]: ").strip().lower() != 'n'

    if use_default_optimizer:
        config['learning_rate'] = 3e-4
        config['weight_decay'] = 1e-1
        config['beta1'] = 0.9
        config['beta2'] = 0.95
        config['grad_clip'] = 1.0
    else:
        config['learning_rate'] = get_user_input("Learning rate", 3e-4, float)
        config['weight_decay'] = get_user_input("Weight decay", 1e-1, float)
        config['beta1'] = get_user_input("Adam beta1", 0.9, float)
        config['beta2'] = get_user_input("Adam beta2", 0.95, float)
        config['grad_clip'] = get_user_input("Gradient clipping", 1.0, float)

    # Learning rate schedule
    config['decay_lr'] = True
    config['warmup_iters'] = int(config['max_iters'] * 0.1)  # 10% warmup
    config['lr_decay_iters'] = config['max_iters']
    config['min_lr'] = config['learning_rate'] / 10

    # Hardware
    print("\n💻 HARDWARE SETTINGS")
    print("-" * 70)
    config['device'] = device or 'cpu'
    config['dtype'] = dtype or 'float32'
    config['compile_model'] = False

    print(f"Device: {config['device']}")
    print(f"Dtype: {config['dtype']}")

    # Advanced
    config['gradient_accumulation_steps'] = 1
    config['gradient_checkpointing'] = False
    config['always_save_checkpoint'] = True

    # Output
    config['out_dir'] = 'out'

    # Generate config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = f"train_{config.get('model_preset', 'custom')}_{timestamp}.py"
    config_path = Path('config') / config_name

    # Write config file
    write_config_file(config_path, config)

    print("\n" + "="*70)
    print(f"✓ Configuration saved to: {config_path}")
    print("="*70)

    return str(config_path)


def write_config_file(path, config):
    """Write configuration to Python file"""
    with open(path, 'w') as f:
        f.write('"""\n')
        f.write(f'Training configuration generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('"""\n\n')

        # Model configuration
        if config.get('model_preset'):
            f.write(f"# Model preset\n")
            f.write(f"model_preset = '{config['model_preset']}'\n\n")
        else:
            f.write(f"# Custom model architecture\n")
            f.write(f"model_preset = None\n")
            f.write(f"n_layer = {config['n_layer']}\n")
            f.write(f"n_head = {config['n_head']}\n")
            f.write(f"n_kv_head = {config['n_kv_head']}\n")
            f.write(f"n_embd = {config['n_embd']}\n")
            f.write(f"block_size = {config['block_size']}\n")
            f.write(f"mlp_ratio = {config['mlp_ratio']}\n")
            f.write(f"dropout = {config['dropout']}\n\n")

        # Data
        f.write(f"# Dataset\n")
        f.write(f"dataset = '{config['dataset']}'\n")
        f.write(f"batch_size = {config['batch_size']}\n")
        if 'block_size' in config and not config.get('model_preset'):
            f.write(f"block_size = {config['block_size']}\n")
        f.write("\n")

        # Training
        f.write(f"# Training\n")
        f.write(f"max_iters = {config['max_iters']}\n")
        f.write(f"eval_interval = {config['eval_interval']}\n")
        f.write(f"eval_iters = {config['eval_iters']}\n")
        f.write(f"log_interval = {config['log_interval']}\n")
        f.write(f"always_save_checkpoint = {config['always_save_checkpoint']}\n\n")

        # Optimizer
        f.write(f"# Optimizer\n")
        f.write(f"learning_rate = {config['learning_rate']}\n")
        f.write(f"weight_decay = {config['weight_decay']}\n")
        f.write(f"beta1 = {config['beta1']}\n")
        f.write(f"beta2 = {config['beta2']}\n")
        f.write(f"grad_clip = {config['grad_clip']}\n\n")

        # Learning rate schedule
        f.write(f"# Learning rate schedule\n")
        f.write(f"decay_lr = {config['decay_lr']}\n")
        f.write(f"warmup_iters = {config['warmup_iters']}\n")
        f.write(f"lr_decay_iters = {config['lr_decay_iters']}\n")
        f.write(f"min_lr = {config['min_lr']}\n\n")

        # Hardware
        f.write(f"# Hardware\n")
        f.write(f"device = '{config['device']}'\n")
        f.write(f"dtype = '{config['dtype']}'\n")
        f.write(f"compile_model = {config['compile_model']}\n\n")

        # Advanced
        f.write(f"# Advanced\n")
        f.write(f"gradient_accumulation_steps = {config['gradient_accumulation_steps']}\n")
        f.write(f"gradient_checkpointing = {config['gradient_checkpointing']}\n\n")

        # Output
        f.write(f"# Output\n")
        f.write(f"out_dir = '{config['out_dir']}'\n")


def interactive_config_builder():
    """Run the interactive config builder standalone"""
    try:
        from utils.hardware_detector import HardwareDetector, auto_detect_device

        # Auto-detect hardware
        device, dtype = auto_detect_device()

        # Handle MPS/CPU
        if device in ['mps', 'cpu']:
            dtype = 'float32'

        config_path = create_training_config(device, dtype)

        print(f"\n✓ Configuration ready!")
        print(f"\nTrain with: python train.py --config {config_path}")

    except ImportError:
        print("Error: Hardware detector not available")
        config_path = create_training_config()


if __name__ == '__main__':
    interactive_config_builder()

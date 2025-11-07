#!/usr/bin/env python3
"""
GPT Command Center - Your one-stop CLI for training and running GPT models

Usage (like npm scripts!):
    python gpt.py train          # Interactive training setup
    python gpt.py generate       # Generate text from trained model
    python gpt.py config         # Create/edit configuration
    python gpt.py dataset        # Manage datasets
    python gpt.py hardware       # Check available hardware
    python gpt.py info           # Show current setup info

This is your "package.json scripts" equivalent for GPT training!
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from utils.python_utils import is_in_virtualenv, get_venv_python

# Check Python version early
if sys.version_info < (3, 8):
    print("ERROR: Python 3.8 or higher is required")
    print(f"You are using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("\nPlease upgrade Python or use a newer version:")
    print("  python3.11 gpt.py <command>")
    sys.exit(1)

# Warn if not using recommended version
if sys.version_info < (3, 11):
    print(f"⚠️  Warning: Python 3.11+ is recommended (you have {sys.version_info.major}.{sys.version_info.minor})")
    print("   Your version will work, but consider upgrading for best compatibility.\n")

# Warn if not in virtual environment
if not is_in_virtualenv() and os.path.exists('venv'):
    print("⚠️  Warning: Virtual environment detected but not activated!")
    print("   For best results, activate it first:")
    print("   source venv/bin/activate  # Mac/Linux")
    print("   .\\venv\\Scripts\\activate  # Windows\n")


class Colors:
    """ANSI color codes for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print a fancy header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")




def run_python_script(script_path, args=None):
    """
    Run a Python script with the best available Python interpreter
    
    Args:
        script_path: Path to the Python script to run
        args: Optional list of command-line arguments to pass to the script
        
    Returns:
        bool: True if the script ran successfully, False otherwise
    """
    try:
        from utils.python_utils import get_venv_python
        python_cmd = get_venv_python()
    except ImportError:
        print_error("Could not import utils.python_utils")
        print_info("Make sure to install the package in development mode: pip install -e .")
        return False

    cmd = [python_cmd, script_path]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Script failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print_error(f"Python interpreter not found: {python_cmd}")
        print_info("Make sure virtual environment is set up: python3 -m venv venv")
        return False


def cmd_train(args):
    """Interactive training command"""
    print_header("GPT Training Setup")

    # Import here to avoid issues if deps not installed
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from utils.hardware_detector import HardwareDetector
        from config_builder import create_training_config

        # Step 1: Hardware selection
        print_info("Step 1: Hardware Selection")
        detector = HardwareDetector()

        available = detector.get_available_devices()
        print("\nAvailable hardware:")
        for i, device in enumerate(available):
            print(f"  [{i+1}] {device.type.value.upper()} - {device.name}")

        if len(available) > 1:
            choice = input(f"\nSelect hardware (1-{len(available)}) or Enter for recommended: ").strip()
            if choice:
                device = available[int(choice) - 1]
            else:
                device = detector.get_best_device()
        else:
            device = available[0]

        device_str = detector.get_device_string(device)
        dtype_str = detector.get_optimal_dtype(device)

        # Handle MPS/CPU specific dtype
        if device_str in ['mps', 'cpu']:
            dtype_str = 'float32'

        print_success(f"Selected: {device.type.value.upper()} with {dtype_str}")

        # Step 2: Dataset selection
        print_info("\nStep 2: Dataset Selection")
        datasets = list_available_datasets()

        if not datasets:
            print_warning("No datasets prepared yet!")
            prepare = input("Prepare Shakespeare dataset now? (y/n): ").strip().lower()
            if prepare == 'y':
                print_info("Preparing Shakespeare dataset...")
                python = get_venv_python()
                subprocess.run([python, 'data/prepare.py'], check=True)
                datasets = list_available_datasets()

        print("\nAvailable datasets:")
        for i, ds in enumerate(datasets):
            print(f"  [{i+1}] {ds}")

        dataset_choice = input(f"\nSelect dataset (1-{len(datasets)}) [default: 1]: ").strip()
        dataset = datasets[int(dataset_choice) - 1] if dataset_choice else datasets[0]
        print_success(f"Selected: {dataset}")

        # Step 3: Model configuration
        print_info("\nStep 3: Model Configuration")
        config = create_training_config(device_str, dtype_str, dataset)

        # Step 4: Start training
        print_info("\nStep 4: Starting Training")
        print(f"Configuration saved to: {config}")
        confirm = input("\nStart training now? (y/n): ").strip().lower()

        if confirm == 'y':
            run_python_script('train.py', ['--config', config])
        else:
            print_info(f"Train later with: python gpt.py train --config {config}")

    except ImportError as e:
        print_error(f"Missing dependencies: {e}")
        print_info("Install with: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print_info("\nTraining setup cancelled")


def cmd_generate(args):
    """Interactive generation command"""
    print_header("GPT Text Generation")

    # Check for checkpoint
    if not os.path.exists('out/ckpt.pt'):
        print_error("No trained model found!")
        print_info("Train a model first with: python gpt.py train")
        return

    # Run generation script with proper Python
    # This ensures it uses venv Python with all dependencies
    if not run_python_script('generate_interactive.py'):
        # Fallback to simple generation
        print_info("Trying simple generation...")
        run_python_script('generate_demo.py')


def cmd_config(args):
    """Create/edit configuration"""
    print_header("Configuration Builder")

    # Run as script to ensure proper Python with dependencies
    if not run_python_script('config_builder.py'):
        print_error("Config builder failed")
        print_info("Make sure dependencies are installed: pip install -r requirements.txt")


def cmd_dataset(args):
    """Manage datasets"""
    print_header("Dataset Manager")

    # Run as script to ensure proper Python with dependencies
    if not run_python_script('dataset_manager.py'):
        print_error("Dataset manager failed")
        print_info("Make sure dependencies are installed: pip install -r requirements.txt")


def cmd_hardware(args):
    """Check hardware"""
    print_header("Hardware Detection")
    run_python_script('check_hardware.py')


def cmd_info(args):
    """Show current setup info"""
    print_header("Current Setup Information")

    print(f"{Colors.BOLD}Project:{Colors.END} Custom GPT from Scratch")
    print(f"{Colors.BOLD}Location:{Colors.END} {os.getcwd()}")
    print()

    # Check virtual environment
    if os.path.exists('venv'):
        print_success("Virtual environment: Ready")
    else:
        print_error("Virtual environment: Not found")
        print_info("Create with: python3 -m venv venv")

    # Check datasets
    datasets = list_available_datasets()
    if datasets:
        print_success(f"Datasets: {len(datasets)} available")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        print_warning("Datasets: None prepared")
        print_info("Prepare with: cd data && python prepare.py")

    # Check checkpoints
    if os.path.exists('out/ckpt.pt'):
        size = os.path.getsize('out/ckpt.pt') / (1024 * 1024)
        print_success(f"Trained model: Yes ({size:.1f} MB)")
    else:
        print_warning("Trained model: None")
        print_info("Train with: python gpt.py train")

    # Check configs
    configs = list(Path('config').glob('*.py'))
    print_success(f"Configurations: {len(configs)} available")
    for cfg in configs:
        print(f"  - {cfg.name}")


def list_available_datasets():
    """List available prepared datasets"""
    datasets = []
    data_dir = Path('data')

    if (data_dir / 'train.bin').exists():
        datasets.append('shakespeare (current)')

    return datasets if datasets else []


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='GPT Command Center - Easy training and generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpt.py train              # Interactive training setup
  python gpt.py generate           # Generate text
  python gpt.py config             # Build a configuration
  python gpt.py hardware           # Check available hardware
  python gpt.py info               # Show setup information

Think of this as your 'npm run' for GPT models!
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Commands (like npm scripts!)
    subparsers.add_parser('train', help='Start interactive training')
    subparsers.add_parser('generate', help='Generate text from trained model')
    subparsers.add_parser('config', help='Create/edit configuration')
    subparsers.add_parser('dataset', help='Manage datasets')
    subparsers.add_parser('hardware', help='Check available hardware')
    subparsers.add_parser('info', help='Show current setup')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print(f"\n{Colors.BOLD}Quick Start:{Colors.END}")
        print("  python gpt.py info       # Check your setup")
        print("  python gpt.py train      # Start training")
        print("  python gpt.py generate   # Generate text")
        return

    # Route to appropriate command
    commands = {
        'train': cmd_train,
        'generate': cmd_generate,
        'config': cmd_config,
        'dataset': cmd_dataset,
        'hardware': cmd_hardware,
        'info': cmd_info,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        print_error(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == '__main__':
    main()

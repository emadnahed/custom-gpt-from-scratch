""
Command-line interface for GPT from Scratch
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

from .utils.python_utils import is_in_virtualenv, get_venv_python

class Colors:
    """ANSI color codes for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str) -> None:
    """Print a fancy header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}» {text.upper()}{Colors.ENDC}\n")

def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}", file=sys.stderr)

def print_info(text: str) -> None:
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def print_warning(text: str) -> None:
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def run_python_script(script_path: str, args: Optional[List[str]] = None) -> int:
    """Run a Python script with the best available Python interpreter"""
    python = get_venv_python()
    cmd = [python, script_path]
    if args:
        cmd.extend(args)
    
    try:
        return subprocess.run(cmd, check=True).returncode
    except subprocess.CalledProcessError as e:
        return e.returncode

def cmd_train(args: argparse.Namespace) -> int:
    """Interactive training command"""
    print_header("Starting Training")
    from .train import main as train_main
    return train_main(args)

def cmd_generate(args: argparse.Namespace) -> int:
    """Interactive generation command"""
    print_header("Text Generation")
    from .generate import main as generate_main
    return generate_main(args)

def cmd_config(args: argparse.Namespace) -> int:
    """Create/edit configuration"""
    print_header("Configuration")
    from .config_builder import main as config_main
    return config_main(args)

def cmd_dataset(args: argparse.Namespace) -> int:
    """Manage datasets"""
    print_header("Dataset Management")
    from .dataset_manager import main as dataset_main
    return dataset_main(args)

def cmd_hardware(args: argparse.Namespace) -> int:
    """Check hardware"""
    from .utils.hardware_detector import HardwareDetector
    detector = HardwareDetector()
    detector.print_hardware_summary()
    return 0

def cmd_info(args: argparse.Namespace) -> int:
    """Show current setup info"""
    print_header("System Information")
    
    # Python info
    print(f"Python: {sys.version.split()[0]}")
    print(f"Virtual Environment: {'Active' if is_in_virtualenv() else 'Not Active'}")
    
    # Check PyTorch
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nPyTorch is not installed")
    
    return 0

def main() -> int:
    """Main CLI entry point"""
    # Check Python version early
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        print(f"You are using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print("\nPlease upgrade Python or use a newer version:")
        print("  python3.11 -m pip install -e .")
        return 1

    # Warn if not using recommended version
    if sys.version_info < (3, 11):
        print_warning(f"Python 3.11+ is recommended (you have {sys.version_info.major}.{sys.version_info.minor})")
        print("   Your version will work, but consider upgrading for best compatibility.\n")

    # Warn if not in virtual environment
    if not is_in_virtualenv() and os.path.exists('venv'):
        print_warning("Virtual environment detected but not activated!")
        print("   For best results, activate it first:")
        print("   source venv/bin/activate  # Mac/Linux")
        print("   .\\venv\\Scripts\\activate  # Windows\n")

    parser = argparse.ArgumentParser(description="GPT from Scratch - Train and run GPT models")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from a trained model")
    gen_parser.add_argument("--model", type=str, help="Path to model checkpoint")
    gen_parser.add_argument("--prompt", type=str, help="Input prompt")
    gen_parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")

    # Config command
    config_parser = subparsers.add_parser("config", help="Create or edit configuration")
    config_parser.add_argument("action", choices=["new", "edit"], help="Action to perform")
    config_parser.add_argument("--name", type=str, help="Configuration name")

    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Manage datasets")
    dataset_parser.add_argument("action", choices=["list", "prepare", "info"], help="Action to perform")
    dataset_parser.add_argument("--name", type=str, help="Dataset name")
    dataset_parser.add_argument("--path", type=str, help="Path to dataset")

    # Hardware command
    subparsers.add_parser("hardware", help="Show hardware information")
    
    # Info command
    subparsers.add_parser("info", help="Show system and environment information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    command_handlers = {
        "train": cmd_train,
        "generate": cmd_generate,
        "config": cmd_config,
        "dataset": cmd_dataset,
        "hardware": cmd_hardware,
        "info": cmd_info,
    }

    handler = command_handlers.get(args.command)
    if not handler:
        print_error(f"Unknown command: {args.command}")
        parser.print_help()
        return 1

    return handler(args)

if __name__ == "__main__":
    sys.exit(main())

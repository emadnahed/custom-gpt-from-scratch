"""
Dataset Manager - Easy dataset preparation and switching

Makes it easy to:
- Prepare new datasets
- Switch between datasets
- Add custom text files
- Use Hugging Face datasets
"""

import os
import sys
import subprocess
from pathlib import Path

# Import utility for getting correct Python executable
def get_python_for_scripts():
    """
    Get the best Python executable to use for running scripts
    Fallback implementation if utils not available
    """
    import os

    # Check if we're in a virtualenv
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV')
    )

    # If in venv, use current Python
    if in_venv:
        return sys.executable, None

    # If venv exists but not activated, use it
    venv_paths = [
        'venv/bin/python',
        'venv/Scripts/python.exe',
        '.venv/bin/python',
        '.venv/Scripts/python.exe',
    ]

    for venv_path in venv_paths:
        if os.path.exists(venv_path):
            return os.path.abspath(venv_path), None

    # No venv available - warn user
    warning = (
        "⚠️  Virtual environment not detected!\n"
        "   For best results, activate the virtual environment:\n"
        "   source venv/bin/activate  (Mac/Linux)\n"
        "   venv\\Scripts\\activate     (Windows)"
    )

    return sys.executable, warning


def print_header(text):
    """Print a fancy header"""
    print(f"\n{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}\n")


def list_datasets():
    """List all available datasets"""
    data_dir = Path('data')
    datasets = []

    # Check for prepared datasets
    if (data_dir / 'train.bin').exists():
        size = (data_dir / 'train.bin').stat().st_size / (1024 * 1024)
        datasets.append(('shakespeare', f'Current dataset ({size:.1f} MB)'))

    # Check for custom text files
    for txt_file in data_dir.glob('*.txt'):
        if txt_file.name != 'input.txt':  # Skip the prepared file
            size = txt_file.stat().st_size / 1024
            datasets.append((txt_file.stem, f'Text file ({size:.1f} KB)'))

    return datasets


def prepare_shakespeare():
    """Prepare the Shakespeare dataset"""
    print("Preparing Shakespeare dataset...")
    print("This will download ~1MB of text and create train/val splits.\n")

    # Get correct Python executable (venv if available)
    python_cmd, warning = get_python_for_scripts()

    if warning:
        print(warning)
        print()

    result = subprocess.run(
        [python_cmd, 'data/prepare.py'],
        capture_output=False
    )

    if result.returncode == 0:
        print("\n✓ Shakespeare dataset prepared successfully!")
    else:
        print("\n✗ Failed to prepare dataset")
        print("Make sure dependencies are installed:")
        print("  pip install -r requirements.txt")


def prepare_custom_text():
    """Prepare a custom text file"""
    print("Add Custom Text File")
    print("-" * 70)
    print("\nOptions:")
    print("  1. Copy an existing text file to data/ directory")
    print("  2. Enter text file path to import")
    print("  3. Enter text manually (for small datasets)")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == '1':
        print("\n✓ Copy your .txt file to the 'data/' directory")
        print("  Then run this again to prepare it")

    elif choice == '2':
        file_path = input("\nEnter path to text file: ").strip()
        if not os.path.exists(file_path):
            print("✗ File not found!")
            return

        # Copy to data directory
        import shutil
        dest = Path('data') / Path(file_path).name
        shutil.copy(file_path, dest)
        print(f"✓ Copied to {dest}")

        # Prepare it
        prepare = input("Prepare this dataset now? (y/n): ").strip().lower()
        if prepare == 'y':
            prepare_from_file(dest)

    elif choice == '3':
        print("\nEnter your text (press Ctrl+D when done):")
        print("-" * 70)
        try:
            lines = []
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        text = '\n'.join(lines)
        name = input("\nDataset name: ").strip() or "custom"

        output_file = Path('data') / f"{name}.txt"
        with open(output_file, 'w') as f:
            f.write(text)

        print(f"✓ Saved to {output_file}")

        prepare = input("Prepare this dataset now? (y/n): ").strip().lower()
        if prepare == 'y':
            prepare_from_file(output_file)


def prepare_from_file(file_path):
    """Prepare dataset from a text file"""
    print(f"\nPreparing dataset from {file_path}...")

    # Simple preparation script (creates train.bin and val.bin)
    code = f"""
import os
import numpy as np

# Read text
with open('{file_path}', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {{vocab_size}} characters")

# Create mappings
stoi = {{ch: i for i, ch in enumerate(chars)}}
itos = {{i: ch for i, ch in enumerate(chars)}}

# Encode
data = np.array([stoi[c] for c in text], dtype=np.uint16)

# Split train/val (90/10)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Save
train_data.tofile('data/train.bin')
val_data.tofile('data/val.bin')

# Save vocab
import pickle
with open('data/meta.pkl', 'wb') as f:
    pickle.dump({{'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}}, f)

print(f"Train: {{len(train_data):,}} tokens")
print(f"Val: {{len(val_data):,}} tokens")
print("✓ Dataset prepared!")
"""

    # Run preparation
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_script = f.name

    # Get correct Python executable
    python_cmd, warning = get_python_for_scripts()

    if warning:
        print(warning)
        print()

    try:
        subprocess.run([python_cmd, temp_script], check=True)
        os.unlink(temp_script)
        print("\n✓ Dataset prepared successfully!")
    except subprocess.CalledProcessError:
        print("\n✗ Failed to prepare dataset")
        print("Make sure dependencies are installed:")
        print("  pip install -r requirements.txt")
        os.unlink(temp_script)


def manage_datasets():
    """Main dataset management interface"""
    print_header("DATASET MANAGER")

    while True:
        print("\nDataset Options:")
        print("  [1] List available datasets")
        print("  [2] Prepare Shakespeare dataset")
        print("  [3] Add custom text file")
        print("  [4] Info about current dataset")
        print("  [5] Back to main menu")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == '1':
            datasets = list_datasets()
            if datasets:
                print("\nAvailable datasets:")
                for name, desc in datasets:
                    print(f"  - {name}: {desc}")
            else:
                print("\n⚠ No datasets prepared yet")
                print("Prepare one with option 2 or 3")

        elif choice == '2':
            prepare_shakespeare()

        elif choice == '3':
            prepare_custom_text()

        elif choice == '4':
            # Show info about current dataset
            if os.path.exists('data/train.bin'):
                train_size = os.path.getsize('data/train.bin')
                val_size = os.path.getsize('data/val.bin')

                # Calculate tokens (assuming uint16)
                train_tokens = train_size // 2
                val_tokens = val_size // 2

                print("\nCurrent Dataset Information:")
                print(f"  Training tokens: {train_tokens:,}")
                print(f"  Validation tokens: {val_tokens:,}")
                print(f"  Total: {train_tokens + val_tokens:,} tokens")
                print(f"  Size: {(train_size + val_size)/(1024*1024):.2f} MB")

                # Try to load vocab info
                try:
                    import pickle
                    with open('data/meta.pkl', 'rb') as f:
                        meta = pickle.load(f)
                        print(f"  Vocabulary size: {meta['vocab_size']}")
                except:
                    pass
            else:
                print("\n⚠ No dataset prepared")

        elif choice == '5':
            break

        else:
            print("Invalid option")


if __name__ == '__main__':
    manage_datasets()

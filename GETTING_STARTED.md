# Getting Started with GPT Training

Welcome! This guide will help you get started with training your GPT model. If you're coming from a JavaScript/React background, think of this as similar to setting up and running a React project, but for machine learning.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Hardware Setup](#hardware-setup)
4. [Data Preparation](#data-preparation)
5. [Training Your First Model](#training-your-first-model)
6. [Monitoring Training](#monitoring-training)
7. [Common Issues](#common-issues)
8. [Next Steps](#next-steps)

---

## Prerequisites

### Python Installation

You'll need Python 3.8 or later. Check if you have Python installed:

```bash
python --version
# or
python3 --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/).

### Comparison with JavaScript/React

| JavaScript/React | Python/ML |
|-----------------|-----------|
| `node` / `npm` / `yarn` | `python` / `pip` |
| `package.json` | `requirements.txt` |
| `node_modules/` | `venv/` (virtual environment) |
| `npm install` | `pip install` |
| `npm start` | `python train.py` |

---

## Installation

### Step 1: Create a Virtual Environment

A virtual environment is like `node_modules` - it keeps your project dependencies isolated.

```bash
# Create virtual environment (like npm init)
python3 -m venv venv

# Activate it (like entering your project directory)
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when activated.

### Step 2: Install Dependencies

This is similar to running `yarn install` or `npm install`.

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

This will install:
- `torch` - The deep learning framework (like React for ML)
- `numpy` - Numerical computing library
- `tqdm` - Progress bars for training
- `datasets` - For loading datasets

**Hardware-Specific PyTorch:**

The default PyTorch installation supports CPU and NVIDIA CUDA. For other hardware:

- **Apple Silicon (M1/M2/M3)**: The default installation includes MPS (Metal) support
- **AMD GPUs (ROCm)**: Visit [PyTorch ROCm](https://pytorch.org/get-started/locally/) for installation
- **Intel GPUs (XPU)**: Install [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Hardware Setup

### Check Available Hardware

First, let's see what hardware you have available:

```bash
python check_hardware.py
```

This will show:
- ✓ Available hardware (green)
- ✗ Unavailable hardware (greyed out)
- Recommended device and settings
- Supported precision (float16, bfloat16, float32)

**Example output:**
```
======================================================================
HARDWARE DETECTION SUMMARY
======================================================================

[1] CUDA: ✓ AVAILABLE
    Name: NVIDIA GeForce RTX 3080
    Device Count: 1
    Memory: 10.0 GB
    Compute Capability: 8.6
    Supported Precisions: bfloat16, float16, float32
    Details: 1 device(s), 10.0 GB memory

[2] ROCM: ✗ UNAVAILABLE
    Name: AMD ROCm
    Details: ROCm not available or no AMD GPU detected

[3] MPS: ✗ UNAVAILABLE
    Name: Apple Metal (MPS)
    Details: Not on macOS

[4] XPU: ✗ UNAVAILABLE
    Name: Intel XPU
    Details: Intel Extension for PyTorch not installed

[5] CPU: ✓ AVAILABLE
    Name: CPU (x86_64)
    Supported Precisions: float16, float32
    Details: Darwin - x86_64

======================================================================
RECOMMENDED: CUDA - NVIDIA GeForce RTX 3080
Device String: cuda
Optimal Dtype: bfloat16
======================================================================
```

### Interactive Hardware Selection

If you have multiple hardware options, you can interactively select one:

```bash
python check_hardware.py --interactive
```

---

## Data Preparation

### Using the Shakespeare Dataset (Default)

The project comes with a data preparation script. Let's prepare the Shakespeare dataset:

```bash
# Navigate to the data directory
cd data

# Prepare the Shakespeare dataset (this is the default)
python prepare.py

# Go back to the project root
cd ..
```

This will:
1. Download the Shakespeare text dataset
2. Tokenize it
3. Split it into train/validation sets
4. Save it as `train.bin` and `val.bin`

### Using Your Own Dataset

You can modify `data/prepare.py` to use your own text data. The script should:
1. Load your text
2. Tokenize it
3. Save as train/val binary files

---

## Training Your First Model

Now you're ready to train! This is like running `yarn start` or `npm start`.

### Quick Start (Automatic Hardware Detection)

```bash
python train.py
```

The script will:
1. Auto-detect your best available hardware
2. Load the prepared dataset
3. Initialize the GPT model
4. Start training
5. Save checkpoints to the `out/` directory

### Training with Interactive Hardware Selection

```bash
python train.py --interactive
```

This lets you choose which hardware to use if you have multiple options.

### View Available Hardware

```bash
python train.py --show-hardware
```

This shows all available hardware and exits without training.

### Custom Configuration

You can modify training parameters in `config/train_default.py` or create your own config file:

```bash
python train.py --config config/my_config.py
```

**Key parameters to adjust:**

```python
# Model size
model_preset = 'small'  # Options: 'small', 'medium', 'large'

# Training
batch_size = 12          # Increase if you have more GPU memory
max_iters = 5000         # Total training steps
learning_rate = 6e-4     # Learning rate

# Data
dataset = 'shakespeare'  # Your dataset name
block_size = 256         # Context length (sequence length)

# Hardware (auto-detected by default)
device = 'auto'          # or 'cuda', 'mps', 'cpu'
dtype = 'auto'           # or 'bfloat16', 'float16', 'float32'
```

---

## Monitoring Training

### Understanding the Output

During training, you'll see logs like this:

```
iter 0: loss 4.2345, time 123.45ms, mfu 0.00%
iter 1: loss 4.1234, time 98.76ms, mfu 12.34%
...
step 2000: train loss 1.2345, val loss 1.3456
Saving checkpoint to out
```

**What do these mean?**

- `iter`: Current training iteration (like a step counter)
- `loss`: How wrong the model's predictions are (lower is better)
- `time`: Time per iteration
- `mfu`: Model FLOPs Utilization (GPU efficiency, higher is better)
- `train loss` / `val loss`: Performance on training and validation data

### When is Training Complete?

Training completes when:
1. You reach `max_iters` (defined in config)
2. You manually stop it (Ctrl+C)
3. The validation loss stops improving

### Checkpoints

The model is saved to `out/ckpt.pt` periodically. This is like a save file - you can resume training or use it for inference later.

---

## Common Issues

### Issue 1: Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solution:** Reduce `batch_size` in your config:

```python
batch_size = 8  # Try smaller values: 4, 2, or even 1
```

### Issue 2: CUDA Not Available

**Error:** Device shows as CPU even though you have a GPU

**Solution:**
1. Verify CUDA installation: `nvidia-smi` (for NVIDIA GPUs)
2. Reinstall PyTorch with CUDA support: Visit [pytorch.org](https://pytorch.org)
3. Check that your GPU drivers are up to date

### Issue 3: MPS Not Available (Mac)

**Error:** MPS not detected on Apple Silicon

**Solution:**
- Requires macOS 12.3+ and Apple Silicon (M1/M2/M3)
- Update to the latest macOS version
- Ensure you have the latest PyTorch: `pip install --upgrade torch`

### Issue 4: Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
1. Make sure your virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`

### Issue 5: Data Not Found

**Error:** `FileNotFoundError: data/train.bin not found`

**Solution:**
Run the data preparation script:
```bash
cd data
python prepare.py
cd ..
```

---

## Next Steps

### 1. Generate Text

After training, generate text with your model:

```bash
python generate.py --checkpoint out/ckpt.pt --prompt "Once upon a time"
```

### 2. Experiment with Model Sizes

Try different model presets in `config/train_default.py`:

```python
model_preset = 'small'   # Fastest, least memory
model_preset = 'medium'  # Balanced
model_preset = 'large'   # Best quality, more resources
```

### 3. Fine-tune on Your Own Data

1. Prepare your text data
2. Modify `data/prepare.py` to load your data
3. Run training with your dataset

### 4. Distributed Training (Multiple GPUs)

If you have multiple GPUs, you can train faster with distributed training. Check the `README.md` for more details.

---

## Quick Reference

```bash
# Virtual Environment
python3 -m venv venv                  # Create
source venv/bin/activate              # Activate (Mac/Linux)
venv\Scripts\activate                 # Activate (Windows)
deactivate                            # Deactivate

# Installation
pip install -r requirements.txt       # Install dependencies
pip list                              # List installed packages

# Hardware
python check_hardware.py              # Check hardware
python check_hardware.py --interactive # Interactive selection
python check_hardware.py --json       # JSON output

# Data Preparation
cd data && python prepare.py && cd .. # Prepare dataset

# Training
python train.py                       # Auto-detect hardware
python train.py --interactive         # Choose hardware
python train.py --show-hardware       # Show hardware options
python train.py --config config.py    # Custom config

# Monitoring
# Watch the loss values decrease over time
# Lower loss = better model
# Training time depends on hardware and model size
```

---

## Help and Resources

- **Project README**: Check `README.md` for more details
- **Configuration**: See `config/train_default.py` for all options
- **Hardware Issues**: Run `python check_hardware.py` for diagnostics

Happy training! 🚀

# Getting Started with GPT Training

Complete guide for training your GPT model from scratch.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Hardware Setup](#hardware-setup)
4. [Training Your Model](#training-your-model)
5. [Generating Text](#generating-text)
6. [Common Commands](#common-commands)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Fastest Way to Get Started

**One-Command Setup:**
```bash
./setup.sh           # Mac/Linux
setup.bat            # Windows
```

This automatically:
- ✅ Checks/installs Python
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Detects your hardware
- ✅ Prepares dataset
- ✅ Gets you ready to train!

**Start Training:**
```bash
source venv/bin/activate   # Activate venv (recommended)
python gpt.py train        # Start interactive training
```

That's it! ✅

---

## 🛠️ Installation

### Prerequisites

**Python Requirements:**
- **Recommended:** Python 3.11
- **Minimum:** Python 3.8

**Check your Python version:**
```bash
python3 --version
python check_python_version.py  # Detailed version check
```

**If you have multiple Python versions:**
```bash
# Use specific version
python3.11 -m venv venv
python3.11 gpt.py info
```

### Automated Setup (Recommended)

```bash
# Mac/Linux
./setup.sh

# Windows
setup.bat
```

The script will:
1. Check Python version (installs if missing on Mac)
2. Create virtual environment
3. Install dependencies (PyTorch, numpy, etc.)
4. Detect hardware (CPU, GPU, MPS)
5. Prepare Shakespeare dataset
6. Offer to start training

### Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare dataset
cd data && python prepare.py && cd ..

# 5. Check setup
python gpt.py info
```

---

## 🖥️ Hardware Setup

### Check Available Hardware

```bash
python gpt.py hardware
```

**Output shows:**
- ✅ **Available** (green) - Can use this hardware
- ✗ **Unavailable** (grey) - Not available
- **Recommended** - Best option for your system

### Supported Hardware

| Hardware | Description | Speed | Your Mac |
|----------|-------------|-------|----------|
| **CUDA** | NVIDIA GPU | Fastest | ✗ Not available |
| **ROCm** | AMD GPU | Fast | ✗ Not available |
| **MPS** | Apple Silicon GPU | Medium | ✅ Available |
| **CPU** | Universal fallback | Slowest | ✅ Available |

**Your Mac will use:** MPS (Apple Metal) if you have Apple Silicon, or CPU

### Virtual Environment Usage

**Important:** The system now works whether you activate venv or not!

**Option 1: Activate venv (Recommended - cleaner output)**
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Now use commands
python gpt.py train
```

**Option 2: Don't activate (Still works!)**
```bash
# System automatically uses venv Python
python3 gpt.py train

# You'll see a friendly warning, but it works!
```

---

## 🎓 Training Your Model

### Interactive Training (Easiest)

```bash
python gpt.py train
```

**You'll be asked:**

1. **Hardware?** (Auto-selected: MPS or CPU)
2. **Dataset?** (Shakespeare, or your own)
3. **Model size?**
   - `tiny` - Fast, ~6M params (good for testing)
   - `small` - Balanced, ~25M params (good for laptops)
   - `medium` - Powerful, ~80M params (needs good GPU)
   - `large` - Maximum, ~350M params (needs lots of memory)
4. **Number of layers?** (4, 8, 12, 24, or custom)
5. **How long?**
   - `quick` - 200 iterations (~5 min)
   - `short` - 1000 iterations (~20 min)
   - `medium` - 5000 iterations (~1-2 hours)
   - `long` - 20000+ iterations (several hours)
6. **Start now?** (yes!)

### Custom Configuration

Create a custom configuration:
```bash
python gpt.py config
```

**You can customize:**
- Number of layers (more = better quality)
- Model architecture (heads, embedding size)
- Training duration
- Learning rate and optimizer settings
- Hardware and precision

**Adjusting Layers:**
```python
n_layer = 4    # Fast, okay results
n_layer = 8    # Balanced
n_layer = 12   # Good quality (recommended)
n_layer = 24   # Best quality (slow on CPU)
```

### Training with Specific Config

```bash
python train.py --config config/my_config.py
```

### Understanding Training Output

During training you'll see:
```
iter 0: loss 4.2345, time 123.45ms, mfu 0.00%
iter 1: loss 4.1234, time 98.76ms, mfu 12.34%
...
step 2000: train loss 1.2345, val loss 1.3456
```

**What these mean:**
- `iter` - Current iteration (step number)
- `loss` - How wrong predictions are (lower = better)
- `time` - Time per iteration
- `mfu` - GPU efficiency (higher = better)
- `train loss` / `val loss` - Performance on train/validation data

**Good progress:**
- Loss starts around 4.0
- Should decrease to ~2.0 (good!)
- Below 1.5 is excellent!

**Checkpoints:**
- Saved to `out/ckpt.pt` automatically
- You can stop (Ctrl+C) and resume later

---

## 🎨 Generating Text

### Interactive Generation

```bash
python gpt.py generate
```

**Options:**
- Enter your prompt (or press Enter for random)
- Set max tokens (default: 200)
- Set temperature (0.1-2.0, default: 0.8)
- Set top-k filtering (default: 200)

### Temperature Guide

```
0.1 - 0.5   # Conservative, coherent (factual text)
0.6 - 0.9   # Balanced (recommended)
1.0 - 2.0   # Creative, random (poetry/fiction)
```

### Simple Generation Script

```bash
python generate_demo.py
```

Shows multiple samples with different prompts.

---

## 📝 Common Commands

### Main Commands (Like npm scripts!)

```bash
# Most used commands
python gpt.py train          # Train a model
python gpt.py generate       # Generate text
python gpt.py info           # Check setup status
python gpt.py hardware       # View hardware options

# Management commands
python gpt.py config         # Create configurations
python gpt.py dataset        # Manage datasets
```

### Dataset Management

```bash
python gpt.py dataset
```

**Options:**
1. List available datasets
2. Prepare Shakespeare dataset
3. Add your own text file
4. View dataset info

**Adding your own dataset:**
1. Copy your .txt file to `data/` directory
2. Run `python gpt.py dataset`
3. Choose option 3 (Add custom text file)
4. Follow prompts to prepare it

### Traditional Commands (Still Work)

```bash
# Training
python train.py --config config/train_demo.py

# Generation
python generate_demo.py

# Hardware check
python check_hardware.py

# Check Python version
python check_python_version.py
```

---

## 🔧 Troubleshooting

### Common Issues

#### "Out of Memory"
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `batch_size` in your config
- Use smaller model (`tiny` or `small`)
- Or add this to config:
```python
batch_size = 4  # or even 2, 1
model_preset = 'tiny'
```

#### "No module named 'torch'"

**Solution 1: Activate venv**
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**Solution 2: Reinstall dependencies**
```bash
pip install -r requirements.txt
```

**Note:** If using `python3 gpt.py` (without activating venv), this should now work automatically!

#### "MPS errors on Mac"

**Solution:** Use CPU instead (more stable)
```bash
# In your config or when prompted:
device = 'cpu'
```

#### "Dataset not found"

**Solution:**
```bash
python gpt.py dataset
# Choose option 2 to prepare Shakespeare
```

#### "Virtual environment not activated"

You'll see a warning, but commands still work!

**To remove warning:**
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### Training is Slow

**Expected times (CPU):**
- Quick test (200 iter): ~5 minutes
- Short (1000 iter): ~20 minutes
- Medium (5000 iter): ~2 hours

**To speed up:**
1. Use smaller model (`tiny`)
2. Reduce iterations
3. Use GPU if available (MPS on Mac)

### Loss Not Decreasing

**If loss stays high:**
1. Train longer (more iterations)
2. Check learning rate (try 3e-4 to 6e-4)
3. Use larger model
4. Check dataset is prepared correctly

---

## 💡 Tips & Best Practices

### Tip 1: Start Small

```bash
# First time? Use tiny model with quick training
python gpt.py train
# Choose: tiny, quick, 200 iterations
# See results in 5 minutes!
```

### Tip 2: Watch the Loss

During training, loss should decrease:
- Start: ~4.0
- After 500 iter: ~2.5
- After 2000 iter: ~1.8-2.0
- Good results: < 1.5

### Tip 3: Experiment with Layers

```python
# More layers = better but slower
n_layer = 8   # Good starting point
n_layer = 12  # Better quality
n_layer = 24  # Best (needs good hardware)
```

### Tip 4: Save Multiple Configs

```bash
# Create and save different configs
python gpt.py config

# Name them descriptively:
config/train_quick_test.py      # 200 iterations
config/train_production.py      # 20000 iterations
config/train_my_novel.py        # Custom dataset
```

### Tip 5: Use Your Own Text

```bash
# 1. Copy your text file
cp my_novel.txt data/

# 2. Prepare it
python gpt.py dataset
# Choose option 3

# 3. Train on it
python gpt.py train
# Select your dataset
```

---

## 📊 Model Size Guide

| Preset | Params | Layers | Use Case | Time (CPU) |
|--------|--------|--------|----------|------------|
| **tiny** | ~6M | 4 | Testing, quick experiments | Minutes |
| **small** | ~25M | 6 | Laptops, learning | 10-30 min |
| **medium** | ~80M | 12 | Good results, needs GPU | 1-3 hours |
| **large** | ~350M | 24 | Best quality, high-end GPU | Several hours |

### Recommended for Your Mac

- **For testing:** `tiny` model
- **For actual training:** `small` or `medium`
- **With MPS:** `small` works well
- **CPU only:** Stick to `tiny` or `small`

---

## 🎯 Workflows

### Workflow 1: Quick Test

```bash
source venv/bin/activate
python gpt.py train
# tiny model, quick duration
# Wait 5 minutes
python gpt.py generate
# Test with prompt "Hello"
```

### Workflow 2: Serious Training

```bash
source venv/bin/activate
python gpt.py config
# small or medium model
# 12 layers
# medium or long duration
python train.py --config config/train_*.py
# Wait 1-2 hours
python gpt.py generate
```

### Workflow 3: Custom Dataset

```bash
source venv/bin/activate
# Add your text
cp my_data.txt data/
python gpt.py dataset
# Option 3: Add custom text
python gpt.py train
# Select your dataset
# medium model, medium duration
```

---

## 📦 Project Structure

```
custom-gpt-from-scratch/
├── gpt_from_scratch/        # Main Python package
│   ├── __init__.py
│   ├── cli.py               # Optional packaged CLI
│   ├── model/               # Model implementation
│   │   ├── __init__.py
│   │   └── transformer.py   # GPT implementation
│   ├── utils/               # Utilities (single source of truth)
│   │   ├── __init__.py
│   │   ├── hardware_detector.py
│   │   └── python_utils.py
│   └── data/
│       ├── __init__.py
│       └── utils.py         # Data loading helpers
│
├── gpt.py                   # Main command center (use this!)
├── train.py                 # Training script
├── generate_demo.py         # Simple generation
├── generate_interactive.py  # Interactive generation
├── check_hardware.py        # Hardware checker
│
├── config/                  # Training configurations
│   ├── train_default.py
│   ├── train_demo.py
│   └── train_*.py           # Your custom configs
│
├── data/                    # Datasets
│   ├── prepare.py           # Data preparation script
│   ├── train.bin            # Prepared training data
│   ├── val.bin              # Validation data
│   └── *.txt                # Your text files
│
├── out/                     # Trained models
│   └── ckpt.pt              # Your trained model!
```

---

## 🚀 Quick Reference

### Setup
```bash
./setup.sh                          # One-command setup
source venv/bin/activate            # Activate venv
python check_python_version.py     # Check Python version
```

### Training
```bash
python gpt.py train                 # Interactive training
python gpt.py config                # Create config
python train.py --config <config>   # Train with config
```

### Generation
```bash
python gpt.py generate              # Interactive generation
python generate_demo.py             # Simple generation
```

### Management
```bash
python gpt.py info                  # Check setup
python gpt.py hardware              # Check hardware
python gpt.py dataset               # Manage datasets
```

---

## 📖 Additional Resources

- **README.md** - Complete technical documentation
- **scripts.json** - Available commands reference
- Run `python gpt.py --help` - Command help

---

## 🎉 You're Ready!

```bash
# Start here:
python gpt.py info      # Check your setup
python gpt.py train     # Start training
python gpt.py generate  # Generate text

# Enjoy! 🚀
```

**Happy training!** If you get stuck, check the troubleshooting section or re-read the relevant parts of this guide.

# GPT Quick Reference Guide

Your cheat sheet for common operations - like package.json scripts!

## 🚀 Quick Commands (Most Used)

```bash
# Like "npm start" - Start interactive training
python gpt.py train

# Like "npm run dev" - Generate text
python gpt.py generate

# Like "npm run build" - Check your setup
python gpt.py info

# Like "npm run test" - Check hardware
python gpt.py hardware
```

---

## 📋 Complete Command Reference

### Main Commands

| Command | Description | Example |
|---------|-------------|---------|
| `python gpt.py train` | Interactive training setup | Walks you through hardware, dataset, model setup |
| `python gpt.py generate` | Generate text from trained model | Interactive text generation |
| `python gpt.py config` | Create/edit configuration | Build custom training configs |
| `python gpt.py dataset` | Manage datasets | Add, prepare, switch datasets |
| `python gpt.py hardware` | Check available hardware | Shows CPU, GPU, MPS options |
| `python gpt.py info` | Show current setup | Lists datasets, models, configs |

### Traditional Commands (Still Work!)

```bash
# Traditional training (with specific config)
python train.py --config config/train_demo.py

# Traditional generation
python generate_demo.py

# Hardware check
python check_hardware.py

# Interactive hardware selection
python train.py --interactive
```

---

## 🎯 Common Workflows

### First Time Setup

```bash
# 1. Check what you have
python gpt.py info

# 2. Check your hardware
python gpt.py hardware

# 3. Prepare dataset (if needed)
python gpt.py dataset
# Select option 2 to prepare Shakespeare

# 4. Start training
python gpt.py train
# Follow the prompts!
```

### Regular Training Workflow

```bash
# Quick training session
python gpt.py train
# Choose: quick test (200 iterations)
# Takes ~5 minutes

# View results
python gpt.py generate
```

### Custom Model Configuration

```bash
# Create custom config
python gpt.py config

# When asked about model architecture:
# - Choose "n" for custom architecture
# - Set number of layers (e.g., 12 for more capacity)
# - Set other parameters

# Train with custom config
python train.py --config config/train_custom_*.py
```

---

## ⚙️ Model Architecture Options

### Preset Models (Easy)

| Preset | Layers | Params | Best For | Training Time |
|--------|--------|--------|----------|---------------|
| `tiny` | 4 | ~6M | Testing, quick experiments | Minutes |
| `small` | 6 | ~25M | Laptops, learning | 10-30 min |
| `medium` | 12 | ~80M | Good GPU | 1-3 hours |
| `large` | 24 | ~350M | High-end GPU | Several hours |

### Custom Architecture (Advanced)

When you choose custom architecture in `python gpt.py config`:

```python
n_layer = 8           # Number of transformer layers (more = better, slower)
n_head = 8            # Number of attention heads
n_kv_head = 4         # Number of KV heads (for GQA, usually n_head/2)
n_embd = 512          # Embedding dimension
block_size = 256      # Context length (max sequence length)
mlp_ratio = 4.0       # MLP expansion ratio
dropout = 0.1         # Dropout rate
```

**Quick Tips:**
- **More layers** = Better quality but slower training
- **Larger embedding** = More capacity but more memory
- **Longer context** = Can see more text but uses more memory

---

## 💾 Dataset Management

### Add Your Own Text Dataset

```bash
# Option 1: Interactive
python gpt.py dataset
# Choose option 3 (Add custom text file)

# Option 2: Manual
# 1. Copy your .txt file to data/ directory
cp my_novel.txt data/
# 2. Prepare it
python gpt.py dataset
# Choose option 3 and follow prompts
```

### Supported Formats

- `.txt` files (UTF-8 text)
- Any plain text format
- Size: Any (larger = better results)

---

## 🖥️ Hardware Selection

### Auto-Select (Easiest)

```bash
python gpt.py train
# Hardware will be auto-selected
```

### Manual Selection

```bash
# See all options
python gpt.py hardware

# Train with specific hardware
# Edit your config file:
device = 'cpu'      # For CPU
device = 'cuda'     # For NVIDIA GPU
device = 'mps'      # For Apple Silicon GPU
```

### Hardware Priority (Auto-Selected)

1. **CUDA** (NVIDIA GPU) - Fastest
2. **ROCm** (AMD GPU) - Fast
3. **MPS** (Apple Silicon) - Medium
4. **CPU** - Slowest but always works

---

## 📊 Training Duration Guide

| Duration | Iterations | Time (CPU) | Time (GPU) | When to Use |
|----------|-----------|------------|------------|-------------|
| Quick test | 200 | ~5 min | ~1 min | Testing setup |
| Short | 1,000 | ~20 min | ~5 min | Quick results |
| Medium | 5,000 | ~2 hours | ~20 min | Good quality |
| Long | 20,000+ | ~8+ hours | ~2 hours | Best quality |

---

## 🎨 Generation Parameters

When generating text:

### Temperature
```bash
0.1 - 0.5   # Conservative, coherent (good for factual text)
0.6 - 0.9   # Balanced (recommended)
1.0 - 2.0   # Creative, random (good for poetry/fiction)
```

### Max Tokens
```bash
50-100      # Short responses
100-200     # Paragraphs
200-500     # Long form
```

### Top-K
```bash
10-50       # Very focused
50-200      # Balanced (default: 200)
200+        # More diverse
```

---

## 🔧 Troubleshooting Quick Fixes

### "Out of Memory"
```bash
# Reduce batch size in config
batch_size = 4  # or even 1

# Or use smaller model
model_preset = 'tiny'
```

### "No module named 'torch'"
```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### "No trained model found"
```bash
# Train first!
python gpt.py train
```

### "MPS errors on Mac"
```bash
# Use CPU instead (more stable)
# In config: device = 'cpu'
```

### Training is slow
```bash
# Check hardware
python gpt.py hardware

# Use smaller model
model_preset = 'tiny'

# Reduce iterations
max_iters = 200
```

---

## 📁 File Structure

```
Where things are:
├── gpt.py                 # Main command center (use this!)
├── train.py               # Traditional training script
├── generate_demo.py       # Simple generation
├── config/                # Your training configurations
│   ├── train_demo.py
│   └── train_*.py         # Generated configs
├── data/                  # Your datasets
│   ├── train.bin          # Prepared training data
│   ├── val.bin            # Validation data
│   └── *.txt              # Source text files
├── out/                   # Trained models
│   └── ckpt.pt            # Your trained model!
└── model/                 # Model code (don't edit)
```

---

## 💡 Pro Tips

### Tip 1: Save Your Configs
```bash
# When you create a good config, save it!
cp config/train_custom_*.py config/my_best_config.py
```

### Tip 2: Experiment with Layers
```python
# More layers = Better (but slower)
n_layer = 4   # Very fast, okay results
n_layer = 8   # Balanced
n_layer = 12  # Good results
n_layer = 24  # Best results (needs good hardware)
```

### Tip 3: Watch the Loss
```bash
# During training, watch the loss value:
# - Should decrease over time
# - 4.0 → 2.0 is good progress
# - 2.0 → 1.5 is great!
# - Below 1.0 is excellent
```

### Tip 4: Quick Test Before Long Training
```bash
# Always test with quick training first
python gpt.py train
# Choose "quick" duration
# Make sure everything works
# Then do longer training
```

### Tip 5: Multiple Configs for Different Tasks
```bash
# Keep different configs for different purposes
config/train_quick_test.py      # For testing (200 iters)
config/train_shakespeare.py     # For Shakespeare (5000 iters)
config/train_my_novel.py        # For your custom data
```

---

## 🎓 Learning Path

### Beginner (Day 1)
```bash
1. python gpt.py info           # Understand setup
2. python gpt.py hardware       # Check your hardware
3. python gpt.py train          # First training (quick)
4. python gpt.py generate       # See results!
```

### Intermediate (Week 1)
```bash
1. python gpt.py dataset        # Add your own text
2. python gpt.py config         # Create custom config
3. Train for longer (1000+ iterations)
4. Experiment with temperature in generation
```

### Advanced (Month 1)
```bash
1. Customize model architecture (layers, embedding size)
2. Train large models (5000+ iterations)
3. Fine-tune on specific text styles
4. Compare different configurations
```

---

## 📞 Quick Help

```bash
# Any command with --help shows options
python gpt.py --help
python train.py --help

# Check setup
python gpt.py info

# Check this guide
cat QUICK_REFERENCE.md

# Full documentation
cat GETTING_STARTED.md
```

---

## 🎯 Most Common Use Cases

### "I want to train on my own text"
```bash
1. Copy your .txt file to data/
2. python gpt.py dataset
3. Choose option 3, prepare your file
4. python gpt.py train
5. Select your dataset when prompted
```

### "I want more/fewer layers"
```bash
1. python gpt.py config
2. Choose "n" for custom architecture
3. Set n_layer to your desired number (4, 8, 12, 24, etc.)
4. Complete the config
5. Train with that config
```

### "I want to use my GPU instead of CPU"
```bash
1. python gpt.py hardware  # Check if GPU detected
2. python gpt.py train     # Will auto-select GPU if available
# Or manually: Edit your config file and set device = 'cuda' or 'mps'
```

### "I want better quality text"
```bash
Option 1: Train longer
  - Increase max_iters to 5000-20000

Option 2: Bigger model
  - Use 'small' or 'medium' preset
  - Or increase n_layer to 12-24

Option 3: More data
  - Add more text files to train on
```

---

**Remember:** Like npm/yarn, `python gpt.py` is your main interface for everything!

Start with: `python gpt.py info` 🚀

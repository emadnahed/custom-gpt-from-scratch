# 🚀 START HERE - Ultra Quick Setup

Welcome! Get started in **one command**.

## Installation (Choose Your OS)

### macOS / Linux

```bash
./setup.sh
```

That's it! The script will:
- ✅ Check if Python is installed (install if needed)
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Detect your hardware (CPU/GPU)
- ✅ Prepare training dataset
- ✅ Ask if you want to start training now!

### Windows

```cmd
setup.bat
```

**Note for Windows:** If Python isn't installed, the script will guide you to download it.

### Manual Setup (If scripts don't work)

```bash
# 1. Install Python 3.8+ from python.org
# 2. Run these commands:
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python check_hardware.py
cd data && python prepare.py && cd ..
```

---

## After Setup - Your Main Commands

Think of these like `npm run` commands:

```bash
# Check your setup
python gpt.py info

# Start training (interactive - recommended!)
python gpt.py train

# Generate text from trained model
python gpt.py generate

# Check available hardware
python gpt.py hardware

# Manage datasets
python gpt.py dataset

# Create custom config
python gpt.py config
```

---

## Quick Training (5 minutes)

```bash
# This is all you need!
python gpt.py train

# The interactive prompt will ask you:
# 1. Which hardware? (auto-detected)
# 2. Which dataset? (Shakespeare is prepared)
# 3. Model size? (choose 'tiny' for quick test)
# 4. How long? (choose 'quick' for 200 iterations)
# 5. Start now? (yes!)

# Then sit back and watch it train! ☕
```

---

## Your First Training Session

1. **Start training:**
   ```bash
   python gpt.py train
   ```

2. **Choose these options for a quick test:**
   - Hardware: (auto-selected)
   - Dataset: Shakespeare
   - Model: `tiny`
   - Duration: `quick` (200 iterations)

3. **Watch it train** (~5 minutes on CPU)
   - Loss will decrease: 4.2 → 2.0 ✓
   - Model saves automatically

4. **Generate text:**
   ```bash
   python gpt.py generate
   ```
   - Try prompt: "ROMEO:"
   - See Shakespeare-style text!

---

## What's Your Mac's Hardware?

Your Mac has:
- **CPU**: Intel or Apple Silicon
- **MPS**: Apple Metal GPU (if Apple Silicon)

The `python gpt.py hardware` command shows:
- ✓ **Available** (green) - Can use this
- ✗ **Unavailable** (grey) - Can't use this

**Auto-select** will pick the best available automatically!

---

## Adjust Number of Layers (Model Size)

```bash
# Create custom config
python gpt.py config

# When asked about model architecture:
# - Choose "n" for custom
# - Set number of layers:
#   * 4 layers  = Very fast, okay results
#   * 8 layers  = Balanced
#   * 12 layers = Good quality (recommended)
#   * 24 layers = Best quality (slow on CPU)

# Then train with that config
```

**Rule of thumb:** More layers = better quality but slower training

---

## Use Your Own Text Dataset

```bash
# Option 1: Interactive
python gpt.py dataset
# Choose option 3 (Add custom text)

# Option 2: Manual
# 1. Copy your .txt file to data/ folder
# 2. Run: python gpt.py dataset
# 3. Follow prompts to prepare it
```

---

## Hardware Selection Made Easy

```bash
# See all hardware options
python gpt.py hardware

# Output shows:
# ✓ AVAILABLE (green)   - You can use this
# ✗ UNAVAILABLE (grey)  - You can't use this
#
# RECOMMENDED: Best option for you

# Auto-select (easiest):
python gpt.py train
# Hardware is auto-selected!

# Manual select:
# Edit your config file and set:
# device = 'cpu'   # For CPU
# device = 'mps'   # For Apple Silicon GPU
# device = 'cuda'  # For NVIDIA GPU
```

---

## Common Questions

### "How long does training take?"
- Quick test: 5 minutes (200 iterations)
- Short training: 20 minutes (1000 iterations)
- Good results: 1-2 hours (5000 iterations)
- Best results: Several hours (20000+ iterations)

### "Which model size should I use?"
- **Testing/Learning**: `tiny` (6M params)
- **Your Mac (CPU)**: `small` (25M params)
- **Good GPU**: `medium` (80M params)
- **High-end GPU**: `large` (350M params)

### "How do I know if it's working?"
Watch the loss value during training:
- Starts around 4.0
- Should decrease to ~2.0 (good!)
- Below 1.5 is excellent!

### "Can I stop and resume training?"
- Press Ctrl+C to stop gracefully
- Model saves automatically at checkpoints
- To resume: Just train again (loads last checkpoint)

---

## File Structure (Where Things Are)

```
📁 Your Project
├── setup.sh / setup.bat     # ← Run this first!
├── gpt.py                   # ← Your main command (use this!)
│
├── 📄 Documentation
│   ├── START_HERE.md        # ← You are here!
│   ├── QUICK_REFERENCE.md   # Commands cheat sheet
│   ├── GETTING_STARTED.md   # Detailed guide
│   └── README.md            # Full documentation
│
├── config/                  # Training configurations
│   ├── train_demo.py
│   └── train_*.py           # Your custom configs appear here
│
├── data/                    # Your datasets
│   ├── train.bin            # Prepared training data
│   ├── val.bin              # Validation data
│   └── *.txt                # Source text files
│
└── out/                     # Your trained models
    └── ckpt.pt              # Your trained model! (appears after training)
```

---

## Next Steps

### Day 1: Learn the Basics
```bash
1. ./setup.sh                    # Setup everything
2. python gpt.py info           # Check your setup
3. python gpt.py hardware       # See your hardware
4. python gpt.py train          # First training (quick)
5. python gpt.py generate       # See results!
```

### Day 2: Customize
```bash
1. python gpt.py config         # Create custom config
2. Try different model sizes
3. Experiment with layers (4, 8, 12, 24)
4. Train for longer (1000+ iterations)
```

### Week 1: Your Own Data
```bash
1. python gpt.py dataset        # Add your text
2. Train on your data
3. Generate in your style!
```

---

## Getting Help

```bash
# Command help
python gpt.py --help
python gpt.py info

# Read documentation
cat QUICK_REFERENCE.md      # Command cheat sheet
cat GETTING_STARTED.md      # Detailed guide
cat README.md               # Full technical docs

# Check your setup
python gpt.py info          # Shows what's ready

# Test hardware
python gpt.py hardware      # Shows CPU/GPU options
```

---

## Troubleshooting

### "Command not found"
```bash
# Make sure you activated the virtual environment:
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### "No module named torch"
```bash
# Install dependencies:
pip install -r requirements.txt
```

### "Out of memory"
```bash
# Use smaller model or batch size
# In config: model_preset = 'tiny'
#           batch_size = 4
```

### "MPS errors on Mac"
```bash
# Use CPU instead (more stable)
# Choose CPU when prompted, or
# In config: device = 'cpu'
```

---

## Summary: The Essentials

**Setup once:**
```bash
./setup.sh
```

**Regular use (like npm scripts):**
```bash
python gpt.py train      # Train a model
python gpt.py generate   # Generate text
python gpt.py info       # Check status
```

**That's it!** Everything else is optional customization.

---

## Ready?

```bash
# Start here:
./setup.sh

# Or if already setup:
python gpt.py train
```

**Have fun training! 🚀**

For more details:
- **Quick commands**: See `QUICK_REFERENCE.md`
- **Full guide**: See `GETTING_STARTED.md`
- **Technical**: See `README.md`

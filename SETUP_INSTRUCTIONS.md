# Quick Setup Instructions

Welcome! This is your quick-start guide to get training immediately.

## 🚀 Fastest Way to Get Started

### For macOS/Linux Users:

```bash
./quickstart.sh
```

This one command will:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Detect your hardware
- ✅ Prepare the dataset
- ✅ Get you ready to train!

### For Windows Users:

```cmd
quickstart.bat
```

### Manual Setup (All Platforms):

If you prefer step-by-step:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Check your hardware
python check_hardware.py

# 4. Prepare data
cd data
python prepare.py
cd ..

# 5. Start training!
python train.py
```

---

## 📖 Documentation Overview

### For Beginners (Start Here!)
**📄 GETTING_STARTED.md** - Comprehensive guide covering:
- Installation and setup
- Hardware detection
- Data preparation
- Training your first model
- Common issues and solutions

### Hardware Information
**📄 HARDWARE_FEATURE_SUMMARY.md** - Details about:
- Supported hardware platforms
- Auto-detection features
- Usage examples
- Hardware-specific optimizations

### Main Documentation
**📄 README.md** - Project overview and technical details:
- Architecture deep dive
- Advanced configuration
- Performance optimization
- API reference

---

## ⚡ Quick Commands Reference

### Hardware Detection
```bash
python check_hardware.py              # Show all hardware
python check_hardware.py --interactive # Choose hardware
python check_hardware.py --recommended # Show best device
```

### Training
```bash
python train.py                       # Auto-detect and train
python train.py --interactive         # Choose hardware first
python train.py --show-hardware       # Just show hardware
```

### Data Preparation
```bash
cd data && python prepare.py && cd ..  # Prepare Shakespeare dataset
```

---

## 🎯 What's New in Your Project

### New Files Added:

1. **utils/hardware_detector.py** - Core hardware detection logic
2. **utils/__init__.py** - Utils module initialization
3. **check_hardware.py** - Standalone hardware detection CLI
4. **quickstart.sh** - Automated setup for Unix systems
5. **quickstart.bat** - Automated setup for Windows
6. **GETTING_STARTED.md** - Beginner-friendly comprehensive guide
7. **HARDWARE_FEATURE_SUMMARY.md** - Hardware features documentation
8. **SETUP_INSTRUCTIONS.md** - This file!

### Modified Files:

1. **train.py** - Now includes auto hardware detection with:
   - `--interactive` flag for choosing hardware
   - `--show-hardware` flag for displaying hardware info
   - Auto-detection by default
   - Optimal precision selection

2. **README.md** - Updated with:
   - Hardware auto-detection section
   - Updated project structure
   - Links to new documentation

---

## 🖥️ Expected Hardware Detection on Your Mac

Since you're on a Mac (Darwin system), here's what you should see:

```
======================================================================
HARDWARE DETECTION SUMMARY
======================================================================

[1] CUDA: ✗ UNAVAILABLE
    Name: NVIDIA CUDA
    Details: CUDA not available or no NVIDIA GPU detected

[2] ROCM: ✗ UNAVAILABLE
    Name: AMD ROCm
    Details: Not on Linux

[3] MPS: ✓ AVAILABLE  ← THIS IS YOUR DEVICE!
    Name: Apple Metal (MPS)
    Supported Precisions: float16, float32
    Details: Apple Silicon: arm

[4] XPU: ✗ UNAVAILABLE
    Name: Intel XPU
    Details: Intel Extension for PyTorch not installed

[5] CPU: ✓ AVAILABLE
    Name: CPU (arm)
    Supported Precisions: float16, float32
    Details: Darwin - arm

======================================================================
RECOMMENDED: MPS - Apple Metal (MPS)
Device String: mps
Optimal Dtype: float16
======================================================================
```

**What this means:**
- Your Mac will use **MPS (Metal Performance Shaders)** for GPU acceleration
- Training will be significantly faster than CPU-only
- The optimal precision is **float16** (Apple Silicon doesn't support bfloat16 yet)

---

## 🔧 What Happens When You Run `train.py`

With the new auto-detection, you'll see:

```
================================================================================
HARDWARE SETUP
================================================================================
Auto-detected device: mps
Auto-selected dtype: float16

Final configuration:
  Device: mps
  Device Type: mps
  Dtype: float16
================================================================================

Loading prepared dataset...
Train dataset: 301,966 tokens
Val dataset: 33,552 tokens
Vocabulary size: 65

Initializing model...
Starting training...
Total iterations: 5000
Batch size: 12
Gradient accumulation steps: 1
Effective batch size: 12
Device: mps, dtype: float16
================================================================================

iter 0: loss 4.2345, time 123.45ms, mfu 0.00%
iter 1: loss 4.1234, time 98.76ms, mfu 12.34%
...
```

---

## 🎓 Learning Path for JavaScript/React Developers

Since you're coming from React/yarn, here's a helpful comparison:

### Ecosystem Parallels

| Task | JavaScript/React | Python/ML |
|------|-----------------|-----------|
| Run project | `yarn start` | `python train.py` |
| Install packages | `yarn install` | `pip install -r requirements.txt` |
| Package manager | yarn/npm | pip |
| Dependencies file | package.json | requirements.txt |
| Virtual env | node_modules/ | venv/ |
| Dev server | webpack-dev-server | (training runs directly) |
| Build output | build/, dist/ | out/ (checkpoints) |

### Key Differences

1. **No Hot Reload**: Training is a batch process, not a dev server
2. **Long-Running**: Training takes minutes to hours (not instant like React)
3. **Checkpoints**: Models are saved periodically (like build artifacts)
4. **Hardware Matters**: GPU makes a huge difference (unlike web dev)
5. **Virtual Environments**: More important in Python (like node_modules but activated)

---

## 💡 Pro Tips

### Activate Your Virtual Environment!

Always activate your venv before running commands:

```bash
# Activate (you'll see "venv" in your prompt)
source venv/bin/activate

# Now you can run:
python train.py

# When done:
deactivate
```

### Monitor Training

Watch these metrics:
- **Loss**: Should decrease over time (4.0 → 1.5 for Shakespeare)
- **Time per iter**: Consistency means stable training
- **MFU**: Higher is better (but don't worry if low on Mac)

### Save Your Work

Training creates checkpoints in `out/ckpt.pt`. This is your trained model!

### Keyboard Shortcuts

- `Ctrl+C`: Stop training gracefully (saves last checkpoint)
- `Ctrl+Z`: Suspend (use `fg` to resume)

---

## 🐛 Quick Troubleshooting

### "No module named 'torch'"
→ Activate your virtual environment: `source venv/bin/activate`

### "CUDA out of memory"
→ Reduce `batch_size` in `config/train_default.py`

### "MPS not available"
→ Normal on Intel Macs or older macOS. Will use CPU instead.

### "Data files not found"
→ Run: `cd data && python prepare.py && cd ..`

### Training is slow
→ Expected on CPU. Check your detected hardware: `python check_hardware.py`

---

## 🎯 Your Next Steps

1. **Run the quickstart**: `./quickstart.sh`
2. **Check your hardware**: `python check_hardware.py`
3. **Start training**: `python train.py`
4. **Watch it train**: Loss should decrease over ~10-20 minutes
5. **Generate text**: `python sample.py --prompt "Hello"`

---

## 📚 Need More Help?

- **Getting started?** → Read `GETTING_STARTED.md`
- **Hardware issues?** → Read `HARDWARE_FEATURE_SUMMARY.md`
- **Technical details?** → Read `README.md`
- **Configuration?** → Check `config/train_default.py`

---

**Happy Training! 🚀**

Remember: Machine learning training is different from web development. It's normal for training to take time and for you to experiment with different configurations. Start small, learn from the logs, and scale up gradually!

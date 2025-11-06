# Hardware Auto-Detection Feature Summary

This document summarizes the new hardware auto-detection features added to your GPT training project.

## What's New

### 1. Comprehensive Hardware Detection (`utils/hardware_detector.py`)

A new utility module that detects and manages all available hardware accelerators:

**Supported Hardware:**
- ✅ **NVIDIA CUDA** (GPU) - For NVIDIA GPUs
- ✅ **AMD ROCm** (GPU) - For AMD GPUs on Linux
- ✅ **Apple Metal (MPS)** - For Apple Silicon (M1/M2/M3)
- ✅ **Intel XPU** - For Intel GPUs
- ✅ **CPU** - Universal fallback

**Features:**
- Auto-detects best available hardware
- Shows all hardware (available and unavailable)
- Displays detailed info (memory, compute capability, precision support)
- Recommends optimal precision (bfloat16, float16, float32)
- Color-coded output (green for available, grey for unavailable)

### 2. Updated Training Script (`train.py`)

The training script now includes:

**Command Line Options:**
```bash
python train.py                  # Auto-detect hardware
python train.py --interactive    # Choose hardware interactively
python train.py --show-hardware  # Show hardware and exit
```

**Configuration Options:**
```python
device = 'auto'  # Auto-detect, or set to 'cuda', 'mps', 'cpu', etc.
dtype = 'auto'   # Auto-select precision, or set to 'bfloat16', 'float16', 'float32'
interactive_hardware = False  # Set to True for interactive selection
```

**Visual Feedback:**
- Clear hardware setup section in output
- Shows detected device and precision
- Platform-specific optimizations (e.g., TF32 for CUDA)

### 3. Standalone Hardware Checker (`check_hardware.py`)

A dedicated CLI tool for hardware detection:

```bash
python check_hardware.py              # Show all hardware
python check_hardware.py --interactive # Interactive selection
python check_hardware.py --json        # JSON output
python check_hardware.py --recommended # Show only recommended device
```

**Use Cases:**
- Check hardware before training
- Verify GPU drivers are working
- Debug hardware issues
- Get device strings for configuration

### 4. Getting Started Guide (`GETTING_STARTED.md`)

A comprehensive guide for beginners, especially those coming from JavaScript/React:

**Topics Covered:**
- Prerequisites and installation
- Virtual environment setup (compared to npm/yarn)
- Hardware detection and selection
- Data preparation
- Training your first model
- Monitoring and understanding training logs
- Common issues and troubleshooting
- Next steps and advanced features

## File Structure

```
custom-gpt-from-scratch/
├── train.py                       # Updated with auto-detection
├── check_hardware.py             # New: Hardware checker CLI
├── GETTING_STARTED.md            # New: Beginner's guide
├── HARDWARE_FEATURE_SUMMARY.md   # This file
├── utils/
│   ├── __init__.py               # New: Utils module init
│   └── hardware_detector.py      # New: Hardware detection logic
├── requirements.txt              # Existing dependencies
└── ... (other project files)
```

## How It Works

### Auto-Detection Flow

1. **Scan Hardware**: Detects all available accelerators
   - Checks for CUDA support (NVIDIA)
   - Checks for ROCm (AMD)
   - Checks for MPS (Apple Silicon)
   - Checks for Intel XPU
   - CPU is always available

2. **Rank by Priority**: CUDA > ROCm > MPS > XPU > CPU

3. **Select Optimal Precision**:
   - `bfloat16` if supported (best for training)
   - `float16` if bfloat16 not available
   - `float32` as fallback

4. **Return Configuration**: Device string and dtype

### Hardware Detection Details

**NVIDIA CUDA:**
- Uses `torch.cuda.is_available()`
- Gets device name, memory, compute capability
- Checks bfloat16 support (requires Ampere+ GPUs)

**AMD ROCm:**
- Detects on Linux systems
- Uses `rocm-smi` command for verification
- ROCm uses 'cuda' backend in PyTorch

**Apple MPS:**
- Only on macOS with Apple Silicon
- Uses `torch.backends.mps.is_available()`
- Supports float16 (not bfloat16 yet)

**Intel XPU:**
- Requires Intel Extension for PyTorch
- Detects via `torch.xpu.is_available()`

**CPU:**
- Always available as fallback
- Shows system information

## Usage Examples

### Example 1: Quick Start (Auto-Detection)

```bash
python train.py
```

Output:
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
```

### Example 2: Interactive Selection

```bash
python train.py --interactive
```

Output shows all hardware with selection menu:
```
======================================================================
HARDWARE DETECTION SUMMARY
======================================================================

[1] CUDA: ✗ UNAVAILABLE
[2] ROCM: ✗ UNAVAILABLE
[3] MPS: ✓ AVAILABLE
    Name: Apple Metal (MPS)
    Supported Precisions: float16, float32
    Details: Apple Silicon: arm

[4] XPU: ✗ UNAVAILABLE
[5] CPU: ✓ AVAILABLE

======================================================================
RECOMMENDED: MPS - Apple Metal (MPS)
======================================================================

Multiple devices available. Please select one:
  [1] MPS - Apple Metal (MPS)
  [2] CPU - CPU (arm)

Enter choice (1-2) or press Enter for recommended:
```

### Example 3: Check Hardware Only

```bash
python check_hardware.py
```

Shows detailed hardware report without starting training.

## Benefits

1. **User-Friendly**: No need to manually configure device/dtype
2. **Cross-Platform**: Works on Windows, Linux, macOS
3. **Multi-Hardware**: Supports NVIDIA, AMD, Apple, Intel
4. **Visual Feedback**: Clear indication of available vs unavailable hardware
5. **Flexible**: Auto mode or manual selection
6. **Informative**: Shows hardware capabilities and limitations
7. **Beginner-Friendly**: Includes comprehensive guide for newcomers

## Next Steps for You

1. **Install Dependencies** (see GETTING_STARTED.md):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Check Your Hardware**:
   ```bash
   python check_hardware.py
   ```

3. **Prepare Data**:
   ```bash
   cd data
   python prepare.py
   cd ..
   ```

4. **Start Training**:
   ```bash
   python train.py
   ```

## For Your Reference

**On your Mac (Apple Silicon):**
- Your device will be detected as `mps` (Metal Performance Shaders)
- Optimal dtype will be `float16`
- Training will use Apple's GPU acceleration
- Much faster than CPU training

**Expected Hardware Detection on Your System:**
```
[3] MPS: ✓ AVAILABLE
    Name: Apple Metal (MPS)
    Supported Precisions: float16, float32
    Details: Apple Silicon: arm

RECOMMENDED: MPS - Apple Metal (MPS)
Device String: mps
Optimal Dtype: float16
```

## Troubleshooting

If hardware detection doesn't work:

1. **Check PyTorch Installation**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **Check MPS Availability (Mac)**:
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

3. **Check CUDA Availability (NVIDIA)**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Force CPU Mode** (if needed):
   ```python
   # In train.py, set:
   device = 'cpu'
   dtype = 'float32'
   ```

---

**Happy Training!** 🚀

For more details, see `GETTING_STARTED.md`.

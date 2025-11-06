# Quick Start Guide

This guide will help you get started with training and using your GPT model in under 10 minutes.

## 1. Setup (1 minute)

```bash
# Activate your virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 2. Prepare Data (2 minutes)

Download and prepare the Shakespeare dataset:

```bash
python data/prepare.py
```

This will:
- Download ~1MB of Shakespeare text
- Process it into train/val splits
- Create character-level vocabulary
- Save binary files for fast loading

Expected output:
```
Downloading Shakespeare dataset...
Vocabulary size: 65 characters
Train: 1,003,854 tokens
Val: 111,540 tokens
Dataset prepared successfully!
```

## 3. Train the Model (5-15 minutes)

Start training with default settings:

```bash
python train.py
```

### What to expect:

**On GPU (NVIDIA):**
```
Using device: cuda
Train dataset: 1,003,854 tokens
Val dataset: 111,540 tokens
Vocabulary size: 65
Number of parameters: 10.65M
Starting training...
step 0: train loss 4.1765, val loss 4.1893
iter 10: loss 3.2341, time 45.21ms, mfu 12.34%
...
step 2000: train loss 1.4523, val loss 1.5123
```
Training time: ~5-10 minutes

**On CPU:**
```
Using device: cpu
...
iter 10: loss 3.2341, time 523.45ms, mfu 2.15%
```
Training time: ~30-60 minutes

**On Mac (M1/M2):**
```
Using device: mps
...
iter 10: loss 3.2341, time 112.34ms, mfu 5.67%
```
Training time: ~15-30 minutes

### Monitoring Training

Watch the loss decrease:
- **Loss > 3.0**: Model is learning basic patterns
- **Loss 2.0-3.0**: Learning word structure
- **Loss 1.5-2.0**: Generating coherent text
- **Loss 1.0-1.5**: High quality generation

The model auto-saves checkpoints to `out/ckpt.pt`.

## 4. Generate Text (< 1 minute)

After training, generate Shakespeare-like text:

```bash
# Basic generation
python sample.py --prompt "To be or not to be"

# More creative
python sample.py --prompt "Once upon" --temperature 0.9 --max_tokens 200

# Interactive mode
python sample.py --interactive
```

### Example Output

After 2000 iterations:
```
To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
```

After 5000 iterations, quality improves significantly!

## 5. Experiment! (optional)

### Try different temperatures:

```bash
# Conservative (more repetitive)
python sample.py --prompt "Hello" --temperature 0.5

# Balanced
python sample.py --prompt "Hello" --temperature 0.8

# Creative (more random)
python sample.py --prompt "Hello" --temperature 1.2
```

### Train longer:

Edit `config/train_default.py`:
```python
max_iters = 10000  # Double the training time
```

### Use a larger model:

Edit `config/train_default.py`:
```python
model_preset = 'medium'  # ~80M parameters (requires GPU)
```

## Common Issues

### 1. Out of Memory (OOM)

**Solution**: Reduce batch size in `config/train_default.py`:
```python
batch_size = 4  # Down from 12
```

### 2. Training is slow

**Solutions**:
- Use smaller model: `model_preset = 'tiny'`
- Reduce iterations: `max_iters = 1000`
- Enable mixed precision (GPU only): `dtype = 'bfloat16'`

### 3. Model not generating well

**Solutions**:
- Train longer (wait for loss < 1.5)
- Adjust temperature (try 0.8)
- Try different prompts

### 4. "No such file" error when sampling

**Solution**: Make sure training completed and created `out/ckpt.pt`:
```bash
ls out/ckpt.pt
```

## Next Steps

1. **Try your own data**: Modify `data/prepare.py` to load your text files
2. **Experiment with hyperparameters**: Edit `config/train_default.py`
3. **Use the model in code**: See `README.md` for API examples
4. **Scale up**: Try larger models and longer training on GPU

## Tips for Best Results

1. **Train for at least 3000 iterations** for coherent text
2. **Loss should be < 1.5** for good quality
3. **Use temperature 0.7-0.9** for balanced generation
4. **Longer context helps**: Increase `block_size` if you have GPU memory
5. **Save multiple checkpoints**: Training can take time, don't lose progress!

## Performance Benchmarks

### Small Model (~25M params)

| Hardware | Batch Size | Iterations | Time | MFU |
|----------|------------|------------|------|-----|
| RTX 3090 | 32 | 5000 | 8 min | 35% |
| RTX 2080 | 16 | 5000 | 15 min | 25% |
| M1 Mac | 8 | 5000 | 25 min | 8% |
| CPU (i7) | 4 | 5000 | 90 min | 2% |

### Tiny Model (~10M params)

| Hardware | Batch Size | Iterations | Time | MFU |
|----------|------------|------------|------|-----|
| RTX 3090 | 64 | 5000 | 5 min | 28% |
| CPU (i7) | 8 | 5000 | 45 min | 3% |

## Example Workflow

```bash
# Complete workflow in one go
python data/prepare.py && \
python train.py && \
python sample.py --prompt "Hello world" --max_tokens 200
```

That's it! You now have a working GPT model. Happy experimenting!

For more advanced usage, see the main `README.md`.

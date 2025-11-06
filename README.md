# Custom GPT from Scratch

This project implements a highly optimized GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. The implementation includes state-of-the-art architectural improvements, a complete training pipeline, and flexible text generation capabilities.

## 🚀 Features

### Core Architecture
- **Rotary Position Embeddings (RoPE)** - Better length generalization than learned positional embeddings
- **Grouped Query Attention (GQA)** - Memory-efficient attention with fewer KV heads
- **SwiGLU Activation** - Superior performance compared to GELU
- **RMSNorm** - Faster normalization than LayerNorm
- **Flash Attention Support** - Optimized attention computation when available
- **Weight Tying** - Shared embeddings between input and output layers

### Training Optimizations
- **Mixed Precision Training** (FP16/BF16) - Faster training with lower memory usage
- **Gradient Checkpointing** - Trade compute for memory on large models
- **Fused AdamW** - Optimized optimizer for CUDA devices
- **Learning Rate Scheduling** - Warmup and cosine decay
- **Gradient Accumulation** - Simulate larger batch sizes
- **torch.compile() Support** - JIT compilation for PyTorch 2.0+

### Generation Features
- **Temperature Sampling** - Control randomness in generation
- **Top-k Filtering** - Sample from top k most likely tokens
- **Top-p (Nucleus) Sampling** - Dynamic vocabulary truncation
- **KV-Cache** - Efficient autoregressive generation
- **Interactive Mode** - Chat-like interface for generation

### Model Presets
- **Tiny** (~10M params) - Fast testing and prototyping
- **Small** (~25M params) - Runnable on CPU/Mac
- **Medium** (~80M params) - GPU recommended
- **Large** (~350M params) - Requires substantial GPU memory

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/emadnahed/custom-gpt-from-scratch.git
   cd custom-gpt-from-scratch
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Project Structure

```
custom-gpt-from-scratch/
│
├── data/                 # Data loading and preprocessing
│   └── prepare.py        # Data preparation scripts
│
├── model/                # Model architecture
│   ├── __init__.py
│   └── transformer.py    # Core transformer implementation
│
├── config/               # Configuration files
│   └── train_default.py  # Default training configuration
│
├── train.py              # Training script
├── sample.py             # Text generation script
├── requirements.txt      # Project dependencies
└── README.md
```

## 🚦 Getting Started

### 1. Prepare Your Data

First, prepare a dataset for training. The easiest way to start is with the Shakespeare dataset:

```bash
python data/prepare.py
```

This downloads and prepares the Tiny Shakespeare dataset (~1MB) for quick experimentation.

For custom datasets, modify `data/prepare.py` to load your text data. The script supports:
- Character-level tokenization (built-in)
- Hugging Face datasets integration
- Custom text files

### 2. Train the Model

Train with the default configuration:

```bash
python train.py
```

The default configuration trains a small model (~25M parameters) on the Shakespeare dataset. Training should take 5-15 minutes on a modern GPU or 30-60 minutes on CPU.

#### Training Output

During training, you'll see:
- Loss metrics (train/val) every N iterations
- Training speed (tokens/sec)
- Model FLOPS Utilization (MFU) - efficiency metric
- Checkpoints saved to `out/ckpt.pt`

#### Custom Configuration

Modify `config/train_default.py` to customize:
- Model size (tiny/small/medium/large presets)
- Training hyperparameters (learning rate, batch size, etc.)
- Hardware settings (device, mixed precision)
- Optimization features (gradient checkpointing, compilation)

Example configurations:

```python
# Fast training on CPU
model_preset = 'tiny'
batch_size = 4
max_iters = 1000
device = 'cpu'

# GPU training with larger model
model_preset = 'medium'
batch_size = 32
max_iters = 10000
device = 'cuda'
dtype = 'bfloat16'
compile_model = True  # PyTorch 2.0+ for speedup
```

### 3. Generate Text

After training, generate text with your model:

```bash
# Basic generation
python sample.py --prompt "To be or not to be" --max_tokens 100

# Control creativity
python sample.py --prompt "Once upon a time" --temperature 0.8 --top_k 50

# Interactive mode
python sample.py --interactive
```

#### Generation Parameters

- `--prompt`: Starting text (empty for random start)
- `--max_tokens`: Number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature - higher = more random (default: 0.8)
  - `0.1-0.5`: Conservative, coherent
  - `0.6-0.9`: Balanced creativity
  - `1.0+`: Very creative, potentially incoherent
- `--top_k`: Top-k filtering - only sample from top k tokens (default: 200)
- `--top_p`: Nucleus sampling - cumulative probability threshold (default: 0.9)
- `--seed`: Random seed for reproducibility
- `--interactive`: Launch interactive generation mode

## 📊 Dependencies

- Python 3.11+
- PyTorch 2.2.2+
- torchvision
- torchaudio
- NumPy
- tqdm
- Hugging Face Datasets (for data loading)

All dependencies are listed in `requirements.txt`.

## 🏛️ Architecture Deep Dive

This section provides a detailed breakdown of the model architecture at different levels of abstraction, from the simplest overview to the complete technical implementation.

### Level 1: Basic Flow (Simplest View)
```
Input → Tokenization → Token Embeddings → Transformer Blocks (×N layers) → Final Norm → Linear Head → Logits → Loss (if training)
```

### Level 2: Components Breakdown
```
Input Tokens → Token Embeddings → Dropout → 
[Grouped Query Attention → Residual Add → Feedforward (SwiGLU/MLP) → Residual Add] (×N layers) → 
Final Norm → Linear Head → Logits → Softmax (during generation)
```

### Level 3: With Residual Connections
```
Input Tokens → Token Embeddings → Dropout → 
[x + GQA(x) → (x + GQA(x)) + MLP(x + GQA(x))] (×N layers) → 
Final Norm → Linear Head → Logits
```

### Level 4: With Normalization (Complete Flow - Pre-LN Architecture)
```
Input Tokens → Token Embeddings → Dropout → 
[RMSNorm → GQA → Add & Residual → RMSNorm → SwiGLU → Add & Residual] (×N layers) → 
Final RMSNorm → Linear Head (weight-tied) → Logits → Cross-Entropy Loss
```

### Level 5: Advanced - Full Technical Flow
```
Input Tokens (B, T) 
  ↓
Token Embeddings: wte(idx) → (B, T, n_embd)
  ↓
Dropout(p=0.1)
  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Transformer Block (×n_layer) - Pre-Norm Architecture:              │
│                                                                     │
│  x_norm = RMSNorm(x)  [or LayerNorm if configured]                │
│    ↓                                                                │
│  Grouped Query Attention (GQA):                                    │
│    • Q = Linear(x_norm) → (B, n_head, T, head_dim)                │
│    • K = Linear(x_norm) → (B, n_kv_head, T, head_dim)             │
│    • V = Linear(x_norm) → (B, n_kv_head, T, head_dim)             │
│    • Q, K = RoPE(Q, K)  [Rotary Position Embeddings]              │
│    • K, V = repeat_interleave(K, V, n_rep)  [if GQA]              │
│    • attn_out = scaled_dot_product_attention(Q, K, V, causal=True)│
│    • attn_out = Linear_o(attn_out) + Dropout                      │
│  x = x + attn_out  [Residual Connection 1]                        │
│    ↓                                                                │
│  x_norm2 = RMSNorm(x)                                              │
│    ↓                                                                │
│  SwiGLU Feedforward:                                               │
│    • gate = SiLU(W1(x_norm2))                                      │
│    • hidden = W3(x_norm2)                                          │
│    • mlp_out = W2(gate * hidden) + Dropout                        │
│  x = x + mlp_out  [Residual Connection 2]                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
  ↓
Final Normalization: RMSNorm(x) → (B, T, n_embd)
  ↓
Language Model Head: Linear(x) → (B, T, vocab_size) [Weight-Tied with wte]
  ↓
Output Logits (Training) OR Logits[:, -1, :] (Inference)
  ↓
Loss Calculation (if targets provided):
  • Flatten: logits → (B*T, vocab_size), targets → (B*T)
  • Cross-Entropy Loss with ignore_index=-1
```

### Key Architecture Features

#### 1. Pre-Norm Architecture
- Normalization **before** each sub-layer (attention and feedforward)
- More stable training than Post-Norm
- Formula: `x = x + SubLayer(Norm(x))`

#### 2. RoPE (Rotary Position Embeddings)
- Applied **inside** attention mechanism (not as separate layer)
- No learned positional parameters
- Better length generalization
- Applied to Q and K before attention computation

#### 3. Grouped Query Attention (GQA)
- Query heads: `n_head` (e.g., 6)
- Key/Value heads: `n_kv_head` (e.g., 3)
- Repetition factor: `n_rep = n_head / n_kv_head`
- Memory efficient: reduces KV cache size by 2-4x

#### 4. SwiGLU Activation
- Three projection matrices: W1, W2, W3
- Formula: `W2(SiLU(W1(x)) ⊙ W3(x))`
- Better performance than standard GELU
- Used in PaLM and LLaMA models

#### 5. RMSNorm
- Faster than LayerNorm (no mean centering)
- Formula: `x * rsqrt(mean(x²) + ε) * γ`
- Single learnable parameter: `weight (γ)`
- ~10-15% faster than LayerNorm

#### 6. Weight Tying
- Token embeddings and output head share weights
- Reduces parameters by ~vocab_size * n_embd
- Formula: `lm_head.weight = wte.weight`

#### 7. Optimization Features
- Flash Attention support (when available)
- Gradient Checkpointing option
- Mixed Precision compatible (FP16/BF16)
- KV-Cache for efficient generation

### Data Flow Dimensions

```
Input: (Batch, Sequence) → (B, T)
  ↓
Embeddings: (B, T, n_embd)
  ↓
Attention Reshaping:
  Q: (B, T, n_embd) → (B, n_head, T, head_dim)
  K: (B, T, n_kv_head*head_dim) → (B, n_kv_head, T, head_dim)
  V: (B, T, n_kv_head*head_dim) → (B, n_kv_head, T, head_dim)
  ↓
After Attention: (B, n_head, T, head_dim) → (B, T, n_embd)
  ↓
MLP: (B, T, n_embd) → (B, T, hidden_dim) → (B, T, n_embd)
  ↓
Final: (B, T, n_embd) → (B, T, vocab_size)
```

Where:
- `B` = Batch size
- `T` = Sequence length (context window)
- `n_embd` = Embedding dimension
- `n_head` = Number of query attention heads
- `n_kv_head` = Number of key/value attention heads (for GQA)
- `head_dim` = n_embd / n_head
- `hidden_dim` = n_embd * mlp_ratio (typically 4.0)

Note: The core architecture remains identical both while training and generating/inference, but the execution flow and some behaviors differ based on the mode.

## 🏗️ Architecture Details

### What Makes This Implementation Efficient?

1. **RoPE (Rotary Position Embeddings)**
   - Better extrapolation to longer sequences than learned embeddings
   - Relative position encoding with rotation matrices
   - No learned parameters for positions

2. **Grouped Query Attention (GQA)**
   - Reduces KV cache memory by 2-4x
   - Fewer key/value heads than query heads
   - Near-identical performance to full Multi-Head Attention

3. **SwiGLU Activation**
   - Combination of Swish activation and gating mechanism
   - Empirically better than GELU for language modeling
   - Used in PaLM, LLaMA models

4. **RMSNorm**
   - Simpler than LayerNorm (no mean centering)
   - ~10-15% faster
   - Same performance as LayerNorm

5. **Flash Attention**
   - Automatically used if available (PyTorch 2.0+)
   - 2-4x speedup on attention computation
   - Reduced memory usage

### Model Configuration

The model is highly configurable through `GPTConfig`:

```python
@dataclass
class GPTConfig:
    block_size: int = 256        # Context length
    vocab_size: int = 8192       # Vocabulary size
    n_layer: int = 6             # Number of transformer layers
    n_head: int = 6              # Number of attention heads
    n_kv_head: int = 3           # Number of KV heads (GQA)
    n_embd: int = 384            # Embedding dimension
    mlp_ratio: float = 4.0       # MLP expansion ratio
    dropout: float = 0.1         # Dropout probability
    bias: bool = False           # Use bias in linear layers
    use_rms_norm: bool = True    # Use RMSNorm vs LayerNorm
    use_swiglu: bool = True      # Use SwiGLU vs GELU MLP
    gradient_checkpointing: bool = False  # Memory optimization
```

## 💡 Performance Tips

### Training Faster

1. **Use mixed precision**: Set `dtype = 'bfloat16'` or `'float16'` in config
2. **Enable compilation**: Set `compile_model = True` (PyTorch 2.0+ with CUDA)
3. **Increase batch size**: Max out your GPU memory with larger batches
4. **Use gradient accumulation**: Simulate larger batches without memory increase
5. **Optimize data loading**: Use memory-mapped files for large datasets

### Training on Limited Hardware

1. **Use smaller model**: Start with 'tiny' or 'small' presets
2. **Enable gradient checkpointing**: Trades compute for memory
3. **Reduce batch size**: Compensate with gradient accumulation
4. **Reduce context length**: Shorter sequences use less memory
5. **Use CPU**: Training is slower but works for small models

### Example: Training on M1/M2 Mac

```python
# config/train_default.py
model_preset = 'small'
batch_size = 8
max_iters = 5000
device = 'mps'  # Metal Performance Shaders
dtype = 'float32'  # MPS doesn't support bfloat16 yet
compile_model = False  # Not supported on MPS
```

### Example: Large Model on GPU

```python
# config/train_default.py
model_preset = 'large'
batch_size = 16
gradient_accumulation_steps = 4  # Effective batch size: 64
max_iters = 50000
device = 'cuda'
dtype = 'bfloat16'
compile_model = True
gradient_checkpointing = True  # If OOM
```

## 🔬 Advanced Usage

### Custom Model Architecture

Instead of using presets, you can define custom architectures:

```python
from model.transformer import GPT, GPTConfig

config = GPTConfig(
    block_size=512,
    vocab_size=50257,  # GPT-2 vocab size
    n_layer=12,
    n_head=12,
    n_kv_head=4,       # GQA with 4 KV heads
    n_embd=768,
    dropout=0.1,
)

model = GPT(config)
```

### Using the Model in Your Code

```python
import torch
from model.transformer import GPT, create_model

# Create a model
model = create_model('small')

# Forward pass
input_ids = torch.randint(0, 8192, (1, 128))
logits, loss = model(input_ids)

# Generation
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)
```

### Monitoring Training

The training script outputs Model FLOPS Utilization (MFU), which estimates how efficiently you're using your hardware:

- **MFU < 10%**: Bottleneck in data loading or system
- **MFU 10-30%**: Normal for small models or CPU training
- **MFU 30-50%**: Good GPU utilization
- **MFU 50%+**: Excellent (difficult to achieve)

### Loading Checkpoints

```python
import torch
from model.transformer import GPT

# Load checkpoint
checkpoint = torch.load('out/ckpt.pt')
model_config = checkpoint['model_config']
model = GPT(model_config)
model.load_state_dict(checkpoint['model'])

# Access vocabulary
vocab = checkpoint['vocab']
stoi = vocab['stoi']  # string to int
itos = vocab['itos']  # int to string
```

## 🧪 Testing the Model

Test the model implementation:

```bash
python model/transformer.py
```

This runs a forward pass and generation test to verify everything works.

## 📈 Expected Results

With the default Shakespeare dataset:

- **Training loss**: Should drop from ~4.0 to ~1.0-1.5 after 5000 iterations
- **Validation loss**: Should be similar to training loss (little overfitting)
- **Generation quality**: After 2000-3000 iterations, should generate recognizable Shakespeare-like text
- **Training time** (small model on RTX 3090): ~10 minutes for 5000 iterations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for improvement:
- BPE tokenizer integration
- Multi-GPU training support
- Weights & Biases logging
- More efficient data loading
- Additional sampling strategies

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This implementation incorporates techniques from:
- **Attention Is All You Need** (Vaswani et al., 2017) - Original Transformer
- **RoFormer** (Su et al., 2021) - Rotary Position Embeddings
- **PaLM** (Chowdhery et al., 2022) - SwiGLU activation
- **LLaMA** (Touvron et al., 2023) - RMSNorm, GQA architecture
- **GPT-2** (Radford et al., 2019) - Language model pretraining
- **Flash Attention** (Dao et al., 2022) - Efficient attention

## 📚 Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this project
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
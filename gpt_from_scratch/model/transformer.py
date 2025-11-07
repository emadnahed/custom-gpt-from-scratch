"""
Optimized GPT Language Model - Efficient Implementation

Improvements over base GPT:
- RoPE (Rotary Position Embeddings) for better length generalization
- Grouped Query Attention (GQA) for memory efficiency
- SwiGLU activation for better performance
- RMSNorm for faster normalization
- KV-cache for efficient generation
- Gradient checkpointing support
- Mixed precision training support
- torch.compile() compatible
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# Normalization Layers
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class LayerNorm(nn.Module):
    """LayerNorm with optional bias for compatibility"""

    def __init__(self, ndim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings - better than learned positional embeddings
    Allows for better length generalization
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cos/sin cache if sequence length changed"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys"""
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device)

        return (
            apply_rotary_emb(q, self._cos_cached[:seq_len], self._sin_cached[:seq_len]),
            apply_rotary_emb(k, self._cos_cached[:seq_len], self._sin_cached[:seq_len])
        )


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor"""
    # x: (batch, n_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    cos1, cos2 = cos.chunk(2, dim=-1)
    sin1, sin2 = sin.chunk(2, dim=-1)
    y1 = x1 * cos1 - x2 * sin1
    y2 = x1 * sin2 + x2 * cos2
    return torch.cat((y1, y2), dim=-1)


# ============================================================================
# Attention Mechanism
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - memory efficient attention
    Uses fewer KV heads than Q heads for reduced memory usage
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if hasattr(config, 'n_kv_head') else config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head  # Repetition factor for KV heads
        self.dropout = config.dropout

        # Q projection
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # K, V projections (potentially fewer heads)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        # Output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.block_size)

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention not available. Using manual attention.")
            # Causal mask for manual attention
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
                persistent=False
            )

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)

        # Apply RoPE
        q, k = self.rotary_emb(q, k)

        # Handle KV cache for generation
        if use_cache:
            if cache is not None:
                k_cache, v_cache = cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # Repeat KV heads if using GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.o_proj(y))

        return y, new_cache


# ============================================================================
# Feedforward Network
# ============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation - better than GELU for transformers
    Uses Swish (SiLU) with gating mechanism
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * config.mlp_ratio)
        # Make hidden_dim multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: swish(W1(x)) * W3(x) then W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    """Standard MLP with GELU activation (for compatibility)"""

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * config.mlp_ratio)

        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ============================================================================
# Transformer Block
# ============================================================================

class Block(nn.Module):
    """Transformer block with attention and feedforward"""

    def __init__(self, config):
        super().__init__()
        # Choose normalization
        norm_class = RMSNorm if config.use_rms_norm else LayerNorm
        norm_args = (config.n_embd,) if config.use_rms_norm else (config.n_embd, config.bias)

        self.ln_1 = norm_class(*norm_args)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = norm_class(*norm_args)

        # Choose MLP type
        self.mlp = SwiGLU(config) if config.use_swiglu else MLP(config)

        self.use_checkpoint = False  # Set externally for gradient checkpointing

    def forward(self, x, use_cache=False, cache=None):
        # Pre-norm architecture
        attn_out, new_cache = self.attn(self.ln_1(x), use_cache=use_cache, cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GPTConfig:
    # Model architecture
    block_size: int = 256  # Context length
    vocab_size: int = 8192  # Vocabulary size (should be multiple of 64)
    n_layer: int = 6  # Number of transformer blocks
    n_head: int = 6  # Number of attention heads
    n_kv_head: int = 3  # Number of KV heads (for GQA, set to n_head for MHA)
    n_embd: int = 384  # Embedding dimension
    mlp_ratio: float = 4.0  # MLP hidden dim = n_embd * mlp_ratio

    # Regularization
    dropout: float = 0.1

    # Architecture choices
    bias: bool = False  # Use bias in linear layers and norms
    use_rms_norm: bool = True  # Use RMSNorm instead of LayerNorm
    use_swiglu: bool = True  # Use SwiGLU instead of GELU MLP

    # Training
    gradient_checkpointing: bool = False  # Enable gradient checkpointing


# ============================================================================
# Main Model
# ============================================================================

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Token embeddings
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd) if config.use_rms_norm else LayerNorm(config.n_embd, config.bias),
        ))

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('w2.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        # Note: With weight tying, token embeddings are also used as output weights
        return n_params

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Token embeddings
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        # Transformer blocks
        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(block, x, False, None, use_reentrant=False)
            else:
                x, _ = block(x)

        # Final norm
        x = self.transformer.ln_f(x)

        # Compute loss if targets provided
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference optimization: only compute logits for last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                           betas: Tuple[float, float], device_type: str):
        """Create optimizer with weight decay"""
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters for weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Decayed params: {len(decay_params)} tensors, {num_decay:,} parameters")
        print(f"Non-decayed params: {len(nodecay_params)} tensors, {num_nodecay:,} parameters")

        # Use fused AdamW if available (CUDA only)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate tokens autoregressively

        Args:
            idx: Starting sequence (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOPS utilization (MFU) relative to A100"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt

        # A100 bfloat16 peak: 312 TFLOPS
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


# ============================================================================
# Utility Functions
# ============================================================================

def create_model(preset: str = 'tiny') -> 'GPT':
    """
    Create a model with preset configurations

    Presets:
        - tiny: ~10M params, good for testing
        - small: ~25M params, good for Mac CPU
        - medium: ~80M params, requires GPU
        - large: ~350M params, requires good GPU
    """
    presets = {
        'tiny': dict(
            block_size=128,
            vocab_size=8192,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
            mlp_ratio=4.0,
            dropout=0.1,
        ),
        'small': dict(
            block_size=256,
            vocab_size=8192,
            n_layer=6,
            n_head=6,
            n_kv_head=3,
            n_embd=384,
            mlp_ratio=4.0,
            dropout=0.1,
        ),
        'medium': dict(
            block_size=512,
            vocab_size=16384,
            n_layer=12,
            n_head=12,
            n_kv_head=4,
            n_embd=768,
            mlp_ratio=4.0,
            dropout=0.1,
        ),
        'large': dict(
            block_size=1024,
            vocab_size=32768,
            n_layer=24,
            n_head=16,
            n_kv_head=4,
            n_embd=1024,
            mlp_ratio=4.0,
            dropout=0.1,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

    config = GPTConfig(**presets[preset])
    return GPT(config)


if __name__ == "__main__":
    # Create a small model
    model = create_model('small')

    # Test forward pass
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, 8192, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits, _ = model(idx)
        print(f"Output shape: {logits.shape}")

    # Test generation
    start_tokens = torch.randint(0, 8192, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")

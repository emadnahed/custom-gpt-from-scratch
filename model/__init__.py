"""
GPT Model Package

This package contains the optimized GPT transformer implementation.
"""

from model.transformer import (
    GPT,
    GPTConfig,
    create_model,
    RMSNorm,
    LayerNorm,
    RotaryEmbedding,
    GroupedQueryAttention,
    SwiGLU,
    MLP,
    Block,
)

__all__ = [
    'GPT',
    'GPTConfig',
    'create_model',
    'RMSNorm',
    'LayerNorm',
    'RotaryEmbedding',
    'GroupedQueryAttention',
    'SwiGLU',
    'MLP',
    'Block',
]

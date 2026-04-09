"""KV cache memory helpers."""
from __future__ import annotations

from .operator import Dtype


def kv_cache_bytes(batch_size: int, seq_len: int, num_layers: int,
                   n_kv_heads: int, head_dim: int, dtype: Dtype) -> int:
    """Total KV cache memory in bytes.

    KV cache = 2 (K+V) * batch * seq * layers * kv_heads * head_dim * dtype_bytes
    """
    return int(2 * batch_size * seq_len * num_layers * n_kv_heads * head_dim * dtype.bytes)


def kv_cache_per_token_bytes(num_layers: int, n_kv_heads: int,
                              head_dim: int, dtype: Dtype) -> int:
    """KV cache bytes per token per sequence."""
    return int(2 * num_layers * n_kv_heads * head_dim * dtype.bytes)

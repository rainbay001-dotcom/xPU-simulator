"""Serving configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServingConfig:
    """Configuration for serving simulation.

    Attributes:
        max_batch_size: Maximum number of sequences in a batch.
        max_seq_len: Maximum total sequence length (prompt + generated).
        max_tokens_budget: Maximum tokens per iteration (prefill + decode).
            Controls how many new tokens can be admitted while decoding.
        block_size: KV cache block size in tokens.
        num_kv_blocks: Total number of KV cache blocks available.
    """
    max_batch_size: int = 32
    max_seq_len: int = 4096
    max_tokens_budget: int = 4096
    block_size: int = 16
    num_kv_blocks: int = 1024

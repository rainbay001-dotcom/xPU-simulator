"""Batch scheduler with continuous batching."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import ServingConfig
from .request import Request, RequestState
from .kv_cache import KVCacheAllocator


@dataclass
class ScheduledBatch:
    """A batch of requests scheduled for one iteration."""
    prefill_requests: list[Request] = field(default_factory=list)
    decode_requests: list[Request] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens in this batch (prefill prompt tokens + decode tokens)."""
        prefill_tokens = sum(r.prompt_len for r in self.prefill_requests)
        decode_tokens = len(self.decode_requests)
        return prefill_tokens + decode_tokens

    @property
    def total_sequences(self) -> int:
        return len(self.prefill_requests) + len(self.decode_requests)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests


class BatchScheduler:
    """Continuous batching scheduler.

    Each iteration:
    1. Continue decoding for all active decode requests.
    2. Admit new prefill requests if budget and batch size allow.
    """

    def __init__(self, config: ServingConfig, kv_allocator: KVCacheAllocator):
        self.config = config
        self.kv_allocator = kv_allocator

    def schedule(self, waiting: list[Request],
                 decoding: list[Request]) -> ScheduledBatch:
        """Schedule one iteration.

        Args:
            waiting: Requests waiting to be prefilled (FCFS order).
            decoding: Requests currently in decode phase.

        Returns:
            ScheduledBatch with prefill and decode requests for this iteration.
        """
        batch = ScheduledBatch()

        # 1. All active decode requests continue
        for req in decoding:
            if batch.total_sequences >= self.config.max_batch_size:
                break
            batch.decode_requests.append(req)

        budget_left = self.config.max_tokens_budget - batch.total_tokens

        # 2. Admit new prefill requests if budget allows
        admitted = []
        for req in waiting:
            if batch.total_sequences >= self.config.max_batch_size:
                break
            if req.prompt_len > budget_left:
                break  # Can't fit this prompt
            # Check KV cache
            blocks_needed = req.blocks_needed(self.kv_allocator.block_size)
            if not self.kv_allocator.allocate(req.id, blocks_needed):
                break  # No KV cache space
            batch.prefill_requests.append(req)
            budget_left -= req.prompt_len
            admitted.append(req)

        return batch

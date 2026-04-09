"""Block-based KV cache allocator for serving simulation."""
from __future__ import annotations


class KVCacheAllocator:
    """Simple block-based KV cache allocator.

    Tracks allocated blocks per request. Each block holds ``block_size`` tokens.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._allocated: dict[int, int] = {}  # request_id -> num_blocks

    @property
    def free_blocks(self) -> int:
        return self.num_blocks - sum(self._allocated.values())

    @property
    def used_blocks(self) -> int:
        return sum(self._allocated.values())

    def allocate(self, request_id: int, num_blocks: int) -> bool:
        """Try to allocate blocks for a request. Returns False if insufficient."""
        current = self._allocated.get(request_id, 0)
        needed = num_blocks - current
        if needed <= 0:
            return True  # Already have enough
        if needed > self.free_blocks:
            return False
        self._allocated[request_id] = num_blocks
        return True

    def free(self, request_id: int) -> int:
        """Free all blocks for a request. Returns number freed."""
        freed = self._allocated.pop(request_id, 0)
        return freed

    def blocks_for(self, request_id: int) -> int:
        """Number of blocks currently allocated for a request."""
        return self._allocated.get(request_id, 0)

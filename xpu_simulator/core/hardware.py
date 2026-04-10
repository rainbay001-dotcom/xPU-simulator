"""Hardware specification abstractions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MemLevel:
    """A level in the memory hierarchy."""
    name: str
    size_bytes: int
    bandwidth_GBs: float  # GB/s


class HardwareSpec(ABC):
    """Abstract base class for hardware specifications."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def peak_flops(self) -> dict[str, float]:
        """Peak FLOPS by dtype, e.g. {"fp16": 3.12e14, "fp32": 1.56e13}."""
        ...

    @property
    @abstractmethod
    def memory_hierarchy(self) -> list[MemLevel]:
        """Memory levels from fastest/smallest to slowest/largest."""
        ...

    # Fallback chain: if a dtype isn't supported, use the next-wider dtype
    _DTYPE_FALLBACKS = {
        "fp8": ["fp16", "bf16", "fp32"],
        "int4": ["int8", "fp16", "bf16", "fp32"],
        "int8": ["fp16", "bf16", "fp32"],
        "bf16": ["fp16", "fp32"],
        "fp16": ["bf16", "fp32"],
        "fp32": [],
    }

    def peak_flops_for(self, dtype: str) -> float:
        """Get peak FLOPS for a given dtype, with graceful fallback.

        If the hardware doesn't support a dtype (e.g., FP8 on A100),
        falls back to the next-wider dtype in the chain.
        """
        if dtype in self.peak_flops:
            return self.peak_flops[dtype]
        for fallback in self._DTYPE_FALLBACKS.get(dtype, []):
            if fallback in self.peak_flops:
                return self.peak_flops[fallback]
        return self.peak_flops.get("fp32", 0)

    def main_memory_bandwidth(self) -> float:
        """Bandwidth of the outermost (main) memory level in GB/s."""
        return self.memory_hierarchy[-1].bandwidth_GBs

    def effective_bandwidth(self, working_set_bytes: int) -> float:
        """Return bandwidth (GB/s) of the smallest memory level that fits the working set.

        Walks the hierarchy from smallest/fastest to largest/slowest.
        If the working set fits in a level, that level's bandwidth is used.
        Falls back to main memory bandwidth.
        """
        for level in self.memory_hierarchy:
            if working_set_bytes <= level.size_bytes:
                return level.bandwidth_GBs
        return self.main_memory_bandwidth()

    def roofline_limit(self, flops: float, bytes_accessed: float, dtype: str = "fp16") -> float:
        """
        Roofline model: returns estimated latency in microseconds.

        The operation is bound by either:
        - Compute: flops / peak_flops
        - Memory: bytes / bandwidth
        """
        peak = self.peak_flops_for(dtype)
        bw = self.main_memory_bandwidth() * 1e9  # Convert GB/s to B/s

        compute_time_s = flops / peak if peak > 0 else float("inf")
        memory_time_s = bytes_accessed / bw if bw > 0 else float("inf")

        return max(compute_time_s, memory_time_s) * 1e6  # to microseconds

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

    def peak_flops_for(self, dtype: str) -> float:
        """Get peak FLOPS for a given dtype."""
        return self.peak_flops.get(dtype, self.peak_flops.get("fp32", 0))

    def main_memory_bandwidth(self) -> float:
        """Bandwidth of the outermost (main) memory level in GB/s."""
        return self.memory_hierarchy[-1].bandwidth_GBs

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

"""Cost model abstractions and roofline implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .hardware import HardwareSpec
from .operator import OpSpec


@dataclass
class OpCost:
    """Cost estimate for a single operation."""
    compute_us: float       # Compute-limited latency (microseconds)
    memory_us: float        # Memory-limited latency (microseconds)
    latency_us: float       # Actual estimated latency = max(compute, memory)
    bound: str              # "compute" or "memory"
    flops: int              # Total FLOPs
    bytes_accessed: int     # Total bytes read + written
    utilization: float      # Fraction of peak utilized (0-1)

    @property
    def arithmetic_intensity(self) -> float:
        if self.bytes_accessed == 0:
            return float("inf")
        return self.flops / self.bytes_accessed


class CostModel(ABC):
    """Abstract base class for cost models."""

    def __init__(self, hw: HardwareSpec):
        self.hw = hw

    @abstractmethod
    def estimate(self, op: OpSpec) -> OpCost:
        """Estimate cost of executing an operation on this hardware."""
        ...


class RooflineCostModel(CostModel):
    """Basic roofline cost model — works for any hardware."""

    def estimate(self, op: OpSpec) -> OpCost:
        dtype = op.inputs[0].dtype.value if op.inputs else "fp16"
        flops = op.flops
        mem_bytes = op.memory_bytes

        peak = self.hw.peak_flops_for(dtype)
        bw = self.hw.main_memory_bandwidth() * 1e9  # B/s

        compute_us = (flops / peak * 1e6) if peak > 0 else 0.0
        memory_us = (mem_bytes / bw * 1e6) if bw > 0 else 0.0
        latency_us = max(compute_us, memory_us)

        bound = "compute" if compute_us >= memory_us else "memory"

        # Utilization: how much of peak we'd use if running at this latency
        if latency_us > 0 and peak > 0:
            utilization = flops / (latency_us * 1e-6 * peak)
        else:
            utilization = 0.0

        return OpCost(
            compute_us=compute_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=min(utilization, 1.0),
        )

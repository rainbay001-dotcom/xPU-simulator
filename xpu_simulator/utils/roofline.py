"""Roofline analysis utilities."""
from __future__ import annotations

from ..core.hardware import HardwareSpec


def arithmetic_intensity(flops: int, bytes_accessed: int) -> float:
    """Compute arithmetic intensity (FLOPs/byte)."""
    if bytes_accessed == 0:
        return float("inf")
    return flops / bytes_accessed


def roofline_bound(
    flops: int,
    bytes_accessed: int,
    peak_flops: float,
    bandwidth_GBs: float,
) -> tuple[float, str]:
    """
    Determine if an operation is compute-bound or memory-bound.

    Returns:
        (latency_us, bound_type) where bound_type is "compute" or "memory"
    """
    bw = bandwidth_GBs * 1e9  # to B/s
    compute_us = (flops / peak_flops * 1e6) if peak_flops > 0 else 0.0
    memory_us = (bytes_accessed / bw * 1e6) if bw > 0 else 0.0

    if compute_us >= memory_us:
        return compute_us, "compute"
    return memory_us, "memory"


def ridge_point(peak_flops: float, bandwidth_GBs: float) -> float:
    """
    Compute the ridge point — arithmetic intensity where compute and memory
    ceilings intersect. Operations above this are compute-bound.

    Returns:
        FLOPs/byte at ridge point
    """
    bw = bandwidth_GBs * 1e9
    if bw == 0:
        return float("inf")
    return peak_flops / bw

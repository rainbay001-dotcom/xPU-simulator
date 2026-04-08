"""NVIDIA GPU hardware specifications."""
from __future__ import annotations

from ...core.hardware import HardwareSpec, MemLevel


class GPUSpec(HardwareSpec):
    """NVIDIA GPU hardware spec."""

    def __init__(
        self,
        name: str,
        sm_count: int,
        clock_ghz: float,
        peak_flops_map: dict[str, float],
        l1_size_kb: int,
        l1_bw_GBs: float,
        l2_size_mb: int,
        l2_bw_GBs: float,
        hbm_size_gb: int,
        hbm_bw_GBs: float,
    ):
        self._name = name
        self.sm_count = sm_count
        self.clock_ghz = clock_ghz
        self._peak_flops = peak_flops_map
        self._memory_hierarchy = [
            MemLevel("L1", l1_size_kb * 1024, l1_bw_GBs),
            MemLevel("L2", l2_size_mb * 1024 * 1024, l2_bw_GBs),
            MemLevel("HBM", hbm_size_gb * 1024**3, hbm_bw_GBs),
        ]

    @property
    def name(self) -> str:
        return self._name

    @property
    def peak_flops(self) -> dict[str, float]:
        return self._peak_flops

    @property
    def memory_hierarchy(self) -> list[MemLevel]:
        return self._memory_hierarchy


# --- Presets ---

A100_80GB = GPUSpec(
    name="NVIDIA A100 80GB",
    sm_count=108,
    clock_ghz=1.41,
    peak_flops_map={
        "fp32": 19.5e12,
        "fp16": 312e12,     # with tensor cores
        "bf16": 312e12,
        "int8": 624e12,
    },
    l1_size_kb=192,          # per SM
    l1_bw_GBs=19200,         # aggregate across SMs
    l2_size_mb=40,
    l2_bw_GBs=6400,
    hbm_size_gb=80,
    hbm_bw_GBs=2039,
)

H100_80GB = GPUSpec(
    name="NVIDIA H100 80GB",
    sm_count=132,
    clock_ghz=1.83,
    peak_flops_map={
        "fp32": 67e12,
        "fp16": 989e12,
        "bf16": 989e12,
        "fp8": 1979e12,
        "int8": 1979e12,
    },
    l1_size_kb=256,
    l1_bw_GBs=33792,
    l2_size_mb=50,
    l2_bw_GBs=12000,
    hbm_size_gb=80,
    hbm_bw_GBs=3350,
)

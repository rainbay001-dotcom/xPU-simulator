"""NVIDIA GPU hardware specifications."""
from __future__ import annotations

from ...core.hardware import HardwareSpec, MemLevel
from ...core.parallel import InterconnectSpec


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
        cuda_core_flops_map: dict[str, float] | None = None,
        efficiency_factors: dict[str, float] | None = None,
        interconnect: InterconnectSpec | None = None,
    ):
        self._name = name
        self.interconnect = interconnect
        self.sm_count = sm_count
        self.clock_ghz = clock_ghz
        self._peak_flops = peak_flops_map
        self._cuda_core_flops = cuda_core_flops_map or {}
        self.efficiency_factors = efficiency_factors or {}
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

    def cuda_core_flops_for(self, dtype: str) -> float:
        """Get CUDA core (SIMT) peak FLOPS for a given dtype."""
        if self._cuda_core_flops:
            return self._cuda_core_flops.get(dtype, self._cuda_core_flops.get("fp32", 0))
        return self.peak_flops_for(dtype) / 4.0

    def get_efficiency(self, category: str) -> float:
        """Get efficiency factor for an op category. Returns 1.0 if not set."""
        return self.efficiency_factors.get(category, 1.0)


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
    cuda_core_flops_map={
        "fp32": 19.5e12,
        "fp16": 78e12,      # SIMT FP16: ~1/4 of TC peak
        "bf16": 78e12,
        "int8": 156e12,
    },
    efficiency_factors={
        "matmul_fp16": 0.70,
        "matmul_fp32": 0.65,
        "elementwise_fp16": 0.85,
        "elementwise_fp32": 0.85,
        "memory": 0.80,
        "static_tc_us": 5.0,     # per-op overhead for Tensor Core ops
        "static_cuda_us": 2.0,   # per-op overhead for CUDA Core ops
    },
    interconnect=InterconnectSpec("NVLink", 600, 0.5),
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
    cuda_core_flops_map={
        "fp32": 67e12,
        "fp16": 247e12,     # SIMT FP16: ~1/4 of TC peak
        "bf16": 247e12,
        "fp8": 495e12,
        "int8": 495e12,
    },
    efficiency_factors={
        "matmul_fp16": 0.70,
        "matmul_fp32": 0.65,
        "elementwise_fp16": 0.85,
        "elementwise_fp32": 0.85,
        "memory": 0.80,
        "static_tc_us": 5.0,
        "static_cuda_us": 2.0,
    },
    interconnect=InterconnectSpec("NVLink", 900, 0.5),
)

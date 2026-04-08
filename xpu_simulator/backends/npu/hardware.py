"""Huawei Ascend NPU hardware specifications."""
from __future__ import annotations

from ...core.hardware import HardwareSpec, MemLevel


class AscendSpec(HardwareSpec):
    """Huawei Ascend NPU hardware spec.

    Ascend architecture has:
    - AI Cores with CUBE (matrix), VECTOR, and SCALAR units
    - Memory hierarchy: GM -> L2 -> L1 -> L0A/L0B/L0C -> UB
    - Data pipes: PIPE_M (matrix), PIPE_V (vector), PIPE_S (scalar), MTE (data move)
    """

    def __init__(
        self,
        name: str,
        ai_core_count: int,
        cube_peak_tflops: dict[str, float],   # by dtype
        vector_peak_tflops: dict[str, float],  # by dtype
        cube_tile_size: int,                    # e.g., 16 for fp16
        ub_size_kb: int,       # Unified Buffer per core
        l0a_size_kb: int,
        l0b_size_kb: int,
        l0c_size_kb: int,
        l1_size_kb: int,       # L1 buffer (shared)
        l2_size_mb: int,
        gm_size_gb: int,
        ub_bw_GBs: float,     # UB bandwidth
        l0_bw_GBs: float,     # L0 bandwidth
        l1_bw_GBs: float,     # L1 bandwidth
        l2_bw_GBs: float,
        gm_bw_GBs: float,     # Global memory (HBM) bandwidth
    ):
        self._name = name
        self.ai_core_count = ai_core_count
        self._cube_peak = cube_peak_tflops
        self._vector_peak = vector_peak_tflops
        self.cube_tile_size = cube_tile_size
        self.ub_size_kb = ub_size_kb

        self._peak_flops = {}
        for dtype in set(list(cube_peak_tflops.keys()) + list(vector_peak_tflops.keys())):
            cube = cube_peak_tflops.get(dtype, 0)
            vector = vector_peak_tflops.get(dtype, 0)
            self._peak_flops[dtype] = max(cube, vector) * 1e12  # TFLOPS -> FLOPS

        self._memory_hierarchy = [
            MemLevel("UB", ub_size_kb * 1024, ub_bw_GBs),
            MemLevel("L0A", l0a_size_kb * 1024, l0_bw_GBs),
            MemLevel("L0B", l0b_size_kb * 1024, l0_bw_GBs),
            MemLevel("L0C", l0c_size_kb * 1024, l0_bw_GBs),
            MemLevel("L1", l1_size_kb * 1024, l1_bw_GBs),
            MemLevel("L2", l2_size_mb * 1024 * 1024, l2_bw_GBs),
            MemLevel("GM", gm_size_gb * 1024**3, gm_bw_GBs),
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

    def cube_peak_for(self, dtype: str) -> float:
        """CUBE unit peak FLOPS for a given dtype."""
        return self._cube_peak.get(dtype, 0) * 1e12

    def vector_peak_for(self, dtype: str) -> float:
        """VECTOR unit peak FLOPS for a given dtype."""
        return self._vector_peak.get(dtype, 0) * 1e12

    def get_mem_level(self, name: str) -> MemLevel:
        """Get a specific memory level by name."""
        for level in self._memory_hierarchy:
            if level.name == name:
                return level
        raise ValueError(f"Unknown memory level: {name}")


# --- Presets ---

# Ascend 910B (Atlas 800T A2)
ASCEND_910B = AscendSpec(
    name="Ascend 910B",
    ai_core_count=30,
    cube_peak_tflops={"fp16": 320, "bf16": 320, "int8": 640, "fp32": 160},
    vector_peak_tflops={"fp16": 10, "bf16": 10, "fp32": 5},
    cube_tile_size=16,
    ub_size_kb=192,
    l0a_size_kb=64,
    l0b_size_kb=64,
    l0c_size_kb=256,
    l1_size_kb=1024,
    l2_size_mb=192,
    gm_size_gb=64,
    ub_bw_GBs=8192,
    l0_bw_GBs=16384,
    l1_bw_GBs=4096,
    l2_bw_GBs=3200,
    gm_bw_GBs=1600,
)

# Ascend 910C
ASCEND_910C = AscendSpec(
    name="Ascend 910C",
    ai_core_count=32,
    cube_peak_tflops={"fp16": 400, "bf16": 400, "int8": 800, "fp32": 200},
    vector_peak_tflops={"fp16": 12.5, "bf16": 12.5, "fp32": 6.25},
    cube_tile_size=16,
    ub_size_kb=256,
    l0a_size_kb=64,
    l0b_size_kb=64,
    l0c_size_kb=256,
    l1_size_kb=2048,
    l2_size_mb=256,
    gm_size_gb=64,
    ub_bw_GBs=10240,
    l0_bw_GBs=20480,
    l1_bw_GBs=5120,
    l2_bw_GBs=4000,
    gm_bw_GBs=1800,
)

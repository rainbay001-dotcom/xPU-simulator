"""Huawei Ascend NPU hardware specifications."""
from __future__ import annotations

from ...core.hardware import HardwareSpec, MemLevel
from ...core.parallel import InterconnectSpec


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
        *,
        # CA-model pipeline parameters
        mte_bw_GBs: float = 0.0,           # Per-core MTE bandwidth GM→UB
        mte_l2_bw_GBs: float = 0.0,        # Per-core MTE bandwidth L2→UB
        double_buffer: bool = True,          # Double-buffering enabled
        pipeline_startup_us: float = 2.0,    # Pipeline fill latency
        pipeline_drain_us: float = 1.0,      # Pipeline drain latency
        interconnect: InterconnectSpec | None = None,
    ):
        self._name = name
        self.interconnect = interconnect
        self.ai_core_count = ai_core_count
        self._cube_peak = cube_peak_tflops
        self._vector_peak = vector_peak_tflops
        self.cube_tile_size = cube_tile_size
        self.ub_size_kb = ub_size_kb
        # CA-model pipeline params (default MTE bw from aggregate / core_count)
        self.mte_bw_GBs = mte_bw_GBs if mte_bw_GBs > 0 else gm_bw_GBs / ai_core_count
        self.mte_l2_bw_GBs = mte_l2_bw_GBs if mte_l2_bw_GBs > 0 else self.mte_bw_GBs * 2
        self.double_buffer = double_buffer
        self.pipeline_startup_us = pipeline_startup_us
        self.pipeline_drain_us = pipeline_drain_us

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

    def per_core_cube_peak(self, dtype: str) -> float:
        """CUBE peak FLOPS for a single AI core."""
        return self.cube_peak_for(dtype) / self.ai_core_count

    def per_core_vector_peak(self, dtype: str) -> float:
        """VECTOR peak FLOPS for a single AI core."""
        return self.vector_peak_for(dtype) / self.ai_core_count

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
    cube_peak_tflops={"fp16": 320, "bf16": 320, "int8": 640, "fp8": 640, "fp32": 160},
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
    mte_bw_GBs=53.3,        # 1600 / 30 cores
    mte_l2_bw_GBs=106.7,    # ~2x GM per-core
    pipeline_startup_us=2.0,
    pipeline_drain_us=1.0,
    interconnect=InterconnectSpec("HCCS", 392, 1.0),
)

# Ascend 910C
ASCEND_910C = AscendSpec(
    name="Ascend 910C",
    ai_core_count=32,
    cube_peak_tflops={"fp16": 400, "bf16": 400, "int8": 800, "fp8": 800, "fp32": 200},
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
    mte_bw_GBs=56.3,        # 1800 / 32 cores
    mte_l2_bw_GBs=125.0,    # ~2x GM per-core
    pipeline_startup_us=1.5,
    pipeline_drain_us=0.8,
    interconnect=InterconnectSpec("HCCS", 600, 1.0),
)

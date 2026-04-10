"""Ascend NPU cost model with CA-style tiled pipeline simulation.

Models the DaVinci AI core's 3-stage pipeline (MTE → CUBE → VECTOR)
with double-buffering, tile-level cost breakdown, and L2 reuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ...core.cost_model import CostModel, OpCost
from ...core.operator import OpSpec, OpType
from .hardware import AscendSpec


# Ops that run on the CUBE unit (matrix engine)
_CUBE_OPS = {OpType.MATMUL, OpType.CONV2D}

# Ops that run on the VECTOR unit
_VECTOR_OPS = {
    OpType.RELU, OpType.ADD, OpType.GELU, OpType.SILU, OpType.MUL,
    OpType.LAYER_NORM, OpType.SOFTMAX, OpType.ROPE, OpType.EMBEDDING,
    OpType.GATHER, OpType.ALL_REDUCE, OpType.ALL_TO_ALL,
    OpType.ALL_GATHER, OpType.REDUCE_SCATTER, OpType.DEQUANT,
}

# Data layout preferences per op type
_LAYOUT_MAP = {
    OpType.MATMUL: "Fractal_NZ",
    OpType.CONV2D: "NC1HWC0",
    OpType.RELU: "NC1HWC0",
    OpType.ADD: "NC1HWC0",
    OpType.GELU: "ND",
    OpType.SILU: "ND",
    OpType.MUL: "NC1HWC0",
    OpType.ROPE: "ND",
    OpType.EMBEDDING: "ND",
    OpType.GATHER: "ND",
    OpType.ALL_REDUCE: "ND",
    OpType.ALL_TO_ALL: "ND",
    OpType.ALL_GATHER: "ND",
    OpType.REDUCE_SCATTER: "ND",
    OpType.DEQUANT: "ND",
    OpType.LAYER_NORM: "ND",
    OpType.SOFTMAX: "ND",
}


@dataclass
class _TilingConfig:
    """Tile dimensions and counts for a tiled MatMul."""
    Tm: int
    Tn: int
    Tk: int
    num_m_tiles: int
    num_n_tiles: int
    num_k_tiles: int
    tiles_per_core: int


class NPUCostModel(CostModel):
    """Cost model for Ascend NPU with CA-style tiled pipeline simulation.

    Models the DaVinci architecture's pipeline stages:
    - MTE (Memory Transfer Engine): data movement GM/L2 → UB
    - CUBE: matrix multiplication on systolic array
    - VECTOR: elementwise/reduction operations
    - MTE-out: write results back

    Stages overlap via double-buffering for pipelined execution.
    """

    hw: AscendSpec

    def __init__(self, hw: AscendSpec):
        super().__init__(hw)
        self.hw: AscendSpec = hw

    def estimate(self, op: OpSpec) -> OpCost:
        if op.op_type in _CUBE_OPS:
            return self._estimate_cube(op)
        elif op.op_type in _VECTOR_OPS:
            return self._estimate_vector(op)
        else:
            return self._estimate_vector(op)  # fallback

    # ------------------------------------------------------------------ #
    # CUBE pipeline (MatMul / Conv2D)
    # ------------------------------------------------------------------ #

    def _estimate_cube(self, op: OpSpec) -> OpCost:
        """Estimate cost for CUBE ops using hybrid roofline + tiling model.

        Uses chip-level aggregate bandwidth for memory bound (L2 caches tile
        data transparently) and per-core tiled CUBE model for compute bound.
        The final latency = max(compute, memory) + overhead — matching the
        roofline principle while preserving tile-level accuracy for compute.
        """
        dtype_str = op.inputs[0].dtype.value if op.inputs else "fp16"
        dtype_bytes = op.inputs[0].dtype.bytes if op.inputs else 2
        flops = op.flops
        mem_bytes = op.memory_bytes
        tile = self.hw.cube_tile_size

        # Tile alignment utilization (captures padding waste)
        utilization = self._cube_utilization(op, tile)

        # --- Compute bound: per-core tiled CUBE model ---
        cube_efficiency = self.hw.get_efficiency(f"cube_{dtype_str}")
        chip_cube_peak = self.hw.cube_peak_for(dtype_str) * utilization * cube_efficiency
        compute_us = (flops / chip_cube_peak * 1e6) if chip_cube_peak > 0 else 0.0

        # --- Memory bound: chip-level aggregate bandwidth ---
        # L2 transparently caches tile data; actual GM traffic ≈ unique data
        mem_efficiency = self.hw.get_efficiency("memory")
        gm_bw = self.hw.main_memory_bandwidth() * mem_efficiency * 1e9  # B/s
        memory_us = (mem_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0

        # Pipeline startup + drain (per-op overhead for tiled execution)
        pipeline_us = self.hw.pipeline_startup_us + self.hw.pipeline_drain_us

        # Format conversion overhead
        if op.attrs.get("skip_format_conversion"):
            format_overhead_us = 0.0
        else:
            format_overhead_us = self._format_conversion_cost(op, dtype_str)

        # Static per-op overhead (kernel dispatch, synchronization, etc.)
        static_overhead_us = self.hw.get_efficiency("static_cube_us")

        # Roofline: max(compute, memory) + fixed overhead
        latency_us = max(compute_us, memory_us) + pipeline_us + format_overhead_us + static_overhead_us

        bound = "compute (PIPE_M)" if compute_us >= memory_us else "memory"

        return OpCost(
            compute_us=compute_us + format_overhead_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=utilization,
        )

    # ------------------------------------------------------------------ #
    # VECTOR pipeline (elementwise / reduction ops)
    # ------------------------------------------------------------------ #

    def _estimate_vector(self, op: OpSpec) -> OpCost:
        """Estimate cost for VECTOR ops using tiled 3-stage pipeline.

        Pipeline per tile: MTE_in → VECTOR → MTE_out
        """
        dtype_str = op.inputs[0].dtype.value if op.inputs else "fp16"
        dtype_bytes = op.inputs[0].dtype.bytes if op.inputs else 2
        flops = op.flops
        mem_bytes = op.memory_bytes

        vector_peak = self.hw.per_core_vector_peak(dtype_str)
        vector_efficiency = self.hw.get_efficiency("vector")
        mem_efficiency = self.hw.get_efficiency("memory")
        effective_vector_peak = vector_peak * vector_efficiency
        mte_bw = self.hw.mte_bw_GBs * mem_efficiency * 1e9  # B/s

        # Tiling: partition data into UB-sized chunks
        tile_elements, num_tiles, tiles_per_core = self._compute_vector_tiling(op, dtype_bytes)

        if tile_elements <= 0 or tiles_per_core <= 0:
            # Degenerate case: zero-compute ops (embedding, gather, reshape)
            effective_bw = self.hw.effective_bandwidth(mem_bytes) * mem_efficiency * 1e9
            memory_us = (mem_bytes / effective_bw * 1e6) if mem_bytes > 0 else 0.0
            static_overhead_us = self.hw.get_efficiency("static_vector_us")
            return OpCost(
                compute_us=0.0, memory_us=memory_us,
                latency_us=memory_us + static_overhead_us,
                bound="memory", flops=flops, bytes_accessed=mem_bytes,
                utilization=0.0,
            )

        total_elements = sum(t.numel for t in op.inputs)
        if total_elements == 0:
            total_elements = 1

        # Per-tile costs
        bytes_per_tile_in = tile_elements * dtype_bytes
        bytes_per_tile_out = tile_elements * dtype_bytes
        flops_per_tile = flops * tile_elements / total_elements if total_elements > 0 else 0

        mte_in_us = (bytes_per_tile_in / mte_bw * 1e6) if mte_bw > 0 else 0.0
        vector_us = (flops_per_tile / effective_vector_peak * 1e6) if effective_vector_peak > 0 else 0.0
        mte_out_us = (bytes_per_tile_out / mte_bw * 1e6) if mte_bw > 0 else 0.0

        # Pipeline overlap: per_tile = max(mte_in, vector, mte_out)
        per_tile_us = max(mte_in_us, vector_us, mte_out_us)

        # Total
        per_core_us = tiles_per_core * per_tile_us
        static_overhead_us = self.hw.get_efficiency("static_vector_us")
        total_us = per_core_us + self.hw.pipeline_startup_us + self.hw.pipeline_drain_us + static_overhead_us

        total_vector_us = tiles_per_core * vector_us
        total_mte_us = tiles_per_core * (mte_in_us + mte_out_us)
        bound = "compute (PIPE_V)" if total_vector_us >= total_mte_us else "memory"

        if total_us > 0 and effective_vector_peak > 0:
            effective_time = total_us - self.hw.pipeline_startup_us - self.hw.pipeline_drain_us - static_overhead_us
            util = flops / (effective_time * 1e-6 * effective_vector_peak * self.hw.ai_core_count) if effective_time > 0 else 0.0
        else:
            util = 0.0

        return OpCost(
            compute_us=total_vector_us,
            memory_us=total_mte_us,
            latency_us=total_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=min(util, 1.0),
        )

    # ------------------------------------------------------------------ #
    # Tiling helpers
    # ------------------------------------------------------------------ #

    def _compute_matmul_tiling(self, op: OpSpec, dtype_bytes: int) -> _TilingConfig:
        """Compute tile sizes for MatMul that fit in the UB.

        For [M, K] x [K, N], tiles must satisfy:
          double_factor * (Tm*Tk + Tk*Tn) * dtype_bytes + Tm*Tn*4 <= UB_size
        where double_factor=2 for double-buffering (ping-pong input tiles).
        """
        tile = self.hw.cube_tile_size
        ub_bytes = self.hw.ub_size_kb * 1024
        double_factor = 2 if self.hw.double_buffer else 1

        # Extract M, K, N
        if op.op_type == OpType.CONV2D:
            # Treat Conv2D as implicit matmul
            if len(op.inputs) >= 2:
                inp, kernel = op.inputs[0], op.inputs[1]
                C_out, C_in = kernel.shape[0], kernel.shape[1]
                Kh = kernel.shape[2] if len(kernel.shape) > 2 else 1
                Kw = kernel.shape[3] if len(kernel.shape) > 3 else 1
                N_batch = inp.shape[0] if len(inp.shape) > 0 else 1
                H_out = op.outputs[0].shape[2] if len(op.outputs[0].shape) > 2 else 1
                W_out = op.outputs[0].shape[3] if len(op.outputs[0].shape) > 3 else 1
                M = N_batch * H_out * W_out
                K = C_in * Kh * Kw
                N = C_out
            else:
                M = K = N = tile
        elif len(op.inputs) >= 2:
            a, b = op.inputs[0], op.inputs[1]
            M = a.shape[-2] if len(a.shape) >= 2 else 1
            K = a.shape[-1] if len(a.shape) >= 1 else 1
            N = b.shape[-1] if len(b.shape) >= 2 else 1
            # Handle batched matmul: batch dims are implicit parallelism
            batch = 1
            for d in a.shape[:-2]:
                batch *= d
            # Effective M includes batch
            M = M * batch
        else:
            M = K = N = tile

        # Align to tile size
        def align_up(x):
            return max(tile, math.ceil(x / tile) * tile)

        # Try K-tile sizes from large to small, find largest Tm/Tn that fits UB
        best = None
        for Tk_candidate in self._tile_candidates(K, tile):
            # Budget: double_factor * (Tm*Tk + Tk*Tn) * dtype + Tm*Tn*4 <= ub
            # Assume Tm = Tn = T for simplicity, solve for T:
            # double_factor * 2 * T * Tk * dtype + T*T*4 <= ub
            # This is quadratic in T; use a simpler iterative approach
            Tk = min(Tk_candidate, align_up(K))

            # Max Tm/Tn: try square tiles first
            for T in reversed(self._tile_candidates(min(M, N, 1024), tile)):
                Tm = min(T, align_up(M))
                Tn = min(T, align_up(N))
                needed = double_factor * (Tm * Tk + Tk * Tn) * dtype_bytes + Tm * Tn * 4
                if needed <= ub_bytes:
                    best = (Tm, Tn, Tk)
                    break
            if best is not None:
                break

        if best is None:
            # Minimum tile: single cube tile
            Tm = Tn = Tk = tile
        else:
            Tm, Tn, Tk = best

        num_m = math.ceil(M / Tm) if M > 0 else 1
        num_n = math.ceil(N / Tn) if N > 0 else 1
        num_k = math.ceil(K / Tk) if K > 0 else 1
        total_output_tiles = num_m * num_n
        tiles_per_core = math.ceil(total_output_tiles / self.hw.ai_core_count)

        return _TilingConfig(
            Tm=Tm, Tn=Tn, Tk=Tk,
            num_m_tiles=num_m, num_n_tiles=num_n, num_k_tiles=num_k,
            tiles_per_core=max(tiles_per_core, 1),
        )

    def _compute_vector_tiling(self, op: OpSpec, dtype_bytes: int) -> tuple[int, int, int]:
        """Compute tiling for VECTOR ops. Returns (tile_elements, num_tiles, tiles_per_core)."""
        ub_bytes = self.hw.ub_size_kb * 1024
        double_factor = 2 if self.hw.double_buffer else 1

        # UB holds input + output tiles, double-buffered
        # double_factor * (in_tile + out_tile) * dtype_bytes <= ub_bytes
        tile_elements = ub_bytes // (double_factor * 2 * dtype_bytes)
        if tile_elements <= 0:
            tile_elements = 1

        total_elements = sum(t.numel for t in op.inputs)
        if total_elements == 0:
            return (0, 0, 0)

        num_tiles = math.ceil(total_elements / tile_elements)
        tiles_per_core = math.ceil(num_tiles / self.hw.ai_core_count)

        return (tile_elements, num_tiles, max(tiles_per_core, 1))

    @staticmethod
    def _tile_candidates(max_dim: int, tile: int) -> list[int]:
        """Generate tile size candidates aligned to `tile`, from small to large."""
        candidates = []
        t = tile
        while t <= max_dim:
            candidates.append(t)
            t *= 2
        if not candidates:
            candidates = [tile]
        return candidates

    # ------------------------------------------------------------------ #
    # Shared helpers (unchanged)
    # ------------------------------------------------------------------ #

    def _cube_utilization(self, op: OpSpec, tile: int) -> float:
        """Compute CUBE utilization based on tile alignment.

        For matmul [M,K] x [K,N], CUBE processes (tile x tile) blocks.
        Misalignment causes wasted computation.
        """
        if op.op_type == OpType.MATMUL and len(op.inputs) >= 2:
            a, b = op.inputs[0], op.inputs[1]
            M = a.shape[-2] if len(a.shape) >= 2 else 1
            K = a.shape[-1] if len(a.shape) >= 1 else 1
            N = b.shape[-1] if len(b.shape) >= 2 else 1

            # Padded dimensions
            M_pad = math.ceil(M / tile) * tile
            K_pad = math.ceil(K / tile) * tile
            N_pad = math.ceil(N / tile) * tile

            actual = M * K * N
            padded = M_pad * K_pad * N_pad

            return actual / padded if padded > 0 else 1.0

        elif op.op_type == OpType.CONV2D and len(op.inputs) >= 2:
            kernel = op.inputs[1]
            C_out = kernel.shape[0]
            C_in = kernel.shape[1]

            C_out_pad = math.ceil(C_out / tile) * tile
            C_in_pad = math.ceil(C_in / tile) * tile

            actual = C_out * C_in
            padded = C_out_pad * C_in_pad

            return actual / padded if padded > 0 else 1.0

        return 1.0

    def _format_conversion_cost(self, op: OpSpec, dtype: str) -> float:
        """Estimate cost of data format conversion (e.g., ND -> Fractal_NZ).

        Format conversion is essentially a memory copy with reshape,
        costing ~1.5x a simple memory copy.
        """
        target_layout = _LAYOUT_MAP.get(op.op_type, "ND")
        if target_layout == "ND":
            return 0.0

        # Format conversion: CANN compiler keeps intermediate tensors in
        # Fractal_NZ format between consecutive CUBE ops, so full conversion
        # only happens at format boundaries (input/output of fused regions).
        # We model this as ~0.15x of a simple copy — most data stays in-format.
        total_input_bytes = sum(t.size_bytes for t in op.inputs)
        mem_efficiency = self.hw.get_efficiency("memory")
        gm_bw = self.hw.main_memory_bandwidth() * mem_efficiency * 1e9
        if gm_bw == 0:
            return 0.0

        conversion_bytes = total_input_bytes * 0.15
        return conversion_bytes / gm_bw * 1e6  # microseconds

    @staticmethod
    def preferred_layout(op_type: OpType) -> str:
        """Return the preferred data layout for an op type on Ascend."""
        return _LAYOUT_MAP.get(op_type, "ND")

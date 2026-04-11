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

# Ops that run on the MIX_AIC path (CUBE + VECTOR in one fused kernel).
# FlashAttentionScore on Ascend is dispatched here: QK matmul + softmax + SV
# matmul all fused, cube and vector pipes overlap within the kernel.
_FUSED_ATTN_OPS = {OpType.ATTENTION_FUSED}

# Memory-plumbing ops that move tensors around without compute.
# TRANSPOSE/KV_SLICE/KV_CONCAT stream data through UB; TRIU is a tiny
# constant-build kernel. All pay per-op launch/floor but minimal compute.
_PLUMBING_OPS = {
    OpType.KV_CONCAT, OpType.KV_SLICE, OpType.TRANSPOSE, OpType.TRIU,
}

# Sub-kernel counts: some "logical" ops are implemented on real NPU as a
# chain of small kernels, each paying its own static + host-dispatch floor.
# Calibrated to msprof Qwen3-0.6B where RMSNorm unrolls to 5 kernels
# (Pows + ReduceMean + Rsqrt + Mul + Mul, with Cast ops interleaved) and
# RoPE unrolls to ~3 kernels (Neg + Mul + Add for the half-rotation).
_SUB_KERNEL_COUNT: dict[OpType, int] = {
    OpType.LAYER_NORM: 5,
    OpType.ROPE: 2,  # Neg + (Mul+Add fused); calibrated vs Qwen3 msprof Neg share
}

# Ops that run on the VECTOR unit
_VECTOR_OPS = {
    OpType.RELU, OpType.ADD, OpType.GELU, OpType.SILU, OpType.MUL,
    OpType.LAYER_NORM, OpType.SOFTMAX, OpType.ROPE, OpType.EMBEDDING,
    OpType.GATHER, OpType.ALL_REDUCE, OpType.ALL_TO_ALL,
    OpType.ALL_GATHER, OpType.REDUCE_SCATTER, OpType.DEQUANT,
}

# Memory-pass multipliers for VECTOR ops.
# Multi-pass ops (LayerNorm, Softmax) read data multiple times internally:
#   LayerNorm: pass1 (mean), pass2 (variance), pass3 (normalize+write)
#   Softmax:   pass1 (max), pass2 (exp+sum), pass3 (normalize+write)
# Values are lower than the nominal pass count because inner passes hit
# UB/L2 rather than HBM. Re-calibrated 2026-04-11 against 910B msprof
# large-shape sweeps (1024x4096 ... 4096x4096) where mem_passes=3.5
# overpredicted LN by 35% and softmax mem_passes=2.3 overpredicted by 49%.
_VECTOR_MEM_PASSES: dict[OpType, float] = {
    OpType.LAYER_NORM: 2.8,
    OpType.SOFTMAX: 1.7,
    OpType.ROPE: 2.5,       # CA sim: ~2.5x bytes/cycle cost of ADD due to trig unit serialization
    OpType.RELU: 1.0,
    OpType.ADD: 1.0,
    OpType.GELU: 1.0,
    OpType.SILU: 1.0,
    OpType.MUL: 1.0,
    OpType.EMBEDDING: 1.0,
    OpType.GATHER: 1.0,
    OpType.DEQUANT: 1.0,
}

# Additional per-op static overhead (us) on top of `static_vector_us`.
# Some kernels have higher fixed setup than the bare vector floor:
#   LayerNorm: reduction tree init + epsilon/gamma/beta scalar broadcast
#   Softmax:   max-reduction + exp table setup + normalize pass orchestration
# Calibrated against 910B msprof small-shape measurements where the bare
# 1.7 us floor underpredicted LN by ~2 us and softmax by ~3 us.
_VECTOR_STATIC_US_EXTRA: dict[OpType, float] = {
    OpType.LAYER_NORM: 2.0,
    OpType.SOFTMAX: 3.0,
}

# Simple elementwise ops whose operands commonly stay L2-resident in inference
# loops (activations only, no weights, small intermediate footprint).
# Calibrated 2026-04-11 against 910B msprof benchmarks where repeated ops on
# warm tensors hit the L2 cache rather than HBM.
_L2_RESIDENT_OPS = {OpType.ADD, OpType.MUL, OpType.RELU}

# Cap L2 residency to realistic warm working sets. Full L2 is 192 MB but
# LRU effective capacity is ~60% and benchmark rerun footprints must fit
# in it. Compared against 910B msprof sweeps:
#   - Mul 2048x4096 (32 MB unique input) → warm (measured 11 us)
#   - Mul 4096x4096 (64 MB unique input) → cold (measured 75 us)
#   - Relu 4096x4096 (32 MB unique input) → warm (measured 12.8 us)
# The transition sits at ~48 MB of *unique input bytes* (excluding output,
# which is a streaming write). This is the right accounting: mem_bytes
# double-counts and inflates the footprint artificially.
_L2_WARM_CAP_INPUT_BYTES = 48 * 1024 * 1024

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

    def __init__(self, hw: AscendSpec, *, warm_l2: bool = True):
        """
        Args:
            hw: Ascend hardware spec.
            warm_l2: If True (default), elementwise ops on small activation
                tensors hit L2 instead of HBM — matches warm microbenchmark
                runs and steady-state LLM inference. Set False for cold-path
                validation (CA sim, first-iteration execution).
        """
        super().__init__(hw)
        self.hw: AscendSpec = hw
        self.warm_l2 = warm_l2

    def estimate(self, op: OpSpec) -> OpCost:
        if op.op_type in _CUBE_OPS:
            cost = self._estimate_cube(op)
        elif op.op_type in _FUSED_ATTN_OPS:
            cost = self._estimate_fused_attention(op)
        elif op.op_type in _PLUMBING_OPS:
            cost = self._estimate_plumbing(op)
        elif op.op_type in _VECTOR_OPS:
            cost = self._estimate_vector(op)
        else:
            cost = self._estimate_vector(op)  # fallback

        # Per-op host dispatch floor (PyTorch/ACL launch overhead). Distinct
        # from static_vector_us which is the on-device kernel floor; this
        # captures host-side launch cost that applies to every kernel.
        # Only applied in warm_l2 (runtime inference) mode — CA-sim and
        # cold-path benchmarks measure device cycles only and must not
        # include host-side launch.
        if self.warm_l2:
            host_us = self.hw.efficiency_factors.get("host_dispatch_us", 0.0)
            if host_us > 0:
                cost.latency_us += host_us
                cost.scalar_us = (cost.scalar_us or 0.0) + host_us

            # Sub-kernel multiplier: logical ops that unroll to N real
            # kernels on device each pay N static + N host-dispatch floors.
            n_sub = _SUB_KERNEL_COUNT.get(op.op_type, 1)
            if n_sub > 1:
                extra_static = (n_sub - 1) * (
                    self.hw.get_efficiency("static_vector_us")
                    + host_us
                )
                cost.latency_us += extra_static
                cost.scalar_us = (cost.scalar_us or 0.0) + extra_static
        return cost

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

        # Per-pipe microarchitectural breakdown.
        # CUBE: inputs flow GM→L1→L0A/L0B via MTE2; outputs drain L0C→GM via
        # fixpipe (accounted on MTE3 here). CUBE pipe busy = compute_us.
        input_bytes = sum(t.size_bytes for t in op.inputs)
        output_bytes = sum(t.size_bytes for t in op.outputs)
        mte2_us = (input_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0
        mte3_us = (output_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0

        return OpCost(
            compute_us=compute_us + format_overhead_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=utilization,
            mte2_us=mte2_us,
            mte3_us=mte3_us,
            cube_us=compute_us,
            scalar_us=static_overhead_us,
        )

    # ------------------------------------------------------------------ #
    # VECTOR pipeline (elementwise / reduction ops)
    # ------------------------------------------------------------------ #

    def _estimate_vector(self, op: OpSpec) -> OpCost:
        """Estimate cost for VECTOR ops using chip-level roofline.

        Uses aggregate HBM bandwidth (like CUBE) with per-op memory-pass
        multipliers to account for multi-pass ops (LayerNorm, Softmax).
        Calibrated against real Ascend 910C profiling data.
        """
        dtype_str = op.inputs[0].dtype.value if op.inputs else "fp16"
        flops = op.flops
        mem_bytes = op.memory_bytes

        # --- Memory bound: chip-level aggregate bandwidth ---
        # Apply per-op memory-pass multiplier for multi-pass ops
        mem_passes = _VECTOR_MEM_PASSES.get(op.op_type, 1.0)
        effective_mem_bytes = mem_bytes * mem_passes

        mem_efficiency = self.hw.get_efficiency("memory")
        # L2 residency: simple elementwise ops (ADD/MUL/RELU) on tensors that
        # fit in L2 run at L2 bandwidth, not HBM. This captures the real hw
        # behavior observed in msprof benchmarks where repeated ops on warm
        # activations hit L2 cache.
        unique_input_bytes = sum(t.size_bytes for t in op.inputs)
        if (self.warm_l2
                and op.op_type in _L2_RESIDENT_OPS
                and unique_input_bytes <= _L2_WARM_CAP_INPUT_BYTES):
            l2_level = self.hw.get_mem_level("L2")
            bw = l2_level.bandwidth_GBs * mem_efficiency * 1e9
        else:
            bw = self.hw.main_memory_bandwidth() * mem_efficiency * 1e9  # B/s
        memory_us = (effective_mem_bytes / bw * 1e6) if bw > 0 else 0.0

        # --- Compute bound: chip-level VECTOR peak ---
        vector_peak = self.hw.vector_peak_for(dtype_str)
        vector_efficiency = self.hw.get_efficiency("vector")
        effective_peak = vector_peak * vector_efficiency
        compute_us = (flops / effective_peak * 1e6) if effective_peak > 0 else 0.0

        # Static per-op overhead (bare floor + optional op-specific extra)
        static_overhead_us = (
            self.hw.get_efficiency("static_vector_us")
            + _VECTOR_STATIC_US_EXTRA.get(op.op_type, 0.0)
        )

        # Roofline: max(compute, memory) + overhead
        latency_us = max(compute_us, memory_us) + static_overhead_us

        bound = "compute (PIPE_V)" if compute_us >= memory_us else "memory"

        # Utilization
        if latency_us > static_overhead_us and effective_peak > 0:
            effective_time = latency_us - static_overhead_us
            util = flops / (effective_time * 1e-6 * effective_peak) if effective_time > 0 else 0.0
        else:
            util = 0.0

        # Per-pipe microarchitectural breakdown.
        # VECTOR ops: inputs stream GM→UB via MTE2, outputs UB→GM via MTE3.
        # Multi-pass ops (LN/Softmax) re-read from UB (near-free on MTE2);
        # we charge the extra passes to MTE2 for a best-effort attribution.
        input_bytes = sum(t.size_bytes for t in op.inputs)
        output_bytes = sum(t.size_bytes for t in op.outputs)
        mte2_bytes = input_bytes * max(mem_passes, 1.0)
        mte2_us = (mte2_bytes / bw * 1e6) if bw > 0 else 0.0
        mte3_us = (output_bytes / bw * 1e6) if bw > 0 else 0.0

        return OpCost(
            compute_us=compute_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=min(util, 1.0),
            mte2_us=mte2_us,
            mte3_us=mte3_us,
            vec_us=compute_us,
            scalar_us=static_overhead_us,
        )

    # ------------------------------------------------------------------ #
    # Fused attention (FlashAttentionScore-style)
    # ------------------------------------------------------------------ #

    def _estimate_fused_attention(self, op: OpSpec) -> OpCost:
        """Cost model for a fused FA kernel.

        Pipe-overlap within the kernel means cube and vector work run in
        parallel; latency = max(cube_time, vector_time, mem_time) + launch.
        Calibrated to ~25 µs/call on 910C for Qwen3-0.6B decode shapes
        (msprof FlashAttentionScore avg).
        """
        dtype_str = op.inputs[0].dtype.value if op.inputs else "bf16"

        q_seq = op.attrs.get("q_seq", 1)
        kv_seq = op.attrs.get("kv_seq", 1)
        n_heads = op.attrs.get("n_heads", 1)
        qk_d = op.attrs.get("qk_head_dim", 128)
        v_d = op.attrs.get("v_head_dim", 128)
        B = op.attrs.get("batch", 1)

        flops = op.attrs.get("_fused_flops", 0)
        mem_bytes = op.memory_bytes

        # Cube work: QK + SV matmuls. Reuse the same cube efficiency + peak
        # because FA is implemented on top of the cube unit.
        cube_efficiency = self.hw.get_efficiency(f"cube_{dtype_str}")
        chip_cube_peak = self.hw.cube_peak_for(dtype_str) * cube_efficiency
        cube_us = (flops / chip_cube_peak * 1e6) if chip_cube_peak > 0 else 0.0

        # Vector work: online softmax (rescale + exp + normalize). Small but
        # real — charge 5 ops/element across q_seq·kv_seq·n_heads·B tiles.
        sm_elems = B * n_heads * q_seq * kv_seq
        vector_peak = self.hw.vector_peak_for(dtype_str) * self.hw.get_efficiency("vector")
        vec_us = ((5 * sm_elems) / vector_peak * 1e6) if vector_peak > 0 else 0.0

        # Memory: Q, K, V, O streaming through UB. Mostly tile-local.
        mem_efficiency = self.hw.get_efficiency("memory")
        gm_bw = self.hw.main_memory_bandwidth() * mem_efficiency * 1e9
        memory_us = (mem_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0

        # Kernel launch + tile setup. FA has a higher floor than plain cube
        # because it runs on MIX_AIC and needs extra synchronization.
        static_overhead_us = self.hw.efficiency_factors.get(
            "static_fa_us",
            self.hw.get_efficiency("static_cube_us") * 2.0)

        # Pipes overlap inside the fused kernel → max, not sum.
        latency_us = max(cube_us, vec_us, memory_us) + static_overhead_us
        bound = "compute (MIX_AIC)" if cube_us >= max(vec_us, memory_us) else "memory"

        input_bytes = sum(t.size_bytes for t in op.inputs)
        output_bytes = sum(t.size_bytes for t in op.outputs)

        return OpCost(
            compute_us=cube_us + vec_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=(flops / (latency_us * 1e-6 * chip_cube_peak)
                         if latency_us > 0 and chip_cube_peak > 0 else 0.0),
            mte2_us=(input_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0,
            mte3_us=(output_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0,
            cube_us=cube_us,
            vec_us=vec_us,
            scalar_us=static_overhead_us,
        )

    # ------------------------------------------------------------------ #
    # KV-cache plumbing (ConcatD / Slice)
    # ------------------------------------------------------------------ #

    def _estimate_plumbing(self, op: OpSpec) -> OpCost:
        """Cost model for memory-only plumbing ops (KV_CONCAT).

        Msprof on Qwen3-0.6B shows ConcatD ≈ 3.8 µs/call avg. That is
        dominated by the static kernel-launch overhead, with a small
        memory copy on top. We reflect both.
        """
        mem_bytes = op.memory_bytes
        mem_efficiency = self.hw.get_efficiency("memory")
        gm_bw = self.hw.main_memory_bandwidth() * mem_efficiency * 1e9
        memory_us = (mem_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0

        static_overhead_us = self.hw.efficiency_factors.get(
            "static_plumbing_us",
            self.hw.get_efficiency("static_vector_us"))

        latency_us = memory_us + static_overhead_us

        input_bytes = sum(t.size_bytes for t in op.inputs)
        output_bytes = sum(t.size_bytes for t in op.outputs)

        return OpCost(
            compute_us=0.0,
            memory_us=memory_us,
            latency_us=latency_us,
            bound="memory",
            flops=0,
            bytes_accessed=mem_bytes,
            utilization=0.0,
            mte2_us=(input_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0,
            mte3_us=(output_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0,
            scalar_us=static_overhead_us,
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

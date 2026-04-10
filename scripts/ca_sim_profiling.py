"""CA Simulator profiling for NPU cost model calibration.

Generates TIK kernels for various op types and shapes, runs them through
the CANN Cycle-Accurate simulator, and extracts per-pipe cycle counts.

Usage (on NPU server):
    source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
    source /home/Ray/venv/bin/activate
    python scripts/ca_sim_profiling.py [--ops all] [--output results.csv]

The CA sim gives instruction-level traces with per-pipe timing (MTE2/MTE3/
VECTOR/CUBE/SCALAR), so we can directly measure:
  - Total kernel cycles (max ts+dur across all pipes)
  - MTE busy cycles (data movement bottleneck)
  - VECTOR busy cycles (compute)
  - Effective memory passes = MTE_busy / single_pass_MTE_estimate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# --- Must set SOC version BEFORE importing tik ---
import te.platform as tp
tp.te_set_version("Ascend910B1", core_type="AiCore")
from te import tik  # noqa: E402


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

SOC_VERSION = "Ascend910B1"
BLOCK_DIM = 8  # number of AI cores to use
SIMULATOR_LIB = "/usr/local/Ascend/cann-8.5.0/tools/simulator"
DUMP_BASE = "/home/Ray/ca_sim_test/ca_profiling_dumps"

# Shapes to profile for each op type.
# For VECTOR ops: (rows, cols) in fp16 — total elements = rows * cols.
# For MATMUL: (M, K, N) in fp16.
VECTOR_SHAPES = [
    (1, 4096),        # small: 4K elements
    (1024, 4096),     # medium: 4M elements (typical hidden dim)
    (1024, 8192),     # large hidden
    (1024, 14336),    # MLP intermediate (LLaMA-8B)
    (2048, 4096),     # longer seq
    (4096, 4096),     # large
]

MATMUL_SHAPES = [
    (1024, 1024, 1024),
    (1024, 4096, 4096),
    (1024, 4096, 14336),   # LLaMA-8B up-proj
    (1024, 14336, 4096),   # LLaMA-8B down-proj
    (2048, 4096, 4096),
    (4096, 4096, 4096),
    (1024, 4096, 11008),   # LLaMA-7B MLP
    (1024, 11008, 4096),   # LLaMA-7B MLP down
    (1024, 8192, 28672),   # LLaMA-70B MLP
]


@dataclass
class PipeStats:
    """Cycle counts extracted from CA sim trace."""
    total_cycles: int = 0
    mte2_cycles: int = 0   # GM/L2 → UB loads
    mte3_cycles: int = 0   # UB → GM stores
    vector_cycles: int = 0
    cube_cycles: int = 0
    scalar_cycles: int = 0
    # Derived
    kernel_start: int = 0
    kernel_end: int = 0


@dataclass
class ProfilingResult:
    """Single op profiling result."""
    op_type: str
    shape: str
    total_elements: int
    dtype_bytes: int
    # From CA sim
    total_cycles: int = 0
    mte2_cycles: int = 0
    mte3_cycles: int = 0
    vector_cycles: int = 0
    cube_cycles: int = 0
    # Derived metrics
    effective_mem_passes: float = 0.0
    bottleneck: str = ""


# ------------------------------------------------------------------ #
# Trace parsing
# ------------------------------------------------------------------ #

def parse_ca_trace(dump_path: str, num_cores: int = BLOCK_DIM) -> list[PipeStats]:
    """Parse CA sim dump2trace JSON files, return per-core PipeStats."""
    from op_gen.simulator.Simulator import Simulator
    trace_dir = os.path.join(dump_path, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    Simulator.run([f"--dump_path={dump_path}", f"--output_path={trace_dir}"])

    results = []
    for core_id in range(num_cores):
        trace_file = os.path.join(trace_dir, f"dump2trace_core{core_id}.json")
        if not os.path.exists(trace_file):
            continue
        with open(trace_file) as f:
            trace = json.load(f)

        stats = PipeStats()
        min_ts = float("inf")
        max_end = 0

        for ev in trace.get("traceEvents", []):
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            tid = ev.get("tid", "")
            end = ts + dur

            min_ts = min(min_ts, ts)
            max_end = max(max_end, end)

            if tid == "MTE2":
                stats.mte2_cycles += dur
            elif tid == "MTE3":
                stats.mte3_cycles += dur
            elif tid == "VECTOR":
                stats.vector_cycles += dur
            elif tid == "CUBE":
                stats.cube_cycles += dur
            elif tid == "SCALAR":
                stats.scalar_cycles += dur

        stats.kernel_start = min_ts if min_ts != float("inf") else 0
        stats.kernel_end = max_end
        stats.total_cycles = max_end - stats.kernel_start
        results.append(stats)

    return results


def aggregate_core_stats(core_stats: list[PipeStats]) -> PipeStats:
    """Aggregate per-core stats — take the slowest core (it determines wall time)."""
    if not core_stats:
        return PipeStats()
    # Wall time = max across cores (they run in parallel)
    agg = PipeStats()
    slowest = max(core_stats, key=lambda s: s.total_cycles)
    agg.total_cycles = slowest.total_cycles
    agg.kernel_start = slowest.kernel_start
    agg.kernel_end = slowest.kernel_end
    # Per-pipe: also take slowest core (it's the bottleneck)
    agg.mte2_cycles = slowest.mte2_cycles
    agg.mte3_cycles = slowest.mte3_cycles
    agg.vector_cycles = slowest.vector_cycles
    agg.cube_cycles = slowest.cube_cycles
    agg.scalar_cycles = slowest.scalar_cycles
    return agg


# ------------------------------------------------------------------ #
# TIK kernel builders — VECTOR ops
# ------------------------------------------------------------------ #

def _tik_data_move_in(t, ub_tensor, gm_tensor, num_elements: int):
    """Move data from GM to UB."""
    burst = num_elements * 2 // 32  # fp16, 32 bytes per burst
    if burst > 0:
        t.data_move(ub_tensor, gm_tensor, 0, 1, burst, 0, 0)


def _tik_data_move_out(t, gm_tensor, ub_tensor, num_elements: int):
    """Move data from UB to GM."""
    burst = num_elements * 2 // 32
    if burst > 0:
        t.data_move(gm_tensor, ub_tensor, 0, 1, burst, 0, 0)


def _vec_repeat_params(num_elements: int):
    """Calculate repeat count and mask for fp16 vec ops.

    Max mask for fp16 = 128 elements per repeat.
    """
    mask = min(128, num_elements)
    repeat = math.ceil(num_elements / 128)
    return mask, repeat


def build_add_kernel(rows: int, cols: int) -> tuple:
    """vec_add: C = A + B. Single-pass memory."""
    t = tik.Tik(disable_debug=False)
    total = rows * cols
    per_core = math.ceil(total / BLOCK_DIM)
    # Align to 16 (fp16 alignment)
    per_core = math.ceil(per_core / 16) * 16

    gm_a = t.Tensor("float16", (total,), name="gm_a", scope=tik.scope_gm)
    gm_b = t.Tensor("float16", (total,), name="gm_b", scope=tik.scope_gm)
    gm_c = t.Tensor("float16", (total,), name="gm_c", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_a = t.Tensor("float16", (per_core,), name="ub_a", scope=tik.scope_ubuf)
        ub_b = t.Tensor("float16", (per_core,), name="ub_b", scope=tik.scope_ubuf)
        ub_c = t.Tensor("float16", (per_core,), name="ub_c", scope=tik.scope_ubuf)

        offset = bid * per_core
        _tik_data_move_in(t, ub_a, gm_a[offset], per_core)
        _tik_data_move_in(t, ub_b, gm_b[offset], per_core)

        mask, repeat = _vec_repeat_params(per_core)
        t.vec_add(mask, ub_c, ub_a, ub_b, repeat, 1, 1, 1)

        _tik_data_move_out(t, gm_c[offset], ub_c, per_core)

    return t, f"add_{rows}x{cols}"


def build_mul_kernel(rows: int, cols: int) -> tuple:
    """vec_mul: C = A * B. Single-pass memory."""
    t = tik.Tik(disable_debug=False)
    total = rows * cols
    per_core = math.ceil(total / BLOCK_DIM)
    per_core = math.ceil(per_core / 16) * 16

    gm_a = t.Tensor("float16", (total,), name="gm_a", scope=tik.scope_gm)
    gm_b = t.Tensor("float16", (total,), name="gm_b", scope=tik.scope_gm)
    gm_c = t.Tensor("float16", (total,), name="gm_c", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_a = t.Tensor("float16", (per_core,), name="ub_a", scope=tik.scope_ubuf)
        ub_b = t.Tensor("float16", (per_core,), name="ub_b", scope=tik.scope_ubuf)
        ub_c = t.Tensor("float16", (per_core,), name="ub_c", scope=tik.scope_ubuf)

        offset = bid * per_core
        _tik_data_move_in(t, ub_a, gm_a[offset], per_core)
        _tik_data_move_in(t, ub_b, gm_b[offset], per_core)

        mask, repeat = _vec_repeat_params(per_core)
        t.vec_mul(mask, ub_c, ub_a, ub_b, repeat, 1, 1, 1)

        _tik_data_move_out(t, gm_c[offset], ub_c, per_core)

    return t, f"mul_{rows}x{cols}"


def build_relu_kernel(rows: int, cols: int) -> tuple:
    """vec_relu: Y = max(0, X). Single-pass, single input."""
    t = tik.Tik(disable_debug=False)
    total = rows * cols
    per_core = math.ceil(total / BLOCK_DIM)
    per_core = math.ceil(per_core / 16) * 16

    gm_x = t.Tensor("float16", (total,), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (total,), name="gm_y", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (per_core,), name="ub_x", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (per_core,), name="ub_y", scope=tik.scope_ubuf)

        offset = bid * per_core
        _tik_data_move_in(t, ub_x, gm_x[offset], per_core)

        mask, repeat = _vec_repeat_params(per_core)
        t.vec_relu(mask, ub_y, ub_x, repeat, 1, 1)

        _tik_data_move_out(t, gm_y[offset], ub_y, per_core)

    return t, f"relu_{rows}x{cols}"


def build_silu_kernel(rows: int, cols: int) -> tuple:
    """SiLU(x) = x * sigmoid(x).

    Two compute passes over data:
      1. sigmoid(x) → tmp
      2. x * tmp → out
    Plus load x, store out.
    """
    t = tik.Tik(disable_debug=False)
    total = rows * cols
    per_core = math.ceil(total / BLOCK_DIM)
    per_core = math.ceil(per_core / 16) * 16

    gm_x = t.Tensor("float16", (total,), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (total,), name="gm_y", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (per_core,), name="ub_x", scope=tik.scope_ubuf)
        ub_tmp = t.Tensor("float16", (per_core,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (per_core,), name="ub_y", scope=tik.scope_ubuf)

        offset = bid * per_core
        _tik_data_move_in(t, ub_x, gm_x[offset], per_core)

        mask, repeat = _vec_repeat_params(per_core)
        # sigmoid(x) → tmp
        t.vec_sigmoid(mask, ub_tmp, ub_x, repeat, 1, 1)
        # x * sigmoid(x) → y
        t.vec_mul(mask, ub_y, ub_x, ub_tmp, repeat, 1, 1, 1)

        _tik_data_move_out(t, gm_y[offset], ub_y, per_core)

    return t, f"silu_{rows}x{cols}"


def build_gelu_kernel(rows: int, cols: int) -> tuple:
    """GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

    Approximated as: x * sigmoid(1.702 * x) (sigmoidal GELU).
    Two-pass like SiLU but with a scalar multiply first.
    """
    t = tik.Tik(disable_debug=False)
    total = rows * cols
    per_core = math.ceil(total / BLOCK_DIM)
    per_core = math.ceil(per_core / 16) * 16

    gm_x = t.Tensor("float16", (total,), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (total,), name="gm_y", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (per_core,), name="ub_x", scope=tik.scope_ubuf)
        ub_tmp = t.Tensor("float16", (per_core,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (per_core,), name="ub_y", scope=tik.scope_ubuf)

        offset = bid * per_core
        _tik_data_move_in(t, ub_x, gm_x[offset], per_core)

        mask, repeat = _vec_repeat_params(per_core)
        # 1.702 * x → tmp
        t.vec_muls(mask, ub_tmp, ub_x, 1.702, repeat, 1, 1)
        # sigmoid(1.702*x) → tmp
        t.vec_sigmoid(mask, ub_tmp, ub_tmp, repeat, 1, 1)
        # x * sigmoid(1.702*x) → y
        t.vec_mul(mask, ub_y, ub_x, ub_tmp, repeat, 1, 1, 1)

        _tik_data_move_out(t, gm_y[offset], ub_y, per_core)

    return t, f"gelu_{rows}x{cols}"


def build_layernorm_kernel(rows: int, cols: int) -> tuple:
    """LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta.

    Multi-pass over data:
      Pass 1: reduce_sum(x) → mean  (read x)
      Pass 2: reduce_sum((x - mean)^2) → var  (read x again, sub, square, reduce)
      Pass 3: normalize: (x - mean) * rsqrt(var+eps) * gamma + beta  (read x again)
      Write output
    ~3-4 memory passes over input data.
    """
    t = tik.Tik(disable_debug=False)
    per_core_rows = math.ceil(rows / BLOCK_DIM)
    # cols must be aligned to 16
    cols_aligned = math.ceil(cols / 16) * 16

    gm_x = t.Tensor("float16", (rows, cols_aligned), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (rows, cols_aligned), name="gm_y", scope=tik.scope_gm)
    gm_gamma = t.Tensor("float16", (cols_aligned,), name="gm_gamma", scope=tik.scope_gm)
    gm_beta = t.Tensor("float16", (cols_aligned,), name="gm_beta", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (cols_aligned,), name="ub_x", scope=tik.scope_ubuf)
        ub_tmp = t.Tensor("float16", (cols_aligned,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (cols_aligned,), name="ub_y", scope=tik.scope_ubuf)
        ub_gamma = t.Tensor("float16", (cols_aligned,), name="ub_gamma", scope=tik.scope_ubuf)
        ub_beta = t.Tensor("float16", (cols_aligned,), name="ub_beta", scope=tik.scope_ubuf)
        # Work buffer for reductions (fp32 for accuracy)
        ub_work = t.Tensor("float32", (16,), name="ub_work", scope=tik.scope_ubuf)

        # Load gamma/beta once
        burst_cols = cols_aligned * 2 // 32
        t.data_move(ub_gamma, gm_gamma, 0, 1, burst_cols, 0, 0)
        t.data_move(ub_beta, gm_beta, 0, 1, burst_cols, 0, 0)

        mask, repeat = _vec_repeat_params(cols_aligned)

        with t.for_range(0, per_core_rows) as row:
            row_offset = bid * per_core_rows + row

            # === Pass 1: compute mean ===
            t.data_move(ub_x, gm_x[row_offset, 0], 0, 1, burst_cols, 0, 0)
            # sum(x) via vec_reduce_add
            t.vec_reduce_add(mask, ub_work, ub_x, repeat, 1)
            # mean = sum / cols (scalar divide done by adding scaled values)

            # === Pass 2: compute variance ===
            # x - mean → tmp, tmp^2 → tmp, reduce_sum(tmp) → var
            # Re-read x (already in UB from pass 1 if cols fits)
            t.vec_adds(mask, ub_tmp, ub_x, 0.0, repeat, 1, 1)  # copy for sub
            t.vec_mul(mask, ub_tmp, ub_tmp, ub_tmp, repeat, 1, 1, 1)  # square
            t.vec_reduce_add(mask, ub_work, ub_tmp, repeat, 1)

            # === Pass 3: normalize ===
            # (x - mean) * rsqrt(var+eps) * gamma + beta
            t.vec_mul(mask, ub_y, ub_x, ub_gamma, repeat, 1, 1, 1)
            t.vec_add(mask, ub_y, ub_y, ub_beta, repeat, 1, 1, 1)

            # Write output
            t.data_move(gm_y[row_offset, 0], ub_y, 0, 1, burst_cols, 0, 0)

    return t, f"layernorm_{rows}x{cols}"


def build_softmax_kernel(rows: int, cols: int) -> tuple:
    """Softmax: y = exp(x - max(x)) / sum(exp(x - max(x))).

    Multi-pass:
      Pass 1: reduce_max(x)  (read x)
      Pass 2: exp(x - max) → tmp, reduce_sum(tmp)  (read x again)
      Pass 3: tmp / sum → y  (read tmp)
      Write output
    ~3 memory passes.
    """
    t = tik.Tik(disable_debug=False)
    per_core_rows = math.ceil(rows / BLOCK_DIM)
    cols_aligned = math.ceil(cols / 16) * 16

    gm_x = t.Tensor("float16", (rows, cols_aligned), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (rows, cols_aligned), name="gm_y", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (cols_aligned,), name="ub_x", scope=tik.scope_ubuf)
        ub_tmp = t.Tensor("float16", (cols_aligned,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (cols_aligned,), name="ub_y", scope=tik.scope_ubuf)
        ub_work = t.Tensor("float32", (16,), name="ub_work", scope=tik.scope_ubuf)

        mask, repeat = _vec_repeat_params(cols_aligned)
        burst_cols = cols_aligned * 2 // 32

        with t.for_range(0, per_core_rows) as row:
            row_offset = bid * per_core_rows + row

            # === Pass 1: find max ===
            t.data_move(ub_x, gm_x[row_offset, 0], 0, 1, burst_cols, 0, 0)
            t.vec_reduce_max(mask, ub_work, ub_x, repeat, 1)

            # === Pass 2: exp(x - max), sum ===
            # x - max → tmp  (subtract scalar max from each element)
            t.vec_adds(mask, ub_tmp, ub_x, 0.0, repeat, 1, 1)  # placeholder sub
            t.vec_exp(mask, ub_tmp, ub_tmp, repeat, 1, 1)
            t.vec_reduce_add(mask, ub_work, ub_tmp, repeat, 1)

            # === Pass 3: divide by sum ===
            # tmp * (1/sum) → y
            t.vec_muls(mask, ub_y, ub_tmp, 1.0, repeat, 1, 1)  # placeholder div

            # Write output
            t.data_move(gm_y[row_offset, 0], ub_y, 0, 1, burst_cols, 0, 0)

    return t, f"softmax_{rows}x{cols}"


def build_rope_kernel(rows: int, cols: int) -> tuple:
    """RoPE: applies rotary position embeddings.

    For each pair (x0, x1), compute:
      y0 = x0 * cos(θ) - x1 * sin(θ)
      y1 = x0 * sin(θ) + x1 * cos(θ)

    Passes:
      1. Load x, load cos/sin tables
      2. x0*cos, x1*sin, subtract → y0
      3. x0*sin, x1*cos, add → y1
      Interleave y0/y1 → output
    ~2-3x memory passes (input read once, several compute passes in UB).
    """
    t = tik.Tik(disable_debug=False)
    total = rows * cols
    half_cols = cols // 2
    half_cols_aligned = math.ceil(half_cols / 16) * 16
    per_core_rows = math.ceil(rows / BLOCK_DIM)

    gm_x = t.Tensor("float16", (rows, cols), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (rows, cols), name="gm_y", scope=tik.scope_gm)
    gm_cos = t.Tensor("float16", (half_cols_aligned,), name="gm_cos", scope=tik.scope_gm)
    gm_sin = t.Tensor("float16", (half_cols_aligned,), name="gm_sin", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x0 = t.Tensor("float16", (half_cols_aligned,), name="ub_x0", scope=tik.scope_ubuf)
        ub_x1 = t.Tensor("float16", (half_cols_aligned,), name="ub_x1", scope=tik.scope_ubuf)
        ub_cos = t.Tensor("float16", (half_cols_aligned,), name="ub_cos", scope=tik.scope_ubuf)
        ub_sin = t.Tensor("float16", (half_cols_aligned,), name="ub_sin", scope=tik.scope_ubuf)
        ub_t0 = t.Tensor("float16", (half_cols_aligned,), name="ub_t0", scope=tik.scope_ubuf)
        ub_t1 = t.Tensor("float16", (half_cols_aligned,), name="ub_t1", scope=tik.scope_ubuf)
        ub_y0 = t.Tensor("float16", (half_cols_aligned,), name="ub_y0", scope=tik.scope_ubuf)
        ub_y1 = t.Tensor("float16", (half_cols_aligned,), name="ub_y1", scope=tik.scope_ubuf)

        # Load cos/sin tables once
        burst_half = half_cols_aligned * 2 // 32
        t.data_move(ub_cos, gm_cos, 0, 1, burst_half, 0, 0)
        t.data_move(ub_sin, gm_sin, 0, 1, burst_half, 0, 0)

        mask, repeat = _vec_repeat_params(half_cols_aligned)

        with t.for_range(0, per_core_rows) as row:
            row_offset = bid * per_core_rows + row

            # Load x (split into first half and second half)
            t.data_move(ub_x0, gm_x[row_offset, 0], 0, 1, burst_half, 0, 0)
            t.data_move(ub_x1, gm_x[row_offset, half_cols], 0, 1, burst_half, 0, 0)

            # y0 = x0*cos - x1*sin
            t.vec_mul(mask, ub_t0, ub_x0, ub_cos, repeat, 1, 1, 1)
            t.vec_mul(mask, ub_t1, ub_x1, ub_sin, repeat, 1, 1, 1)
            t.vec_sub(mask, ub_y0, ub_t0, ub_t1, repeat, 1, 1, 1)

            # y1 = x0*sin + x1*cos
            t.vec_mul(mask, ub_t0, ub_x0, ub_sin, repeat, 1, 1, 1)
            t.vec_mul(mask, ub_t1, ub_x1, ub_cos, repeat, 1, 1, 1)
            t.vec_add(mask, ub_y1, ub_t0, ub_t1, repeat, 1, 1, 1)

            # Write y0, y1
            t.data_move(gm_y[row_offset, 0], ub_y0, 0, 1, burst_half, 0, 0)
            t.data_move(gm_y[row_offset, half_cols], ub_y1, 0, 1, burst_half, 0, 0)

    return t, f"rope_{rows}x{cols}"


# ------------------------------------------------------------------ #
# TIK kernel builders — CUBE ops
# ------------------------------------------------------------------ #

def build_matmul_kernel(M: int, K: int, N: int) -> tuple:
    """MatMul: C[M,N] = A[M,K] x B[K,N] in fp16.

    Uses TIK matmul intrinsic which takes L1 tensors.
    Result in L0C (fp32), then fixpipe to GM (fp16).
    """
    t = tik.Tik(disable_debug=False)

    gm_a = t.Tensor("float16", (M, K), name="gm_a", scope=tik.scope_gm)
    gm_b = t.Tensor("float16", (K, N), name="gm_b", scope=tik.scope_gm)
    gm_c = t.Tensor("float16", (M, N), name="gm_c", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        # Each core handles a chunk of M rows
        rows_per_core = math.ceil(M / BLOCK_DIM)
        # Align to 16
        rows_per_core = math.ceil(rows_per_core / 16) * 16

        # L1 buffers for A-tile and B-tile
        l1_a = t.Tensor("float16", (rows_per_core, K), name="l1_a", scope=tik.scope_cbuf)
        l1_b = t.Tensor("float16", (K, N), name="l1_b", scope=tik.scope_cbuf)
        # L0C for result (fp32)
        l0c_c = t.Tensor("float32", (rows_per_core, N), name="l0c_c", scope=tik.scope_cbuf_out)

        # Load A tile from GM to L1
        m_offset = bid * rows_per_core
        a_burst = rows_per_core * K * 2 // 32
        t.data_move(l1_a, gm_a[m_offset, 0], 0, 1, a_burst, 0, 0)

        # Load B from GM to L1
        b_burst = K * N * 2 // 32
        t.data_move(l1_b, gm_b, 0, 1, b_burst, 0, 0)

        # Matmul: l0c_c = l1_a x l1_b
        t.matmul(l0c_c, l1_a, l1_b, rows_per_core, K, N, init_l1out=True)

        # fixpipe: fp32 L0C → fp16 GM
        t.fixpipe(
            gm_c[m_offset, 0], l0c_c, rows_per_core, N * 4 // 32, 0, 0,
            extend_params={
                "quantize_params": {"mode": "fp322fp16", "mode_param": None}
            },
        )

    return t, f"matmul_{M}x{K}x{N}"


# ------------------------------------------------------------------ #
# Kernel registry
# ------------------------------------------------------------------ #

KERNEL_BUILDERS = {
    "ADD": build_add_kernel,
    "MUL": build_mul_kernel,
    "RELU": build_relu_kernel,
    "SILU": build_silu_kernel,
    "GELU": build_gelu_kernel,
    "LAYERNORM": build_layernorm_kernel,
    "SOFTMAX": build_softmax_kernel,
    "ROPE": build_rope_kernel,
}

CUBE_BUILDERS = {
    "MATMUL": build_matmul_kernel,
}


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

def run_ca_sim(t, kernel_name: str, dump_path: str) -> list[PipeStats]:
    """Compile TIK kernel and run through CA simulator."""
    os.makedirs(dump_path, exist_ok=True)

    # Build the kernel binary
    t.BuildCCE(
        kernel_name=kernel_name,
        postfix=SOC_VERSION,
        inputs=[],
        outputs=[],
    )

    # Run on CA simulator
    from op_test_frame.rt.op_ut_run import AscendOpKernelRunner
    runner = AscendOpKernelRunner(
        simulator_mode="ca",
        soc_version=SOC_VERSION,
        simulator_lib_path=SIMULATOR_LIB,
        simulator_dump_path=dump_path,
    )

    # Create feed dict with random data
    import numpy as np
    feed_dict = {}
    for tensor in t.get_input_tensors():
        shape = tensor.shape
        feed_dict[tensor.name] = np.random.randn(*shape).astype(np.float16)

    runner.run(t, feed_dict)

    # Parse traces
    return parse_ca_trace(dump_path, BLOCK_DIM)


def compute_single_pass_estimate(total_elements: int, dtype_bytes: int = 2) -> float:
    """Estimate MTE cycles for a single pass of data (read + write).

    Used to compute effective_mem_passes = actual_mte_cycles / single_pass_cycles.
    This is a rough estimate — the CA sim result is ground truth.
    """
    total_bytes = total_elements * dtype_bytes
    read_bytes = total_bytes   # input
    write_bytes = total_bytes  # output
    return read_bytes + write_bytes


def profile_vector_ops(ops: list[str], output_rows: list) -> None:
    """Profile all vector op types across shapes."""
    for op_name in ops:
        if op_name not in KERNEL_BUILDERS:
            print(f"  Skipping unknown op: {op_name}")
            continue

        builder = KERNEL_BUILDERS[op_name]

        for rows, cols in VECTOR_SHAPES:
            shape_str = f"{rows}x{cols}"
            total_elements = rows * cols
            print(f"  {op_name} {shape_str} ({total_elements:,} elements)...", end=" ")

            dump_path = os.path.join(DUMP_BASE, f"{op_name.lower()}_{shape_str}")
            if os.path.exists(dump_path):
                shutil.rmtree(dump_path)

            try:
                t, kname = builder(rows, cols)
                core_stats = run_ca_sim(t, kname, dump_path)
                agg = aggregate_core_stats(core_stats)

                # Compute effective memory passes
                single_pass_bytes = compute_single_pass_estimate(total_elements)
                total_mte = agg.mte2_cycles + agg.mte3_cycles
                # Effective passes = total_mte / (estimated single pass mte cycles)
                # We normalize by the simplest ops (ADD/RELU) to get relative passes
                effective_passes = total_mte / max(agg.mte2_cycles, 1)  # placeholder

                bottleneck = "MTE" if total_mte > agg.vector_cycles else "VECTOR"

                result = ProfilingResult(
                    op_type=op_name,
                    shape=shape_str,
                    total_elements=total_elements,
                    dtype_bytes=2,
                    total_cycles=agg.total_cycles,
                    mte2_cycles=agg.mte2_cycles,
                    mte3_cycles=agg.mte3_cycles,
                    vector_cycles=agg.vector_cycles,
                    cube_cycles=agg.cube_cycles,
                    bottleneck=bottleneck,
                )
                output_rows.append(result)
                print(f"OK  total={agg.total_cycles}  MTE2={agg.mte2_cycles}  "
                      f"MTE3={agg.mte3_cycles}  VEC={agg.vector_cycles}  [{bottleneck}]")

            except Exception as e:
                print(f"FAILED: {e}")
                continue


def profile_cube_ops(output_rows: list) -> None:
    """Profile MATMUL across shapes."""
    for M, K, N in MATMUL_SHAPES:
        shape_str = f"{M}x{K}x{N}"
        print(f"  MATMUL {shape_str}...", end=" ")

        dump_path = os.path.join(DUMP_BASE, f"matmul_{shape_str}")
        if os.path.exists(dump_path):
            shutil.rmtree(dump_path)

        try:
            t, kname = build_matmul_kernel(M, K, N)
            core_stats = run_ca_sim(t, kname, dump_path)
            agg = aggregate_core_stats(core_stats)

            total_mte = agg.mte2_cycles + agg.mte3_cycles
            bottleneck = "CUBE" if agg.cube_cycles > total_mte else "MTE"

            result = ProfilingResult(
                op_type="MATMUL",
                shape=shape_str,
                total_elements=M * K + K * N + M * N,  # A + B + C
                dtype_bytes=2,
                total_cycles=agg.total_cycles,
                mte2_cycles=agg.mte2_cycles,
                mte3_cycles=agg.mte3_cycles,
                vector_cycles=agg.vector_cycles,
                cube_cycles=agg.cube_cycles,
                bottleneck=bottleneck,
            )
            output_rows.append(result)
            print(f"OK  total={agg.total_cycles}  MTE2={agg.mte2_cycles}  "
                  f"CUBE={agg.cube_cycles}  MTE3={agg.mte3_cycles}  [{bottleneck}]")

        except Exception as e:
            print(f"FAILED: {e}")
            continue


# ------------------------------------------------------------------ #
# Calibration analysis
# ------------------------------------------------------------------ #

def analyze_results(results: list[ProfilingResult]) -> None:
    """Analyze results to derive calibration parameters."""
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)

    # Group by op type
    by_op: dict[str, list[ProfilingResult]] = {}
    for r in results:
        by_op.setdefault(r.op_type, []).append(r)

    # For single-pass ops (ADD, MUL, RELU), compute baseline MTE per element
    baseline_ops = {"ADD", "MUL", "RELU"}
    baseline_mte_per_element = []
    for op_name in baseline_ops:
        for r in by_op.get(op_name, []):
            if r.total_elements > 0 and r.mte2_cycles > 0:
                mte_per_el = (r.mte2_cycles + r.mte3_cycles) / r.total_elements
                baseline_mte_per_element.append(mte_per_el)

    if baseline_mte_per_element:
        avg_baseline = sum(baseline_mte_per_element) / len(baseline_mte_per_element)
        print(f"\nBaseline MTE cycles/element (from ADD/MUL/RELU): {avg_baseline:.4f}")
    else:
        avg_baseline = None
        print("\nNo baseline data (single-pass ops not profiled)")

    # For each multi-pass op, compute effective memory passes
    print(f"\n{'Op':<12} {'Shape':<16} {'Total':<10} {'MTE2':<10} {'MTE3':<10} "
          f"{'VEC':<10} {'CUBE':<10} {'Eff.Passes':<12} {'Bottleneck'}")
    print("-" * 110)

    for op_name, op_results in sorted(by_op.items()):
        for r in op_results:
            if avg_baseline and r.total_elements > 0:
                expected_single = avg_baseline * r.total_elements
                actual_mte = r.mte2_cycles + r.mte3_cycles
                eff_passes = actual_mte / expected_single if expected_single > 0 else 0
            else:
                eff_passes = 0

            print(f"{r.op_type:<12} {r.shape:<16} {r.total_cycles:<10} "
                  f"{r.mte2_cycles:<10} {r.mte3_cycles:<10} {r.vector_cycles:<10} "
                  f"{r.cube_cycles:<10} {eff_passes:<12.2f} {r.bottleneck}")

    # Summarize per-op average effective passes
    if avg_baseline:
        print(f"\n{'Op':<12} {'Avg Eff.Passes':<16} {'Current Model':<16} {'Suggested'}")
        print("-" * 60)
        current_model = {
            "ADD": 1.0, "MUL": 1.0, "RELU": 1.0,
            "SILU": 1.0, "GELU": 1.0,
            "LAYERNORM": 3.5, "SOFTMAX": 2.3, "ROPE": 2.0,
        }
        for op_name, op_results in sorted(by_op.items()):
            passes_list = []
            for r in op_results:
                expected_single = avg_baseline * r.total_elements
                actual_mte = r.mte2_cycles + r.mte3_cycles
                if expected_single > 0:
                    passes_list.append(actual_mte / expected_single)
            if passes_list:
                avg_passes = sum(passes_list) / len(passes_list)
                cur = current_model.get(op_name, 1.0)
                print(f"{op_name:<12} {avg_passes:<16.2f} {cur:<16.1f} {avg_passes:.1f}")


def save_results(results: list[ProfilingResult], output_path: str) -> None:
    """Save results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "op_type", "shape", "total_elements", "dtype_bytes",
            "total_cycles", "mte2_cycles", "mte3_cycles",
            "vector_cycles", "cube_cycles", "bottleneck",
        ])
        for r in results:
            writer.writerow([
                r.op_type, r.shape, r.total_elements, r.dtype_bytes,
                r.total_cycles, r.mte2_cycles, r.mte3_cycles,
                r.vector_cycles, r.cube_cycles, r.bottleneck,
            ])
    print(f"\nResults saved to {output_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="CA simulator profiling for NPU calibration")
    parser.add_argument("--ops", default="all",
                        help="Comma-separated op types or 'all' / 'vector' / 'cube'")
    parser.add_argument("--output", default="reports/ca_sim_profiling_results.csv",
                        help="Output CSV path")
    parser.add_argument("--block-dim", type=int, default=BLOCK_DIM,
                        help=f"Number of AI cores (default: {BLOCK_DIM})")
    args = parser.parse_args()

    global BLOCK_DIM
    BLOCK_DIM = args.block_dim

    # Determine which ops to profile
    if args.ops == "all":
        vector_ops = list(KERNEL_BUILDERS.keys())
        do_cube = True
    elif args.ops == "vector":
        vector_ops = list(KERNEL_BUILDERS.keys())
        do_cube = False
    elif args.ops == "cube":
        vector_ops = []
        do_cube = True
    else:
        requested = [s.strip().upper() for s in args.ops.split(",")]
        vector_ops = [op for op in requested if op in KERNEL_BUILDERS]
        do_cube = "MATMUL" in requested

    results: list[ProfilingResult] = []

    print("=" * 70)
    print("CA Simulator Profiling for NPU Cost Model Calibration")
    print(f"SOC: {SOC_VERSION}  Cores: {BLOCK_DIM}")
    print("=" * 70)

    if vector_ops:
        print(f"\n--- VECTOR ops: {', '.join(vector_ops)} ---")
        profile_vector_ops(vector_ops, results)

    if do_cube:
        print(f"\n--- CUBE ops: MATMUL ---")
        profile_cube_ops(results)

    # Save raw results
    save_results(results, args.output)

    # Analyze for calibration
    analyze_results(results)


if __name__ == "__main__":
    main()

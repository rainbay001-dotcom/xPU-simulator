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

# UB on 910B is 192KB per core. We leave headroom for workspaces.
# Per-tile fp16 element cap: each builder can allocate multiple UB tensors
# of size UB_TILE_ELEMS; keep the sum under ~128KB (64K fp16 elts total).
UB_TILE_ELEMS = 16384  # 32 KB per tensor; ADD/MUL uses 3 tensors = 96 KB

# Shapes to profile for each op type.
# For VECTOR ops: (rows, cols) in fp16 — total elements = rows * cols.
# For MATMUL: (M, K, N) in fp16.
VECTOR_SHAPES = [
    (1, 4096),        # tiny
    (1, 8192),
    (4, 4096),
    (16, 4096),
    (64, 4096),
    (128, 4096),
    (256, 4096),      # single-core heavy tiling
    (512, 4096),
    (1024, 4096),     # LLM hidden state
    (2048, 4096),
]

MATMUL_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
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

import re as _re

# Matches e.g.  "[info] [00000697] (PC: 0x10d11000) SCALAR   : ..."
_INSTR_LINE_RE = _re.compile(
    r"\[info\]\s*\[(\d+)\]\s*\(PC:[^)]*\)\s*(SCALAR|VEC|MTE1|MTE2|MTE3|CUBE|FIXPIPE)\s*:"
)


def _parse_instr_log(path: str) -> dict[str, set[int]]:
    """Return {pipe -> set of cycles where that pipe popped an instruction}."""
    buckets: dict[str, set[int]] = {}
    if not os.path.exists(path):
        return buckets
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = _INSTR_LINE_RE.match(line)
            if not m:
                continue
            cycle = int(m.group(1))
            pipe = m.group(2)
            buckets.setdefault(pipe, set()).add(cycle)
    return buckets


def _parse_network_total_cycles(dump_path: str) -> int:
    """Extract total_cycles from profile_network_log0.toml."""
    path = os.path.join(dump_path, "profile_network_log0.toml")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("total_cycles"):
                # total_cycles = 1659
                parts = line.split("=")
                if len(parts) == 2:
                    try:
                        return int(parts[1].strip())
                    except ValueError:
                        return 0
    return 0


def parse_ca_trace(dump_path: str, num_cores: int = BLOCK_DIM) -> list[PipeStats]:
    """Parse CA sim instruction dumps, return per-core PipeStats.

    Per-pipe busy cycles are estimated as the number of distinct simulator
    cycles at which the pipe popped an instruction. This is a good proxy
    for pipe utilization since popped instructions execute for at least 1
    cycle and our interest is in ratios (MTE-bound vs VEC-bound), not
    absolute instruction latency.

    Total cycles come from profile_network_log0.toml.
    """
    network_total = _parse_network_total_cycles(dump_path)

    results: list[PipeStats] = []
    for core_id in range(num_cores):
        stats = PipeStats()
        # Each core has one vec-core (veccore0/1) and one cube-core (cubecore0).
        vec_path_0 = os.path.join(dump_path, f"core{core_id}.veccore0.instr_popped_log.dump")
        vec_path_1 = os.path.join(dump_path, f"core{core_id}.veccore1.instr_popped_log.dump")
        cube_path = os.path.join(dump_path, f"core{core_id}.cubecore0.instr_popped_log.dump")

        vec_buckets = _parse_instr_log(vec_path_0)
        # Merge veccore1 into the same pipe buckets
        for pipe, cycles in _parse_instr_log(vec_path_1).items():
            vec_buckets.setdefault(pipe, set()).update(cycles)
        cube_buckets = _parse_instr_log(cube_path)

        all_cycles: set[int] = set()
        for cycles in vec_buckets.values():
            all_cycles |= cycles
        for cycles in cube_buckets.values():
            all_cycles |= cycles

        if not all_cycles and not vec_buckets and not cube_buckets:
            continue

        stats.mte2_cycles = len(vec_buckets.get("MTE2", set())) + len(cube_buckets.get("MTE2", set()))
        stats.mte3_cycles = len(vec_buckets.get("MTE3", set())) + len(cube_buckets.get("MTE3", set()))
        stats.vector_cycles = len(vec_buckets.get("VEC", set()))
        stats.cube_cycles = len(cube_buckets.get("CUBE", set())) + len(cube_buckets.get("FIXPIPE", set()))
        stats.scalar_cycles = len(vec_buckets.get("SCALAR", set())) + len(cube_buckets.get("SCALAR", set()))

        if all_cycles:
            stats.kernel_start = min(all_cycles)
            stats.kernel_end = max(all_cycles)
            stats.total_cycles = stats.kernel_end - stats.kernel_start + 1
        else:
            stats.total_cycles = network_total

        results.append(stats)

    # Fallback: if per-core parsing gave nothing, synthesize a single entry
    # from the network total so the pipeline still produces a row.
    if not results and network_total > 0:
        stats = PipeStats()
        stats.total_cycles = network_total
        stats.kernel_end = network_total
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


def _tile_dims(rows: int, cols: int) -> tuple[int, int, int, int]:
    """Return (total_padded, per_core, tile, num_tiles) for vector ops.

    Pads the total so every core runs an equal number of fixed-size tiles.
    Each tile is UB_TILE_ELEMS elements (fp16), aligned to 16.
    """
    total_raw = rows * cols
    tile = UB_TILE_ELEMS
    per_core_raw = math.ceil(total_raw / BLOCK_DIM)
    num_tiles = max(1, math.ceil(per_core_raw / tile))
    per_core = num_tiles * tile
    total_padded = BLOCK_DIM * per_core
    return total_padded, per_core, tile, num_tiles


def build_add_kernel(rows: int, cols: int) -> tuple:
    """vec_add: C = A + B. Single-pass memory, UB-tiled."""
    t = tik.Tik(disable_debug=False)
    total, per_core, tile, num_tiles = _tile_dims(rows, cols)

    gm_a = t.Tensor("float16", (total,), name="gm_a", scope=tik.scope_gm)
    gm_b = t.Tensor("float16", (total,), name="gm_b", scope=tik.scope_gm)
    gm_c = t.Tensor("float16", (total,), name="gm_c", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_a = t.Tensor("float16", (tile,), name="ub_a", scope=tik.scope_ubuf)
        ub_b = t.Tensor("float16", (tile,), name="ub_b", scope=tik.scope_ubuf)
        ub_c = t.Tensor("float16", (tile,), name="ub_c", scope=tik.scope_ubuf)

        mask, repeat = _vec_repeat_params(tile)
        with t.for_range(0, num_tiles) as ti:
            offset = bid * per_core + ti * tile
            _tik_data_move_in(t, ub_a, gm_a[offset], tile)
            _tik_data_move_in(t, ub_b, gm_b[offset], tile)
            t.vec_add(mask, ub_c, ub_a, ub_b, repeat, 1, 1, 1)
            _tik_data_move_out(t, gm_c[offset], ub_c, tile)

    return t, f"add_{rows}x{cols}", [gm_a, gm_b], [gm_c]


def build_mul_kernel(rows: int, cols: int) -> tuple:
    """vec_mul: C = A * B. Single-pass memory, UB-tiled."""
    t = tik.Tik(disable_debug=False)
    total, per_core, tile, num_tiles = _tile_dims(rows, cols)

    gm_a = t.Tensor("float16", (total,), name="gm_a", scope=tik.scope_gm)
    gm_b = t.Tensor("float16", (total,), name="gm_b", scope=tik.scope_gm)
    gm_c = t.Tensor("float16", (total,), name="gm_c", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_a = t.Tensor("float16", (tile,), name="ub_a", scope=tik.scope_ubuf)
        ub_b = t.Tensor("float16", (tile,), name="ub_b", scope=tik.scope_ubuf)
        ub_c = t.Tensor("float16", (tile,), name="ub_c", scope=tik.scope_ubuf)

        mask, repeat = _vec_repeat_params(tile)
        with t.for_range(0, num_tiles) as ti:
            offset = bid * per_core + ti * tile
            _tik_data_move_in(t, ub_a, gm_a[offset], tile)
            _tik_data_move_in(t, ub_b, gm_b[offset], tile)
            t.vec_mul(mask, ub_c, ub_a, ub_b, repeat, 1, 1, 1)
            _tik_data_move_out(t, gm_c[offset], ub_c, tile)

    return t, f"mul_{rows}x{cols}", [gm_a, gm_b], [gm_c]


def build_relu_kernel(rows: int, cols: int) -> tuple:
    """vec_relu: Y = max(0, X). Single-pass, UB-tiled."""
    t = tik.Tik(disable_debug=False)
    total, per_core, tile, num_tiles = _tile_dims(rows, cols)

    gm_x = t.Tensor("float16", (total,), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (total,), name="gm_y", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (tile,), name="ub_x", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (tile,), name="ub_y", scope=tik.scope_ubuf)

        mask, repeat = _vec_repeat_params(tile)
        with t.for_range(0, num_tiles) as ti:
            offset = bid * per_core + ti * tile
            _tik_data_move_in(t, ub_x, gm_x[offset], tile)
            t.vec_relu(mask, ub_y, ub_x, repeat, 1, 1)
            _tik_data_move_out(t, gm_y[offset], ub_y, tile)

    return t, f"relu_{rows}x{cols}", [gm_x], [gm_y]


def build_silu_kernel(rows: int, cols: int) -> tuple:
    """SiLU(x) = x * sigmoid(x). UB-tiled."""
    t = tik.Tik(disable_debug=False)
    total, per_core, tile, num_tiles = _tile_dims(rows, cols)

    gm_x = t.Tensor("float16", (total,), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (total,), name="gm_y", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (tile,), name="ub_x", scope=tik.scope_ubuf)
        ub_tmp = t.Tensor("float16", (tile,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (tile,), name="ub_y", scope=tik.scope_ubuf)

        mask, repeat = _vec_repeat_params(tile)
        with t.for_range(0, num_tiles) as ti:
            offset = bid * per_core + ti * tile
            _tik_data_move_in(t, ub_x, gm_x[offset], tile)
            t.vec_sigmoid(mask, ub_tmp, ub_x, repeat, 1, 1)
            t.vec_mul(mask, ub_y, ub_x, ub_tmp, repeat, 1, 1, 1)
            _tik_data_move_out(t, gm_y[offset], ub_y, tile)

    return t, f"silu_{rows}x{cols}", [gm_x], [gm_y]


def build_gelu_kernel(rows: int, cols: int) -> tuple:
    """Sigmoidal GELU ≈ x * sigmoid(1.702 * x). UB-tiled."""
    t = tik.Tik(disable_debug=False)
    total, per_core, tile, num_tiles = _tile_dims(rows, cols)

    gm_x = t.Tensor("float16", (total,), name="gm_x", scope=tik.scope_gm)
    gm_y = t.Tensor("float16", (total,), name="gm_y", scope=tik.scope_gm)

    with t.for_range(0, BLOCK_DIM, block_num=BLOCK_DIM) as bid:
        ub_x = t.Tensor("float16", (tile,), name="ub_x", scope=tik.scope_ubuf)
        ub_tmp = t.Tensor("float16", (tile,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_y = t.Tensor("float16", (tile,), name="ub_y", scope=tik.scope_ubuf)

        mask, repeat = _vec_repeat_params(tile)
        with t.for_range(0, num_tiles) as ti:
            offset = bid * per_core + ti * tile
            _tik_data_move_in(t, ub_x, gm_x[offset], tile)
            t.vec_muls(mask, ub_tmp, ub_x, 1.702, repeat, 1, 1)
            t.vec_sigmoid(mask, ub_tmp, ub_tmp, repeat, 1, 1)
            t.vec_mul(mask, ub_y, ub_x, ub_tmp, repeat, 1, 1, 1)
            _tik_data_move_out(t, gm_y[offset], ub_y, tile)

    return t, f"gelu_{rows}x{cols}", [gm_x], [gm_y]


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

    return t, f"layernorm_{rows}x{cols}", [gm_x, gm_gamma, gm_beta], [gm_y]


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

    return t, f"softmax_{rows}x{cols}", [gm_x], [gm_y]


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

    return t, f"rope_{rows}x{cols}", [gm_x, gm_cos, gm_sin], [gm_y]


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

    return t, f"matmul_{M}x{K}x{N}", [gm_a, gm_b], [gm_c]


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

_TIK_TO_NP_DTYPE = {
    "float16": "float16",
    "float32": "float32",
    "int32": "int32",
    "int8": "int8",
    "uint8": "uint8",
}


def _tik_tensor_shape(tensor) -> tuple:
    """Extract a concrete shape tuple from a TIK Tensor."""
    shape = tensor.shape
    out = []
    for d in shape:
        out.append(int(d))
    return tuple(out)


def _tik_tensor_dtype(tensor) -> str:
    dt = tensor.dtype
    return _TIK_TO_NP_DTYPE.get(dt, dt)


def run_ca_sim(t, kernel_name: str, dump_path: str,
               input_tensors: list, output_tensors: list) -> list[PipeStats]:
    """Compile TIK kernel and run through CA simulator.

    Parameters
    ----------
    t : tik.Tik
        TIK instance with the kernel program built up.
    kernel_name : str
        Name used for BuildCCE output (.o and .json in ./kernel_meta/).
    dump_path : str
        Directory for CA simulator dump output.
    input_tensors, output_tensors : list
        Lists of GM TIK Tensors that are kernel params (in order).
    """
    import numpy as np
    from op_test_frame.common.ascend_tbe_op import AscendOpKernel, AscendOpKernelRunner

    os.makedirs(dump_path, exist_ok=True)

    # Build the kernel binary. TIK writes ./kernel_meta/<kernel_name>.o + .json.
    t.BuildCCE(
        kernel_name=kernel_name,
        inputs=list(input_tensors),
        outputs=list(output_tensors),
    )
    bin_path = os.path.join("kernel_meta", f"{kernel_name}.o")
    json_path = os.path.join("kernel_meta", f"{kernel_name}.json")

    # Derive input/output info lists from TIK tensor metadata.
    input_info_list = []
    input_data_list = []
    for tensor in input_tensors:
        shape = _tik_tensor_shape(tensor)
        dtype = _tik_tensor_dtype(tensor)
        np_data = np.random.uniform(-1.0, 1.0, size=shape).astype(dtype)
        input_info_list.append({"shape": shape, "dtype": dtype, "value": np_data})
        input_data_list.append(np_data)

    output_info_list = []
    for tensor in output_tensors:
        shape = _tik_tensor_shape(tensor)
        dtype = _tik_tensor_dtype(tensor)
        output_info_list.append({"shape": shape, "dtype": dtype})

    op_kernel = AscendOpKernel(bin_path, json_path)
    op_kernel.set_input_info(input_info_list)
    op_kernel.set_output_info(output_info_list)

    with AscendOpKernelRunner(
        simulator_mode="ca",
        soc_version=SOC_VERSION,
        simulator_lib_path=SIMULATOR_LIB,
        simulator_dump_path=dump_path,
    ) as runner:
        runner.run(op_kernel, inputs=input_data_list, block_dim=BLOCK_DIM)

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
                t, kname, inputs, outputs = builder(rows, cols)
                core_stats = run_ca_sim(t, kname, dump_path, inputs, outputs)
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

            except BaseException as e:
                print(f"FAILED: {type(e).__name__}: {e}")
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
            t, kname, inputs, outputs = build_matmul_kernel(M, K, N)
            core_stats = run_ca_sim(t, kname, dump_path, inputs, outputs)
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
    global BLOCK_DIM
    parser = argparse.ArgumentParser(description="CA simulator profiling for NPU calibration")
    parser.add_argument("--ops", default="all",
                        help="Comma-separated op types or 'all' / 'vector' / 'cube'")
    parser.add_argument("--output", default="reports/ca_sim_profiling_results.csv",
                        help="Output CSV path")
    parser.add_argument("--block-dim", type=int, default=BLOCK_DIM,
                        help=f"Number of AI cores (default: {BLOCK_DIM})")
    parser.add_argument("--shapes", default="all",
                        help="'all' or 'small' for a single smoke-test shape")
    args = parser.parse_args()

    BLOCK_DIM = args.block_dim

    if args.shapes == "small":
        global VECTOR_SHAPES, MATMUL_SHAPES
        VECTOR_SHAPES = [(1, 4096)]
        MATMUL_SHAPES = [(256, 256, 256)]

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

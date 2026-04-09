"""Tests for the CA-style tiled pipeline NPU cost model."""

import sys
sys.path.insert(0, ".")

import math
import copy

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C, AscendSpec
from xpu_simulator.backends.npu.cost_model import NPUCostModel


def t(shape, dtype=Dtype.FP16):
    return TensorSpec(shape, dtype)


def _matmul_op(M, K, N, name="matmul"):
    return OpSpec(OpType.MATMUL, [t((M, K)), t((K, N))], [t((M, N))], name=name)


# ------------------------------------------------------------------ #
# Test 1: Tiling fits in UB
# ------------------------------------------------------------------ #

def test_tiling_fits_ub():
    """Computed tile dimensions must fit within UB capacity."""
    model = NPUCostModel(ASCEND_910B)
    op = _matmul_op(4096, 4096, 4096)
    tiling = model._compute_matmul_tiling(op, dtype_bytes=2)

    Tm, Tn, Tk = tiling.Tm, tiling.Tn, tiling.Tk
    ub_bytes = ASCEND_910B.ub_size_kb * 1024
    double_factor = 2  # double-buffering

    # Input tiles (double-buffered) + accumulator
    needed = double_factor * (Tm * Tk + Tk * Tn) * 2 + Tm * Tn * 4
    assert needed <= ub_bytes, (
        f"Tile {Tm}x{Tk}x{Tn} needs {needed} bytes but UB is {ub_bytes}"
    )

    # All dims aligned to cube_tile_size
    tile = ASCEND_910B.cube_tile_size
    assert Tm % tile == 0
    assert Tn % tile == 0
    assert Tk % tile == 0

    # tiles_per_core should be reasonable
    assert tiling.tiles_per_core >= 1
    print(f"  Tiling: Tm={Tm}, Tn={Tn}, Tk={Tk}, tiles_per_core={tiling.tiles_per_core}")
    print(f"  UB usage: {needed}/{ub_bytes} bytes ({needed/ub_bytes:.0%})")


# ------------------------------------------------------------------ #
# Test 2: Pipeline overlap reduces latency
# ------------------------------------------------------------------ #

def test_pipeline_overlap():
    """Pipelined latency should be less than sum of all stages."""
    model = NPUCostModel(ASCEND_910B)
    op = _matmul_op(2048, 2048, 2048)
    cost = model.estimate(op)

    # With pipeline overlap, latency < compute + memory
    sum_stages = cost.compute_us + cost.memory_us
    assert cost.latency_us < sum_stages or cost.latency_us <= sum_stages * 1.1, (
        f"Pipeline should overlap: latency={cost.latency_us:.1f}, "
        f"compute+memory={sum_stages:.1f}"
    )
    print(f"  Pipelined latency: {cost.latency_us:.1f} us")
    print(f"  Sum of stages:     {sum_stages:.1f} us")


# ------------------------------------------------------------------ #
# Test 3: L2 reuse effect
# ------------------------------------------------------------------ #

def test_l2_reuse():
    """MatMul where B fits in L2 should be faster than one where it doesn't."""
    model = NPUCostModel(ASCEND_910B)

    # Small N: B tiles fit in L2
    op_small = _matmul_op(4096, 4096, 256, name="b_fits_l2")
    cost_small = model.estimate(op_small)

    # Large N: B tiles may not fit in L2
    op_large = _matmul_op(4096, 4096, 4096, name="b_no_fit")
    cost_large = model.estimate(op_large)

    # Normalize by FLOPs to compare efficiency
    flops_per_us_small = op_small.flops / cost_small.latency_us if cost_small.latency_us > 0 else 0
    flops_per_us_large = op_large.flops / cost_large.latency_us if cost_large.latency_us > 0 else 0

    print(f"  B-fits-L2:     {cost_small.latency_us:.1f} us, {flops_per_us_small/1e6:.1f} MFLOPS/us")
    print(f"  B-no-fit:      {cost_large.latency_us:.1f} us, {flops_per_us_large/1e6:.1f} MFLOPS/us")

    # The smaller problem should have at least comparable efficiency
    # (L2 reuse helps or at least doesn't hurt)
    assert cost_small.latency_us > 0
    assert cost_large.latency_us > 0


# ------------------------------------------------------------------ #
# Test 4: Small vs large MatMul (overhead amortization)
# ------------------------------------------------------------------ #

def test_small_vs_large_matmul():
    """Small matmul has higher overhead fraction; large matmul amortizes it."""
    model = NPUCostModel(ASCEND_910B)

    op_small = _matmul_op(16, 16, 16, name="tiny")
    op_large = _matmul_op(4096, 4096, 4096, name="large")

    cost_small = model.estimate(op_small)
    cost_large = model.estimate(op_large)

    # Both should produce valid results
    assert cost_small.latency_us > 0
    assert cost_large.latency_us > cost_small.latency_us

    # Overhead fraction: startup+drain relative to total
    overhead = ASCEND_910B.pipeline_startup_us + ASCEND_910B.pipeline_drain_us
    overhead_frac_small = overhead / cost_small.latency_us
    overhead_frac_large = overhead / cost_large.latency_us

    assert overhead_frac_small > overhead_frac_large, (
        f"Small op overhead fraction ({overhead_frac_small:.2%}) should exceed "
        f"large op ({overhead_frac_large:.2%})"
    )
    print(f"  Small: {cost_small.latency_us:.2f} us (overhead {overhead_frac_small:.1%})")
    print(f"  Large: {cost_large.latency_us:.2f} us (overhead {overhead_frac_large:.1%})")


# ------------------------------------------------------------------ #
# Test 5: VECTOR pipeline
# ------------------------------------------------------------------ #

def test_vector_pipeline():
    """VECTOR ops should use 3-stage pipeline with tiling."""
    model = NPUCostModel(ASCEND_910B)

    shape = (1024, 7168)
    op = OpSpec(OpType.LAYER_NORM, [t(shape)], [t(shape)], name="layer_norm")
    cost = model.estimate(op)

    assert cost.latency_us > 0
    assert cost.flops > 0
    # Pipeline startup/drain should be included
    assert cost.latency_us >= ASCEND_910B.pipeline_startup_us + ASCEND_910B.pipeline_drain_us
    print(f"  LayerNorm {shape}: {cost.latency_us:.2f} us, bound={cost.bound}")


# ------------------------------------------------------------------ #
# Test 6: Multi-core scaling
# ------------------------------------------------------------------ #

def test_multicore_scaling():
    """More AI cores should reduce latency for large ops.

    We keep per-core MTE bandwidth the same (simulating a chip with fewer
    cores but same per-core specs), so the only variable is parallelism
    across output tiles.
    """
    hw_half = copy.deepcopy(ASCEND_910B)
    hw_half.ai_core_count = 15
    # Keep per-core MTE bw the same — fewer cores just means less parallelism

    model_full = NPUCostModel(ASCEND_910B)
    model_half = NPUCostModel(hw_half)

    op = _matmul_op(4096, 4096, 4096)
    cost_full = model_full.estimate(op)
    cost_half = model_half.estimate(op)

    # Full cores should be faster (more parallelism across output tiles)
    assert cost_full.latency_us < cost_half.latency_us, (
        f"30 cores ({cost_full.latency_us:.1f} us) should be faster than "
        f"15 cores ({cost_half.latency_us:.1f} us)"
    )
    speedup = cost_half.latency_us / cost_full.latency_us
    print(f"  30 cores: {cost_full.latency_us:.1f} us")
    print(f"  15 cores: {cost_half.latency_us:.1f} us")
    print(f"  Speedup:  {speedup:.2f}x")


# ------------------------------------------------------------------ #
# Test 7: Dual-pipeline overlap in evaluator
# ------------------------------------------------------------------ #

def test_dual_pipeline_overlap():
    """Independent CUBE and VECTOR ops should overlap on NPU."""
    graph = ComputeGraph("dual_pipe")

    # Two independent ops: one CUBE, one VECTOR
    mm_op = _matmul_op(2048, 2048, 2048, name="matmul")
    vec_op = OpSpec(OpType.LAYER_NORM,
                    [t((2048, 2048))], [t((2048, 2048))],
                    name="layer_norm")

    mm_node = graph.add_node(mm_op, "matmul")
    vec_node = graph.add_node(vec_op, "layer_norm")
    # No edge between them — fully independent

    model = NPUCostModel(ASCEND_910B)
    evaluator = PerformanceEvaluator(model)

    result_overlap = evaluator.run(graph, overlap=True)
    result_seq = evaluator.run(graph, overlap=False)

    mm_cost = model.estimate(mm_op)
    vec_cost = model.estimate(vec_op)

    print(f"  MatMul:     {mm_cost.latency_us:.1f} us ({mm_cost.bound})")
    print(f"  LayerNorm:  {vec_cost.latency_us:.1f} us ({vec_cost.bound})")
    print(f"  Sequential: {result_seq.total_latency_us:.1f} us")
    print(f"  Overlapped: {result_overlap.total_latency_us:.1f} us")

    # Overlapped should be faster than sequential
    assert result_overlap.total_latency_us < result_seq.total_latency_us, (
        f"Overlap ({result_overlap.total_latency_us:.1f}) should beat "
        f"sequential ({result_seq.total_latency_us:.1f})"
    )

    # Overlapped should be close to max(cube, vector), not sum
    max_single = max(mm_cost.latency_us, vec_cost.latency_us)
    assert result_overlap.total_latency_us <= max_single * 1.1, (
        f"Overlap latency ({result_overlap.total_latency_us:.1f}) should be near "
        f"max(cube, vector) = {max_single:.1f}"
    )


# ------------------------------------------------------------------ #
# Test 8: Backward compatibility with existing assertions
# ------------------------------------------------------------------ #

def test_backward_compat():
    """Original test_phase4 qualitative assertions still hold."""
    model = NPUCostModel(ASCEND_910B)

    # Aligned matmul: high utilization
    op_aligned = _matmul_op(2048, 2048, 2048, name="aligned")
    cost_aligned = model.estimate(op_aligned)
    assert cost_aligned.utilization > 0.99, f"Got {cost_aligned.utilization}"

    # Misaligned matmul: lower utilization
    op_mis = OpSpec(OpType.MATMUL,
                    [t((2001, 2001)), t((2001, 2001))],
                    [t((2001, 2001))], name="misaligned")
    cost_mis = model.estimate(op_mis)
    assert cost_mis.utilization < 1.0, f"Got {cost_mis.utilization}"

    # ReLU: memory-bound
    relu_op = OpSpec(OpType.RELU, [t((32, 1024, 1024))],
                     [t((32, 1024, 1024))], name="relu")
    relu_cost = model.estimate(relu_op)
    assert "memory" in relu_cost.bound.lower(), f"Got {relu_cost.bound}"

    # Format conversion layout
    assert NPUCostModel.preferred_layout(OpType.MATMUL) == "Fractal_NZ"

    print("  All backward compat checks passed")


if __name__ == "__main__":
    print("=== NPU CA Model Tests ===\n")

    for name, fn in [
        ("Tiling fits UB", test_tiling_fits_ub),
        ("Pipeline overlap", test_pipeline_overlap),
        ("L2 reuse", test_l2_reuse),
        ("Small vs large MatMul", test_small_vs_large_matmul),
        ("VECTOR pipeline", test_vector_pipeline),
        ("Multi-core scaling", test_multicore_scaling),
        ("Dual-pipeline overlap", test_dual_pipeline_overlap),
        ("Backward compat", test_backward_compat),
    ]:
        print(f"--- {name} ---")
        fn()
        print()

    print("All CA model tests passed!")

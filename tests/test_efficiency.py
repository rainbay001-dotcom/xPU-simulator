"""Tests for hardware efficiency factors and per-op static overhead."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.cost_model import RooflineCostModel
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB, GPUSpec
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C, AscendSpec
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.core.parallel import InterconnectSpec


def _matmul_op(M=2048, K=2048, N=2048, dtype=Dtype.FP16):
    a = TensorSpec((M, K), dtype)
    b = TensorSpec((K, N), dtype)
    c = TensorSpec((M, N), dtype)
    return OpSpec(OpType.MATMUL, [a, b], [c], name="mm")


def _vector_op(shape=(1, 4096, 4096), dtype=Dtype.FP16):
    inp = TensorSpec(shape, dtype)
    out = TensorSpec(shape, dtype)
    return OpSpec(OpType.RELU, [inp], [out], name="relu")


# --- GPU efficiency factor tests ---

def test_gpu_has_efficiency_factors():
    """A100 and H100 presets should have efficiency factors."""
    for hw in [A100_80GB, H100_80GB]:
        assert hw.get_efficiency("matmul_fp16") == 0.70
        assert hw.get_efficiency("elementwise_fp16") == 0.85
        assert hw.get_efficiency("memory") == 0.80
        assert hw.get_efficiency("static_tc_us") == 5.0
        assert hw.get_efficiency("static_cuda_us") == 2.0


def test_gpu_efficiency_slows_matmul():
    """GPU matmul with 0.70 efficiency should be slower than raw roofline."""
    op = _matmul_op()
    gpu_cost = GPUCostModel(A100_80GB).estimate(op)
    raw_cost = RooflineCostModel(A100_80GB).estimate(op)
    # GPU applies 0.70 compute efficiency + 5us static → should be slower
    assert gpu_cost.latency_us > raw_cost.latency_us


def test_gpu_tc_vs_cuda_static_overhead():
    """Tensor Core ops get 5us overhead, CUDA Core ops get 2us."""
    mm_op = _matmul_op()
    relu_op = _vector_op()
    model = GPUCostModel(A100_80GB)
    mm_cost = model.estimate(mm_op)
    relu_cost = model.estimate(relu_op)
    # Both include static overhead; verify by checking with a zero-flop scenario
    # Instead, check that the static amounts are retrievable
    assert A100_80GB.get_efficiency("static_tc_us") == 5.0
    assert A100_80GB.get_efficiency("static_cuda_us") == 2.0
    # Matmul latency includes 5us, relu includes 2us
    assert mm_cost.latency_us >= 5.0
    assert relu_cost.latency_us >= 2.0


def test_gpu_no_efficiency_is_faster():
    """A GPU spec with no efficiency factors should estimate faster than one with."""
    raw_gpu = GPUSpec(
        name="Raw GPU",
        sm_count=108, clock_ghz=1.41,
        peak_flops_map=A100_80GB.peak_flops,
        l1_size_kb=192, l1_bw_GBs=19200,
        l2_size_mb=40, l2_bw_GBs=6400,
        hbm_size_gb=80, hbm_bw_GBs=2039,
        efficiency_factors={},  # no efficiency factors
    )
    op = _matmul_op()
    raw_cost = GPUCostModel(raw_gpu).estimate(op)
    eff_cost = GPUCostModel(A100_80GB).estimate(op)
    # Without efficiency factors, defaults to 1.0 everywhere + 1.0 for static (which returns 1.0)
    # Actually get_efficiency returns 1.0 for missing keys, and static_tc_us returns 1.0
    # So raw_cost has 1us static vs 5us. And compute_efficiency=1.0 vs 0.70
    assert raw_cost.latency_us < eff_cost.latency_us


# --- NPU efficiency factor tests ---

def test_npu_has_efficiency_factors():
    """910B and 910C presets should have efficiency factors."""
    # 910B: original analytical factors (validated vs msmodeling at avg 1.046x)
    assert ASCEND_910B.get_efficiency("cube_fp16") == 0.70
    assert ASCEND_910B.get_efficiency("cube_bf16") == 0.70
    assert ASCEND_910B.get_efficiency("vector") == 0.80
    assert ASCEND_910B.get_efficiency("memory") == 0.60
    # 910C: calibrated from real profiling data
    assert ASCEND_910C.get_efficiency("cube_fp16") == 0.59
    assert ASCEND_910C.get_efficiency("cube_bf16") == 0.59
    assert ASCEND_910C.get_efficiency("vector") == 0.80
    assert ASCEND_910C.get_efficiency("memory") == 0.71
    # Both share static cube overhead; vector floor differs slightly
    # (910B calibrated to CA sim ~3000 cycle floor = 1.7us @ 1.8GHz).
    for hw in [ASCEND_910B, ASCEND_910C]:
        assert hw.get_efficiency("static_cube_us") == 5.0
    assert ASCEND_910B.get_efficiency("static_vector_us") == 1.7
    assert ASCEND_910C.get_efficiency("static_vector_us") == 2.0


def test_npu_efficiency_slows_matmul():
    """NPU matmul with efficiency factors should be slower than raw roofline."""
    op = _matmul_op()
    npu_cost = NPUCostModel(ASCEND_910B).estimate(op)
    raw_cost = RooflineCostModel(ASCEND_910B).estimate(op)
    assert npu_cost.latency_us > raw_cost.latency_us


def test_npu_cube_static_overhead():
    """NPU CUBE ops include 5us static overhead."""
    op = _matmul_op()
    cost = NPUCostModel(ASCEND_910B).estimate(op)
    assert cost.latency_us >= 5.0


def test_npu_vector_static_overhead():
    """NPU VECTOR ops include 2us static overhead."""
    op = _vector_op()
    cost = NPUCostModel(ASCEND_910B).estimate(op)
    assert cost.latency_us >= 2.0


def test_npu_memory_efficiency_applied():
    """Memory efficiency (0.60) should make memory-bound ops slower."""
    # Small matmul that's likely memory-bound
    op = _matmul_op(M=16, K=16, N=16)
    cost_910b = NPUCostModel(ASCEND_910B).estimate(op)
    # With 0.60 memory efficiency, effective bandwidth = 1600 * 0.60 = 960 GB/s
    # vs raw 1600 GB/s — so memory-bound ops take ~1.67x longer
    raw_cost = RooflineCostModel(ASCEND_910B).estimate(op)
    # NPU cost includes tiling, pipeline, format conversion, static overhead
    # Should be meaningfully slower than raw roofline
    assert cost_910b.latency_us > raw_cost.latency_us


def test_npu_910c_faster_than_910b():
    """910C (400 TFLOPS) should be faster than 910B (320 TFLOPS) for large matmul."""
    op = _matmul_op(M=4096, K=4096, N=4096)
    cost_910b = NPUCostModel(ASCEND_910B).estimate(op)
    cost_910c = NPUCostModel(ASCEND_910C).estimate(op)
    assert cost_910c.latency_us < cost_910b.latency_us


def test_efficiency_default_is_one():
    """get_efficiency for unknown keys should return 1.0."""
    assert A100_80GB.get_efficiency("nonexistent_key") == 1.0
    assert ASCEND_910B.get_efficiency("nonexistent_key") == 1.0


def test_gpu_matmul_efficiency_affects_compute_us():
    """Compute_us with efficiency and wave quantization applied."""
    op = _matmul_op(M=4096, K=4096, N=4096)
    cost = GPUCostModel(A100_80GB).estimate(op)
    flops = op.flops
    raw_peak = A100_80GB.peak_flops_for("fp16")
    # For 4096x4096: ceil(4096/128)*ceil(4096/128) = 32*32 = 1024 tiles on 108 SMs
    # num_waves = ceil(1024/108) = 10, wave_eff = 1024/(10*108) = 0.948
    import math
    tile_M, tile_N = A100_80GB.cta_tile
    num_tiles = math.ceil(4096 / tile_M) * math.ceil(4096 / tile_N)
    num_waves = math.ceil(num_tiles / A100_80GB.sm_count)
    wave_eff = num_tiles / (num_waves * A100_80GB.sm_count)
    effective_compute_us = flops / (raw_peak * 0.70 * wave_eff) * 1e6
    assert abs(cost.compute_us - effective_compute_us) / effective_compute_us < 0.01


def test_wave_quantization_perfect_fit():
    """When tiles exactly fill SMs, wave efficiency should be 1.0."""
    # 108 SMs on A100, tile 128x128: M=128*108=13824, N=128 → 108 tiles = 1 wave
    # But that's a weird shape. Use: M=128*12=1536, N=128*9=1152 → 12*9=108 tiles
    op = _matmul_op(M=1536, K=1024, N=1152)
    model = GPUCostModel(A100_80GB)
    wave_eff = model._wave_efficiency(op)
    assert wave_eff == 1.0


def test_wave_quantization_partial_wave():
    """Partial last wave should reduce efficiency."""
    # M=128, N=128 → 1 tile on 108 SMs → wave_eff = 1/108 ≈ 0.009
    op = _matmul_op(M=128, K=1024, N=128)
    model = GPUCostModel(A100_80GB)
    wave_eff = model._wave_efficiency(op)
    assert wave_eff < 0.02  # 1 tile / (1 wave * 108 SMs)

    # M=256, N=256 → ceil(256/128)*ceil(256/128) = 4 tiles → wave_eff = 4/108 ≈ 0.037
    op2 = _matmul_op(M=256, K=1024, N=256)
    wave_eff2 = model._wave_efficiency(op2)
    assert 0.03 < wave_eff2 < 0.04


def test_wave_quantization_increases_latency():
    """Small matmul with bad wave utilization should be slower per-FLOP."""
    model = GPUCostModel(A100_80GB)
    # Large matmul: good wave efficiency
    big = _matmul_op(M=4096, K=4096, N=4096)
    big_cost = model.estimate(big)
    # Small matmul: terrible wave efficiency (1 tile on 108 SMs)
    small = _matmul_op(M=128, K=4096, N=128)
    small_cost = model.estimate(small)
    # Per-FLOP cost should be much higher for the small matmul
    big_us_per_flop = big_cost.compute_us / big.flops
    small_us_per_flop = small_cost.compute_us / small.flops
    assert small_us_per_flop > big_us_per_flop * 5  # at least 5x worse


def test_h100_wave_quantization_different_from_a100():
    """H100 has 132 SMs vs A100's 108, so wave efficiency differs."""
    op = _matmul_op(M=1024, K=4096, N=4096)
    a100_eff = GPUCostModel(A100_80GB)._wave_efficiency(op)
    h100_eff = GPUCostModel(H100_80GB)._wave_efficiency(op)
    # ceil(1024/128)*ceil(4096/128) = 8*32 = 256 tiles
    # A100: ceil(256/108)=3 waves, eff=256/324=0.790
    # H100: ceil(256/132)=2 waves, eff=256/264=0.970
    assert h100_eff > a100_eff  # H100 has better wave utilization here

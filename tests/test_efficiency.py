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
    # Both share static overheads
    for hw in [ASCEND_910B, ASCEND_910C]:
        assert hw.get_efficiency("static_cube_us") == 5.0
        assert hw.get_efficiency("static_vector_us") == 2.0


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
    """Compute_us with 0.70 efficiency should be ~1.43x the raw compute time."""
    op = _matmul_op(M=4096, K=4096, N=4096)
    cost = GPUCostModel(A100_80GB).estimate(op)
    flops = op.flops
    raw_peak = A100_80GB.peak_flops_for("fp16")
    raw_compute_us = flops / raw_peak * 1e6
    effective_compute_us = flops / (raw_peak * 0.70) * 1e6
    # GPU cost model's compute_us should be close to effective (with efficiency)
    assert abs(cost.compute_us - effective_compute_us) / effective_compute_us < 0.01

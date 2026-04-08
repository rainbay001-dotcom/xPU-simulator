"""Phase 4 verification: Ascend NPU backend."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


def test_aligned_matmul():
    """16-aligned matmul should have ~100% CUBE utilization."""
    M, K, N = 2048, 2048, 2048  # all divisible by 16
    a = TensorSpec((M, K), Dtype.FP16)
    b = TensorSpec((K, N), Dtype.FP16)
    c = TensorSpec((M, N), Dtype.FP16)
    op = OpSpec(OpType.MATMUL, [a, b], [c], name="matmul_aligned")

    model = NPUCostModel(ASCEND_910B)
    cost = model.estimate(op)

    print(f"=== Aligned MatMul {M}x{K}x{N} on {ASCEND_910B.name} ===")
    print(f"  Utilization: {cost.utilization:.1%}")
    print(f"  Latency:     {cost.latency_us:.2f} us")
    print(f"  Bound:       {cost.bound}")
    print()

    assert cost.utilization > 0.99, f"Aligned matmul should have ~100% util, got {cost.utilization}"


def test_misaligned_matmul():
    """Non-16-aligned matmul should have lower utilization."""
    M, K, N = 2001, 2001, 2001  # NOT divisible by 16
    a = TensorSpec((M, K), Dtype.FP16)
    b = TensorSpec((K, N), Dtype.FP16)
    c = TensorSpec((M, N), Dtype.FP16)
    op = OpSpec(OpType.MATMUL, [a, b], [c], name="matmul_misaligned")

    model = NPUCostModel(ASCEND_910B)
    cost = model.estimate(op)

    print(f"=== Misaligned MatMul {M}x{K}x{N} on {ASCEND_910B.name} ===")
    print(f"  Utilization: {cost.utilization:.1%}")
    print(f"  Latency:     {cost.latency_us:.2f} us")
    print(f"  Bound:       {cost.bound}")
    print()

    assert cost.utilization < 1.0, "Misaligned matmul should have <100% util"


def test_vector_op():
    """Elementwise ops should use VECTOR pipeline and be memory-bound."""
    shape = (32, 1024, 1024)
    inp = TensorSpec(shape, Dtype.FP16)
    out = TensorSpec(shape, Dtype.FP16)
    op = OpSpec(OpType.RELU, [inp], [out], name="relu")

    model = NPUCostModel(ASCEND_910B)
    cost = model.estimate(op)

    print(f"=== ReLU {shape} on {ASCEND_910B.name} ===")
    print(f"  Latency:     {cost.latency_us:.2f} us")
    print(f"  Bound:       {cost.bound}")
    print()

    assert "memory" in cost.bound.lower(), "ReLU should be memory-bound"


def test_gpu_vs_npu():
    """Compare GPU and NPU for the same workload."""
    M, K, N = 4096, 4096, 4096
    a = TensorSpec((M, K), Dtype.FP16)
    b = TensorSpec((K, N), Dtype.FP16)
    c = TensorSpec((M, N), Dtype.FP16)
    op = OpSpec(OpType.MATMUL, [a, b], [c])

    gpu_cost = GPUCostModel(A100_80GB).estimate(op)
    npu_cost = NPUCostModel(ASCEND_910B).estimate(op)
    npu_c_cost = NPUCostModel(ASCEND_910C).estimate(op)

    print(f"=== MatMul {M}x{K}x{N}: GPU vs NPU ===")
    print(f"  A100:     {gpu_cost.latency_us:.2f} us ({gpu_cost.bound})")
    print(f"  910B:     {npu_cost.latency_us:.2f} us ({npu_cost.bound})")
    print(f"  910C:     {npu_c_cost.latency_us:.2f} us ({npu_c_cost.bound})")
    print()


def test_format_conversion():
    """MatMul on NPU should include Fractal_NZ format conversion cost."""
    M, K, N = 1024, 1024, 1024
    a = TensorSpec((M, K), Dtype.FP16)
    b = TensorSpec((K, N), Dtype.FP16)
    c = TensorSpec((M, N), Dtype.FP16)
    op = OpSpec(OpType.MATMUL, [a, b], [c])

    model = NPUCostModel(ASCEND_910B)
    cost = model.estimate(op)

    # Format conversion should add overhead
    layout = NPUCostModel.preferred_layout(OpType.MATMUL)
    print(f"=== Format Conversion ===")
    print(f"  Preferred layout for MatMul: {layout}")
    print(f"  Total latency: {cost.latency_us:.2f} us (includes conversion overhead)")
    print()

    assert layout == "Fractal_NZ"


if __name__ == "__main__":
    test_aligned_matmul()
    test_misaligned_matmul()
    test_vector_op()
    test_gpu_vs_npu()
    test_format_conversion()
    print("All Phase 4 tests passed!")

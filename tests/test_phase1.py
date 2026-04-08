"""Phase 1 verification: roofline cost model on GPU."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.cost_model import RooflineCostModel
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


def test_matmul():
    """Test MatMul [2048, 2048] x [2048, 2048] on A100."""
    M, K, N = 2048, 2048, 2048
    a = TensorSpec(shape=(M, K), dtype=Dtype.FP16)
    b = TensorSpec(shape=(K, N), dtype=Dtype.FP16)
    c = TensorSpec(shape=(M, N), dtype=Dtype.FP16)

    op = OpSpec(op_type=OpType.MATMUL, inputs=[a, b], outputs=[c], name="matmul_2048")

    model = GPUCostModel(A100_80GB)
    cost = model.estimate(op)

    print(f"=== MatMul {M}x{K}x{N} on {A100_80GB.name} ===")
    print(f"  FLOPs:       {cost.flops:,.0f}")
    print(f"  Bytes:       {cost.bytes_accessed:,.0f}")
    print(f"  AI:          {cost.arithmetic_intensity:.1f} FLOPs/byte")
    print(f"  Compute:     {cost.compute_us:.2f} us")
    print(f"  Memory:      {cost.memory_us:.2f} us")
    print(f"  Latency:     {cost.latency_us:.2f} us")
    print(f"  Bound:       {cost.bound}")
    print(f"  Utilization: {cost.utilization:.1%}")
    print()

    assert cost.flops == 2 * M * K * N
    assert cost.bound == "compute", "Large matmul should be compute-bound"
    assert cost.latency_us > 0


def test_elementwise():
    """Test ReLU on a large tensor — should be memory-bound."""
    shape = (32, 1024, 1024)
    inp = TensorSpec(shape=shape, dtype=Dtype.FP16)
    out = TensorSpec(shape=shape, dtype=Dtype.FP16)

    op = OpSpec(op_type=OpType.RELU, inputs=[inp], outputs=[out], name="relu")

    model = GPUCostModel(A100_80GB)
    cost = model.estimate(op)

    print(f"=== ReLU {shape} on {A100_80GB.name} ===")
    print(f"  FLOPs:       {cost.flops:,.0f}")
    print(f"  Bytes:       {cost.bytes_accessed:,.0f}")
    print(f"  AI:          {cost.arithmetic_intensity:.4f} FLOPs/byte")
    print(f"  Compute:     {cost.compute_us:.2f} us")
    print(f"  Memory:      {cost.memory_us:.2f} us")
    print(f"  Latency:     {cost.latency_us:.2f} us")
    print(f"  Bound:       {cost.bound}")
    print()

    assert cost.bound == "memory", "Elementwise op should be memory-bound"


def test_h100_vs_a100():
    """H100 should be faster than A100 for the same matmul."""
    M, K, N = 4096, 4096, 4096
    a = TensorSpec(shape=(M, K), dtype=Dtype.FP16)
    b = TensorSpec(shape=(K, N), dtype=Dtype.FP16)
    c = TensorSpec(shape=(M, N), dtype=Dtype.FP16)
    op = OpSpec(op_type=OpType.MATMUL, inputs=[a, b], outputs=[c])

    a100_cost = GPUCostModel(A100_80GB).estimate(op)
    h100_cost = GPUCostModel(H100_80GB).estimate(op)

    print(f"=== MatMul {M}x{K}x{N}: A100 vs H100 ===")
    print(f"  A100: {a100_cost.latency_us:.2f} us")
    print(f"  H100: {h100_cost.latency_us:.2f} us")
    print(f"  Speedup: {a100_cost.latency_us / h100_cost.latency_us:.2f}x")
    print()

    assert h100_cost.latency_us < a100_cost.latency_us


if __name__ == "__main__":
    test_matmul()
    test_elementwise()
    test_h100_vs_a100()
    print("All Phase 1 tests passed!")

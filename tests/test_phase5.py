"""Phase 5 verification: overlap modeling, profiling, CLI."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.utils.profiling import print_timeline, to_chrome_trace


def test_overlap():
    """Two independent matmuls should overlap, halving latency."""
    M, K, N = 2048, 2048, 2048
    a = TensorSpec((M, K), Dtype.FP16)
    b = TensorSpec((K, N), Dtype.FP16)
    c = TensorSpec((M, N), Dtype.FP16)

    op1 = OpSpec(OpType.MATMUL, [a, b], [c], name="matmul_1")
    op2 = OpSpec(OpType.MATMUL, [a, b], [c], name="matmul_2")

    # Two independent ops (no edges between them)
    graph = ComputeGraph("parallel_matmuls")
    n1 = graph.add_node(op1, "matmul_1")
    n2 = graph.add_node(op2, "matmul_2")
    # No edge — they are independent

    model = GPUCostModel(A100_80GB)
    evaluator = PerformanceEvaluator(model)

    seq_result = evaluator.run(graph, overlap=False)
    par_result = evaluator.run(graph, overlap=True)

    print(f"=== Overlap Test: Two Independent MatMuls ===")
    print(f"  Sequential: {seq_result.total_latency_us:.2f} us")
    print(f"  Parallel:   {par_result.total_latency_us:.2f} us")
    print(f"  Speedup:    {par_result.speedup_from_overlap:.2f}x")
    print()

    # Parallel should be ~half of sequential (both start at t=0)
    assert par_result.total_latency_us < seq_result.total_latency_us * 0.6, \
        "Parallel should be significantly faster than sequential"
    assert par_result.speedup_from_overlap > 1.5


def test_chain_no_overlap():
    """A chain of dependent ops should have no speedup from overlap."""
    shape = (32, 1024)
    t = TensorSpec(shape, Dtype.FP16)

    ops = [
        OpSpec(OpType.RELU, [t], [t], name=f"relu_{i}")
        for i in range(4)
    ]

    graph = ComputeGraph("chain")
    nodes = [graph.add_node(op, op.name) for op in ops]
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i], nodes[i + 1], t)

    model = GPUCostModel(A100_80GB)
    evaluator = PerformanceEvaluator(model)

    seq_result = evaluator.run(graph, overlap=False)
    par_result = evaluator.run(graph, overlap=True)

    print(f"=== Chain Test: 4 Dependent ReLUs ===")
    print(f"  Sequential: {seq_result.total_latency_us:.2f} us")
    print(f"  Parallel:   {par_result.total_latency_us:.2f} us")
    print(f"  Speedup:    {par_result.speedup_from_overlap:.2f}x")
    print()

    # No speedup — all ops are dependent
    assert abs(seq_result.total_latency_us - par_result.total_latency_us) < 0.01


def test_timeline_and_trace():
    """Test profiling output."""
    M, K, N = 1024, 1024, 1024
    a = TensorSpec((M, K), Dtype.FP16)
    b = TensorSpec((K, N), Dtype.FP16)
    c = TensorSpec((M, N), Dtype.FP16)
    t = TensorSpec((32, 1024), Dtype.FP16)

    graph = ComputeGraph("mixed")
    n1 = graph.add_node(OpSpec(OpType.MATMUL, [a, b], [c], name="matmul"), "matmul")
    n2 = graph.add_node(OpSpec(OpType.RELU, [t], [t], name="relu"), "relu")
    n3 = graph.add_node(OpSpec(OpType.MATMUL, [a, b], [c], name="matmul2"), "matmul2")
    graph.add_edge(n1, n3)
    # n2 is independent of n1 and n3

    model = GPUCostModel(A100_80GB)
    result = PerformanceEvaluator(model).run(graph, overlap=True)

    print("=== Timeline ===")
    print_timeline(result)
    print()

    # Export trace
    trace_file = to_chrome_trace(result, "/tmp/xpu_sim_test_trace.json")
    print(f"Chrome trace exported to: {trace_file}")
    print()


if __name__ == "__main__":
    test_overlap()
    test_chain_no_overlap()
    test_timeline_and_trace()
    print("All Phase 5 tests passed!")

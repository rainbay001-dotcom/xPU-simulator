"""Phase 2 verification: computation graph + evaluator."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


def test_linear_relu_linear():
    """Test a simple 3-op graph: linear -> relu -> linear."""
    B, D = 32, 1024

    # Op 1: Linear (matmul) [B, D] x [D, D] -> [B, D]
    op1 = OpSpec(
        op_type=OpType.MATMUL,
        inputs=[TensorSpec((B, D), Dtype.FP16), TensorSpec((D, D), Dtype.FP16)],
        outputs=[TensorSpec((B, D), Dtype.FP16)],
        name="linear1",
    )

    # Op 2: ReLU [B, D] -> [B, D]
    op2 = OpSpec(
        op_type=OpType.RELU,
        inputs=[TensorSpec((B, D), Dtype.FP16)],
        outputs=[TensorSpec((B, D), Dtype.FP16)],
        name="relu",
    )

    # Op 3: Linear (matmul) [B, D] x [D, D] -> [B, D]
    op3 = OpSpec(
        op_type=OpType.MATMUL,
        inputs=[TensorSpec((B, D), Dtype.FP16), TensorSpec((D, D), Dtype.FP16)],
        outputs=[TensorSpec((B, D), Dtype.FP16)],
        name="linear2",
    )

    # Build graph
    graph = ComputeGraph("linear_relu_linear")
    n1 = graph.add_node(op1, "linear1")
    n2 = graph.add_node(op2, "relu")
    n3 = graph.add_node(op3, "linear2")
    graph.add_edge(n1, n2, TensorSpec((B, D), Dtype.FP16))
    graph.add_edge(n2, n3, TensorSpec((B, D), Dtype.FP16))

    print(f"Graph: {graph}")
    print(f"Topo order: {graph.topo_order()}")
    print()

    # Evaluate
    model = GPUCostModel(A100_80GB)
    evaluator = PerformanceEvaluator(model)
    result = evaluator.run(graph)

    print(result.summary())
    print()

    # Per-op breakdown
    print("Per-op breakdown:")
    for r in result.per_op:
        print(f"  {r.node}: {r.cost.latency_us:.2f} us ({r.cost.bound})")
    print()

    assert result.total_latency_us > 0
    assert len(result.per_op) == 3
    assert result.bottleneck_op is not None

    # Total should equal sum of individual ops (sequential, no overlap)
    expected = sum(r.cost.latency_us for r in result.per_op)
    assert abs(result.total_latency_us - expected) < 0.01, \
        f"Total {result.total_latency_us} != sum {expected}"


def test_graph_structure():
    """Test graph API basics."""
    graph = ComputeGraph("test")
    op = OpSpec(OpType.RELU, [TensorSpec((10,))], [TensorSpec((10,))])

    n1 = graph.add_node(op, "a")
    n2 = graph.add_node(op, "b")
    n3 = graph.add_node(op, "c")
    graph.add_edge(n1, n2)
    graph.add_edge(n1, n3)

    assert graph.num_nodes == 3
    assert graph.num_edges == 2
    assert graph.successors(n1) == [n2, n3] or set(graph.successors(n1)) == {n2, n3}
    assert graph.predecessors(n2) == [n1]
    print("Graph structure test passed!")


if __name__ == "__main__":
    test_graph_structure()
    test_linear_relu_linear()
    print("All Phase 2 tests passed!")

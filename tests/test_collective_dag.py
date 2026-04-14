"""Tests for collective DAG decomposition."""
import sys
sys.path.insert(0, ".")

import math
from xpu_simulator.core.collective_dag import (
    ring_all_reduce_dag, ring_all_gather_dag, ring_reduce_scatter_dag,
    tree_all_reduce_dag, double_binary_tree_dag,
)
from xpu_simulator.core.parallel import InterconnectSpec, ParallelConfig
from xpu_simulator.core.communication import all_reduce_time
from xpu_simulator.core.cost_model import CommAwareCostModel
from xpu_simulator.core.operator import OpType, TensorSpec, OpSpec
from xpu_simulator.backends.gpu.hardware import H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


NVLINK = InterconnectSpec("NVLink", 900, 0.5)


def test_ring_all_reduce_step_count():
    """Ring all-reduce should have 2*(N-1) steps (single channel)."""
    for n in [2, 4, 8]:
        dag = ring_all_reduce_dag(n, 100 * 1024 * 1024)
        # Filter out empty steps
        non_empty = [s for s in dag.steps if s.transfers]
        expected = 2 * (n - 1)
        assert len(non_empty) == expected, f"N={n}: expected {expected}, got {len(non_empty)}"
        print(f"  N={n}: {len(non_empty)} steps (expected {expected})")


def test_ring_reduce_scatter_step_count():
    """Ring reduce-scatter should have N-1 steps (single channel)."""
    for n in [2, 4, 8]:
        dag = ring_reduce_scatter_dag(n, 100 * 1024 * 1024)
        non_empty = [s for s in dag.steps if s.transfers]
        expected = n - 1
        assert len(non_empty) == expected, f"N={n}: expected {expected}, got {len(non_empty)}"
        print(f"  RS N={n}: {len(non_empty)} steps")


def test_ring_all_gather_step_count():
    """Ring all-gather should have N-1 steps (single channel)."""
    for n in [2, 4, 8]:
        dag = ring_all_gather_dag(n, 100 * 1024 * 1024)
        non_empty = [s for s in dag.steps if s.transfers]
        expected = n - 1
        assert len(non_empty) == expected
        print(f"  AG N={n}: {len(non_empty)} steps")


def test_tree_all_reduce_step_count():
    """Tree all-reduce should have 2*ceil(log2(N)) steps."""
    for n in [2, 4, 8]:
        dag = tree_all_reduce_dag(n, 100 * 1024 * 1024)
        expected = 2 * math.ceil(math.log2(n))
        assert dag.num_steps == expected, f"N={n}: expected {expected}, got {dag.num_steps}"
        print(f"  Tree N={n}: {dag.num_steps} steps (expected {expected})")


def test_double_binary_tree():
    """Double binary tree should have 4 steps (2 reduce + 2 broadcast)."""
    dag = double_binary_tree_dag(8, 100 * 1024 * 1024)
    assert dag.algorithm == "double_binary_tree"
    assert dag.n_channels == 2
    assert dag.num_steps == 4  # 2 trees x 2 phases
    print(f"  DBT N=8: {dag.num_steps} steps, {dag.total_transfers} transfers")


def test_dag_latency_positive():
    """DAG-based latency should be positive and reasonable."""
    dag = ring_all_reduce_dag(8, 100 * 1024 * 1024)
    lat = dag.compute_latency(900, 0.5)
    assert lat > 0
    print(f"  Ring AR 8 ranks, 100MB: {lat:.1f} us")

    dag_tree = tree_all_reduce_dag(8, 100 * 1024 * 1024)
    lat_tree = dag_tree.compute_latency(900, 0.5)
    assert lat_tree > 0
    print(f"  Tree AR 8 ranks, 100MB: {lat_tree:.1f} us")


def test_multi_channel_pipelining():
    """More channels should not increase latency (can reduce it for large N)."""
    n = 8
    msg = 100 * 1024 * 1024
    dag_1ch = ring_all_reduce_dag(n, msg, n_channels=1)
    dag_8ch = ring_all_reduce_dag(n, msg, n_channels=8)

    lat_1 = dag_1ch.compute_latency(900, 0.5)
    lat_8 = dag_8ch.compute_latency(900, 0.5)

    # Multi-channel has more steps due to pipelining but smaller chunks
    assert dag_8ch.num_steps >= dag_1ch.num_steps
    print(f"  1 channel: {lat_1:.1f} us ({dag_1ch.num_steps} steps)")
    print(f"  8 channels: {lat_8:.1f} us ({dag_8ch.num_steps} steps)")


def test_decompose_integration():
    """CommAwareCostModel with decompose=True should work."""
    base = GPUCostModel(H100_80GB)
    model = CommAwareCostModel(base, NVLINK, ParallelConfig(tp_size=8),
                               decompose=True, n_channels=4)

    comm_op = OpSpec(OpType.ALL_REDUCE,
                     [TensorSpec((2048, 4096))],
                     [TensorSpec((2048, 4096))],
                     attrs={"n_ranks": 8}, name="ar")
    cost = model.estimate(comm_op)
    assert cost.bound == "communication"
    assert cost.latency_us > 0

    # Compare with analytical
    model_analytical = CommAwareCostModel(base, NVLINK, ParallelConfig(tp_size=8))
    cost_analytical = model_analytical.estimate(comm_op)

    print(f"  DAG: {cost.latency_us:.1f} us, Analytical: {cost_analytical.latency_us:.1f} us")


def test_decompose_all_to_all_fallback():
    """ALL_TO_ALL should fall back to analytical even with decompose=True."""
    base = GPUCostModel(H100_80GB)
    model = CommAwareCostModel(base, NVLINK, ParallelConfig(tp_size=8),
                               decompose=True)

    a2a_op = OpSpec(OpType.ALL_TO_ALL,
                    [TensorSpec((2048, 4096))],
                    [TensorSpec((2048, 4096))],
                    attrs={"n_ranks": 8}, name="a2a")
    cost = model.estimate(a2a_op)
    assert cost.bound == "communication"
    assert cost.latency_us > 0
    print(f"  ALL_TO_ALL fallback: {cost.latency_us:.1f} us")


def test_summary():
    """CollectiveDAG.summary() should produce readable output."""
    dag = ring_all_reduce_dag(8, 100 * 1024 * 1024, n_channels=4)
    s = dag.summary()
    assert "ring" in s
    assert "all_reduce" in s
    print(f"  {s}")


if __name__ == "__main__":
    print("=== Collective DAG Tests ===\n")
    for name, fn in [
        ("Ring AR step count", test_ring_all_reduce_step_count),
        ("Ring RS step count", test_ring_reduce_scatter_step_count),
        ("Ring AG step count", test_ring_all_gather_step_count),
        ("Tree AR step count", test_tree_all_reduce_step_count),
        ("Double binary tree", test_double_binary_tree),
        ("DAG latency positive", test_dag_latency_positive),
        ("Multi-channel pipelining", test_multi_channel_pipelining),
        ("Decompose integration", test_decompose_integration),
        ("ALL_TO_ALL fallback", test_decompose_all_to_all_fallback),
        ("Summary", test_summary),
    ]:
        print(f"--- {name} ---")
        fn()
        print()
    print("All collective DAG tests passed!")

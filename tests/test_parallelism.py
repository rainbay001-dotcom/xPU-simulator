"""Tests for multi-device parallelism and communication modeling."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpType, Dtype, TensorSpec, OpSpec
from xpu_simulator.core.parallel import ParallelConfig, InterconnectSpec
from xpu_simulator.core.communication import (
    all_reduce_time, all_gather_time, reduce_scatter_time, all_to_all_time,
)
from xpu_simulator.core.cost_model import CommAwareCostModel
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.backends.gpu.hardware import H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.frontend.graph_builder import GraphBuilder
from xpu_simulator.frontend.config_extractor import ConfigExtractor


NVLINK = InterconnectSpec("NVLink", 900, 0.5)


# ------------------------------------------------------------------ #
# Communication formula tests
# ------------------------------------------------------------------ #

def test_all_reduce_ring_formula():
    """Ring all-reduce: 2*(N-1)/N * msg/bw."""
    n = 8
    msg = 100 * 1024 * 1024  # 100 MB
    cc = all_reduce_time(msg, n, NVLINK)

    bw = 900e9  # B/s
    expected_bw_us = 2 * (n - 1) / n * msg / bw * 1e6
    expected_lat_us = 2 * (n - 1) * 0.5

    # Should be close to ring formula (ring wins for large messages)
    assert abs(cc.latency_us - (expected_bw_us + expected_lat_us)) < 1.0
    assert cc.algorithm == "ring"
    print(f"  All-reduce {msg/1e6:.0f}MB, {n} ranks: {cc.latency_us:.1f} us ({cc.algorithm})")


def test_all_gather_formula():
    """All-gather formula verification."""
    n = 4
    msg = 50 * 1024 * 1024  # 50 MB per rank
    cc = all_gather_time(msg, n, NVLINK)

    assert cc.latency_us > 0
    # Single rank → zero latency
    cc_single = all_gather_time(msg, 1, NVLINK)
    assert cc_single.latency_us == 0.0
    print(f"  All-gather {msg/1e6:.0f}MB/rank, {n} ranks: {cc.latency_us:.1f} us")


def test_all_to_all_formula():
    """All-to-all formula verification."""
    n = 8
    msg = 10 * 1024 * 1024  # 10 MB per rank pair
    cc = all_to_all_time(msg, n, NVLINK)

    assert cc.latency_us > 0
    assert cc.algorithm == "pairwise"
    print(f"  All-to-all {msg/1e6:.0f}MB, {n} ranks: {cc.latency_us:.1f} us")


def test_comm_algorithm_selection():
    """Small messages should prefer tree, large should prefer ring."""
    n = 8
    # Very small message: tree wins (fewer latency steps)
    cc_small = all_reduce_time(1024, n, NVLINK)  # 1 KB
    # Very large message: ring wins (better bandwidth utilization)
    cc_large = all_reduce_time(1 * 1024**3, n, NVLINK)  # 1 GB

    print(f"  1KB:  {cc_small.algorithm}, {cc_small.latency_us:.4f} us")
    print(f"  1GB:  {cc_large.algorithm}, {cc_large.latency_us:.1f} us")
    # Ring should be preferred for large messages
    assert cc_large.algorithm == "ring"


def test_reduce_scatter():
    """Reduce-scatter formula verification."""
    n = 4
    msg = 100 * 1024 * 1024
    cc = reduce_scatter_time(msg, n, NVLINK)
    assert cc.latency_us > 0
    # Reduce-scatter should be roughly half of all-reduce (one direction)
    cc_ar = all_reduce_time(msg, n, NVLINK)
    assert cc.latency_us < cc_ar.latency_us
    print(f"  Reduce-scatter: {cc.latency_us:.1f} us vs all-reduce: {cc_ar.latency_us:.1f} us")


# ------------------------------------------------------------------ #
# Graph builder parallelism tests
# ------------------------------------------------------------------ #

def test_tp2_linear_graph():
    """TP=2 linear should have ALL_REDUCE and half-sized weight."""
    parallel = ParallelConfig(tp_size=2)
    gb = GraphBuilder("test", parallel=parallel)
    prev = gb.norm("norm", (1024, 4096))
    gb.swiglu_mlp("ffn", 1024, 4096, 11008, prev)
    g = gb.build()

    ar_nodes = [n for n in g.nodes if n.op.op_type == OpType.ALL_REDUCE]
    assert len(ar_nodes) >= 1, f"Expected ALL_REDUCE, found: {[n.name for n in g.nodes if 'reduce' in (n.name or '').lower()]}"

    # w1 should be sharded: output dim = 11008 / 2 = 5504
    w1_nodes = [n for n in g.nodes if n.name == "ffn.w1"]
    assert len(w1_nodes) == 1
    w1_out = w1_nodes[0].op.outputs[0]
    assert w1_out.shape[-1] == 11008 // 2, f"Expected 5504, got {w1_out.shape[-1]}"
    print(f"  TP=2 SwiGLU: {len(g.nodes)} nodes, {len(ar_nodes)} ALL_REDUCE")


def test_tp4_attention_graph():
    """TP=4 attention should shard heads across devices."""
    from xpu_simulator.frontend.config_normalizer import AttentionPattern

    parallel = ParallelConfig(tp_size=4)
    gb = GraphBuilder("test", parallel=parallel)
    prev = gb.norm("norm", (1024, 4096))
    gb.attention("attn", 1024, 1, 1024, 4096, n_heads=32, head_dim=128,
                 pattern=AttentionPattern(), n_kv_heads=8, prev=prev)
    g = gb.build()

    # wq should project to 32/4 = 8 heads * 128 = 1024 dims
    wq_nodes = [n for n in g.nodes if n.name == "attn.wq"]
    assert len(wq_nodes) == 1
    wq_out_dim = wq_nodes[0].op.outputs[0].shape[-1]
    assert wq_out_dim == 8 * 128, f"Expected {8*128}, got {wq_out_dim}"

    # Should have ALL_REDUCE for wo
    ar_nodes = [n for n in g.nodes if n.op.op_type == OpType.ALL_REDUCE]
    assert len(ar_nodes) >= 1
    print(f"  TP=4 attention: {len(g.nodes)} nodes, wq output dim={wq_out_dim}")


def test_ep4_moe_graph():
    """EP=4 MoE should have ALL_TO_ALL for dispatch/combine."""
    parallel = ParallelConfig(ep_size=4)
    gb = GraphBuilder("test", parallel=parallel)
    prev = gb.norm("norm", (1024, 4096))
    gb.moe_layer("moe", 1024, 4096, 2048, n_experts=64, n_activated=8, prev=prev)
    g = gb.build()

    a2a_nodes = [n for n in g.nodes if n.op.op_type == OpType.ALL_TO_ALL]
    assert len(a2a_nodes) >= 2, f"Expected 2 ALL_TO_ALL, found {len(a2a_nodes)}"

    # Expert tokens should be divided by EP
    expert_w1 = [n for n in g.nodes if n.name == "moe.experts_w1"]
    assert len(expert_w1) == 1
    # tokens * n_activated // ep = 1024 * 8 // 4 = 2048
    assert expert_w1[0].op.inputs[0].shape[0] == 1024 * 8 // 4
    print(f"  EP=4 MoE: {len(g.nodes)} nodes, {len(a2a_nodes)} ALL_TO_ALL")


def test_comm_overlap_with_compute():
    """Communication ops should overlap with compute (different resources)."""
    graph = ComputeGraph("overlap_test")

    # A matmul and a comm op with no dependency
    mm = OpSpec(OpType.MATMUL,
                [TensorSpec((2048, 2048)), TensorSpec((2048, 2048))],
                [TensorSpec((2048, 2048))], name="matmul")
    comm = OpSpec(OpType.ALL_REDUCE,
                  [TensorSpec((2048, 2048))],
                  [TensorSpec((2048, 2048))],
                  attrs={"n_ranks": 4}, name="allreduce")

    mm_node = graph.add_node(mm, "matmul")
    comm_node = graph.add_node(comm, "allreduce")
    # No edge → independent

    model = CommAwareCostModel(
        GPUCostModel(H100_80GB), NVLINK, ParallelConfig(tp_size=4))
    evaluator = PerformanceEvaluator(model)

    result_overlap = evaluator.run(graph, overlap=True)
    result_seq = evaluator.run(graph, overlap=False)

    print(f"  Sequential: {result_seq.total_latency_us:.1f} us")
    print(f"  Overlapped: {result_overlap.total_latency_us:.1f} us")

    # Overlapped should be faster (comm and compute on different resources)
    assert result_overlap.total_latency_us < result_seq.total_latency_us


def test_single_device_backward_compat():
    """ParallelConfig(1,1,1) should produce identical graph to no config."""
    gb_no_par = GraphBuilder("no_par")
    prev = gb_no_par.norm("norm", (1024, 4096))
    gb_no_par.swiglu_mlp("ffn", 1024, 4096, 11008, prev)
    g1 = gb_no_par.build()

    gb_par = GraphBuilder("with_par", parallel=ParallelConfig())
    prev = gb_par.norm("norm", (1024, 4096))
    gb_par.swiglu_mlp("ffn", 1024, 4096, 11008, prev)
    g2 = gb_par.build()

    # Same number of nodes and no comm ops
    assert len(g1.nodes) == len(g2.nodes)
    comm_ops = [n for n in g2.nodes if n.op.op_type in
                (OpType.ALL_REDUCE, OpType.ALL_GATHER, OpType.ALL_TO_ALL)]
    assert len(comm_ops) == 0
    print(f"  Single device: {len(g1.nodes)} nodes = {len(g2.nodes)} nodes, no comm ops")


# ------------------------------------------------------------------ #
# CommAwareCostModel test
# ------------------------------------------------------------------ #

def test_comm_aware_cost_model():
    """CommAwareCostModel should dispatch comm ops to formulas."""
    base = GPUCostModel(H100_80GB)
    model = CommAwareCostModel(base, NVLINK, ParallelConfig(tp_size=4))

    # Comm op
    comm_op = OpSpec(OpType.ALL_REDUCE,
                     [TensorSpec((2048, 4096))],
                     [TensorSpec((2048, 4096))],
                     attrs={"n_ranks": 4}, name="ar")
    cost = model.estimate(comm_op)
    assert cost.bound == "communication"
    assert cost.latency_us > 0

    # Compute op should fall through to base
    mm_op = OpSpec(OpType.MATMUL,
                   [TensorSpec((2048, 2048)), TensorSpec((2048, 2048))],
                   [TensorSpec((2048, 2048))], name="mm")
    cost_mm = model.estimate(mm_op)
    assert cost_mm.bound != "communication"
    print(f"  ALL_REDUCE: {cost.latency_us:.1f} us, MatMul: {cost_mm.latency_us:.1f} us")


if __name__ == "__main__":
    print("=== Parallelism Tests ===\n")

    for name, fn in [
        ("All-reduce ring formula", test_all_reduce_ring_formula),
        ("All-gather formula", test_all_gather_formula),
        ("All-to-all formula", test_all_to_all_formula),
        ("Comm algorithm selection", test_comm_algorithm_selection),
        ("Reduce-scatter", test_reduce_scatter),
        ("TP=2 linear graph", test_tp2_linear_graph),
        ("TP=4 attention graph", test_tp4_attention_graph),
        ("EP=4 MoE graph", test_ep4_moe_graph),
        ("Comm overlap with compute", test_comm_overlap_with_compute),
        ("Single device compat", test_single_device_backward_compat),
        ("CommAwareCostModel", test_comm_aware_cost_model),
    ]:
        print(f"--- {name} ---")
        fn()
        print()

    print("All parallelism tests passed!")

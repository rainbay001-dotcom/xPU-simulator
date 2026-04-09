"""Smoke tests for visualization and HTML report utilities."""

import os
import sys
import tempfile

sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.cost_model import RooflineCostModel
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.utils.categories import categorize_op, CATEGORY_COLORS
from xpu_simulator.utils.visualize import (
    export_architecture_overview,
    export_block_detail,
    export_dataflow_graph,
)
from xpu_simulator.utils.html_report import export_html_report


def _make_test_graph():
    """Build a small two-layer transformer-like graph for testing."""
    graph = ComputeGraph("test-model")
    B, S, D = 1, 16, 64
    dtype = Dtype.FP16

    def t(shape):
        return TensorSpec(shape, dtype)

    prev = None
    for layer_id in range(2):
        prefix = f"L{layer_id}"

        # Attention norm
        norm_op = OpSpec(OpType.LAYER_NORM, [t((B * S, D))], [t((B * S, D))], name=f"{prefix}.attn_norm")
        norm = graph.add_node(norm_op, f"{prefix}.attn_norm")
        if prev:
            graph.add_edge(prev, norm)

        # Q projection
        wq_op = OpSpec(OpType.MATMUL, [t((B * S, D)), t((D, D))], [t((B * S, D))], name=f"{prefix}.wq_a")
        wq = graph.add_node(wq_op, f"{prefix}.wq_a")
        graph.add_edge(norm, wq)

        # Attention score
        attn_op = OpSpec(OpType.MATMUL, [t((B, S, D)), t((B, D, S))], [t((B, S, S))], name=f"{prefix}.attn_score")
        attn = graph.add_node(attn_op, f"{prefix}.attn_score")
        graph.add_edge(wq, attn)

        # Softmax
        sm_op = OpSpec(OpType.SOFTMAX, [t((B, S, S))], [t((B, S, S))], name=f"{prefix}.attn_softmax")
        sm = graph.add_node(sm_op, f"{prefix}.attn_softmax")
        graph.add_edge(attn, sm)

        # Output proj
        wo_op = OpSpec(OpType.MATMUL, [t((B * S, D)), t((D, D))], [t((B * S, D))], name=f"{prefix}.wo")
        wo = graph.add_node(wo_op, f"{prefix}.wo")
        graph.add_edge(sm, wo)

        # FFN norm
        ffn_norm_op = OpSpec(OpType.LAYER_NORM, [t((B * S, D))], [t((B * S, D))], name=f"{prefix}.ffn_norm")
        ffn_norm = graph.add_node(ffn_norm_op, f"{prefix}.ffn_norm")
        graph.add_edge(wo, ffn_norm)

        # FFN w1
        w1_op = OpSpec(OpType.MATMUL, [t((B * S, D)), t((D, D * 4))], [t((B * S, D * 4))], name=f"{prefix}.ffn.w1")
        w1 = graph.add_node(w1_op, f"{prefix}.ffn.w1")
        graph.add_edge(ffn_norm, w1)

        # FFN w2
        w2_op = OpSpec(OpType.MATMUL, [t((B * S, D * 4)), t((D * 4, D))], [t((B * S, D))], name=f"{prefix}.ffn.w2")
        w2 = graph.add_node(w2_op, f"{prefix}.ffn.w2")
        graph.add_edge(w1, w2)

        prev = w2

    return graph


def _make_result(graph):
    """Run the test graph through a cost model."""
    cost_model = GPUCostModel(A100_80GB)
    evaluator = PerformanceEvaluator(cost_model)
    return evaluator.run(graph, overlap=True)


def test_categorize_op():
    """Test that categorize_op returns correct categories."""
    assert categorize_op("L0.attn_score") == "Attention Compute"
    assert categorize_op("L0.attn_v") == "Attention Compute"
    assert categorize_op("L0.wq_a") == "Attention Projections"
    assert categorize_op("L0.wkv_a") == "Attention Projections"
    assert categorize_op("L0.wo") == "Attention Projections"
    assert categorize_op("L0.moe.experts_w1") == "MoE Experts"
    assert categorize_op("L0.moe.shared_w1") == "MoE Shared Expert"
    assert categorize_op("L0.moe.gate") == "MoE Gate"
    assert categorize_op("L0.ffn.w1") == "Dense FFN"
    assert categorize_op("L0.attn_norm") == "Norms"
    assert categorize_op("L0.rope_q") == "RoPE"
    assert categorize_op("embedding") == "Embedding"
    assert categorize_op("lm_head") == "LM Head"
    assert categorize_op("unknown_op") == "Other"
    print("  categorize_op: all categories correct")


def test_category_colors():
    """Test that CATEGORY_COLORS covers all categories from categorize_op."""
    test_names = [
        "L0.attn_score", "L0.wq_a", "L0.moe.experts_w1", "L0.moe.shared_w1",
        "L0.moe.gate", "L0.ffn.w1", "L0.attn_norm", "L0.rope_q",
        "embedding", "lm_head", "unknown_op",
    ]
    for name in test_names:
        cat = categorize_op(name)
        assert cat in CATEGORY_COLORS, f"Category '{cat}' not in CATEGORY_COLORS"
    print("  CATEGORY_COLORS: all categories have colors")


def test_export_block_detail():
    """Test export_block_detail produces a non-empty PNG."""
    graph = _make_test_graph()
    result = _make_result(graph)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fname = f.name

    try:
        ret = export_block_detail(graph, "L0", filename=fname, result=result,
                                  model_name="Test Model")
        assert ret == fname
        assert os.path.exists(fname)
        size = os.path.getsize(fname)
        assert size > 0, f"Block detail PNG is empty (0 bytes)"
        print(f"  export_block_detail: OK ({size} bytes)")
    finally:
        os.unlink(fname)


def test_export_architecture_overview():
    """Test export_architecture_overview produces a non-empty PNG."""
    graph = _make_test_graph()
    result = _make_result(graph)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fname = f.name

    try:
        ret = export_architecture_overview(graph, filename=fname, result=result,
                                           model_name="Test Model", n_dense=1)
        assert ret == fname
        assert os.path.exists(fname)
        size = os.path.getsize(fname)
        assert size > 0, f"Architecture overview PNG is empty (0 bytes)"
        print(f"  export_architecture_overview: OK ({size} bytes)")
    finally:
        os.unlink(fname)


def test_export_dataflow_graph():
    """Test export_dataflow_graph produces a non-empty PNG."""
    graph = _make_test_graph()
    result = _make_result(graph)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fname = f.name

    try:
        ret = export_dataflow_graph(graph, filename=fname, result=result,
                                    model_name="Test Model", layer_filter="L0")
        assert ret == fname
        assert os.path.exists(fname)
        size = os.path.getsize(fname)
        assert size > 0, f"Dataflow graph PNG is empty (0 bytes)"
        print(f"  export_dataflow_graph: OK ({size} bytes)")
    finally:
        os.unlink(fname)


def test_export_html_report():
    """Test export_html_report produces valid non-empty HTML."""
    graph = _make_test_graph()
    result = _make_result(graph)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        fname = f.name

    try:
        ret = export_html_report(
            graph,
            results={"Test GPU": result},
            filename=fname,
            model_name="Test Model",
            config={"dim": 64, "layers": 2},
            n_dense=1,
        )
        assert ret == fname
        assert os.path.exists(fname)
        size = os.path.getsize(fname)
        assert size > 0, f"HTML report is empty (0 bytes)"

        with open(fname) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "Test Model" in content
        assert "Test GPU" in content
        assert "__REPORT_DATA__" not in content, "Template placeholder was not replaced"
        print(f"  export_html_report: OK ({size} bytes)")
    finally:
        os.unlink(fname)


def test_export_html_report_defaults():
    """Test export_html_report works with default parameters."""
    graph = _make_test_graph()
    result = _make_result(graph)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        fname = f.name

    try:
        ret = export_html_report(
            graph,
            results={"GPU": result},
            filename=fname,
        )
        assert ret == fname
        assert os.path.getsize(fname) > 0
        print("  export_html_report (defaults): OK")
    finally:
        os.unlink(fname)


if __name__ == "__main__":
    print("=== Visualization Smoke Tests ===")
    test_categorize_op()
    test_category_colors()
    test_export_block_detail()
    test_export_architecture_overview()
    test_export_dataflow_graph()
    test_export_html_report()
    test_export_html_report_defaults()
    print("\nAll visualization tests passed!")

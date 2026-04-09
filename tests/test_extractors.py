"""Tests for all graph extractors and the GraphBuilder."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, ".")

import torch
import torch.nn as nn

from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.operator import OpType, Dtype
from xpu_simulator.frontend.base import GraphExtractor
from xpu_simulator.frontend.torch_extractor import TorchGraphExtractor
from xpu_simulator.frontend.graph_builder import GraphBuilder
from xpu_simulator.frontend.profiler_extractor import ProfilerExtractor


# ------------------------------------------------------------------ #
# Base ABC
# ------------------------------------------------------------------ #

def test_base_extractor_is_abstract():
    """GraphExtractor cannot be instantiated directly."""
    import pytest
    with pytest.raises(TypeError):
        GraphExtractor()


def test_fx_extractor_is_graph_extractor():
    """TorchGraphExtractor inherits from GraphExtractor."""
    ext = TorchGraphExtractor()
    assert isinstance(ext, GraphExtractor)


# ------------------------------------------------------------------ #
# ExportExtractor
# ------------------------------------------------------------------ #

def test_export_extractor_simple_mlp():
    """ExportExtractor should produce a valid graph from a simple MLP."""
    try:
        from xpu_simulator.frontend.export_extractor import ExportExtractor
    except (ImportError, RuntimeError):
        import pytest
        pytest.skip("torch.export not available")

    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
    ext = ExportExtractor(dtype=Dtype.FP16)

    try:
        graph = ext.extract(model, (torch.randn(4, 64),), "test_mlp")
    except RuntimeError as e:
        if "torch.export" in str(e):
            import pytest
            pytest.skip("torch.export not available in this PyTorch version")
        raise

    assert isinstance(graph, ComputeGraph)
    assert graph.num_nodes > 0
    assert graph.num_edges > 0

    # Should contain at least matmul ops
    op_types = {n.op.op_type for n in graph.topo_order()}
    assert OpType.MATMUL in op_types or OpType.UNKNOWN in op_types


# ------------------------------------------------------------------ #
# ONNXExtractor
# ------------------------------------------------------------------ #

def test_onnx_extractor_from_proto():
    """ONNXExtractor should build a graph from a programmatic ONNX model."""
    try:
        import onnx
        from onnx import helper, TensorProto
        from xpu_simulator.frontend.onnx_extractor import ONNXExtractor
    except ImportError:
        import pytest
        pytest.skip("onnx package not installed")

    # Build a simple: input -> MatMul -> Relu -> output
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 64])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [64, 128])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 128])

    matmul = helper.make_node("MatMul", ["X", "W"], ["mm_out"], name="matmul_1")
    relu = helper.make_node("Relu", ["mm_out"], ["Y"], name="relu_1")

    graph_def = helper.make_graph([matmul, relu], "test", [X, W], [Y])
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 18)])
    model = onnx.shape_inference.infer_shapes(model)

    ext = ONNXExtractor(dtype=Dtype.FP16)
    cg = ext.extract(model, "test_onnx")

    assert isinstance(cg, ComputeGraph)
    assert cg.num_nodes >= 2  # matmul + relu
    assert cg.num_edges >= 1

    op_types = {n.op.op_type for n in cg.topo_order()}
    assert OpType.MATMUL in op_types
    assert OpType.RELU in op_types


def test_onnx_extractor_from_file():
    """ONNXExtractor should load from a file path."""
    try:
        import onnx
        from onnx import helper, TensorProto
        from xpu_simulator.frontend.onnx_extractor import ONNXExtractor
    except ImportError:
        import pytest
        pytest.skip("onnx package not installed")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 32])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 32])
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu_1")
    graph_def = helper.make_graph([relu], "test_file", [X], [Y])
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 18)])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        path = f.name

    try:
        ext = ONNXExtractor(dtype=Dtype.FP16)
        cg = ext.extract(path, "file_test")
        assert cg.num_nodes >= 1
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ #
# ProfilerExtractor
# ------------------------------------------------------------------ #

def _make_trace_json(events):
    """Create a minimal Chrome trace JSON."""
    return {"traceEvents": events}


def test_profiler_extractor_basic():
    """ProfilerExtractor should parse a synthetic Chrome trace."""
    events = [
        {
            "ph": "X", "cat": "cpu_op", "name": "aten::mm",
            "ts": 100, "dur": 50,
            "args": {"Input Shapes": [[4, 64], [64, 128]]}
        },
        {
            "ph": "X", "cat": "cpu_op", "name": "aten::relu",
            "ts": 200, "dur": 10,
            "args": {"Input Shapes": [[4, 128]]}
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(_make_trace_json(events), f)
        path = f.name

    try:
        ext = ProfilerExtractor(dtype=Dtype.FP16)
        graph = ext.extract(path, "profiler_test")

        assert isinstance(graph, ComputeGraph)
        assert graph.num_nodes == 2

        nodes = list(graph.topo_order())
        # Check profiled timing is stored
        assert nodes[0].op.attrs["profiled_latency_us"] == 50
        assert nodes[1].op.attrs["profiled_latency_us"] == 10

        # Check edge inference: mm output [4,128] matches relu input [4,128]
        assert graph.num_edges >= 1
    finally:
        os.unlink(path)


def test_profiler_extractor_skips_short_events():
    """Events shorter than threshold should be skipped."""
    events = [
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 0.01,
         "args": {"Input Shapes": [[2, 2], [2, 2]]}},
        {"ph": "X", "cat": "cpu_op", "name": "aten::relu", "ts": 10, "dur": 5,
         "args": {"Input Shapes": [[2, 2]]}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(_make_trace_json(events), f)
        path = f.name

    try:
        ext = ProfilerExtractor(dtype=Dtype.FP16)
        graph = ext.extract(path, "skip_test")
        assert graph.num_nodes == 1  # only relu kept
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ #
# GraphBuilder
# ------------------------------------------------------------------ #

def test_graph_builder_primitives():
    """Builder primitives create correct nodes and edges."""
    b = GraphBuilder("test", Dtype.FP16)
    n1 = b.matmul("mm1", 32, 64, 128)
    n2 = b.elementwise("relu1", (32, 128), OpType.RELU, n1)
    n3 = b.norm("ln1", (32, 128), n2)
    graph = b.build()

    assert graph.num_nodes == 3
    assert graph.num_edges == 2
    assert n1.op.op_type == OpType.MATMUL
    assert n2.op.op_type == OpType.RELU
    assert n3.op.op_type == OpType.LAYER_NORM


def test_graph_builder_swiglu_mlp():
    """SwiGLU MLP composite should create 5 nodes (w1, w3, silu, mul, w2)."""
    b = GraphBuilder("test_mlp", Dtype.FP16)
    prev = b.norm("input_norm", (1024, 4096))
    out = b.swiglu_mlp("ffn", 1024, 4096, 11008, prev)
    graph = b.build()

    assert graph.num_nodes == 6  # norm + 5 MLP nodes
    assert out.op.op_type == OpType.MATMUL
    assert out.name == "ffn.w2"


def test_graph_builder_moe_layer():
    """MoE composite should create gate + routed experts + shared expert + combine."""
    b = GraphBuilder("test_moe", Dtype.FP16)
    prev = b.norm("norm", (1024, 4096))
    out = b.moe_layer("moe", tokens=1024, dim=4096, inter_dim=1536,
                       n_experts=256, n_activated=8,
                       n_shared=1, shared_inter=1536, prev=prev)
    graph = b.build()

    # gate(1) + gate_softmax(1) + routed(5) + shared(5) + combine(1) + norm(1) = 14
    assert graph.num_nodes == 14
    assert out.op.op_type == OpType.ADD
    assert out.name == "moe.combine"


def test_graph_builder_mla_attention():
    """MLA attention composite should produce the expected structure."""
    b = GraphBuilder("test_mla", Dtype.FP16)
    prev = b.norm("norm", (1024, 7168))
    out = b.mla_attention(
        "attn", tokens=1024, B=1, S=1024, dim=7168, n_heads=128,
        q_lora_rank=1536, kv_lora_rank=512, qk_head_dim=192,
        qk_rope_head_dim=64, v_head_dim=128, prev=prev,
    )
    graph = b.build()

    # wq_a, q_norm, wq_b, wkv_a, kv_norm, wkv_b, rope_q, rope_k,
    # attn_score, attn_softmax, attn_v, wo + input norm = 13
    assert graph.num_nodes == 13
    assert out.name == "attn.wo"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

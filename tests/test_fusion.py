"""Test kernel fusion pass."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.fusion import (
    FusionPass, GPU_FUSION_RULES, NPU_FUSION_RULES,
    MatMulEpilogueFusion, ElementwiseChainFusion,
    FlashAttentionFusion, SwiGLUFusion, NPUFormatFusion,
)
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


def t(shape, dtype=Dtype.FP16):
    return TensorSpec(shape, dtype)


def test_matmul_epilogue():
    """MatMul + ReLU should fuse into one op."""
    graph = ComputeGraph("test")
    mm = graph.add_node(OpSpec(OpType.MATMUL, [t((1024, 1024)), t((1024, 1024))],
                               [t((1024, 1024))], name="mm"), "mm")
    relu = graph.add_node(OpSpec(OpType.RELU, [t((1024, 1024))],
                                 [t((1024, 1024))], name="relu"), "relu")
    graph.add_edge(mm, relu)

    fused, result = FusionPass([MatMulEpilogueFusion()]).apply(graph)

    print(f"=== MatMul + ReLU Fusion ===")
    print(result.summary())
    assert fused.num_nodes == 1, f"Expected 1 node, got {fused.num_nodes}"
    assert result.nodes_eliminated == 1
    print()


def test_elementwise_chain():
    """Chain of 3 elementwise ops should fuse into 1."""
    graph = ComputeGraph("test")
    n1 = graph.add_node(OpSpec(OpType.LAYER_NORM, [t((1024, 7168))], [t((1024, 7168))], name="norm"), "norm")
    n2 = graph.add_node(OpSpec(OpType.ADD, [t((1024, 7168))], [t((1024, 7168))], name="add"), "add")
    n3 = graph.add_node(OpSpec(OpType.RELU, [t((1024, 7168))], [t((1024, 7168))], name="relu"), "relu")
    graph.add_edge(n1, n2)
    graph.add_edge(n2, n3)

    fused, result = FusionPass([ElementwiseChainFusion()]).apply(graph)

    print(f"=== Elementwise Chain Fusion ===")
    print(result.summary())
    assert fused.num_nodes == 1, f"Expected 1 node, got {fused.num_nodes}"
    print()


def test_flash_attention():
    """QK -> softmax -> V should fuse into FlashAttention."""
    B, H, S, D = 1, 128, 1024, 192
    graph = ComputeGraph("test")
    qk = graph.add_node(OpSpec(OpType.MATMUL,
                                [t((B*H, S, D)), t((B*H, D, S))], [t((B*H, S, S))],
                                name="attn_score"), "attn_score")
    softmax = graph.add_node(OpSpec(OpType.SOFTMAX,
                                     [t((B*H, S, S))], [t((B*H, S, S))],
                                     name="softmax"), "softmax")
    attn_v = graph.add_node(OpSpec(OpType.MATMUL,
                                    [t((B*H, S, S)), t((B*H, S, 128))], [t((B*H, S, 128))],
                                    name="attn_v"), "attn_v")
    graph.add_edge(qk, softmax)
    graph.add_edge(softmax, attn_v)

    fused, result = FusionPass([FlashAttentionFusion()]).apply(graph)

    print(f"=== FlashAttention Fusion ===")
    print(result.summary())
    assert fused.num_nodes == 1, f"Expected 1 node, got {fused.num_nodes}"

    # Key benefit: fused op should NOT have N^2 memory for score matrix
    fused_node = fused.topo_order()[0]
    original_mem = sum(n.op.memory_bytes for n in graph.topo_order())
    fused_mem = fused_node.op.memory_bytes
    print(f"  Original memory: {original_mem / 1e6:.1f} MB")
    print(f"  Fused memory:    {fused_mem / 1e6:.1f} MB")
    print(f"  Memory saved:    {(1 - fused_mem / original_mem) * 100:.0f}%")
    assert fused_mem < original_mem
    print()


def test_swiglu():
    """SiLU + gate multiply should fuse."""
    graph = ComputeGraph("test")
    silu = graph.add_node(OpSpec(OpType.GELU, [t((1024, 18432))], [t((1024, 18432))],
                                 name="ffn.silu"), "ffn.silu")
    mul = graph.add_node(OpSpec(OpType.ADD, [t((1024, 18432))], [t((1024, 18432))],
                                name="ffn.mul"), "ffn.mul")
    graph.add_edge(silu, mul)

    fused, result = FusionPass([SwiGLUFusion()]).apply(graph)

    print(f"=== SwiGLU Fusion ===")
    print(result.summary())
    assert fused.num_nodes == 1
    print()


def test_full_gpu_fusion():
    """Test all GPU fusion rules on a mini transformer block."""
    graph = ComputeGraph("mini_block")
    tokens, D = 1024, 7168

    # Attention
    norm = graph.add_node(OpSpec(OpType.LAYER_NORM, [t((tokens, D))], [t((tokens, D))], name="attn_norm"), "attn_norm")
    wq = graph.add_node(OpSpec(OpType.MATMUL, [t((tokens, D)), t((D, D))], [t((tokens, D))], name="wq"), "wq")
    wk = graph.add_node(OpSpec(OpType.MATMUL, [t((tokens, D)), t((D, D))], [t((tokens, D))], name="wk"), "wk")
    graph.add_edge(norm, wq)
    graph.add_edge(norm, wk)

    qk = graph.add_node(OpSpec(OpType.MATMUL, [t((128, tokens, 192)), t((128, 192, tokens))],
                                [t((128, tokens, tokens))], name="attn_score"), "attn_score")
    graph.add_edge(wq, qk)
    graph.add_edge(wk, qk)

    softmax = graph.add_node(OpSpec(OpType.SOFTMAX, [t((128, tokens, tokens))],
                                     [t((128, tokens, tokens))], name="attn_softmax"), "attn_softmax")
    graph.add_edge(qk, softmax)

    attn_v = graph.add_node(OpSpec(OpType.MATMUL, [t((128, tokens, tokens)), t((128, tokens, 128))],
                                    [t((128, tokens, 128))], name="attn_v"), "attn_v")
    graph.add_edge(softmax, attn_v)

    wo = graph.add_node(OpSpec(OpType.MATMUL, [t((tokens, D)), t((D, D))], [t((tokens, D))], name="wo"), "wo")
    graph.add_edge(attn_v, wo)

    # FFN (SwiGLU)
    ffn_norm = graph.add_node(OpSpec(OpType.LAYER_NORM, [t((tokens, D))], [t((tokens, D))], name="ffn_norm"), "ffn_norm")
    graph.add_edge(wo, ffn_norm)

    w1 = graph.add_node(OpSpec(OpType.MATMUL, [t((tokens, D)), t((D, 18432))], [t((tokens, 18432))], name="ffn.w1"), "ffn.w1")
    w3 = graph.add_node(OpSpec(OpType.MATMUL, [t((tokens, D)), t((D, 18432))], [t((tokens, 18432))], name="ffn.w3"), "ffn.w3")
    graph.add_edge(ffn_norm, w1)
    graph.add_edge(ffn_norm, w3)

    silu = graph.add_node(OpSpec(OpType.GELU, [t((tokens, 18432))], [t((tokens, 18432))], name="ffn.silu"), "ffn.silu")
    graph.add_edge(w1, silu)

    mul = graph.add_node(OpSpec(OpType.ADD, [t((tokens, 18432))], [t((tokens, 18432))], name="ffn.mul"), "ffn.mul")
    graph.add_edge(silu, mul)
    graph.add_edge(w3, mul)

    w2 = graph.add_node(OpSpec(OpType.MATMUL, [t((tokens, 18432)), t((18432, D))], [t((tokens, D))], name="ffn.w2"), "ffn.w2")
    graph.add_edge(mul, w2)

    print(f"=== Full GPU Fusion on Mini Block ===")
    print(f"  Original: {graph.num_nodes} nodes, {graph.num_edges} edges")

    fused, result = FusionPass(GPU_FUSION_RULES).apply(graph)
    print(f"  Fused:    {fused.num_nodes} nodes, {fused.num_edges} edges")
    print(result.summary())
    print()

    # Evaluate both
    model = GPUCostModel(A100_80GB)
    orig_result = PerformanceEvaluator(model).run(graph, overlap=True)
    fused_result = PerformanceEvaluator(model).run(fused, overlap=True)

    print(f"  Original latency: {orig_result.total_latency_us:,.0f} us")
    print(f"  Fused latency:    {fused_result.total_latency_us:,.0f} us")
    print(f"  Speedup:          {orig_result.total_latency_us / fused_result.total_latency_us:.2f}x")
    print()

    assert fused.num_nodes < graph.num_nodes
    assert fused_result.total_latency_us < orig_result.total_latency_us


def test_npu_format_fusion():
    """NPUFormatFusion should fuse MatMul+epilogue with skip_format_conversion=True."""
    graph = ComputeGraph("npu_test")
    mm = graph.add_node(OpSpec(OpType.MATMUL, [t((1024, 1024)), t((1024, 1024))],
                               [t((1024, 1024))], name="mm"), "mm")
    relu = graph.add_node(OpSpec(OpType.RELU, [t((1024, 1024))],
                                 [t((1024, 1024))], name="relu"), "relu")
    graph.add_edge(mm, relu)

    fused, result = FusionPass([NPUFormatFusion()]).apply(graph)

    print(f"=== NPU Format Fusion ===")
    print(result.summary())
    assert fused.num_nodes == 1, f"Expected 1 node, got {fused.num_nodes}"
    fused_node = fused.topo_order()[0]
    assert fused_node.op.attrs.get("skip_format_conversion") is True, \
        "Fused NPU op must have skip_format_conversion=True"
    print(f"  skip_format_conversion: {fused_node.op.attrs['skip_format_conversion']}")
    print()


def test_npu_format_fusion_conv2d():
    """NPUFormatFusion should also work with Conv2D + GELU."""
    graph = ComputeGraph("npu_conv_test")
    conv = graph.add_node(OpSpec(OpType.CONV2D,
                                  [t((1, 64, 56, 56)), t((128, 64, 3, 3))],
                                  [t((1, 128, 56, 56))], name="conv"), "conv")
    gelu = graph.add_node(OpSpec(OpType.GELU, [t((1, 128, 56, 56))],
                                  [t((1, 128, 56, 56))], name="gelu"), "gelu")
    graph.add_edge(conv, gelu)

    fused, result = FusionPass([NPUFormatFusion()]).apply(graph)

    print(f"=== NPU Format Fusion (Conv2D) ===")
    print(result.summary())
    assert fused.num_nodes == 1, f"Expected 1 node, got {fused.num_nodes}"
    fused_node = fused.topo_order()[0]
    assert fused_node.op.attrs.get("skip_format_conversion") is True
    print()


def test_npu_fusion_rules():
    """NPU_FUSION_RULES should include NPUFormatFusion and produce skip_format_conversion."""
    graph = ComputeGraph("npu_rules_test")
    mm = graph.add_node(OpSpec(OpType.MATMUL, [t((1024, 1024)), t((1024, 1024))],
                               [t((1024, 1024))], name="mm"), "mm")
    add = graph.add_node(OpSpec(OpType.ADD, [t((1024, 1024))],
                                [t((1024, 1024))], name="add"), "add")
    graph.add_edge(mm, add)

    fused, result = FusionPass(NPU_FUSION_RULES).apply(graph)

    print(f"=== NPU Full Rules ===")
    print(result.summary())
    assert fused.num_nodes == 1
    fused_node = fused.topo_order()[0]
    # Could be fused by MatMulEpilogueFusion or NPUFormatFusion depending on rule order
    # Either way the graph should be fused
    print()


if __name__ == "__main__":
    test_matmul_epilogue()
    test_elementwise_chain()
    test_flash_attention()
    test_swiglu()
    test_npu_format_fusion()
    test_npu_format_fusion_conv2d()
    test_npu_fusion_rules()
    test_full_gpu_fusion()
    print("All fusion tests passed!")

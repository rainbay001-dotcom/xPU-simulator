"""Tests for decode phase and KV cache modeling."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import Dtype, Phase, OpType
from xpu_simulator.core.kv_cache import kv_cache_bytes, kv_cache_per_token_bytes
from xpu_simulator.core.evaluator import PerformanceEvaluator, SimResult
from xpu_simulator.frontend.graph_builder import GraphBuilder
from xpu_simulator.frontend.config_normalizer import AttentionPattern
from xpu_simulator.frontend.config_extractor import ConfigExtractor
from xpu_simulator.backends.gpu.hardware import H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


LLAMA_CONFIG = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "intermediate_size": 11008,
    "vocab_size": 32000,
}


# ------------------------------------------------------------------ #
# KV cache helper tests
# ------------------------------------------------------------------ #

def test_kv_cache_bytes():
    """Verify KV cache bytes formula."""
    # 2 (K+V) * batch=1 * seq=1024 * layers=32 * kv_heads=8 * head_dim=128 * fp16(2)
    result = kv_cache_bytes(1, 1024, 32, 8, 128, Dtype.FP16)
    expected = 2 * 1 * 1024 * 32 * 8 * 128 * 2
    assert result == expected, f"Got {result}, expected {expected}"


def test_kv_cache_per_token():
    """KV cache per token bytes."""
    result = kv_cache_per_token_bytes(32, 8, 128, Dtype.FP16)
    expected = 2 * 32 * 8 * 128 * 2
    assert result == expected


# ------------------------------------------------------------------ #
# GraphBuilder decode tests
# ------------------------------------------------------------------ #

def test_decode_graph_shapes():
    """Decode graph should have [B, 1, D]-shaped matmuls."""
    gb = GraphBuilder("decode_test", phase=Phase.DECODE, kv_seq_len=1024)
    prev = gb.norm("norm", (1, 4096))  # tokens = B*1 = 1
    gb.swiglu_mlp("ffn", 1, 4096, 11008, prev)
    g = gb.build()

    # All matmuls should have M=1 (single token)
    for node in g.nodes:
        if node.op.op_type == OpType.MATMUL:
            M = node.op.inputs[0].shape[0]
            assert M == 1, f"{node.name}: M={M}, expected 1"


def test_decode_attention_kv_read():
    """Decode attention should use kv_seq_len for K/V dimension."""
    kv_len = 2048
    gb = GraphBuilder("decode_test", phase=Phase.DECODE, kv_seq_len=kv_len)
    prev = gb.norm("norm", (1, 4096))
    gb.attention("attn", 1, 1, 1, 4096, n_heads=32, head_dim=128,
                 pattern=AttentionPattern(), n_kv_heads=8, prev=prev)
    g = gb.build()

    # attn_score should be [B*H, 1, kv_len]
    score_nodes = [n for n in g.nodes if "attn_score" in (n.name or "")]
    assert len(score_nodes) == 1
    score = score_nodes[0]
    out_shape = score.op.outputs[0].shape
    assert out_shape[-1] == kv_len, f"Expected kv_len={kv_len}, got {out_shape}"
    assert out_shape[-2] == 1, f"Expected q_seq=1, got {out_shape[-2]}"
    print(f"  attn_score shape: {out_shape}")


def test_decode_memory_bound():
    """Decode phase should be mostly memory-bound (small compute, large KV read)."""
    model = GPUCostModel(H100_80GB)
    evaluator = PerformanceEvaluator(model)

    gb = GraphBuilder("decode", phase=Phase.DECODE, kv_seq_len=4096)
    prev = gb.norm("norm", (1, 4096))
    gb.attention("attn", 1, 1, 1, 4096, n_heads=32, head_dim=128,
                 pattern=AttentionPattern(), n_kv_heads=8, prev=prev)
    g = gb.build()

    result = evaluator.run(g)
    # Most ops should be memory-bound in decode
    assert result.memory_bound_count >= result.compute_bound_count
    print(f"  Decode: {result.memory_bound_count} mem-bound, "
          f"{result.compute_bound_count} compute-bound")


def test_prefill_vs_decode_latency():
    """Prefill latency should be much larger than decode for same model."""
    model = GPUCostModel(H100_80GB)
    evaluator = PerformanceEvaluator(model)
    ext = ConfigExtractor()

    graph_prefill = ext.extract(LLAMA_CONFIG, batch_size=1, seq_len=1024)
    graph_decode = ext.extract(LLAMA_CONFIG, batch_size=1, seq_len=1024,
                               phase="decode", kv_seq_len=1024)

    result_prefill = evaluator.run(graph_prefill, overlap=True)
    result_decode = evaluator.run(graph_decode, overlap=True)

    print(f"  Prefill: {result_prefill.total_latency_us:.1f} us "
          f"({len(graph_prefill.nodes)} ops)")
    print(f"  Decode:  {result_decode.total_latency_us:.1f} us "
          f"({len(graph_decode.nodes)} ops)")

    # Prefill should be >> decode (quadratic attention vs linear)
    # With only 2 layers the lm_head dominates both, so ratio is ~4x not 10x+
    assert result_prefill.total_latency_us > result_decode.total_latency_us * 3


def test_ttft_tpot_metrics():
    """SimResult should report TTFT and TPOT correctly."""
    result = SimResult(
        total_latency_us=5000.0,
        per_op=[],
        bottleneck_op=None,
        phase="prefill",
    )
    assert result.ttft_ms == 5.0
    assert result.tpot_ms is None

    result_decode = SimResult(
        total_latency_us=100.0,
        per_op=[],
        bottleneck_op=None,
        phase="decode",
    )
    assert result_decode.tpot_ms == 0.1
    assert result_decode.ttft_ms is None


def test_config_extractor_decode():
    """ConfigExtractor with phase='decode' should produce valid graph."""
    ext = ConfigExtractor()
    graph = ext.extract(LLAMA_CONFIG, batch_size=4, seq_len=2048,
                        phase="decode", kv_seq_len=2048)

    # Should have nodes
    assert len(graph.nodes) > 0

    # Matmuls should have M = 4 (batch=4, seq=1, so tokens=4)
    matmul_nodes = [n for n in graph.nodes if n.op.op_type == OpType.MATMUL]
    for mm in matmul_nodes:
        if "attn_score" in (mm.name or "") or "attn_v" in (mm.name or ""):
            continue  # attention score/v have different shapes
        M = mm.op.inputs[0].shape[0]
        assert M == 4, f"{mm.name}: M={M}, expected 4 (batch*1)"

    print(f"  Decode graph: {len(graph.nodes)} nodes")


if __name__ == "__main__":
    print("=== Decode Tests ===\n")

    for name, fn in [
        ("KV cache bytes", test_kv_cache_bytes),
        ("KV cache per token", test_kv_cache_per_token),
        ("Decode graph shapes", test_decode_graph_shapes),
        ("Decode attention KV read", test_decode_attention_kv_read),
        ("Decode memory-bound", test_decode_memory_bound),
        ("Prefill vs decode latency", test_prefill_vs_decode_latency),
        ("TTFT/TPOT metrics", test_ttft_tpot_metrics),
        ("ConfigExtractor decode", test_config_extractor_decode),
    ]:
        print(f"--- {name} ---")
        fn()
        print()

    print("All decode tests passed!")

"""Tests for ConfigExtractor, config normalizer, and new GraphBuilder composites."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, ".")

from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.operator import OpType, Dtype
from xpu_simulator.frontend.graph_builder import GraphBuilder
from xpu_simulator.frontend.config_normalizer import ModelConfig, normalize_config
from xpu_simulator.frontend.config_extractor import (
    ConfigExtractor, ArchitectureHandler, StandardTransformerHandler,
)


# ------------------------------------------------------------------ #
# New GraphBuilder composites
# ------------------------------------------------------------------ #

def test_gqa_attention_mha():
    """GQA attention with n_kv_heads == n_heads is MHA."""
    b = GraphBuilder("test", Dtype.FP16)
    prev = b.norm("norm", (1024, 4096))
    out = b.gqa_attention("attn", tokens=1024, B=1, S=1024,
                          dim=4096, n_heads=32, n_kv_heads=32,
                          head_dim=128, rope=True, prev=prev)
    graph = b.build()
    # wq, wk, wv, rope_q, rope_k, attn_score, softmax, attn_v, wo + norm = 10
    assert graph.num_nodes == 10
    assert out.name == "attn.wo"


def test_gqa_attention_gqa():
    """GQA attention with n_kv_heads < n_heads."""
    b = GraphBuilder("test", Dtype.FP16)
    prev = b.norm("norm", (1024, 4096))
    out = b.gqa_attention("attn", tokens=1024, B=1, S=1024,
                          dim=4096, n_heads=32, n_kv_heads=8,
                          head_dim=128, rope=True, prev=prev)
    graph = b.build()
    assert graph.num_nodes == 10
    assert out.name == "attn.wo"


def test_gqa_attention_no_rope():
    """GQA without RoPE (GPT-2 style)."""
    b = GraphBuilder("test", Dtype.FP16)
    prev = b.norm("norm", (512, 768))
    out = b.gqa_attention("attn", tokens=512, B=1, S=512,
                          dim=768, n_heads=12, n_kv_heads=12,
                          head_dim=64, rope=False, prev=prev)
    graph = b.build()
    # wq, wk, wv, attn_score, softmax, attn_v, wo + norm = 8 (no rope nodes)
    assert graph.num_nodes == 8


def test_dense_ffn():
    """Dense 2-matrix FFN."""
    b = GraphBuilder("test", Dtype.FP16)
    prev = b.norm("norm", (512, 768))
    out = b.dense_ffn("ffn", tokens=512, dim=768, inter_dim=3072,
                      activation=OpType.GELU, prev=prev)
    graph = b.build()
    # norm + w1 + act + w2 = 4
    assert graph.num_nodes == 4
    assert out.name == "ffn.w2"


def test_geglu_mlp():
    """GeGLU MLP has same structure as SwiGLU but with GELU."""
    b = GraphBuilder("test", Dtype.FP16)
    prev = b.norm("norm", (512, 768))
    out = b.geglu_mlp("ffn", tokens=512, dim=768, inter_dim=3072, prev=prev)
    graph = b.build()
    # norm + w1, w3, gelu, mul, w2 = 6
    assert graph.num_nodes == 6
    assert out.name == "ffn.w2"


def test_moe_layer_no_shared():
    """MoE layer without shared experts (Mixtral-style)."""
    b = GraphBuilder("test", Dtype.FP16)
    prev = b.norm("norm", (1024, 4096))
    out = b.moe_layer("moe", tokens=1024, dim=4096, inter_dim=14336,
                       n_experts=8, n_activated=2, n_shared=0, prev=prev)
    graph = b.build()
    # norm + gate, gate_softmax, experts_w1, w3, silu, mul, w2 = 8
    assert graph.num_nodes == 8
    # Returns expert_w2 directly (no combine node)
    assert out.name == "moe.experts_w2"


def test_moe_layer_with_shared():
    """MoE layer with shared experts (DeepSeek-style)."""
    b = GraphBuilder("test", Dtype.FP16)
    prev = b.norm("norm", (1024, 4096))
    out = b.moe_layer("moe", tokens=1024, dim=4096, inter_dim=2048,
                       n_experts=256, n_activated=8,
                       n_shared=1, shared_inter=2048, prev=prev)
    graph = b.build()
    # norm + gate(1) + softmax(1) + routed(5) + shared(5) + combine(1) = 14
    assert graph.num_nodes == 14
    assert out.name == "moe.combine"


# ------------------------------------------------------------------ #
# Config normalizer
# ------------------------------------------------------------------ #

def test_normalize_llama():
    """LLaMA-3-8B config normalization."""
    raw = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "hidden_act": "silu",
    }
    cfg = normalize_config(raw)
    assert cfg.model_type == "llama"
    assert cfg.hidden_size == 4096
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.rope is True
    assert cfg.rms_norm is True
    assert cfg.is_moe is False
    assert cfg.is_mla is False


def test_normalize_gpt2():
    """GPT-2 uses non-standard field names."""
    raw = {
        "model_type": "gpt2",
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "vocab_size": 50257,
        "activation_function": "gelu",
    }
    cfg = normalize_config(raw)
    assert cfg.hidden_size == 768
    assert cfg.num_attention_heads == 12
    assert cfg.num_key_value_heads == 12  # MHA
    assert cfg.intermediate_size == 3072  # 4 * 768
    assert cfg.hidden_act == "gelu"
    assert cfg.rope is False
    assert cfg.rms_norm is False


def test_normalize_falcon_mqq():
    """Falcon-7B with multi_query=True."""
    raw = {
        "model_type": "falcon",
        "hidden_size": 4544,
        "num_attention_heads": 71,
        "multi_query": True,
        "num_hidden_layers": 32,
        "vocab_size": 65024,
        "parallel_attn": True,
    }
    cfg = normalize_config(raw)
    assert cfg.num_key_value_heads == 1  # MQA
    assert cfg.intermediate_size == 4 * 4544
    assert cfg.parallel_attn is True
    assert cfg.rms_norm is False


def test_normalize_mixtral():
    """Mixtral-8x7B MoE config."""
    raw = {
        "model_type": "mixtral",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "hidden_act": "silu",
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    cfg = normalize_config(raw)
    assert cfg.is_moe is True
    assert cfg.num_experts == 8
    assert cfg.num_experts_per_tok == 2
    assert cfg.moe_intermediate_size == 14336  # same as intermediate_size
    assert cfg.num_shared_experts is None


def test_normalize_deepseek():
    """DeepSeek-V3 config with MLA and MoE."""
    raw = {
        "model_type": "deepseek_v2",
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "vocab_size": 129280,
        "hidden_act": "silu",
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }
    cfg = normalize_config(raw)
    assert cfg.is_moe is True
    assert cfg.is_mla is True
    assert cfg.num_experts == 256
    assert cfg.first_k_dense_replace == 3
    assert cfg.q_lora_rank == 1536
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_head_dim == 192  # 128 + 64
    assert cfg.shared_expert_intermediate_size == 2048  # 1 * 2048


# ------------------------------------------------------------------ #
# ConfigExtractor end-to-end
# ------------------------------------------------------------------ #

LLAMA_CONFIG = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_hidden_layers": 2,
    "vocab_size": 128256,
    "hidden_act": "silu",
}

MIXTRAL_CONFIG = {
    "model_type": "mixtral",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_hidden_layers": 2,
    "vocab_size": 32000,
    "hidden_act": "silu",
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}

GPT2_CONFIG = {
    "model_type": "gpt2",
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 2,
    "vocab_size": 50257,
    "activation_function": "gelu",
}

DEEPSEEK_CONFIG = {
    "model_type": "deepseek_v2",
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "intermediate_size": 18432,
    "num_hidden_layers": 4,  # 3 dense + 1 MoE for test
    "vocab_size": 129280,
    "hidden_act": "silu",
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,
    "first_k_dense_replace": 3,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
}


def test_config_extractor_llama():
    """LLaMA config produces a valid graph."""
    ext = ConfigExtractor(dtype=Dtype.FP16)
    graph = ext.extract(LLAMA_CONFIG, batch_size=1, seq_len=128)

    assert isinstance(graph, ComputeGraph)
    assert graph.num_nodes > 0
    assert graph.num_edges > 0

    # Per layer: attn_norm, wq, wk, wv, rope_q, rope_k, attn_score, softmax, attn_v, wo,
    #            ffn_norm, w1, w3, silu, mul, w2 = 16 nodes
    # + embedding(1) + final_norm(1) + lm_head(1) = 3
    # 2 layers * 16 + 3 = 35
    assert graph.num_nodes == 35


def test_config_extractor_gpt2():
    """GPT-2 config (no RoPE, dense GELU FFN)."""
    ext = ConfigExtractor(dtype=Dtype.FP16)
    graph = ext.extract(GPT2_CONFIG, batch_size=1, seq_len=128)

    assert isinstance(graph, ComputeGraph)
    assert graph.num_nodes > 0

    # Per layer: attn_norm, wq, wk, wv, attn_score, softmax, attn_v, wo,
    #            ffn_norm, w1, act, w2 = 12 nodes (no rope)
    # + embedding(1) + final_norm(1) + lm_head(1) = 3
    # 2 * 12 + 3 = 27
    assert graph.num_nodes == 27


def test_config_extractor_mixtral():
    """Mixtral MoE config produces MoE layers."""
    ext = ConfigExtractor(dtype=Dtype.FP16)
    graph = ext.extract(MIXTRAL_CONFIG, batch_size=1, seq_len=128)

    assert isinstance(graph, ComputeGraph)
    assert graph.num_nodes > 0

    # Per layer: attn(10) + ffn_norm(1) + moe_no_shared(7) = 18
    # + embedding(1) + final_norm(1) + lm_head(1) = 3
    # 2 * 18 + 3 = 39
    assert graph.num_nodes == 39


def test_config_extractor_deepseek():
    """DeepSeek config with mixed dense/MoE layers."""
    ext = ConfigExtractor(dtype=Dtype.FP16)
    graph = ext.extract(DEEPSEEK_CONFIG, batch_size=1, seq_len=128)

    assert isinstance(graph, ComputeGraph)
    assert graph.num_nodes > 0
    assert graph.num_edges > 0

    # 3 dense layers + 1 MoE layer
    # Dense MLA layer: attn_norm(1) + MLA(12) + ffn_norm(1) + SwiGLU(5) = 19
    # MoE MLA layer: attn_norm(1) + MLA(12) + ffn_norm(1) + MoE_shared(13) = 27
    # + embedding(1) + final_norm(1) + lm_head(1) = 3
    # 3 * 19 + 1 * 27 + 3 = 87
    assert graph.num_nodes == 87


def test_config_extractor_from_file():
    """ConfigExtractor loads from a JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(LLAMA_CONFIG, f)
        path = f.name

    try:
        ext = ConfigExtractor(dtype=Dtype.FP16)
        graph = ext.extract(path, batch_size=1, seq_len=128)
        assert isinstance(graph, ComputeGraph)
        assert graph.num_nodes == 35
    finally:
        os.unlink(path)


def test_config_extractor_unsupported_type():
    """Unsupported model_type raises ValueError."""
    import pytest
    ext = ConfigExtractor(dtype=Dtype.FP16)
    with pytest.raises(ValueError, match="Unsupported model_type"):
        ext.extract({"model_type": "unknown", "hidden_size": 64,
                     "num_attention_heads": 4, "num_hidden_layers": 1,
                     "vocab_size": 100}, batch_size=1, seq_len=8)


def test_config_extractor_custom_handler():
    """Custom handler can be registered and used."""

    class TinyHandler(ArchitectureHandler):
        def build_layer(self, builder, cfg, layer_id, tokens, B, S, prev):
            return builder.norm(f"L{layer_id}.only", (tokens, cfg.hidden_size), prev)

    ConfigExtractor.register_handler("test_custom", TinyHandler)
    try:
        ext = ConfigExtractor(dtype=Dtype.FP16)
        graph = ext.extract({"model_type": "test_custom", "hidden_size": 64,
                             "num_attention_heads": 4, "num_hidden_layers": 2,
                             "vocab_size": 100}, batch_size=1, seq_len=8)
        # embedding + 2 norms + final_norm + lm_head = 5
        assert graph.num_nodes == 5
    finally:
        del ConfigExtractor._handlers["test_custom"]


def test_config_extractor_supported_architectures():
    """supported_architectures returns registered model types."""
    archs = ConfigExtractor.supported_architectures()
    assert "llama" in archs
    assert "deepseek_v2" in archs
    assert "mixtral" in archs
    assert "gpt2" in archs


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

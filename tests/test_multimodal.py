"""Tests for multi-modal VLM simulation support."""
from __future__ import annotations

import math
import sys

sys.path.insert(0, ".")

from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.operator import (
    OpType, Dtype, ImageInput, MultiModalInput, VideoInput, AudioInput,
)
from xpu_simulator.frontend.graph_builder import GraphBuilder
from xpu_simulator.frontend.config_normalizer import (
    ModelConfig, VisionConfig, normalize_config,
)
from xpu_simulator.frontend.config_extractor import (
    ConfigExtractor, VLMHandler, LLaVAHandler, InternVLHandler,
)
from xpu_simulator.utils.categories import categorize_op


# ------------------------------------------------------------------ #
# VisionConfig
# ------------------------------------------------------------------ #

def test_vision_config_patches_per_image():
    vc = VisionConfig(image_size=336, patch_size=14)
    assert vc.patches_per_image == (336 // 14) ** 2  # 576


def test_vision_config_tokens_per_tile_no_merge():
    vc = VisionConfig(image_size=336, patch_size=14, spatial_merge_size=1)
    assert vc.tokens_per_tile == 576


def test_vision_config_tokens_per_tile_with_merge():
    vc = VisionConfig(image_size=448, patch_size=14, spatial_merge_size=2)
    patches = (448 // 14) ** 2  # 1024
    assert vc.tokens_per_tile == 1024 // 4  # 256


def test_vision_config_num_vision_tokens_fixed():
    vc = VisionConfig(image_size=336, patch_size=14, resolution_strategy="fixed")
    assert vc.num_vision_tokens(num_tiles=1) == 576


def test_vision_config_num_vision_tokens_with_thumbnail():
    vc = VisionConfig(image_size=448, patch_size=14, spatial_merge_size=2,
                      use_thumbnail=True, resolution_strategy="dynamic_tile")
    # 1 tile + 1 thumbnail = 2 tiles, 256 tokens each
    assert vc.num_vision_tokens(num_tiles=1) == 2 * 256


def test_vision_config_num_vision_tokens_native():
    vc = VisionConfig(image_size=336, patch_size=14, spatial_merge_size=2,
                      resolution_strategy="native")
    # 672x672 → ceil(672/14)=48 → 48*48=2304 / 4 = 576
    assert vc.num_vision_tokens(img_w=672, img_h=672) == 576


def test_vision_config_encoder_layers_default():
    vc = VisionConfig(num_hidden_layers=24, vision_feature_layer=-1)
    assert vc.num_encoder_layers() == 24


def test_vision_config_encoder_layers_skip_last():
    vc = VisionConfig(num_hidden_layers=24, vision_feature_layer=-2)
    assert vc.num_encoder_layers() == 23


# ------------------------------------------------------------------ #
# Config normalization for VLMs
# ------------------------------------------------------------------ #

LLAVA_NEXT_CONFIG = {
    "model_type": "llava_next",
    "text_config": {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "num_hidden_layers": 2,
        "vocab_size": 32000,
        "hidden_act": "silu",
    },
    "vision_config": {
        "model_type": "clip_vision_model",
        "image_size": 336,
        "patch_size": 14,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_hidden_layers": 2,
    "vocab_size": 32000,
    "hidden_act": "silu",
    "image_grid_pinpoints": [[336, 672], [672, 336], [672, 672]],
    "projector_hidden_act": "gelu",
    "vision_feature_layer": -2,
}

INTERNVL_CONFIG = {
    "model_type": "internvl_chat",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_hidden_layers": 2,
    "vocab_size": 32000,
    "hidden_act": "silu",
    "vision_config": {
        "model_type": "intern_vit_6b",
        "image_size": 448,
        "patch_size": 14,
        "hidden_size": 3200,
        "num_hidden_layers": 45,
        "num_attention_heads": 25,
        "intermediate_size": 12800,
    },
    "downsample_ratio": 0.5,
    "use_thumbnail": True,
    "max_dynamic_patch": 12,
}

QWEN25_VL_CONFIG = {
    "model_type": "qwen2_5_vl",
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "intermediate_size": 18944,
    "num_hidden_layers": 2,
    "vocab_size": 152064,
    "hidden_act": "silu",
    "vision_config": {
        "depth": 32,
        "hidden_size": 1280,
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "in_chans": 3,
        "hidden_act": "silu",
    },
}

DEEPSEEK_VL2_CONFIG = {
    "model_type": "deepseek_vl_v2",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_hidden_layers": 2,
    "vocab_size": 32000,
    "hidden_act": "silu",
    "vision_config": {
        "model_type": "siglip_vision_model",
        "image_size": 384,
        "patch_size": 14,
        "hidden_size": 1152,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "intermediate_size": 4304,
    },
    "projector_config": {
        "model_type": "mlp_projector",
        "n_embed": 2560,
    },
    "candidate_resolutions": [[384, 384], [384, 768], [768, 384]],
}


def test_normalize_llava_next():
    cfg = normalize_config(LLAVA_NEXT_CONFIG)
    assert cfg.is_multimodal
    vc = cfg.vision_config
    assert vc.encoder_type == "clip"
    assert vc.image_size == 336
    assert vc.patch_size == 14
    assert vc.hidden_size == 1024
    assert vc.num_hidden_layers == 24
    assert vc.resolution_strategy == "anyres"
    assert vc.vision_feature_layer == -2
    assert vc.num_encoder_layers() == 23


def test_normalize_internvl():
    cfg = normalize_config(INTERNVL_CONFIG)
    assert cfg.is_multimodal
    vc = cfg.vision_config
    assert vc.encoder_type == "internvit"
    assert vc.image_size == 448
    assert vc.spatial_merge_size == 2  # from downsample_ratio=0.5
    assert vc.projector_type == "pixel_shuffle_mlp"
    assert vc.use_thumbnail is True
    assert vc.resolution_strategy == "dynamic_tile"
    assert vc.max_tiles == 12


def test_normalize_qwen25_vl():
    cfg = normalize_config(QWEN25_VL_CONFIG)
    assert cfg.is_multimodal
    vc = cfg.vision_config
    assert vc.hidden_size == 1280
    assert vc.num_hidden_layers == 32
    assert vc.spatial_merge_size == 2
    assert vc.projector_type == "pixel_shuffle_mlp"
    assert vc.num_channels == 3


def test_normalize_deepseek_vl2():
    cfg = normalize_config(DEEPSEEK_VL2_CONFIG)
    assert cfg.is_multimodal
    vc = cfg.vision_config
    assert vc.encoder_type == "siglip"
    assert vc.image_size == 384
    assert vc.num_hidden_layers == 27
    assert vc.projector_hidden_size == 2560
    assert vc.resolution_strategy == "dynamic_tile"


def test_normalize_text_only_no_vision():
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
    assert not cfg.is_multimodal
    assert cfg.vision_config is None


# ------------------------------------------------------------------ #
# GraphBuilder vision composites
# ------------------------------------------------------------------ #

def test_vision_encoder_node_count():
    """ViT encoder produces correct number of nodes."""
    vcfg = VisionConfig(
        image_size=336, patch_size=14, hidden_size=1024,
        num_hidden_layers=2, num_attention_heads=16,
        intermediate_size=4096, hidden_act="gelu",
    )
    b = GraphBuilder("test", Dtype.FP16)
    out = b.vision_encoder("vision.img0", vcfg, num_patches=576, B=1)
    graph = b.build()

    # patch_embed(1) + per layer: norm1(1) + attn(wq,wk,wv,score,softmax,attn_v,wo=7) + norm2(1) + ffn(w1,act,w2=3)
    # = 1 + 2 * (1 + 7 + 1 + 3) = 1 + 24 = 25
    # Note: gqa_attention without rope = 7 nodes (wq,wk,wv,score,softmax,attn_v,wo)
    assert graph.num_nodes == 25
    assert out.name == "vision.img0.layer1.ffn.w2"


def test_vision_encoder_with_feature_layer():
    """Respects vision_feature_layer=-2 (skip last layer)."""
    vcfg = VisionConfig(
        image_size=336, patch_size=14, hidden_size=1024,
        num_hidden_layers=4, num_attention_heads=16,
        intermediate_size=4096, vision_feature_layer=-2,
    )
    b = GraphBuilder("test", Dtype.FP16)
    b.vision_encoder("v", vcfg, num_patches=576, B=1)
    graph = b.build()

    # 3 layers (not 4) + patch_embed
    # 1 + 3 * 12 = 37
    assert graph.num_nodes == 1 + 3 * 12


def test_vision_projector_linear():
    vcfg = VisionConfig(projector_type="linear", hidden_size=1024)
    b = GraphBuilder("test", Dtype.FP16)
    out = b.vision_projector("vision.img0", vcfg, num_tokens=576,
                              llm_hidden=4096, B=1)
    graph = b.build()
    assert graph.num_nodes == 1
    assert out.name == "vision.img0.proj"


def test_vision_projector_mlp():
    vcfg = VisionConfig(projector_type="mlp", hidden_size=1024)
    b = GraphBuilder("test", Dtype.FP16)
    out = b.vision_projector("vision.img0", vcfg, num_tokens=576,
                              llm_hidden=4096, B=1)
    graph = b.build()
    # fc1 + act + fc2 = 3
    assert graph.num_nodes == 3
    assert out.name == "vision.img0.proj_fc2"


def test_vision_projector_pixel_shuffle():
    vcfg = VisionConfig(projector_type="pixel_shuffle_mlp",
                        hidden_size=1024, spatial_merge_size=2)
    b = GraphBuilder("test", Dtype.FP16)
    out = b.vision_projector("vision.img0", vcfg, num_tokens=256,
                              llm_hidden=4096, B=1)
    graph = b.build()
    # fc1 (input dim = 1024*4=4096) + act + fc2 = 3
    assert graph.num_nodes == 3

    # Verify input dim of first matmul is hidden_size * merge^2
    first_node = list(graph.topo_order())[0]
    assert first_node.op.inputs[0].shape == (256, 4096)  # 1024 * 2^2


def test_vision_projector_qformer():
    vcfg = VisionConfig(projector_type="qformer", hidden_size=1024,
                        num_query_tokens=32)
    b = GraphBuilder("test", Dtype.FP16)
    out = b.vision_projector("vision.img0", vcfg, num_tokens=576,
                              llm_hidden=4096, B=1)
    graph = b.build()
    assert graph.num_nodes == 1
    # Output should use query token count, not input patches
    assert out.op.inputs[0].shape[0] == 32


# ------------------------------------------------------------------ #
# Cross-attention
# ------------------------------------------------------------------ #

def test_cross_attention():
    b = GraphBuilder("test", Dtype.FP16)
    text = b.norm("text_norm", (128, 4096))
    vision = b.norm("vision_norm", (576, 4096))
    out = b.cross_attention("L0", tokens_q=128, tokens_kv=576,
                            dim=4096, n_heads=32, head_dim=128,
                            B=1, prev_q=text, prev_kv=vision)
    graph = b.build()
    # 2 norms + wq, wk, wv, score, softmax, attn_v, wo = 9
    assert graph.num_nodes == 9
    assert out.name == "L0.xattn_wo"


def test_cross_attention_flops():
    """Cross-attention FLOPs reflect T_q × T_kv, not T_q × T_q."""
    b = GraphBuilder("test", Dtype.FP16)
    text = b.norm("t", (128, 4096))
    vis = b.norm("v", (576, 4096))
    b.cross_attention("x", tokens_q=128, tokens_kv=576,
                      dim=4096, n_heads=32, head_dim=128,
                      B=1, prev_q=text, prev_kv=vis)
    graph = b.build()

    score_node = [n for n in graph.nodes if "xattn_score" in (n.name or "")][0]
    # QK^T: [32, 128, 128] × [32, 128, 576] → 2*32*128*128*576
    expected = 2 * 32 * 128 * 128 * 576
    assert score_node.op.flops == expected


# ------------------------------------------------------------------ #
# ConfigExtractor end-to-end with mm_input
# ------------------------------------------------------------------ #

def test_config_extractor_llava_with_image():
    ext = ConfigExtractor(dtype=Dtype.FP16)
    mm = MultiModalInput(
        num_text_tokens=128,
        images=[ImageInput(num_tiles=1)],
    )
    graph = ext.extract(LLAVA_NEXT_CONFIG, batch_size=1, seq_len=128,
                        mm_input=mm)

    assert isinstance(graph, ComputeGraph)
    assert graph.num_nodes > 0

    # Should have vision encoder nodes
    vision_nodes = [n for n in graph.nodes if (n.name or "").startswith("vision.")]
    assert len(vision_nodes) > 0

    # And LLM nodes
    llm_nodes = [n for n in graph.nodes if (n.name or "").startswith("L")]
    assert len(llm_nodes) > 0


def test_config_extractor_llava_no_image_same_as_text():
    """Without mm_input, VLM config still works (text-only path)."""
    ext = ConfigExtractor(dtype=Dtype.FP16)
    graph = ext.extract(LLAVA_NEXT_CONFIG, batch_size=1, seq_len=128)

    assert isinstance(graph, ComputeGraph)
    vision_nodes = [n for n in graph.nodes if (n.name or "").startswith("vision.")]
    assert len(vision_nodes) == 0


def test_config_extractor_internvl_with_image():
    ext = ConfigExtractor(dtype=Dtype.FP16)
    mm = MultiModalInput(images=[ImageInput(num_tiles=4)])
    graph = ext.extract(INTERNVL_CONFIG, batch_size=1, seq_len=64,
                        mm_input=mm)

    assert isinstance(graph, ComputeGraph)
    vision_nodes = [n for n in graph.nodes if (n.name or "").startswith("vision.")]
    assert len(vision_nodes) > 0


def test_config_extractor_multi_image():
    """Multiple images produce multiple vision encoder subgraphs."""
    ext = ConfigExtractor(dtype=Dtype.FP16)
    mm = MultiModalInput(
        images=[ImageInput(num_tiles=1), ImageInput(num_tiles=2)],
    )
    graph = ext.extract(LLAVA_NEXT_CONFIG, batch_size=1, seq_len=64,
                        mm_input=mm)

    img0_nodes = [n for n in graph.nodes if (n.name or "").startswith("vision.img0")]
    img1_nodes = [n for n in graph.nodes if (n.name or "").startswith("vision.img1")]
    assert len(img0_nodes) > 0
    assert len(img1_nodes) > 0


def test_vision_tokens_increase_llm_flops():
    """Adding images should increase LLM FLOPs due to longer sequence."""
    ext = ConfigExtractor(dtype=Dtype.FP16)

    # Text-only
    g_text = ext.extract(LLAVA_NEXT_CONFIG, batch_size=1, seq_len=128)
    f_text = sum(n.op.flops for n in g_text.nodes)

    # With 1 image (576 vision tokens)
    mm = MultiModalInput(images=[ImageInput(num_tiles=1)])
    g_mm = ext.extract(LLAVA_NEXT_CONFIG, batch_size=1, seq_len=128,
                       mm_input=mm)
    f_mm = sum(n.op.flops for n in g_mm.nodes)

    assert f_mm > f_text


def test_vision_encoder_flops_sanity():
    """Vision encoder FLOPs should match manual calculation."""
    vcfg = VisionConfig(
        image_size=336, patch_size=14, hidden_size=1024,
        num_hidden_layers=1, num_attention_heads=16,
        intermediate_size=4096, hidden_act="gelu",
        num_channels=3,
    )
    b = GraphBuilder("test", Dtype.FP16)
    b.vision_encoder("v", vcfg, num_patches=576, B=1)
    graph = b.build()

    total_flops = sum(n.op.flops for n in graph.nodes)
    # Should include: patch_embed + 1 layer of (MHA + FFN) — all positive
    assert total_flops > 0

    # Patch embed: 2 * 576 * (3*14*14) * 1024 = 2 * 576 * 588 * 1024
    patch_embed_flops = 2 * 576 * (3 * 14 * 14) * 1024
    patch_node = [n for n in graph.nodes if "patch_embed" in (n.name or "")][0]
    assert patch_node.op.flops == patch_embed_flops


# ------------------------------------------------------------------ #
# Categorization
# ------------------------------------------------------------------ #

def test_categorize_vision_encoder():
    assert categorize_op("vision.img0.layer0.attn.wq") == "Vision Encoder"
    assert categorize_op("vision.img0.patch_embed") == "Vision Encoder"
    assert categorize_op("vision.img0.layer5.ffn.w1") == "Vision Encoder"


def test_categorize_vision_projector():
    assert categorize_op("vision.img0.proj") == "Vision Projector"
    assert categorize_op("vision.img0.proj_fc1") == "Vision Projector"
    assert categorize_op("vision.img0.proj_fc2") == "Vision Projector"


def test_categorize_cross_attention():
    assert categorize_op("L4.xattn_wq") == "Cross-Attention"
    assert categorize_op("L4.xattn_score") == "Cross-Attention"


def test_categorize_existing_unchanged():
    """Existing categories still work."""
    assert categorize_op("L0.attn.attn_score") == "Attention Compute"
    assert categorize_op("L0.ffn.w1") == "Dense FFN"
    assert categorize_op("embedding") == "Embedding"
    assert categorize_op("L0.attn.wo") == "Attention Projections"
    assert categorize_op("L0.attn.wq_a") == "Attention Projections"


# ------------------------------------------------------------------ #
# VLM handler registration
# ------------------------------------------------------------------ #

def test_vlm_handlers_registered():
    archs = ConfigExtractor.supported_architectures()
    for mt in ("llava", "llava_next", "internvl_chat", "deepseek_vl_v2",
               "qwen2_5_vl", "qwen2_vl", "gemma3", "pixtral"):
        assert mt in archs, f"{mt} not registered"


# ------------------------------------------------------------------ #
# MultiModalInput / ImageInput / VideoInput / AudioInput
# ------------------------------------------------------------------ #

def test_multimodal_input_defaults():
    mm = MultiModalInput()
    assert mm.num_text_tokens == 0
    assert mm.images == []
    assert mm.video_clips == []
    assert mm.audio_segments == []


def test_image_input():
    img = ImageInput(width=672, height=672, num_tiles=4)
    assert img.width == 672
    assert img.num_tiles == 4


def test_video_input():
    vid = VideoInput(num_frames=30, width=672, height=672, temporal_patch_size=2)
    assert vid.num_frames == 30
    assert vid.temporal_patch_size == 2


def test_audio_input():
    aud = AudioInput(duration_seconds=10.0, sample_rate=16000)
    assert aud.duration_seconds == 10.0


# ------------------------------------------------------------------ #
# VLMHandler._compute_patches
# ------------------------------------------------------------------ #

def test_compute_patches_fixed():
    handler = LLaVAHandler()
    vcfg = VisionConfig(image_size=336, patch_size=14, resolution_strategy="fixed")
    img = ImageInput(num_tiles=1)
    assert handler._compute_patches(vcfg, img) == 576


def test_compute_patches_multi_tile():
    handler = LLaVAHandler()
    vcfg = VisionConfig(image_size=336, patch_size=14, resolution_strategy="anyres")
    img = ImageInput(num_tiles=4)
    assert handler._compute_patches(vcfg, img) == 4 * 576


def test_compute_patches_with_thumbnail():
    handler = InternVLHandler()
    vcfg = VisionConfig(image_size=448, patch_size=14, use_thumbnail=True,
                        resolution_strategy="dynamic_tile")
    img = ImageInput(num_tiles=4)
    # 4 + 1 thumbnail = 5 tiles
    assert handler._compute_patches(vcfg, img) == 5 * (448 // 14) ** 2


def test_compute_patches_native():
    handler = LLaVAHandler()
    vcfg = VisionConfig(image_size=336, patch_size=14,
                        resolution_strategy="native")
    img = ImageInput(width=672, height=336)
    # ceil(672/14) * ceil(336/14) = 48 * 24 = 1152
    assert handler._compute_patches(vcfg, img) == 48 * 24


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

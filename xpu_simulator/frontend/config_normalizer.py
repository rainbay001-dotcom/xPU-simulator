"""Normalize HuggingFace config.json into a canonical ModelConfig."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..core.operator import Dtype, QuantConfig


@dataclass
class AttentionPattern:
    """Describes how attention scoring is performed.

    All patterns produce the same canonical node names (attn_score, attn_softmax,
    attn_v) but with different tensor shapes reflecting the effective context length.
    """
    kind: str = "dense"  # "dense", "top_k", "sliding_window", "block_sparse"

    # Top-k (DSA) parameters
    top_k: Optional[int] = None
    num_indexer_heads: Optional[int] = None
    indexer_dim: Optional[int] = None

    # Sliding window parameters
    window_size: Optional[int] = None

    # Block sparse parameters
    block_size: Optional[int] = None


@dataclass
class VisionConfig:
    """Vision encoder configuration, normalized from HF vision_config."""
    encoder_type: str = "clip"       # "clip", "siglip", "internvit", "custom"
    image_size: int = 336            # Base image size
    patch_size: int = 14             # Patch size
    hidden_size: int = 1024          # Vision hidden dim
    num_hidden_layers: int = 24      # ViT layers
    num_attention_heads: int = 16    # ViT attention heads
    intermediate_size: int = 4096    # ViT FFN intermediate dim
    hidden_act: str = "gelu"         # ViT activation
    num_channels: int = 3            # Input channels (RGB)

    # Projector
    projector_type: str = "mlp"      # "linear", "mlp", "pixel_shuffle_mlp",
                                     # "qformer", "perceiver", "cross_attn"
    projector_hidden_size: Optional[int] = None  # Defaults to vision hidden_size
    spatial_merge_size: int = 1      # Per-axis spatial compression (2 → 4:1)

    # Resolution strategy
    resolution_strategy: str = "fixed"  # "fixed", "anyres", "dynamic_tile", "native"
    max_tiles: int = 1
    use_thumbnail: bool = False
    tile_size: Optional[int] = None     # Defaults to image_size
    image_grid_pinpoints: Optional[list] = None

    # Perceiver / Q-Former specific
    num_query_tokens: int = 32

    # Feature extraction
    vision_feature_layer: int = -1   # -2 means skip last layer

    # Cross-attention (Phase 2)
    cross_attn_layer_indices: Optional[list[int]] = None

    @property
    def patches_per_image(self) -> int:
        """Patches for one tile at base resolution."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def tokens_per_tile(self) -> int:
        """Vision tokens per tile after spatial compression."""
        s = self.spatial_merge_size
        return self.patches_per_image // (s * s)

    def num_vision_tokens(self, img_w: int = 0, img_h: int = 0,
                          num_tiles: int = 1) -> int:
        """Compute total vision tokens for one image."""
        if self.resolution_strategy == "native" and img_w > 0 and img_h > 0:
            p = self.patch_size
            s = self.spatial_merge_size
            pw = math.ceil(img_w / p)
            ph = math.ceil(img_h / p)
            return (pw * ph) // max(s * s, 1)

        effective_tiles = num_tiles
        if self.use_thumbnail:
            effective_tiles += 1
        return effective_tiles * self.tokens_per_tile

    def num_encoder_layers(self) -> int:
        """ViT layers actually executed (respecting feature_layer)."""
        if self.vision_feature_layer == -1:
            return self.num_hidden_layers
        elif self.vision_feature_layer < 0:
            return self.num_hidden_layers + self.vision_feature_layer + 1
        return self.vision_feature_layer + 1


@dataclass
class ModelConfig:
    """Canonical model configuration for graph generation.

    Normalizes field-name differences across HuggingFace architectures
    (GPT-2, Falcon, LLaMA, Mixtral, DeepSeek, etc.) into a single schema.
    """

    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    head_dim: int
    hidden_act: str = "silu"
    rope: bool = True
    rms_norm: bool = True
    tie_word_embeddings: bool = False

    # MoE fields (None if dense model)
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    num_shared_experts: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    first_k_dense_replace: Optional[int] = None

    # MLA fields (None if not MLA)
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # DSA fields (kept for backward compatibility, prefer attention_pattern)
    dsa_num_indexer_heads: Optional[int] = None
    dsa_k: Optional[int] = None
    dsa_indexer_dim: Optional[int] = None

    # General attention pattern (populated from DSA fields or directly)
    attention_pattern: AttentionPattern = field(default_factory=AttentionPattern)

    # Quantization
    quant_config: Optional[QuantConfig] = None

    # Falcon-specific
    parallel_attn: bool = False

    # Multi-modal (None if text-only)
    vision_config: Optional[VisionConfig] = None

    @property
    def is_multimodal(self) -> bool:
        return self.vision_config is not None

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 0

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def is_dsa(self) -> bool:
        """True if using DeepSeek Sparse Attention."""
        return self.attention_pattern.kind == "top_k"

    @property
    def qk_head_dim(self) -> int:
        """Full QK head dim for MLA (nope + rope)."""
        if self.qk_nope_head_dim is not None and self.qk_rope_head_dim is not None:
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.head_dim


def normalize_config(raw: dict) -> ModelConfig:
    """Normalize a HuggingFace config dict into a canonical ModelConfig.

    Handles field-name differences across architectures:
    - GPT-2: n_embd, n_head, n_layer
    - Falcon: num_kv_heads, multi_query, parallel_attn
    - Mixtral: num_local_experts
    - DeepSeek: n_routed_experts, first_k_dense_replace, MLA fields
    """
    model_type = raw.get("model_type", "unknown")

    # --- Core dimensions (with GPT-2 remapping) ---
    hidden_size = raw.get("hidden_size") or raw.get("n_embd")
    num_heads = raw.get("num_attention_heads") or raw.get("n_head")
    num_layers = raw.get("num_hidden_layers") or raw.get("n_layer")
    vocab_size = raw.get("vocab_size", 32000)

    # Intermediate size: explicit or implicit (4 * hidden_size for GPT-2/Falcon)
    intermediate_size = (
        raw.get("intermediate_size")
        or raw.get("n_inner")
        or 4 * hidden_size
    )

    # --- KV heads ---
    num_kv_heads = (
        raw.get("num_key_value_heads")
        or raw.get("num_kv_heads")  # Falcon
    )
    # Falcon MQA via boolean flag
    if num_kv_heads is None:
        if raw.get("multi_query", False):
            num_kv_heads = 1
        else:
            num_kv_heads = num_heads

    # --- Head dim ---
    head_dim = raw.get("head_dim") or (hidden_size // num_heads)

    # --- Activation ---
    hidden_act = raw.get("hidden_act") or raw.get("activation_function") or "silu"

    # --- Positional encoding ---
    rope = True
    if model_type in ("gpt2",):
        rope = False
    if raw.get("position_embedding_type") == "absolute":
        rope = False

    # --- Normalization ---
    rms_norm = model_type not in ("gpt2", "falcon", "gpt_neox")

    # --- MoE fields ---
    num_experts = (
        raw.get("num_local_experts")      # Mixtral
        or raw.get("num_experts")          # Qwen2-MoE
        or raw.get("n_routed_experts")     # DeepSeek
    )
    num_experts_per_tok = raw.get("num_experts_per_tok")
    moe_intermediate_size = raw.get("moe_intermediate_size")

    # Mixtral: experts use same intermediate_size as dense layers
    if num_experts and moe_intermediate_size is None:
        moe_intermediate_size = intermediate_size

    num_shared_experts = raw.get("n_shared_experts") or raw.get("num_shared_experts")
    shared_expert_intermediate_size = raw.get("shared_expert_intermediate_size")
    # DeepSeek: derive shared intermediate from n_shared * moe_intermediate
    if num_shared_experts and not shared_expert_intermediate_size and moe_intermediate_size:
        shared_expert_intermediate_size = num_shared_experts * moe_intermediate_size

    first_k_dense_replace = raw.get("first_k_dense_replace")

    # --- MLA fields (DeepSeek-specific) ---
    kv_lora_rank = raw.get("kv_lora_rank")
    q_lora_rank = raw.get("q_lora_rank")
    qk_nope_head_dim = raw.get("qk_nope_head_dim")
    qk_rope_head_dim = raw.get("qk_rope_head_dim")
    v_head_dim = raw.get("v_head_dim")

    # --- DSA fields ---
    dsa_num_indexer_heads = raw.get("dsa_num_indexer_heads")
    dsa_k = raw.get("dsa_k")
    dsa_indexer_dim = raw.get("dsa_indexer_dim")

    # --- Attention pattern ---
    # Can be set directly via "attention_pattern" dict, or inferred from DSA/window fields
    raw_pattern = raw.get("attention_pattern")
    if raw_pattern and isinstance(raw_pattern, dict):
        attention_pattern = AttentionPattern(**raw_pattern)
    elif dsa_k is not None and dsa_k > 0:
        attention_pattern = AttentionPattern(
            kind="top_k", top_k=dsa_k,
            num_indexer_heads=dsa_num_indexer_heads,
            indexer_dim=dsa_indexer_dim,
        )
    elif raw.get("sliding_window"):
        attention_pattern = AttentionPattern(
            kind="sliding_window", window_size=raw["sliding_window"],
        )
    else:
        attention_pattern = AttentionPattern(kind="dense")

    # --- Quantization ---
    quant_config = None
    quant_raw = raw.get("quantization_config")
    if quant_raw and isinstance(quant_raw, dict):
        quant_method = quant_raw.get("quant_method", "")
        bits = quant_raw.get("bits", 8)
        group_size = quant_raw.get("group_size")
        if quant_method in ("gptq", "awq") and bits == 4:
            quant_config = QuantConfig(Dtype.INT4, Dtype.INT8, group_size)
        elif quant_method in ("gptq", "awq") and bits == 8:
            quant_config = QuantConfig(Dtype.INT8, Dtype.INT8, group_size)
        elif quant_method == "fp8":
            quant_config = QuantConfig(Dtype.FP8, Dtype.FP8)
    # Allow direct quant_config dict override
    quant_override = raw.get("quant_config")
    if quant_override and isinstance(quant_override, dict):
        w_dtype = Dtype(quant_override.get("weight_dtype", "fp16"))
        a_dtype = Dtype(quant_override.get("activation_dtype", "fp16"))
        quant_config = QuantConfig(w_dtype, a_dtype,
                                   quant_override.get("group_size"))

    # --- Falcon-specific ---
    parallel_attn = raw.get("parallel_attn", False)

    # --- Vision config ---
    vision_config = None
    raw_vision = raw.get("vision_config")
    if raw_vision and isinstance(raw_vision, dict):
        vision_config = _normalize_vision_config(raw, raw_vision)

    return ModelConfig(
        model_type=model_type,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=head_dim,
        hidden_act=hidden_act,
        rope=rope,
        rms_norm=rms_norm,
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        num_shared_experts=num_shared_experts,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        first_k_dense_replace=first_k_dense_replace,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        dsa_num_indexer_heads=dsa_num_indexer_heads,
        dsa_k=dsa_k,
        dsa_indexer_dim=dsa_indexer_dim,
        attention_pattern=attention_pattern,
        quant_config=quant_config,
        parallel_attn=parallel_attn,
        vision_config=vision_config,
    )


def _normalize_vision_config(raw_top: dict, raw_v: dict) -> VisionConfig:
    """Normalize HF vision_config across model families."""
    encoder_type = raw_v.get("model_type", "custom")
    encoder_map = {
        "clip_vision_model": "clip",
        "siglip_vision_model": "siglip",
        "intern_vit_6b": "internvit",
    }
    encoder_type = encoder_map.get(encoder_type, encoder_type)

    image_size = raw_v.get("image_size", 224)
    patch_size = raw_v.get("patch_size") or raw_v.get("spatial_patch_size", 14)
    hidden_size = raw_v.get("hidden_size") or raw_v.get("width", 1024)
    num_layers = raw_v.get("num_hidden_layers") or raw_v.get("depth", 24)
    num_heads = raw_v.get("num_attention_heads") or raw_v.get("num_heads", 16)
    inter_size = raw_v.get("intermediate_size") or hidden_size * 4
    hidden_act = raw_v.get("hidden_act", "gelu")
    num_channels = raw_v.get("in_chans", raw_v.get("num_channels", 3))

    # Projector type inference
    projector_type = "mlp"
    proj_cfg = raw_top.get("projector_config")
    if proj_cfg and isinstance(proj_cfg, dict):
        pt = proj_cfg.get("model_type", "")
        if "linear" in pt:
            projector_type = "linear"

    spatial_merge = raw_v.get("spatial_merge_size", 1)
    if raw_top.get("downsample_ratio"):
        ratio = raw_top["downsample_ratio"]
        if ratio < 1.0:
            spatial_merge = int(round(1 / ratio))

    if spatial_merge > 1:
        projector_type = "pixel_shuffle_mlp"

    projector_hidden = None
    if proj_cfg and isinstance(proj_cfg, dict):
        projector_hidden = proj_cfg.get("n_embed")

    # Resolution strategy
    resolution_strategy = "fixed"
    max_tiles = 1
    pinpoints = raw_top.get("image_grid_pinpoints")

    if pinpoints:
        resolution_strategy = "anyres"
        max_tiles = max(len(p) // 2 if len(p) > 2 else 1 for p in pinpoints) if pinpoints else 1
    elif raw_top.get("max_dynamic_patch"):
        resolution_strategy = "dynamic_tile"
        max_tiles = raw_top["max_dynamic_patch"]
    elif raw_top.get("candidate_resolutions"):
        resolution_strategy = "dynamic_tile"
        cands = raw_top["candidate_resolutions"]
        max_tiles = max(
            max(1, (r[0] * r[1]) // max(image_size * image_size, 1))
            for r in cands
        ) if cands else 1

    use_thumbnail = raw_top.get("use_thumbnail", False)

    vfl = raw_top.get("vision_feature_layer", -1)
    if isinstance(vfl, str):
        vfl = -1

    return VisionConfig(
        encoder_type=encoder_type,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=inter_size,
        hidden_act=hidden_act,
        num_channels=num_channels,
        projector_type=projector_type,
        projector_hidden_size=projector_hidden,
        spatial_merge_size=spatial_merge,
        resolution_strategy=resolution_strategy,
        max_tiles=max_tiles,
        use_thumbnail=use_thumbnail,
        image_grid_pinpoints=pinpoints,
        vision_feature_layer=vfl,
    )

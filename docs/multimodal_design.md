# Multi-Modal Model Simulation — Design Document

**Date**: 2026-04-16
**Status**: Draft
**Scope**: Add vision-language model (VLM) simulation to xPU-simulator

---

## 1. Motivation

xPU-simulator currently models text-only LLM inference: embedding → N transformer layers → lm_head. Multi-modal models (LLaVA, Qwen2.5-VL, InternVL2.5, DeepSeek-VL2, Gemma3, Pixtral, etc.) prepend a **vision encoder** and **projector/connector** to this pipeline, injecting image tokens into the LLM's sequence. This changes the performance profile significantly:

- **Vision encoder** is a separate ViT model (300M–6B params) that runs entirely in prefill — no decode phase. At high resolutions, FastVLM (Apple, CVPR 2025) shows the vision encoder can **exceed LLM prefill time by 2x+**, making it the TTFT bottleneck.
- **Dynamic resolution** tiling (InternVL, Qwen2.5-VL, LLaVA-NeXT) causes vision token counts to vary from ~256 to ~13,000+ per image, dramatically affecting LLM prefill cost.
- **Token compression** (pixel shuffle, spatial merge, Perceiver resampler) reduces vision tokens before they enter the LLM — a critical parameter for latency.
- **Multi-image and video** inputs can push total token counts to 50K+, making serving throughput estimation fundamentally different from text-only models.

Without multi-modal support, the simulator cannot predict TTFT for the fastest-growing class of production models.

---

## 2. Architecture Survey

### 2.1 VLM Structure

All mainstream VLMs decompose into three stages:

```
Image ──► Vision Encoder ──► Projector/Connector ──► LLM Backbone
                                                      ▲
Text  ──────────────────────────────────────────────────┘
```

The compute graph is a **sequential pipeline**: vision encoder runs first (or concurrently on a separate device), its output tokens are projected and concatenated with text embeddings, then the full sequence enters the LLM.

### 2.2 Vision Encoder Taxonomy

| Encoder | Params | Layers | Hidden | Patch | Used By |
|---------|--------|--------|--------|-------|---------|
| CLIP ViT-L/14-336px | ~300M | 24 | 1024 | 14 | LLaVA-1.5/NeXT, Phi-3.5-V |
| SigLIP-SO400M-384px | ~400M | 27 | 1152 | 14 | DeepSeek-VL2, Gemma3, Janus-Pro |
| InternViT-6B | ~5.5B | 45 | 3200 | 14 | InternVL2/2.5, NVLM |
| InternViT-300M | ~300M | 24 | 1024 | 14 | InternVL2.5 small variants |
| Pixtral-ViT | ~400M | 24 | 1024 | 16 | Pixtral 12B (2D-RoPE) |
| Qwen2.5-VL ViT | ~675M | 32 | 1280 | 14 | Qwen2.5-VL (window attn) |
| EVA02-CLIP-E | ~1B | 40 | 1408 | 14 | CogVLM/CogVLM2 |

Vision encoders are standard ViTs: patch embedding (Conv2D or linear) → N self-attention layers → optional pooling. They are **compute-bound** during prefill (pure GEMMs), with cost scaling **quadratically** in patch count due to self-attention: `O(P^2 * d + P * d^2)` where `P = (H/p) * (W/p)`.

### 2.3 Projector/Connector Taxonomy

| Type | Mechanism | Token Compression | Params | Models |
|------|-----------|-------------------|--------|--------|
| Linear | Single linear layer | 1:1 | ~4M | LLaVA-1.0, PaliGemma |
| MLP (2-layer) | Linear → GELU/SiLU → Linear | 1:1 | ~8M | LLaVA-1.5+, InternVL2, DeepSeek-VL2 |
| MLP + Pixel Shuffle | MLP after spatial 2x2→1 merge | 4:1 | ~8M | InternVL2.5, Qwen2.5-VL |
| Q-Former | Cross-attn with 32 learned queries | N:32 fixed | ~188M | BLIP-2, InstructBLIP |
| Perceiver Resampler | Cross-attn with learned queries | N:K fixed | ~100M | Flamingo, IDEFICS, Apollo |
| Cross-Attention Adapter | Injected cross-attn in LLM layers | N/A (no concat) | varies | Llama 3.2-Vision, Flamingo |
| C-Abstractor | CNN-based spatial compression | configurable | ~20M | Honeybee (CVPR 2024) |

**Design implication**: The MLP projector (with optional spatial merge) covers >80% of production VLMs. Q-Former and Perceiver are important for completeness but can be a second-phase addition.

### 2.4 Dynamic Resolution Strategies

Models handle variable image sizes differently, which directly affects vision token count:

| Strategy | Models | How It Works | Token Formula |
|----------|--------|--------------|---------------|
| Fixed resize | LLaVA-1.5 | Resize all images to 336x336 | `(336/p)^2 = 576` |
| AnyRes grid | LLaVA-NeXT | Fit image into grid of pinpoints (e.g., 2x2) | `N_tiles * (tile/p)^2` |
| Dynamic tiling | InternVL2.5 | Select 1–12 tiles of 448x448 + thumbnail | `(N_tiles+1) * (448/p)^2 / compress` |
| Native dynamic | Qwen2.5-VL | No tiling; ViT runs at native resolution | `ceil(H/p) * ceil(W/p) / merge_ratio` |
| Candidate set | DeepSeek-VL2 | Choose from 23 predefined resolutions | `H_eff/p * W_eff/p` |
| Pan-and-Scan | Gemma3 | Adaptive crops to fixed 896x896 | 256 fixed per image |

**Key parameter**: `num_vision_tokens(image_width, image_height)` — this is the input that drives LLM prefill cost.

### 2.5 Token Merging into the LLM

Three integration paradigms:

1. **Sequence concatenation** (most common): Vision tokens projected to LLM hidden dim, concatenated with text embeddings: `[BOS, <img_start>, v1, v2, ..., vN, <img_end>, t1, t2, ...]`. Standard self-attention over the unified sequence. Used by LLaVA, InternVL, DeepSeek-VL2, Qwen2.5-VL, Pixtral, Gemma3.

2. **Cross-attention injection**: Vision tokens not in the sequence. Dedicated cross-attention layers (every K-th LLM block) attend to vision features. Used by Flamingo, Llama 3.2-Vision. Requires modifying the LLM layer template.

3. **Modality experts** (CogVLM): Parallel QKV + FFN per modality in each LLM layer. Visual tokens use visual expert, text tokens use text expert. Doubles per-layer params but keeps modalities separate.

**Design implication**: Sequence concatenation is the default — it only changes the effective sequence length fed to the existing LLM simulation. Cross-attention and modality experts require new LLM layer variants.

---

## 3. HuggingFace Config Patterns

Multi-modal HF configs nest `vision_config` inside the top-level config:

```json
{
  "model_type": "llava_next",
  "text_config": {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    ...
  },
  "vision_config": {
    "model_type": "clip_vision_model",
    "image_size": 336,
    "patch_size": 14,
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096
  },
  "image_grid_pinpoints": [[336,672],[672,336],[672,672]],
  "projector_hidden_act": "gelu",
  "vision_feature_layer": -2
}
```

**Key observations for the normalizer:**

- `vision_config` fields are consistent across models: `image_size`, `patch_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`.
- The **LLM text config** may be nested under `text_config` or inlined at the top level (varies by model).
- **Projector** config is either implicit (`projector_hidden_act` field) or explicit (`projector_config` block in DeepSeek-VL2).
- **Resolution strategy** is encoded differently per model: `image_grid_pinpoints` (LLaVA-NeXT), `candidate_resolutions` (DeepSeek-VL2), `max_dynamic_patch` + `force_image_size` (InternVL), `spatial_merge_size` (Qwen2.5-VL).
- Multi-modal `model_type` values: `llava`, `llava_next`, `qwen2_5_vl`, `internvl_chat`, `deepseek_vl_v2`, `phi3_v`, `gemma3`, `pixtral`, `idefics2`.

---

## 4. Proposed Design

### 4.1 Design Principles

1. **Reuse, don't duplicate**: The vision encoder is a standard transformer — reuse `GraphBuilder.gqa_attention`, `dense_ffn`, `norm`, etc. The LLM backbone is unchanged; only the effective sequence length increases.
2. **Config-driven**: Follow the existing `normalize_config → ArchitectureHandler.build_layer` pattern. Add a `VisionConfig` dataclass and `VLMHandler` base class. Auto-detect from HF config.json.
3. **Composable**: A VLM handler wraps an existing LLM handler (e.g., `StandardTransformerHandler`) — no copy-paste of LLM layer logic.
4. **Resolution-aware**: The user specifies image dimensions; the handler computes vision token count from the model's resolution strategy.

### 4.2 New Data Structures

#### `VisionConfig` dataclass (`frontend/config_normalizer.py`)

```python
@dataclass
class VisionConfig:
    """Vision encoder configuration, normalized from HF vision_config."""
    encoder_type: str              # "clip", "siglip", "internvit", "custom"
    image_size: int                # Base image size (e.g., 336, 384, 448)
    patch_size: int                # Patch size (e.g., 14, 16)
    hidden_size: int               # Vision hidden dim
    num_hidden_layers: int         # Number of ViT layers
    num_attention_heads: int       # ViT attention heads
    intermediate_size: int         # ViT FFN intermediate dim
    hidden_act: str = "gelu"       # ViT activation
    num_channels: int = 3          # Input channels (RGB)

    # Projector
    projector_type: str = "mlp"    # "linear", "mlp", "qformer", "perceiver",
                                   # "cross_attn", "pixel_shuffle_mlp"
    projector_hidden_size: int | None = None   # Intermediate dim (defaults to vision hidden)
    projector_output_size: int | None = None   # Output dim (defaults to LLM hidden)
    spatial_merge_size: int = 1    # Spatial compression factor per axis
                                   # (2 means 2x2=4 tokens → 1 token)

    # Resolution strategy
    resolution_strategy: str = "fixed"  # "fixed", "anyres", "dynamic_tile", "native"
    max_tiles: int = 1                  # Max tile count for tiled strategies
    use_thumbnail: bool = False         # Add global thumbnail tile
    tile_size: int | None = None        # Per-tile size (defaults to image_size)
    image_grid_pinpoints: list[list[int]] | None = None  # AnyRes pinpoints

    # Perceiver / Q-Former specific
    num_query_tokens: int = 32     # Learned query count for Q-Former/Perceiver

    # Feature extraction
    vision_feature_layer: int = -1  # Which ViT layer to extract features from
                                    # (-2 means skip last layer)

    @property
    def patches_per_image(self) -> int:
        """Number of patches for one tile at base resolution."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def tokens_per_tile(self) -> int:
        """Vision tokens per tile after spatial compression."""
        s = self.spatial_merge_size
        return self.patches_per_image // (s * s)

    def num_vision_tokens(self, img_w: int = 0, img_h: int = 0,
                          num_tiles: int = 1) -> int:
        """Compute total vision tokens given image dimensions or tile count.

        For fixed/anyres: based on num_tiles.
        For native: based on actual pixel dimensions.
        """
        if self.resolution_strategy == "native" and img_w > 0 and img_h > 0:
            p = self.patch_size
            s = self.spatial_merge_size
            pw = math.ceil(img_w / p)
            ph = math.ceil(img_h / p)
            return (pw * ph) // (s * s)

        effective_tiles = num_tiles
        if self.use_thumbnail:
            effective_tiles += 1
        return effective_tiles * self.tokens_per_tile

    def num_encoder_layers(self) -> int:
        """Number of ViT layers actually executed (respecting feature_layer)."""
        if self.vision_feature_layer == -1:
            return self.num_hidden_layers
        elif self.vision_feature_layer < 0:
            return self.num_hidden_layers + self.vision_feature_layer + 1
        return self.vision_feature_layer + 1
```

#### Extend `ModelConfig` with optional `VisionConfig`

```python
@dataclass
class ModelConfig:
    # ... existing fields ...

    # Multi-modal (None if text-only)
    vision_config: VisionConfig | None = None

    @property
    def is_multimodal(self) -> bool:
        return self.vision_config is not None
```

#### `MultiModalInput` — per-request input specification

```python
@dataclass
class MultiModalInput:
    """Describes the multi-modal content of a single request."""
    num_text_tokens: int = 0
    images: list[ImageInput] = field(default_factory=list)
    # Future: video_clips, audio_segments

@dataclass
class ImageInput:
    width: int = 0     # 0 = use model default
    height: int = 0
    num_tiles: int = 1  # Pre-computed tile count (for anyres/dynamic_tile)
```

### 4.3 Vision Encoder Graph Construction

Add a `vision_encoder` composite method to `GraphBuilder`:

```python
def vision_encoder(self, prefix: str, vcfg: VisionConfig,
                   num_patches: int, B: int = 1,
                   prev: Node | None = None) -> Node:
    """Build a ViT vision encoder subgraph.

    Args:
        prefix: Name prefix (e.g., "vision").
        vcfg: Vision encoder configuration.
        num_patches: Total patch count (P = H/p * W/p * num_tiles).
        B: Batch size.
    """
    tokens = B * num_patches

    # 1. Patch embedding: Conv2D or linear projection
    #    Input: [B, C, H, W] → Output: [B*P, D_v]
    #    FLOPs: B * P * (C * p * p) * D_v  (equivalent to a linear)
    patch_dim = vcfg.num_channels * vcfg.patch_size * vcfg.patch_size
    node = self.matmul(f"{prefix}.patch_embed",
                       tokens, patch_dim, vcfg.hidden_size,
                       *([prev] if prev else []))

    # 2. ViT transformer layers
    n_layers = vcfg.num_encoder_layers()
    for i in range(n_layers):
        lp = f"{prefix}.layer{i}"
        # Pre-norm
        node = self.norm(f"{lp}.norm1", (tokens, vcfg.hidden_size), node)
        # Self-attention (ViT uses standard MHA, no GQA/MLA)
        node = self.gqa_attention(
            f"{lp}.attn", tokens, B, num_patches,
            dim=vcfg.hidden_size,
            n_heads=vcfg.num_attention_heads,
            n_kv_heads=vcfg.num_attention_heads,  # MHA
            head_dim=vcfg.hidden_size // vcfg.num_attention_heads,
            rope=False,  # ViTs use learned/absolute pos (except Pixtral 2D-RoPE)
            prev=node,
        )
        # Pre-norm for FFN
        node = self.norm(f"{lp}.norm2", (tokens, vcfg.hidden_size), node)
        # FFN
        act = OpType.GELU if vcfg.hidden_act in ("gelu", "quick_gelu") \
              else OpType.SILU
        node = self.dense_ffn(f"{lp}.ffn", tokens,
                              vcfg.hidden_size, vcfg.intermediate_size,
                              activation=act, prev=node)

    return node
```

### 4.4 Projector Graph Construction

```python
def vision_projector(self, prefix: str, vcfg: VisionConfig,
                     num_tokens: int, llm_hidden: int,
                     B: int = 1, prev: Node | None = None) -> Node:
    """Build a vision-to-LLM projector subgraph."""
    tokens = B * num_tokens

    if vcfg.projector_type == "linear":
        return self.matmul(f"{prefix}.proj",
                           tokens, vcfg.hidden_size, llm_hidden,
                           *([prev] if prev else []))

    elif vcfg.projector_type in ("mlp", "pixel_shuffle_mlp"):
        # If pixel_shuffle, the input token count is already reduced
        # by spatial_merge_size in the caller
        inter = vcfg.projector_hidden_size or vcfg.hidden_size
        # For pixel_shuffle_mlp, input dim is hidden_size * merge^2
        in_dim = vcfg.hidden_size
        if vcfg.projector_type == "pixel_shuffle_mlp":
            in_dim = vcfg.hidden_size * vcfg.spatial_merge_size ** 2
        node = self.matmul(f"{prefix}.proj_fc1",
                           tokens, in_dim, inter,
                           *([prev] if prev else []))
        node = self.elementwise(f"{prefix}.proj_act",
                                (tokens, inter), OpType.GELU, node)
        node = self.matmul(f"{prefix}.proj_fc2",
                           tokens, inter, llm_hidden, node)
        return node

    elif vcfg.projector_type == "qformer":
        # Q-Former: cross-attention with learned queries
        # Simplified: query self-attention + cross-attention + FFN
        # per Q-Former layer (typically 6-12 layers)
        n_queries = vcfg.num_query_tokens  # e.g., 32
        q_tokens = B * n_queries
        # Output is fixed n_queries tokens regardless of input
        # Cross-attn: queries attend to vision tokens
        node = self.matmul(f"{prefix}.qformer_cross",
                           q_tokens, vcfg.hidden_size, llm_hidden,
                           *([prev] if prev else []))
        return node

    elif vcfg.projector_type == "perceiver":
        n_queries = vcfg.num_query_tokens
        q_tokens = B * n_queries
        node = self.matmul(f"{prefix}.perceiver_proj",
                           q_tokens, vcfg.hidden_size, llm_hidden,
                           *([prev] if prev else []))
        return node

    else:
        # Fallback: treat as linear
        return self.matmul(f"{prefix}.proj",
                           tokens, vcfg.hidden_size, llm_hidden,
                           *([prev] if prev else []))
```

### 4.5 VLM Architecture Handlers

#### Base `VLMHandler`

```python
class VLMHandler(ArchitectureHandler):
    """Base handler for vision-language models.

    Wraps an existing LLM handler and prepends vision encoder + projector.
    Subclasses only need to override resolution/tiling logic.
    """

    def __init__(self, llm_handler: ArchitectureHandler):
        self.llm_handler = llm_handler

    def build_vision_stage(self, builder: GraphBuilder, cfg: ModelConfig,
                           mm_input: MultiModalInput,
                           B: int) -> tuple[Node, int]:
        """Build vision encoder + projector. Returns (last_node, num_vision_tokens).

        The returned num_vision_tokens is the count AFTER compression/projection,
        i.e., the number of tokens that will be concatenated into the LLM sequence.
        """
        vcfg = cfg.vision_config
        total_vision_tokens = 0

        last_node = None
        for i, img in enumerate(mm_input.images):
            # Compute raw patch count from resolution strategy
            num_patches = self._compute_patches(vcfg, img)

            # Vision encoder
            enc_node = builder.vision_encoder(
                f"vision.img{i}", vcfg, num_patches, B, last_node)

            # Token count after spatial compression
            if vcfg.projector_type == "pixel_shuffle_mlp":
                s = vcfg.spatial_merge_size
                proj_tokens = num_patches // (s * s)
            elif vcfg.projector_type in ("qformer", "perceiver"):
                proj_tokens = vcfg.num_query_tokens
            else:
                proj_tokens = num_patches

            # Projector
            last_node = builder.vision_projector(
                f"vision.img{i}", vcfg, proj_tokens,
                cfg.hidden_size, B, enc_node)

            total_vision_tokens += proj_tokens

        return last_node, total_vision_tokens

    def _compute_patches(self, vcfg: VisionConfig,
                         img: ImageInput) -> int:
        """Compute raw patch count for one image."""
        if vcfg.resolution_strategy == "native" and img.width > 0:
            p = vcfg.patch_size
            return math.ceil(img.width / p) * math.ceil(img.height / p)

        tile = vcfg.tile_size or vcfg.image_size
        patches_per_tile = (tile // vcfg.patch_size) ** 2
        n_tiles = img.num_tiles or 1
        if vcfg.use_thumbnail:
            n_tiles += 1
        return n_tiles * patches_per_tile
```

#### Concrete VLM handlers (composition over inheritance)

```python
class LLaVAHandler(VLMHandler):
    """LLaVA-1.5, LLaVA-NeXT: CLIP/SigLIP encoder + MLP projector + LLM."""
    def __init__(self):
        super().__init__(StandardTransformerHandler())

class InternVLHandler(VLMHandler):
    """InternVL2/2.5: InternViT + pixel-shuffle MLP + LLM."""
    def __init__(self):
        super().__init__(StandardTransformerHandler())

class DeepSeekVLHandler(VLMHandler):
    """DeepSeek-VL2: SigLIP + MLP + DeepSeek MoE LLM."""
    def __init__(self):
        super().__init__(DeepSeekHandler())
```

### 4.6 ConfigExtractor Integration

Extend `ConfigExtractor.extract()` to handle multi-modal configs:

```python
def extract(self, config, batch_size=1, seq_len=1024,
            mm_input: MultiModalInput | None = None,
            **kwargs) -> ComputeGraph:
    raw = self._load_config(config)
    cfg = normalize_config(raw)  # Now parses vision_config too

    if cfg.is_multimodal and mm_input:
        return self._extract_multimodal(cfg, batch_size, seq_len,
                                        mm_input, **kwargs)
    else:
        return self._extract_text_only(cfg, batch_size, seq_len, **kwargs)
```

The `normalize_config` function gains vision config parsing:

```python
def normalize_config(raw: dict) -> ModelConfig:
    # ... existing LLM normalization ...

    # --- Vision config ---
    vision_config = None
    raw_vision = raw.get("vision_config")
    if raw_vision and isinstance(raw_vision, dict):
        vision_config = _normalize_vision_config(raw, raw_vision)

    return ModelConfig(..., vision_config=vision_config)


def _normalize_vision_config(raw_top: dict, raw_v: dict) -> VisionConfig:
    """Normalize HF vision_config across model families."""
    encoder_type = raw_v.get("model_type", "custom")
    # Map HF model_type to our encoder type
    encoder_map = {
        "clip_vision_model": "clip",
        "siglip_vision_model": "siglip",
        "intern_vit_6b": "internvit",
    }
    encoder_type = encoder_map.get(encoder_type, encoder_type)

    image_size = raw_v.get("image_size", 224)
    patch_size = raw_v.get("patch_size", 14)
    hidden_size = raw_v.get("hidden_size", 1024)
    num_layers = raw_v.get("num_hidden_layers") or raw_v.get("depth", 24)
    num_heads = raw_v.get("num_attention_heads") or raw_v.get("num_heads", 16)
    inter_size = raw_v.get("intermediate_size", hidden_size * 4)
    hidden_act = raw_v.get("hidden_act", "gelu")

    # Projector type inference
    projector_type = "mlp"  # default
    if raw_top.get("projector_config", {}).get("model_type") == "mlp_projector":
        projector_type = "mlp"
    if raw_v.get("spatial_merge_size", 1) > 1 or \
       raw_top.get("downsample_ratio", 1.0) < 1.0:
        projector_type = "pixel_shuffle_mlp"

    spatial_merge = raw_v.get("spatial_merge_size", 1)
    if raw_top.get("downsample_ratio"):
        # InternVL: downsample_ratio=0.5 means 2x2→1
        spatial_merge = int(1 / raw_top["downsample_ratio"])

    # Resolution strategy
    resolution_strategy = "fixed"
    max_tiles = 1
    if raw_top.get("image_grid_pinpoints"):
        resolution_strategy = "anyres"
        max_tiles = max(len(p) for p in raw_top["image_grid_pinpoints"])
    elif raw_top.get("max_dynamic_patch"):
        resolution_strategy = "dynamic_tile"
        max_tiles = raw_top["max_dynamic_patch"]
    elif raw_top.get("candidate_resolutions"):
        resolution_strategy = "dynamic_tile"
        max_tiles = max(
            (r[0] * r[1]) // (image_size * image_size)
            for r in raw_top["candidate_resolutions"]
        )

    use_thumbnail = raw_top.get("use_thumbnail", False)
    vision_feature_layer = raw_top.get("vision_feature_layer",
                                        raw_top.get("vision_feature_select_strategy",
                                                     -1))
    if isinstance(vision_feature_layer, str):
        vision_feature_layer = -1  # "default" or "full" → use all layers

    return VisionConfig(
        encoder_type=encoder_type,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=inter_size,
        hidden_act=hidden_act,
        projector_type=projector_type,
        spatial_merge_size=spatial_merge,
        resolution_strategy=resolution_strategy,
        max_tiles=max_tiles,
        use_thumbnail=use_thumbnail,
        vision_feature_layer=vision_feature_layer,
        image_grid_pinpoints=raw_top.get("image_grid_pinpoints"),
    )
```

### 4.7 Handler Registration

```python
# Register VLM handlers
for _model_type in ("llava", "llava_next", "llava_next_video"):
    ConfigExtractor.register_handler(_model_type, LLaVAHandler)

ConfigExtractor.register_handler("internvl_chat", InternVLHandler)
ConfigExtractor.register_handler("deepseek_vl_v2", DeepSeekVLHandler)

for _model_type in ("qwen2_5_vl", "qwen2_vl"):
    ConfigExtractor.register_handler(_model_type, Qwen2VLHandler)

ConfigExtractor.register_handler("gemma3", Gemma3Handler)
ConfigExtractor.register_handler("pixtral", PixtralHandler)
```

### 4.8 Op Categorization

Extend `categories.py` to recognize vision ops:

```python
CATEGORY_COLORS = {
    # ... existing ...
    "Vision Encoder":    "#FF6B6B",   # coral
    "Vision Projector":  "#FF9F43",   # light orange
}

def categorize_op(name: str) -> str:
    # Vision categories (check first — prefix-based)
    if name.startswith("vision.") and ".proj" in name:
        return "Vision Projector"
    elif name.startswith("vision."):
        return "Vision Encoder"
    # ... existing categories ...
```

---

## 5. Compute Model Considerations

### 5.1 Vision Encoder Cost

The vision encoder is a standard ViT — its ops (MATMUL, LAYER_NORM, SOFTMAX, GELU) are already handled by the existing GPU and NPU cost models. No new op types needed.

Key cost characteristics:
- **Patch embedding**: One MATMUL of shape `[B*P, C*p*p] × [C*p*p, D_v]`. At 336px/p=14: `[576, 588] × [588, 1024]` — small, memory-bound.
- **ViT self-attention**: `[B*H, P, P]` attention matrix. At 336px: P=576, manageable. At 896px with 4 tiles: P=4096+, becomes the bottleneck due to quadratic scaling.
- **ViT FFN**: `[B*P, D_v] × [D_v, 4*D_v]` — standard MATMUL, compute-bound for large P.

### 5.2 Prefill Sequence Length

For a VLM request with text prompt of length T and N images producing V vision tokens total:

```
effective_prefill_seq = V + T
```

This is the sequence length fed to the LLM backbone. The simulator uses this directly — no LLM code changes needed for sequence-concatenation models.

### 5.3 TTFT Breakdown

```
TTFT = vision_encoder_time + projector_time + llm_prefill_time(V + T)
```

The HTML report should show this breakdown. The vision encoder and projector are purely prefill — they do not participate in decode.

### 5.4 Decode Phase

Decode is **unchanged** for sequence-concatenation models: only the new text token is generated, attending to the full KV cache (which includes cached vision token KVs). The KV cache size increases by V entries, affecting memory but not per-token decode latency beyond the KV attention cost already modeled.

### 5.5 Multi-Image and Video

For N images with token counts V_1, V_2, ..., V_N:

```
total_vision_tokens = sum(V_i)
effective_prefill_seq = sum(V_i) + T
```

Video is modeled as a sequence of image frames, optionally with temporal compression (e.g., Qwen2.5-VL `temporal_patch_size=2` halves frame count).

---

## 6. Cross-Attention Models (Phase 2)

For models like Llama 3.2-Vision where vision tokens are not concatenated but accessed via cross-attention in specific LLM layers:

- Add a `cross_attn_layer_indices` field to `VisionConfig` (e.g., every 4th layer).
- Modify the LLM handler's `build_layer` to conditionally insert a cross-attention block: `cross_attn(Q=text_tokens, KV=vision_tokens)` before or after self-attention.
- The cross-attention MATMUL shape is `[B*H, T, V]` for QK and `[B*H, T, D]` for output — different from self-attention's `[B*H, T, T]`.

This is a contained extension to `build_layer` and does not affect the rest of the pipeline.

---

## 7. Implementation Plan

### Phase 1 — Core VLM support (sequence concatenation)

| Step | Files Modified | Description |
|------|---------------|-------------|
| 1 | `config_normalizer.py` | Add `VisionConfig` dataclass, extend `ModelConfig`, add `_normalize_vision_config()` |
| 2 | `graph_builder.py` | Add `vision_encoder()` and `vision_projector()` composite methods |
| 3 | `config_extractor.py` | Add `VLMHandler` base class, `LLaVAHandler`, `InternVLHandler`, etc. |
| 4 | `operator.py` | Add `CONV2D_DEPTHWISE` OpType if needed for CNN-based projectors (likely not needed — patch embed is a linear) |
| 5 | `categories.py` | Add "Vision Encoder" and "Vision Projector" categories |
| 6 | `html_report.py` | Add vision stage to architecture overview, show TTFT breakdown |
| 7 | `tests/test_multimodal.py` | Tests for VisionConfig normalization, graph construction, token counting |

**Target models for Phase 1**: LLaVA-1.5, LLaVA-NeXT, InternVL2.5, Qwen2.5-VL, DeepSeek-VL2, Gemma3, Pixtral.

### Phase 2 — Cross-attention and advanced projectors

| Step | Files Modified | Description |
|------|---------------|-------------|
| 1 | `graph_builder.py` | Add `cross_attention()` method (Q from text, KV from vision) |
| 2 | `config_extractor.py` | Add `CrossAttnVLMHandler` with modified `build_layer` |
| 3 | `graph_builder.py` | Full Q-Former and Perceiver Resampler graph construction |
| 4 | `config_normalizer.py` | Parse cross-attention config (layer indices, gating) |

**Target models**: Llama 3.2-Vision, Flamingo/IDEFICS.

### Phase 3 — Video and audio

| Step | Files Modified | Description |
|------|---------------|-------------|
| 1 | `config_normalizer.py` | Add `VideoConfig` (frame sampling, temporal compression) |
| 2 | `graph_builder.py` | Add 3D patch embedding for video (Conv3D equivalent) |
| 3 | `config_normalizer.py` | Add `AudioConfig` (Whisper-style encoder) |

### Phase 4 — Multi-modal serving simulation

| Step | Files Modified | Description |
|------|---------------|-------------|
| 1 | `serving/` | Extend request model with `MultiModalInput`, variable prefill cost per request |
| 2 | `serving/` | Vision encoder scheduling (batch across requests, pipeline with LLM) |

---

## 8. API Usage Examples

### Basic VLM simulation

```python
from xpu_simulator import ConfigExtractor, MultiModalInput, ImageInput

ext = ConfigExtractor()
graph = ext.extract(
    "path/to/llava-next-config.json",
    batch_size=1,
    seq_len=128,  # text tokens
    mm_input=MultiModalInput(
        num_text_tokens=128,
        images=[ImageInput(width=672, height=672, num_tiles=4)],
    ),
)
# graph now includes: vision encoder (CLIP ViT-L, 4 tiles) +
#                      MLP projector +
#                      LLM prefill over (4*576 + 128) tokens
```

### Comparing vision encoder cost vs LLM prefill

```python
from xpu_simulator import PerformanceEvaluator
from xpu_simulator.backends.gpu import GPUCostModel, gpu_specs

cost = GPUCostModel(gpu_specs["H100"])
evaluator = PerformanceEvaluator(cost)
result = evaluator.run(graph, overlap=True)

# Per-category breakdown
for cat, us in result.category_latency.items():
    print(f"{cat}: {us:.0f} us")
# Vision Encoder: 1234 us
# Vision Projector: 12 us
# Attention Projections: 5678 us
# ...
```

### Resolution sweep

```python
# Sweep image resolution to find TTFT vs quality tradeoff
for tiles in [1, 4, 9, 12]:
    graph = ext.extract(config, batch_size=1, seq_len=128,
                        mm_input=MultiModalInput(
                            images=[ImageInput(num_tiles=tiles)]))
    result = evaluator.run(graph, overlap=True)
    print(f"tiles={tiles}: TTFT={result.ttft_ms:.1f}ms, "
          f"vision_tokens={...}")
```

---

## 9. References

### Model Architecture Papers
- Liu et al. "Visual Instruction Tuning" (LLaVA), NeurIPS 2023
- Liu et al. "LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge", 2024
- Chen et al. "InternVL: Scaling up Vision Foundation Models", CVPR 2024
- Chen et al. "Expanding Performance Boundaries of Open-Source Multimodal Models with InternVL2.5", arXiv 2412.05271, 2024
- Wu et al. "DeepSeek-VL2: Mixture-of-Experts Vision-Language Models", arXiv 2412.10302, 2024
- Agrawal et al. "Pixtral 12B", arXiv 2410.07073, 2024
- Li et al. "CogVLM2: Visual Language Models for Image and Video Understanding", arXiv 2408.16500, 2024
- Wang et al. "Qwen2.5-VL Technical Report", 2025

### Surveys
- "A Survey of Multimodal Models on Language and Vision", sciltp.com/journals/dmml, 2025
- "A Systematic Review of Vision Language Models", ScienceDirect, PRISMA, 2025
- "Efficient Multimodal Large Language Models: A Survey", Visual Intelligence / Springer, 2025
- "Small Vision-Language Models: A Survey on Compact Architectures", arXiv 2503.10665, 2025

### Efficiency and Performance
- Chen et al. "FastVLM: Efficient Vision Encoding for Vision Language Models", Apple, CVPR 2025
- Cha et al. "Honeybee: Locality-Enhanced Projector for Multimodal LLM", CVPR 2024
- Cao et al. "MADTP: Multimodal Alignment-Guided Dynamic Token Pruning", CVPR 2024
- Yang et al. "TopV: Compatible Token Pruning with Inference Time Optimization", CVPR 2025
- Tao et al. "DyCoke: Dynamic Compression of Tokens for Fast Video LLMs", CVPR 2025
- "Nova: Real-Time Agentic VLM Serving with Adaptive Cross-Stage Parallelization", arXiv, 2025

### Design Resources
- "VLM Design Choices in 2024", HuggingFace Blog (gigant)
- "Awesome VLM Architectures" comparison table, github.com/gokayfem/awesome-vlm-architectures

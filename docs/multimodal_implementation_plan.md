# Multi-Modal Implementation Plan

**Date**: 2026-04-16
**Ref**: docs/multimodal_design.md

---

## Phase 1 — Core VLM Support (Sequence Concatenation)

### Step 1: VisionConfig + normalizer (`config_normalizer.py`)
- Add `VisionConfig` dataclass with encoder params, projector config, resolution strategy
- Add `vision_config: VisionConfig | None = None` field to `ModelConfig`
- Add `is_multimodal` property to `ModelConfig`
- Add `_normalize_vision_config(raw_top, raw_vision)` function
- Handle: encoder_type mapping, projector_type inference, spatial_merge detection, resolution strategy detection

### Step 2: MultiModalInput (`core/operator.py`)
- Add `ImageInput` dataclass: width, height, num_tiles
- Add `MultiModalInput` dataclass: num_text_tokens, images list

### Step 3: GraphBuilder composites (`frontend/graph_builder.py`)
- `vision_encoder(prefix, vcfg, num_patches, B, prev)` — ViT subgraph: patch_embed MATMUL + N layers of (norm → MHA → norm → FFN)
- `vision_projector(prefix, vcfg, num_tokens, llm_hidden, B, prev)` — linear/MLP/pixel_shuffle_mlp projector

### Step 4: VLM handlers (`frontend/config_extractor.py`)
- `VLMHandler(ArchitectureHandler)` base: wraps LLM handler, builds vision stage, adjusts effective seq_len
- Concrete: `LLaVAHandler`, `InternVLHandler`, `DeepSeekVLHandler`, `Qwen2VLHandler`, `Gemma3VLHandler`, `PixtralHandler`
- Extend `ConfigExtractor.extract()` with `mm_input` parameter
- Register all VLM model_types

### Step 5: Categories + HTML report
- `categories.py`: Add "Vision Encoder" and "Vision Projector" categories with colors
- `html_report.py`: Add vision pipeline block to architecture overview, show TTFT breakdown (vision + projector + LLM)

### Step 6: Tests (`tests/test_multimodal.py`)
- VisionConfig normalization from real HF config patterns (LLaVA, InternVL, Qwen2.5-VL, DeepSeek-VL2)
- Token counting: fixed, anyres, dynamic_tile, native strategies
- Graph construction: vision_encoder node count, vision_projector variants
- End-to-end: ConfigExtractor with mm_input produces valid graph
- FLOP sanity: vision encoder FLOPs match manual calculation

## Phase 2 — Cross-Attention Models

### Step 1: Cross-attention in GraphBuilder
- `cross_attention(prefix, tokens_q, tokens_kv, dim, n_heads, head_dim, prev_q, prev_kv)` — Q from text, KV from vision

### Step 2: CrossAttnVLMHandler
- Modified `build_layer`: inserts cross-attention at specified layer indices
- VisionConfig gains `cross_attn_layer_indices` and `cross_attn_interval` fields

### Step 3: Tests for cross-attention graph structure

## Phase 3 — Video and Audio

### Step 1: VideoInput extension
- `VideoInput` dataclass: num_frames, fps, temporal_patch_size
- Temporal compression logic in token counting

### Step 2: AudioInput extension
- `AudioInput` dataclass: duration_seconds, sample_rate
- Whisper-style encoder as vision_encoder variant

## Phase 4 — Serving Integration

### Step 1: Extend serving request model with MultiModalInput
### Step 2: Variable prefill cost per request based on image count/resolution

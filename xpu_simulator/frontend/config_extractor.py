"""ConfigExtractor — build compute graphs from HuggingFace config.json files."""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional, Union

from ..core.graph import ComputeGraph, Node
from ..core.operator import OpType, Dtype, Phase, MultiModalInput, ImageInput
from ..core.parallel import ParallelConfig
from .base import GraphExtractor
from .graph_builder import GraphBuilder
from .config_normalizer import ModelConfig, VisionConfig, normalize_config


# ------------------------------------------------------------------ #
# Architecture handlers
# ------------------------------------------------------------------ #

class ArchitectureHandler(ABC):
    """Defines how to build one transformer layer for an architecture."""

    @abstractmethod
    def build_layer(self, builder: GraphBuilder, cfg: ModelConfig,
                    layer_id: int, tokens: int, B: int, S: int,
                    prev: Node) -> Node:
        """Build one transformer layer and return the output node."""
        ...


class StandardTransformerHandler(ArchitectureHandler):
    """Handles LLaMA, Mistral, Qwen2, Mixtral, GPT-2, Falcon, GPT-NeoX.

    Layer structure: norm -> attention -> norm -> FFN
    Falcon parallel_attn: norm -> (attention + FFN) -> add
    """

    def build_layer(self, builder: GraphBuilder, cfg: ModelConfig,
                    layer_id: int, tokens: int, B: int, S: int,
                    prev: Node) -> Node:
        prefix = f"L{layer_id}"

        if cfg.parallel_attn:
            return self._build_parallel_layer(builder, cfg, prefix,
                                              tokens, B, S, prev)

        # Attention norm
        attn_input = builder.norm(f"{prefix}.attn_norm",
                                  (tokens, cfg.hidden_size), prev)

        # Attention
        attn_out = self._build_attention(builder, cfg, prefix, tokens,
                                         B, S, attn_input)

        # FFN norm
        ffn_input = builder.norm(f"{prefix}.ffn_norm",
                                 (tokens, cfg.hidden_size), attn_out)

        # FFN
        ffn_out = self._build_ffn(builder, cfg, prefix, layer_id, tokens,
                                  ffn_input)

        return ffn_out

    def _build_parallel_layer(self, builder: GraphBuilder, cfg: ModelConfig,
                              prefix: str, tokens: int, B: int, S: int,
                              prev: Node) -> Node:
        """Falcon-style parallel attention + FFN from shared norm."""
        norm = builder.norm(f"{prefix}.norm", (tokens, cfg.hidden_size), prev)
        attn_out = self._build_attention(builder, cfg, prefix, tokens,
                                         B, S, norm)
        ffn_out = self._build_ffn(builder, cfg, prefix, 0, tokens, norm)
        combine = builder.elementwise(f"{prefix}.add",
                                      (tokens, cfg.hidden_size),
                                      OpType.ADD, attn_out, ffn_out)
        return combine

    def _build_attention(self, builder: GraphBuilder, cfg: ModelConfig,
                         prefix: str, tokens: int, B: int, S: int,
                         prev: Node) -> Node:
        return builder.attention(
            f"{prefix}.attn", tokens, B, S,
            dim=cfg.hidden_size,
            n_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            pattern=cfg.attention_pattern,
            n_kv_heads=cfg.num_key_value_heads,
            rope=cfg.rope,
            prev=prev,
        )

    def _build_ffn(self, builder: GraphBuilder, cfg: ModelConfig,
                   prefix: str, layer_id: int, tokens: int,
                   prev: Node) -> Node:
        """Select and build the appropriate FFN type."""
        # MoE check — all layers are MoE for Mixtral-type models
        if cfg.is_moe:
            return builder.moe_layer(
                f"{prefix}.moe", tokens, cfg.hidden_size,
                inter_dim=cfg.moe_intermediate_size,
                n_experts=cfg.num_experts,
                n_activated=cfg.num_experts_per_tok or 2,
                n_shared=cfg.num_shared_experts or 0,
                shared_inter=cfg.shared_expert_intermediate_size or 0,
                prev=prev,
            )

        # Dense FFN — select by activation
        if cfg.hidden_act in ("silu", "swish"):
            return builder.swiglu_mlp(f"{prefix}.ffn", tokens,
                                      cfg.hidden_size, cfg.intermediate_size,
                                      prev)
        elif cfg.hidden_act == "gelu_new" or cfg.hidden_act == "geglu":
            return builder.geglu_mlp(f"{prefix}.ffn", tokens,
                                     cfg.hidden_size, cfg.intermediate_size,
                                     prev)
        else:
            # Standard dense FFN (gelu, relu, etc.)
            act_map = {
                "gelu": OpType.GELU,
                "gelu_python": OpType.GELU,
                "gelu_pytorch_tanh": OpType.GELU,
                "relu": OpType.RELU,
            }
            act = act_map.get(cfg.hidden_act, OpType.GELU)
            return builder.dense_ffn(f"{prefix}.ffn", tokens,
                                     cfg.hidden_size, cfg.intermediate_size,
                                     activation=act, prev=prev)


class DeepSeekHandler(ArchitectureHandler):
    """Handles DeepSeek-V2/V3: MLA attention + mixed dense/MoE layers."""

    def build_layer(self, builder: GraphBuilder, cfg: ModelConfig,
                    layer_id: int, tokens: int, B: int, S: int,
                    prev: Node) -> Node:
        prefix = f"L{layer_id}"

        # Attention norm
        attn_input = builder.norm(f"{prefix}.attn_norm",
                                  (tokens, cfg.hidden_size), prev)

        # MLA attention (dense or sparse, dispatched by attention_pattern)
        attn_out = builder.attention(
            f"{prefix}.attn", tokens, B, S,
            dim=cfg.hidden_size,
            n_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            pattern=cfg.attention_pattern,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_head_dim=cfg.qk_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            prev=attn_input,
        )

        # FFN norm
        ffn_input = builder.norm(f"{prefix}.ffn_norm",
                                 (tokens, cfg.hidden_size), attn_out)

        # Dense or MoE based on layer index
        first_k = cfg.first_k_dense_replace or 0
        if layer_id < first_k:
            # Dense SwiGLU
            ffn_out = builder.swiglu_mlp(f"{prefix}.ffn", tokens,
                                         cfg.hidden_size,
                                         cfg.intermediate_size, ffn_input)
        else:
            # MoE with shared experts
            ffn_out = builder.moe_layer(
                f"{prefix}.moe", tokens, cfg.hidden_size,
                inter_dim=cfg.moe_intermediate_size,
                n_experts=cfg.num_experts,
                n_activated=cfg.num_experts_per_tok or 8,
                n_shared=cfg.num_shared_experts or 0,
                shared_inter=cfg.shared_expert_intermediate_size or 0,
                prev=ffn_input,
            )

        return ffn_out


# ------------------------------------------------------------------ #
# VLM handlers (multi-modal)
# ------------------------------------------------------------------ #

class VLMHandler(ArchitectureHandler):
    """Base handler for vision-language models.

    Wraps an existing LLM handler and prepends vision encoder + projector.
    Subclasses can override resolution/tiling logic.
    """

    def __init__(self, llm_handler: ArchitectureHandler):
        self.llm_handler = llm_handler

    def build_layer(self, builder: GraphBuilder, cfg: ModelConfig,
                    layer_id: int, tokens: int, B: int, S: int,
                    prev: Node) -> Node:
        """Delegate LLM layer building to the wrapped handler."""
        return self.llm_handler.build_layer(builder, cfg, layer_id,
                                            tokens, B, S, prev)

    def build_vision_stage(self, builder: GraphBuilder, cfg: ModelConfig,
                           mm_input: MultiModalInput,
                           B: int) -> tuple[Node, int]:
        """Build vision encoder + projector. Returns (last_node, num_vision_tokens)."""
        vcfg = cfg.vision_config
        total_vision_tokens = 0
        last_node = None

        for i, img in enumerate(mm_input.images):
            num_patches = self._compute_patches(vcfg, img)

            enc_node = builder.vision_encoder(
                f"vision.img{i}", vcfg, num_patches, B, last_node)

            # Token count after compression
            if vcfg.projector_type == "pixel_shuffle_mlp":
                s = vcfg.spatial_merge_size
                proj_tokens = num_patches // max(s * s, 1)
            elif vcfg.projector_type in ("qformer", "perceiver"):
                proj_tokens = vcfg.num_query_tokens
            else:
                proj_tokens = num_patches

            last_node = builder.vision_projector(
                f"vision.img{i}", vcfg, proj_tokens,
                cfg.hidden_size, B, enc_node)

            total_vision_tokens += proj_tokens

        return last_node, total_vision_tokens

    def _compute_patches(self, vcfg: VisionConfig, img: ImageInput) -> int:
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


class Qwen2VLHandler(VLMHandler):
    """Qwen2-VL / Qwen2.5-VL: Custom ViT + spatial-merge MLP + Qwen2 LLM."""
    def __init__(self):
        super().__init__(StandardTransformerHandler())


class Gemma3VLHandler(VLMHandler):
    """Gemma3: SigLIP + linear proj + Gemma LLM."""
    def __init__(self):
        super().__init__(StandardTransformerHandler())


class PixtralHandler(VLMHandler):
    """Pixtral: Custom ViT (2D-RoPE) + MLP + Mistral LLM."""
    def __init__(self):
        super().__init__(StandardTransformerHandler())


class CrossAttnVLMHandler(VLMHandler):
    """Cross-attention VLMs (Llama 3.2-Vision, Flamingo).

    Vision tokens not concatenated; cross-attention injected at specific layers.
    """
    def __init__(self):
        super().__init__(StandardTransformerHandler())

    def build_layer(self, builder: GraphBuilder, cfg: ModelConfig,
                    layer_id: int, tokens: int, B: int, S: int,
                    prev: Node) -> Node:
        vcfg = cfg.vision_config
        prev = self.llm_handler.build_layer(builder, cfg, layer_id,
                                            tokens, B, S, prev)

        # Insert cross-attention at specified layer indices
        if vcfg and vcfg.cross_attn_layer_indices:
            if layer_id in vcfg.cross_attn_layer_indices:
                # Vision tokens stored in builder attrs by build_vision_stage
                n_vision = getattr(builder, '_vision_tokens', 0)
                if n_vision > 0:
                    head_dim = cfg.head_dim
                    prev = builder.cross_attention(
                        f"L{layer_id}", tokens, n_vision,
                        cfg.hidden_size, cfg.num_attention_heads,
                        head_dim, B, prev_q=prev)
        return prev


# ------------------------------------------------------------------ #
# ConfigExtractor
# ------------------------------------------------------------------ #

class ConfigExtractor(GraphExtractor):
    """Extract a compute graph from a HuggingFace config.json.

    Accepts a file path, dict, or transformers.PretrainedConfig object.
    Dispatches to a registered ArchitectureHandler based on model_type.

    Usage::

        ext = ConfigExtractor(dtype=Dtype.FP16)
        graph = ext.extract("config.json", batch_size=1, seq_len=1024)
    """

    _handlers: ClassVar[dict[str, type[ArchitectureHandler]]] = {}

    @classmethod
    def register_handler(cls, model_type: str,
                         handler_cls: type[ArchitectureHandler]):
        """Register an architecture handler for a model_type."""
        cls._handlers[model_type] = handler_cls

    @classmethod
    def supported_architectures(cls) -> list[str]:
        """Return list of supported model_type strings."""
        return list(cls._handlers.keys())

    def extract(self, config: Union[str, Path, dict],  # type: ignore[override]
                batch_size: int = 1, seq_len: int = 1024,
                graph_name: Optional[str] = None,
                parallel_config: Optional[ParallelConfig] = None,
                phase: str = "prefill",
                kv_seq_len: int = 0,
                fuse_attention: bool = False,
                model_kv_cache: bool = False,
                mm_input: Optional[MultiModalInput] = None) -> ComputeGraph:
        """Build a compute graph from a HuggingFace model config.

        Args:
            config: Path to config.json, a dict, or a PretrainedConfig object.
            batch_size: Batch size for the simulation.
            seq_len: Sequence length for the simulation.
            graph_name: Optional name for the graph.
            parallel_config: Multi-device parallelism configuration.
            phase: "prefill" or "decode".
            kv_seq_len: KV cache sequence length (decode phase).
            fuse_attention: Emit single ATTENTION_FUSED nodes.
            model_kv_cache: Emit KV cache plumbing nodes.
            mm_input: Multi-modal input (images, video, audio).
        """
        raw = self._load_config(config)
        cfg = normalize_config(raw)

        handler_cls = self._handlers.get(cfg.model_type)
        if handler_cls is None:
            supported = ", ".join(sorted(self._handlers.keys()))
            raise ValueError(
                f"Unsupported model_type '{cfg.model_type}'. "
                f"Supported: {supported}. "
                f"Use ConfigExtractor.register_handler() to add support."
            )
        handler = handler_cls()

        phase_enum = Phase.DECODE if phase == "decode" else Phase.PREFILL

        # Decode: 1 new token per sequence
        if phase_enum == Phase.DECODE:
            effective_seq = 1
            effective_kv = kv_seq_len or seq_len
        else:
            effective_seq = seq_len
            effective_kv = 0

        name = graph_name or f"{cfg.model_type}-{cfg.hidden_size}"
        builder = GraphBuilder(name, self.dtype, quant=cfg.quant_config,
                               parallel=parallel_config,
                               phase=phase_enum, kv_seq_len=effective_kv,
                               fuse_attention=fuse_attention,
                               model_kv_cache=model_kv_cache)

        # --- Multi-modal vision stage (prefill only) ---
        vision_tokens = 0
        if (mm_input and mm_input.images and cfg.is_multimodal
                and isinstance(handler, VLMHandler)
                and phase_enum == Phase.PREFILL):
            vision_node, vision_tokens = handler.build_vision_stage(
                builder, cfg, mm_input, batch_size)
            # Store for cross-attention handlers
            builder._vision_tokens = vision_tokens

        # Effective sequence includes vision tokens
        if vision_tokens > 0:
            effective_seq = effective_seq + vision_tokens

        tokens = batch_size * effective_seq

        # Embedding (text tokens only)
        text_seq = effective_seq - vision_tokens
        prev = builder.embedding("embedding", cfg.vocab_size,
                                 cfg.hidden_size, batch_size, text_seq)

        # Transformer layers
        for i in range(cfg.num_hidden_layers):
            prev = handler.build_layer(builder, cfg, i, tokens,
                                       batch_size, effective_seq, prev)

        # Final norm
        prev = builder.norm("final_norm", (tokens, cfg.hidden_size), prev)

        # LM head (not quantized — always uses builder dtype)
        prev = builder.matmul("lm_head", tokens, cfg.hidden_size,
                              cfg.vocab_size, prev)

        return builder.build()

    @staticmethod
    def _load_config(config) -> dict:
        """Load config from path, dict, or PretrainedConfig."""
        if isinstance(config, dict):
            return config
        if isinstance(config, (str, Path)):
            with open(config) as f:
                return json.load(f)
        # transformers.PretrainedConfig
        if hasattr(config, "to_dict"):
            return config.to_dict()
        raise TypeError(f"Unsupported config type: {type(config)}")


# ------------------------------------------------------------------ #
# Register built-in handlers
# ------------------------------------------------------------------ #

# Standard transformer architectures
for _model_type in ("llama", "mistral", "qwen2", "qwen3", "mixtral",
                     "gpt2", "falcon", "gpt_neox"):
    ConfigExtractor.register_handler(_model_type, StandardTransformerHandler)

# DeepSeek (MLA + mixed dense/MoE)
ConfigExtractor.register_handler("deepseek_v2", DeepSeekHandler)
ConfigExtractor.register_handler("deepseek_v3", DeepSeekHandler)

# Vision-Language Models (VLM)
for _model_type in ("llava", "llava_next", "llava_next_video"):
    ConfigExtractor.register_handler(_model_type, LLaVAHandler)
ConfigExtractor.register_handler("internvl_chat", InternVLHandler)
ConfigExtractor.register_handler("deepseek_vl_v2", DeepSeekVLHandler)
for _model_type in ("qwen2_5_vl", "qwen2_vl"):
    ConfigExtractor.register_handler(_model_type, Qwen2VLHandler)
ConfigExtractor.register_handler("gemma3", Gemma3VLHandler)
ConfigExtractor.register_handler("pixtral", PixtralHandler)

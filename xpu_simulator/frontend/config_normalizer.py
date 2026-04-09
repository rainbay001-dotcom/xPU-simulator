"""Normalize HuggingFace config.json into a canonical ModelConfig."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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

    # Falcon-specific
    parallel_attn: bool = False

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 0

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

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

    # --- Falcon-specific ---
    parallel_attn = raw.get("parallel_attn", False)

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
        parallel_attn=parallel_attn,
    )

"""Shared op categorization and color scheme for visualization and profiling."""
from __future__ import annotations

from typing import Callable


CATEGORY_COLORS = {
    "Attention Projections": "#4A90D9",   # blue
    "Attention Compute":     "#2E5EAA",   # dark blue
    "MoE Experts":           "#E8744F",   # orange
    "MoE Shared Expert":     "#F5A623",   # amber
    "MoE Gate":              "#D0021B",   # red
    "Dense FFN":             "#7ED321",   # green
    "Norms":                 "#9B9B9B",   # gray
    "RoPE":                  "#BD10E0",   # purple
    "Embedding":             "#50E3C2",   # teal
    "LM Head":               "#50E3C2",   # teal
    "Other":                 "#C0C0C0",   # light gray
}


def categorize_op(name: str) -> str:
    """Categorize an op by name into a layer type for visualization and profiling.

    This is the default categorization that works with standard transformer naming
    conventions (e.g., DeepSeek, LLaMA-style naming with .attn_, .moe., .ffn., etc.).
    """
    if ".attn_score" in name or ".attn_v" in name or ".attn_softmax" in name:
        return "Attention Compute"
    elif ".wq_" in name or ".wkv_" in name or ".wo" in name:
        return "Attention Projections"
    elif ".moe.experts" in name:
        return "MoE Experts"
    elif ".moe.shared" in name:
        return "MoE Shared Expert"
    elif ".moe.gate" in name:
        return "MoE Gate"
    elif ".ffn." in name:
        return "Dense FFN"
    elif "norm" in name:
        return "Norms"
    elif "rope" in name:
        return "RoPE"
    elif "embed" in name:
        return "Embedding"
    elif "lm_head" in name:
        return "LM Head"
    return "Other"

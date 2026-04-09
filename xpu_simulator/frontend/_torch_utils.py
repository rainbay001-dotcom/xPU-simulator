"""Shared utilities for PyTorch-based graph extractors."""
from __future__ import annotations

from typing import Optional
import torch

from ..core.operator import Dtype


def torch_dtype_to_dtype(td: torch.dtype) -> Dtype:
    """Convert a torch.dtype to the simulator's Dtype enum."""
    return {
        torch.float32: Dtype.FP32,
        torch.float16: Dtype.FP16,
        torch.bfloat16: Dtype.BF16,
        torch.int8: Dtype.INT8,
    }.get(td, Dtype.FP16)


def shape_from_meta(meta: dict) -> Optional[tuple[int, ...]]:
    """Try to extract tensor shape from FX node metadata."""
    val = meta.get("val")
    if val is None:
        val = meta.get("tensor_meta")
    if val is not None:
        if isinstance(val, torch.Tensor):
            return tuple(val.shape)
        if hasattr(val, "shape"):
            return tuple(val.shape)
    return None


def dtype_from_meta(meta: dict) -> Dtype:
    """Extract Dtype from FX node metadata."""
    val = meta.get("val")
    if val is None:
        val = meta.get("tensor_meta")
    if val is not None:
        if isinstance(val, torch.Tensor):
            return torch_dtype_to_dtype(val.dtype)
        if hasattr(val, "dtype"):
            return torch_dtype_to_dtype(val.dtype)
    return Dtype.FP16

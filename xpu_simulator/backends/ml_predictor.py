"""ML-based execution time prediction using Random Forest models.

Optional dependency on scikit-learn. Falls back to analytical model when
sklearn is not installed or when no training data is available for an op category.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.cost_model import CostModel, OpCost
from ..core.hardware import HardwareSpec
from ..core.operator import OpSpec, OpType

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    import joblib
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Op categories for separate RF models
_CATEGORY_MAP: dict[str, set[OpType]] = {
    "matmul": {OpType.MATMUL, OpType.CONV2D},
    "attention": {OpType.ATTENTION_FUSED},
    "comm": {OpType.ALL_REDUCE, OpType.ALL_GATHER, OpType.REDUCE_SCATTER,
             OpType.ALL_TO_ALL, OpType.PP_SEND_RECV},
    "elementwise": {OpType.RELU, OpType.GELU, OpType.SILU, OpType.ADD,
                    OpType.MUL, OpType.LAYER_NORM, OpType.SOFTMAX,
                    OpType.ROPE, OpType.DEQUANT},
}

# Reverse map: OpType -> category
_OP_TO_CATEGORY: dict[OpType, str] = {}
for _cat, _ops in _CATEGORY_MAP.items():
    for _op in _ops:
        _OP_TO_CATEGORY[_op] = _cat


class MLExecutionTimePredictor(CostModel):
    """ML-augmented cost model using Random Forest per op category.

    Wraps an analytical base model. When a trained RF model exists for an
    op's category, uses the RF prediction. Otherwise falls back to the
    base model.
    """

    def __init__(self, base: CostModel):
        super().__init__(base.hw)
        self.base = base
        self._models: dict = {}  # category -> trained model
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _categorize(op: OpSpec) -> Optional[str]:
        """Map an OpSpec to its RF model category."""
        return _OP_TO_CATEGORY.get(op.op_type)

    @staticmethod
    def _extract_features(op: OpSpec, category: str) -> list[float]:
        """Extract numerical features from an OpSpec for the given category."""
        if category == "matmul":
            return _features_matmul(op)
        elif category == "attention":
            return _features_attention(op)
        elif category == "comm":
            return _features_comm(op)
        elif category == "elementwise":
            return _features_elementwise(op)
        return []

    def train(self, samples: list[tuple[OpSpec, float]],
              n_estimators_list: list[int] | None = None,
              max_depth_list: list[int | None] | None = None,
              cv: int = 3) -> dict[str, float]:
        """Train RF models from (OpSpec, measured_latency_us) pairs.

        Args:
            samples: List of (op_spec, latency_us) tuples.
            n_estimators_list: Hyperparameter grid for n_estimators.
            max_depth_list: Hyperparameter grid for max_depth.
            cv: Cross-validation folds.

        Returns:
            Dict of category -> best R^2 score from GridSearchCV.
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for ML prediction. "
                "Install with: pip install xpu-simulator[ml]"
            )

        if n_estimators_list is None:
            n_estimators_list = [50, 100, 200]
        if max_depth_list is None:
            max_depth_list = [5, 10, None]

        # Group samples by category
        grouped: dict[str, list[tuple[list[float], float]]] = {}
        for op, latency in samples:
            cat = self._categorize(op)
            if cat is None:
                continue
            features = self._extract_features(op, cat)
            if not features:
                continue
            grouped.setdefault(cat, []).append((features, latency))

        scores = {}
        for cat, data in grouped.items():
            if len(data) < cv + 1:
                # Not enough samples for cross-validation
                continue

            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])

            param_grid = {
                "n_estimators": n_estimators_list,
                "max_depth": max_depth_list,
            }
            grid = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid, cv=min(cv, len(data)),
                scoring="r2", n_jobs=-1,
            )
            grid.fit(X, y)
            self._models[cat] = grid.best_estimator_
            scores[cat] = grid.best_score_

        return scores

    def train_from_db(self, db) -> dict[str, float]:
        """Train from a ProfilingDB instance.

        Parses shape_key format (e.g. 'matmul_2048x2048x2048_fp16')
        back into OpSpec-like features.
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for ML prediction. "
                "Install with: pip install xpu-simulator[ml]"
            )

        grouped: dict[str, list[tuple[list[float], float]]] = {}
        for key, latency in db._data.items():
            parts = key.split("_")
            if len(parts) < 3:
                continue
            op_type_str = parts[0]
            dims_str = parts[1]
            dtype_str = parts[2]

            dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1,
                           "fp8": 1, "int4": 0.5}.get(dtype_str, 2)

            if op_type_str == "matmul":
                dim_parts = dims_str.split("x")
                if len(dim_parts) == 3:
                    M, K, N = [float(d) for d in dim_parts]
                    features = [M, K, N, dtype_bytes]
                    grouped.setdefault("matmul", []).append((features, latency))
            elif op_type_str in ("relu", "gelu", "silu", "add", "mul",
                                  "layer_norm", "softmax", "rope", "dequant"):
                dim_parts = dims_str.split("x")
                numel = 1.0
                for d in dim_parts:
                    numel *= float(d)
                features = [numel, dtype_bytes, len(dim_parts)]
                grouped.setdefault("elementwise", []).append((features, latency))
            elif op_type_str in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all"):
                dim_parts = dims_str.split("x")
                numel = 1.0
                for d in dim_parts:
                    numel *= float(d)
                msg_bytes = numel * dtype_bytes
                features = [msg_bytes, 1.0]  # n_ranks unknown from key
                grouped.setdefault("comm", []).append((features, latency))

        # Train on grouped data
        scores = {}
        for cat, data in grouped.items():
            if len(data) < 4:
                continue
            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])

            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42)
            model.fit(X, y)
            self._models[cat] = model
            # Simple R^2 on training data (no CV for DB-trained models)
            scores[cat] = model.score(X, y)

        return scores

    def estimate(self, op: OpSpec) -> OpCost:
        """Predict latency with RF if available, else fall back to base model."""
        category = self._categorize(op)
        if category and category in self._models and HAS_SKLEARN:
            features = self._extract_features(op, category)
            if features:
                try:
                    predicted_us = float(
                        self._models[category].predict(
                            np.array([features])
                        )[0]
                    )
                    self._hits += 1
                    analytical = self.base.estimate(op)
                    return OpCost(
                        compute_us=analytical.compute_us,
                        memory_us=analytical.memory_us,
                        latency_us=max(predicted_us, 0.0),
                        bound=analytical.bound,
                        flops=analytical.flops,
                        bytes_accessed=analytical.bytes_accessed,
                        utilization=analytical.utilization,
                    )
                except Exception:
                    pass  # Fall through to base model
        self._misses += 1
        return self.base.estimate(op)

    def save(self, path: str | Path) -> None:
        """Save trained models to disk."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required to save models")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._models, path)

    def load(self, path: str | Path) -> None:
        """Load trained models from disk."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required to load models")
        path = Path(path)
        self._models = joblib.load(path)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def categories_trained(self) -> list[str]:
        return list(self._models.keys())


# ------------------------------------------------------------------ #
# Feature extraction helpers
# ------------------------------------------------------------------ #

def _features_matmul(op: OpSpec) -> list[float]:
    """Extract [M, K, N, dtype_bytes] from a matmul op."""
    if len(op.inputs) < 2:
        return []
    a, b = op.inputs[0], op.inputs[1]
    dtype_bytes = a.dtype.bytes
    if len(a.shape) >= 2 and len(b.shape) >= 2:
        M, K = float(a.shape[-2]), float(a.shape[-1])
        N = float(b.shape[-1])
        batch = float(math.prod(a.shape[:-2])) if len(a.shape) > 2 else 1.0
        return [M * batch, K, N, dtype_bytes]
    return []


def _features_attention(op: OpSpec) -> list[float]:
    """Extract attention features from a fused attention op."""
    attrs = op.attrs
    tokens = float(attrs.get("tokens", 0))
    n_heads = float(attrs.get("n_heads", 0))
    head_dim = float(attrs.get("head_dim", 0))
    is_decode = 1.0 if attrs.get("is_decode", False) else 0.0
    kv_len = float(attrs.get("kv_len", 0))
    if tokens == 0 and op.inputs:
        tokens = float(op.inputs[0].shape[0]) if len(op.inputs[0].shape) > 0 else 0.0
    return [tokens, n_heads, head_dim, is_decode, kv_len]


def _features_comm(op: OpSpec) -> list[float]:
    """Extract [msg_bytes, n_ranks] from a comm op."""
    msg_bytes = float(op.memory_bytes)
    n_ranks = float(op.attrs.get("n_ranks", 1))
    return [msg_bytes, n_ranks]


def _features_elementwise(op: OpSpec) -> list[float]:
    """Extract [numel, dtype_bytes, n_dims] from an elementwise op."""
    if not op.inputs:
        return []
    inp = op.inputs[0]
    return [float(inp.numel), inp.dtype.bytes, float(len(inp.shape))]

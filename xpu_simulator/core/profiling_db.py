"""Profiling database — stores measured op latencies for calibration."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .operator import OpSpec, OpType


def shape_key(op: OpSpec) -> str:
    """Generate a canonical key for an op shape.

    Format: ``<op_type>_<shape_dims>_<dtype>``

    Examples:
        ``matmul_2048x2048x2048_fp16``
        ``layernorm_1x1024_fp16``
    """
    dtype = op.inputs[0].dtype.value if op.inputs else "unknown"

    if op.op_type == OpType.MATMUL and len(op.inputs) >= 2:
        M, K = op.inputs[0].shape
        _, N = op.inputs[1].shape
        dims = f"{M}x{K}x{N}"
    else:
        # Use first input shape
        if op.inputs:
            dims = "x".join(str(d) for d in op.inputs[0].shape)
        else:
            dims = "scalar"

    return f"{op.op_type.name.lower()}_{dims}_{dtype}"


class ProfilingDB:
    """In-memory database of measured operator latencies.

    Stores (shape_key → latency_us) mappings. Supports CSV persistence.
    """

    def __init__(self):
        self._data: dict[str, float] = {}

    def store(self, op: OpSpec, latency_us: float) -> None:
        """Store a measured latency for an op."""
        key = shape_key(op)
        self._data[key] = latency_us

    def store_key(self, key: str, latency_us: float) -> None:
        """Store a measured latency by key directly."""
        self._data[key] = latency_us

    def lookup(self, op: OpSpec) -> Optional[float]:
        """Look up a measured latency. Returns None on miss."""
        key = shape_key(op)
        return self._data.get(key)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, op: OpSpec) -> bool:
        return shape_key(op) in self._data

    def save(self, path: str | Path) -> None:
        """Save database to CSV."""
        path = Path(path)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "latency_us"])
            for key, lat in sorted(self._data.items()):
                writer.writerow([key, lat])

    def load(self, path: str | Path) -> None:
        """Load database from CSV (merges with existing data)."""
        path = Path(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._data[row["key"]] = float(row["latency_us"])

"""Parallelism and interconnect specifications."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelConfig:
    """Multi-device parallelism configuration.

    tp_size: Tensor parallelism (shards weights across devices)
    dp_size: Data parallelism (replicates model, shards data)
    ep_size: Expert parallelism (shards MoE experts across devices)
    """
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1

    @property
    def world_size(self) -> int:
        return self.tp_size * self.dp_size * self.ep_size


@dataclass
class InterconnectSpec:
    """Interconnect topology between devices.

    bandwidth_GBs: Unidirectional bandwidth in GB/s
    latency_us: Per-hop latency in microseconds
    """
    name: str
    bandwidth_GBs: float
    latency_us: float = 0.5

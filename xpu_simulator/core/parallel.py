"""Parallelism and interconnect specifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


@dataclass
class ParallelConfig:
    """Multi-device parallelism configuration.

    tp_size: Tensor parallelism (shards weights across devices)
    dp_size: Data parallelism (replicates model, shards data)
    ep_size: Expert parallelism (shards MoE experts across devices)
    pp_size: Pipeline parallelism (partitions layers across stages)
    vpp: Virtual pipeline parallelism chunks (interleaved schedule)
    gradient_accumulation: Micro-batch accumulation steps
    gpus_per_node: Devices per physical node (for interconnect scope)
    """
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    vpp: int = 1
    gradient_accumulation: int = 1
    gpus_per_node: int = 8

    @property
    def world_size(self) -> int:
        return self.tp_size * self.dp_size * self.ep_size * self.pp_size

    @property
    def bubble_fraction(self) -> float:
        """Pipeline bubble as fraction of ideal compute time.

        Formula from Megatron-LM: (pp_size - 1) / (gradient_accumulation * vpp)
        Returns 0.0 for pp_size=1.
        """
        if self.pp_size <= 1:
            return 0.0
        return min(1.0, (self.pp_size - 1) / (self.gradient_accumulation * self.vpp))


@dataclass
class InterconnectSpec:
    """Interconnect topology between devices.

    bandwidth_GBs: Unidirectional bandwidth in GB/s
    latency_us: Per-hop latency in microseconds
    """
    name: str
    bandwidth_GBs: float
    latency_us: float = 0.5


class InterconnectLevel(Enum):
    """Levels of the interconnect hierarchy."""
    NVLINK = auto()   # Intra-node GPU-to-GPU (NVLink / HCCS / xGMI)
    NIC = auto()      # Inter-node NIC-to-NIC (InfiniBand / RoCE)


@dataclass
class InterconnectTier:
    """One tier in a hierarchical interconnect."""
    level: InterconnectLevel
    bandwidth_GBs: float    # Unidirectional bus bandwidth in GB/s
    latency_us: float       # Per-hop latency in microseconds


@dataclass
class HierarchicalInterconnect:
    """Multi-level interconnect with intra-node and inter-node tiers.

    Automatically selects the appropriate tier based on whether
    a collective stays within one node or crosses nodes.
    """
    tiers: list[InterconnectTier] = field(default_factory=list)

    def get_interconnect(self, n_ranks: int, gpus_per_node: int) -> InterconnectSpec:
        """Select the effective InterconnectSpec for a collective.

        If all ranks fit within one node, use NVLINK tier.
        Otherwise, use the NIC tier (bottleneck).
        """
        if not self.tiers:
            raise ValueError("HierarchicalInterconnect has no tiers defined")

        if n_ranks <= gpus_per_node:
            # Intra-node: use NVLINK tier
            for tier in self.tiers:
                if tier.level == InterconnectLevel.NVLINK:
                    return InterconnectSpec(
                        name=f"{tier.level.name}",
                        bandwidth_GBs=tier.bandwidth_GBs,
                        latency_us=tier.latency_us,
                    )
        # Inter-node or no NVLINK tier: use NIC tier
        for tier in self.tiers:
            if tier.level == InterconnectLevel.NIC:
                return InterconnectSpec(
                    name=f"{tier.level.name}",
                    bandwidth_GBs=tier.bandwidth_GBs,
                    latency_us=tier.latency_us,
                )
        # Fallback: use first tier
        tier = self.tiers[0]
        return InterconnectSpec(
            name=tier.level.name,
            bandwidth_GBs=tier.bandwidth_GBs,
            latency_us=tier.latency_us,
        )

    @property
    def nvlink_bw(self) -> float:
        """Intra-node bandwidth in GB/s (0 if no NVLINK tier)."""
        for tier in self.tiers:
            if tier.level == InterconnectLevel.NVLINK:
                return tier.bandwidth_GBs
        return 0.0

    @property
    def nic_bw(self) -> float:
        """Inter-node bandwidth in GB/s (0 if no NIC tier)."""
        for tier in self.tiers:
            if tier.level == InterconnectLevel.NIC:
                return tier.bandwidth_GBs
        return 0.0


# --- Hardware presets ---

H100_INTERCONNECT = HierarchicalInterconnect(tiers=[
    InterconnectTier(InterconnectLevel.NVLINK, 370.8, 0.6),   # NVLink 4.0: 900 GB/s bidir / ~2.43 bus_bw factor
    InterconnectTier(InterconnectLevel.NIC, 48.5, 2.7),       # CX-7 400G IB: ~48.5 GB/s unidirectional
])

A100_INTERCONNECT = HierarchicalInterconnect(tiers=[
    InterconnectTier(InterconnectLevel.NVLINK, 240.0, 0.6),   # NVLink 3.0: 600 GB/s bidir
    InterconnectTier(InterconnectLevel.NIC, 24.0, 3.0),       # CX-6 200G IB
])

NPU_910B_INTERCONNECT = HierarchicalInterconnect(tiers=[
    InterconnectTier(InterconnectLevel.NVLINK, 392.0, 1.0),   # HCCS: 392 GB/s
    InterconnectTier(InterconnectLevel.NIC, 24.0, 3.0),       # RoCE 200G
])

NPU_910C_INTERCONNECT = HierarchicalInterconnect(tiers=[
    InterconnectTier(InterconnectLevel.NVLINK, 600.0, 1.0),   # HCCS: 600 GB/s
    InterconnectTier(InterconnectLevel.NIC, 48.5, 2.7),       # RoCE 400G
])

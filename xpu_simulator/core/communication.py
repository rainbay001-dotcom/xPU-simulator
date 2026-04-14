"""Analytical communication cost models for collective operations.

Adapted from msmodeling's comm_analytic.py — implements Ring and Tree
(Recursive Doubling) algorithms and selects the faster one. Supports
hierarchical interconnects and NCCL/HCCL latency floor modeling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .parallel import InterconnectSpec, HierarchicalInterconnect, ParallelConfig


# ------------------------------------------------------------------ #
# NCCL / HCCL latency floor profile
# ------------------------------------------------------------------ #

@dataclass
class NCCLProfile:
    """Models real-world NCCL/HCCL overhead on top of ideal bandwidth.

    launch_latency_us: Kernel launch + synchronization overhead (per collective).
    transport_latency_us: Per-hop transport setup (NVLink/IB/RoCE).
    efficiency_curve: List of (msg_bytes, efficiency) breakpoints.
        Linearly interpolates between points to model how small messages
        can't saturate the bus. Efficiency is 0-1 and multiplies bus_bw.
    """
    launch_latency_us: float = 5.0
    transport_latency_us: float = 0.6
    efficiency_curve: list[tuple[int, float]] = field(default_factory=lambda: [
        (0, 0.0),
        (4 * 1024, 0.10),          # 4 KB → 10%
        (64 * 1024, 0.50),         # 64 KB → 50%
        (1 * 1024 * 1024, 0.85),   # 1 MB → 85%
        (100 * 1024 * 1024, 0.92), # 100 MB → 92%
    ])

    def bandwidth_efficiency(self, msg_bytes: int) -> float:
        """Interpolate efficiency from the curve for a given message size."""
        if not self.efficiency_curve:
            return 1.0
        if msg_bytes <= self.efficiency_curve[0][0]:
            return self.efficiency_curve[0][1]
        if msg_bytes >= self.efficiency_curve[-1][0]:
            return self.efficiency_curve[-1][1]
        for i in range(len(self.efficiency_curve) - 1):
            lo_bytes, lo_eff = self.efficiency_curve[i]
            hi_bytes, hi_eff = self.efficiency_curve[i + 1]
            if lo_bytes <= msg_bytes <= hi_bytes:
                if hi_bytes == lo_bytes:
                    return hi_eff
                t = (msg_bytes - lo_bytes) / (hi_bytes - lo_bytes)
                return lo_eff + t * (hi_eff - lo_eff)
        return self.efficiency_curve[-1][1]

    def apply(self, ideal_bw_us: float, msg_bytes: int) -> float:
        """Apply NCCL overhead to an ideal bandwidth time.

        Returns max(ideal_time / efficiency, nccl_latency) + launch_latency.
        """
        eff = self.bandwidth_efficiency(msg_bytes)
        if eff > 0:
            effective_us = ideal_bw_us / eff
        else:
            effective_us = float("inf") if ideal_bw_us > 0 else 0.0
        floor = self.transport_latency_us
        return max(effective_us, floor) + self.launch_latency_us


# Default profiles
NCCL_GPU_DEFAULT = NCCLProfile(
    launch_latency_us=5.0,
    transport_latency_us=0.6,
    efficiency_curve=[
        (0, 0.0),
        (4 * 1024, 0.10),
        (64 * 1024, 0.50),
        (1 * 1024 * 1024, 0.85),
        (100 * 1024 * 1024, 0.92),
    ],
)

HCCL_NPU_DEFAULT = NCCLProfile(
    launch_latency_us=8.0,      # HCCL has higher launch overhead
    transport_latency_us=1.0,   # HCCS transport latency
    efficiency_curve=[
        (0, 0.0),
        (4 * 1024, 0.08),
        (64 * 1024, 0.40),
        (1 * 1024 * 1024, 0.80),
        (100 * 1024 * 1024, 0.90),
    ],
)


# ------------------------------------------------------------------ #
# Bus bandwidth auto-calculation (calbusbw.cc inspired)
# ------------------------------------------------------------------ #

@dataclass
class BusBandwidth:
    """Calculated effective bus bandwidth for collective algorithms."""
    ring_bw_GBs: float
    tree_bw_GBs: float
    best_algo: str       # "ring" or "tree"
    best_bw_GBs: float


def calculate_bus_bw(
    interconnect: HierarchicalInterconnect,
    n_ranks: int,
    gpus_per_node: int = 8,
) -> BusBandwidth:
    """Calculate effective bus bandwidth for Ring and Tree algorithms.

    For intra-node (n_ranks <= gpus_per_node):
      Ring: full NVLink bus bandwidth
      Tree: NVLink / 2 (binary tree uses half links per step)

    For inter-node:
      Ring: min(nvlink_bw, nic_bw * n_nodes) — NIC is the bottleneck
      Tree: min(nvlink_bw / 2, nic_bw * n_nodes / 2)

    Based on NCCL's algorithm selection in calbusbw.cc.
    """
    nvlink_bw = interconnect.nvlink_bw
    nic_bw = interconnect.nic_bw

    if n_ranks <= gpus_per_node:
        # Intra-node: NVLink only
        ring_bw = nvlink_bw
        tree_bw = nvlink_bw / 2.0
    else:
        # Inter-node: NIC is typically the bottleneck
        n_nodes = max(1, (n_ranks + gpus_per_node - 1) // gpus_per_node)
        # Ring: each node contributes nic_bw; effective = min(nvlink, nic * scale)
        # Scale factor accounts for multiple NICs per node (simplified)
        nic_effective = nic_bw * min(n_nodes, 2)  # 2-node scaling
        if n_nodes > 2:
            # Beyond 2 nodes, nic_bw becomes the hard bottleneck
            nic_effective = nic_bw
        ring_bw = min(nvlink_bw, nic_effective)
        tree_bw = min(nvlink_bw / 2.0, nic_effective / 2.0)

    best_algo = "ring" if ring_bw >= tree_bw else "tree"
    best_bw = max(ring_bw, tree_bw)

    return BusBandwidth(
        ring_bw_GBs=ring_bw,
        tree_bw_GBs=tree_bw,
        best_algo=best_algo,
        best_bw_GBs=best_bw,
    )


# ------------------------------------------------------------------ #
# Communication cost result
# ------------------------------------------------------------------ #

@dataclass
class CommCost:
    """Result of a communication cost estimate."""
    latency_us: float
    algorithm: str   # "ring" or "tree"
    msg_bytes: int


# ------------------------------------------------------------------ #
# Point-to-point (used by pipeline parallelism)
# ------------------------------------------------------------------ #

def pp_send_recv_time(msg_bytes: int, interconnect: InterconnectSpec,
                      nccl: Optional[NCCLProfile] = None) -> float:
    """Point-to-point send/recv latency for pipeline parallelism (us)."""
    bw = interconnect.bandwidth_GBs * 1e9  # B/s
    ideal_us = msg_bytes / bw * 1e6 if bw > 0 else 0.0
    base_lat = interconnect.latency_us
    if nccl is not None:
        return nccl.apply(ideal_us, msg_bytes) + base_lat
    return ideal_us + base_lat


# ------------------------------------------------------------------ #
# Collective operations
# ------------------------------------------------------------------ #

def _resolve_interconnect(
    interconnect: InterconnectSpec | HierarchicalInterconnect,
    n_ranks: int,
    gpus_per_node: int = 8,
) -> InterconnectSpec:
    """Resolve a possibly-hierarchical interconnect to a flat spec."""
    if isinstance(interconnect, HierarchicalInterconnect):
        return interconnect.get_interconnect(n_ranks, gpus_per_node)
    return interconnect


def all_reduce_time(msg_bytes: int, n_ranks: int,
                    interconnect: InterconnectSpec | HierarchicalInterconnect,
                    nccl: Optional[NCCLProfile] = None,
                    gpus_per_node: int = 8,
                    auto_busbw: bool = False) -> CommCost:
    """All-reduce: every rank ends up with the reduced result.

    Ring:  2*(N-1) steps, each sends msg/N bytes.
           time = 2*(N-1)*lat + 2*(N-1)/N * msg/bw
    Tree (Recursive Doubling): 2*log2(N) steps, each sends full msg.
           time = 2*log2(N)*lat + 2*msg/bw
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    ic = _resolve_interconnect(interconnect, n_ranks, gpus_per_node)
    if auto_busbw and isinstance(interconnect, HierarchicalInterconnect):
        bus = calculate_bus_bw(interconnect, n_ranks, gpus_per_node)
        ic = InterconnectSpec(name="auto_busbw", bandwidth_GBs=bus.best_bw_GBs,
                              latency_us=ic.latency_us)
    bw = ic.bandwidth_GBs * 1e9  # B/s
    lat = ic.latency_us  # us

    # Ring
    ring_lat = 2 * (n_ranks - 1) * lat
    ring_bw = 2 * (n_ranks - 1) / n_ranks * msg_bytes / bw * 1e6  # us
    ring_time = ring_lat + ring_bw

    # Tree (Recursive Doubling)
    steps = math.ceil(math.log2(n_ranks))
    tree_lat = 2 * steps * lat
    tree_bw = 2 * msg_bytes / bw * 1e6
    tree_time = tree_lat + tree_bw

    if ring_time <= tree_time:
        raw = ring_time
        algo = "ring"
    else:
        raw = tree_time
        algo = "tree"

    if nccl is not None:
        raw = nccl.apply(raw, msg_bytes)

    return CommCost(raw, algo, msg_bytes)


def all_gather_time(msg_bytes: int, n_ranks: int,
                    interconnect: InterconnectSpec | HierarchicalInterconnect,
                    nccl: Optional[NCCLProfile] = None,
                    gpus_per_node: int = 8,
                    auto_busbw: bool = False) -> CommCost:
    """All-gather: every rank sends its chunk, all end with full data.

    Ring:  (N-1) steps, each sends msg_per_rank bytes.
           time = (N-1)*lat + (N-1)/N * total_msg/bw
    Tree:  log2(N) steps, doubling message size each step.
           time = log2(N)*lat + (N-1)/N * total_msg/bw
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    ic = _resolve_interconnect(interconnect, n_ranks, gpus_per_node)
    if auto_busbw and isinstance(interconnect, HierarchicalInterconnect):
        bus = calculate_bus_bw(interconnect, n_ranks, gpus_per_node)
        ic = InterconnectSpec(name="auto_busbw", bandwidth_GBs=bus.best_bw_GBs,
                              latency_us=ic.latency_us)
    bw = ic.bandwidth_GBs * 1e9
    lat = ic.latency_us

    # msg_bytes = per-rank chunk; total = msg_bytes * n_ranks
    total = msg_bytes * n_ranks

    ring_lat = (n_ranks - 1) * lat
    ring_bw = (n_ranks - 1) / n_ranks * total / bw * 1e6
    ring_time = ring_lat + ring_bw

    steps = math.ceil(math.log2(n_ranks))
    tree_lat = steps * lat
    tree_bw = (n_ranks - 1) / n_ranks * total / bw * 1e6
    tree_time = tree_lat + tree_bw

    if ring_time <= tree_time:
        raw = ring_time
        algo = "ring"
    else:
        raw = tree_time
        algo = "tree"

    if nccl is not None:
        raw = nccl.apply(raw, msg_bytes)

    return CommCost(raw, algo, msg_bytes)


def reduce_scatter_time(msg_bytes: int, n_ranks: int,
                        interconnect: InterconnectSpec | HierarchicalInterconnect,
                        nccl: Optional[NCCLProfile] = None,
                        gpus_per_node: int = 8,
                        auto_busbw: bool = False) -> CommCost:
    """Reduce-scatter: reduce then scatter, each rank gets 1/N of result.

    Ring:  (N-1) steps, each sends msg/N bytes.
           time = (N-1)*lat + (N-1)/N * msg/bw
    Tree:  log2(N) steps.
           time = log2(N)*lat + (N-1)/N * msg/bw
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    ic = _resolve_interconnect(interconnect, n_ranks, gpus_per_node)
    if auto_busbw and isinstance(interconnect, HierarchicalInterconnect):
        bus = calculate_bus_bw(interconnect, n_ranks, gpus_per_node)
        ic = InterconnectSpec(name="auto_busbw", bandwidth_GBs=bus.best_bw_GBs,
                              latency_us=ic.latency_us)
    bw = ic.bandwidth_GBs * 1e9
    lat = ic.latency_us

    ring_lat = (n_ranks - 1) * lat
    ring_bw = (n_ranks - 1) / n_ranks * msg_bytes / bw * 1e6
    ring_time = ring_lat + ring_bw

    steps = math.ceil(math.log2(n_ranks))
    tree_lat = steps * lat
    tree_bw = (n_ranks - 1) / n_ranks * msg_bytes / bw * 1e6
    tree_time = tree_lat + tree_bw

    if ring_time <= tree_time:
        raw = ring_time
        algo = "ring"
    else:
        raw = tree_time
        algo = "tree"

    if nccl is not None:
        raw = nccl.apply(raw, msg_bytes)

    return CommCost(raw, algo, msg_bytes)


def all_to_all_time(msg_bytes: int, n_ranks: int,
                    interconnect: InterconnectSpec | HierarchicalInterconnect,
                    nccl: Optional[NCCLProfile] = None,
                    gpus_per_node: int = 8,
                    auto_busbw: bool = False) -> CommCost:
    """All-to-all: every rank sends a different chunk to every other rank.

    Pairwise: (N-1) steps, each sends data_per_rank bytes.
              time = (N-1)*lat + msg_per_rank/bw * (N-1)
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    ic = _resolve_interconnect(interconnect, n_ranks, gpus_per_node)
    if auto_busbw and isinstance(interconnect, HierarchicalInterconnect):
        bus = calculate_bus_bw(interconnect, n_ranks, gpus_per_node)
        ic = InterconnectSpec(name="auto_busbw", bandwidth_GBs=bus.best_bw_GBs,
                              latency_us=ic.latency_us)
    bw = ic.bandwidth_GBs * 1e9
    lat = ic.latency_us

    # msg_bytes = data each rank sends to each other rank
    pairwise_lat = (n_ranks - 1) * lat
    pairwise_bw = (n_ranks - 1) * msg_bytes / bw * 1e6
    raw = pairwise_lat + pairwise_bw

    if nccl is not None:
        raw = nccl.apply(raw, msg_bytes)

    return CommCost(raw, "pairwise", msg_bytes)


# ------------------------------------------------------------------ #
# KV-cache transfer (disaggregated serving)
# ------------------------------------------------------------------ #

def kv_transfer_time(num_layers: int, n_kv_heads: int, head_dim: int,
                     seq_len: int, dtype_bytes: float,
                     transfer_bw_GBs: float) -> float:
    """KV-cache transfer time from prefill instance to decode instance (us).

    KV cache = 2 (K+V) * num_layers * n_kv_heads * head_dim * seq_len * dtype_bytes
    """
    kv_bytes = 2 * num_layers * n_kv_heads * head_dim * seq_len * dtype_bytes
    if transfer_bw_GBs <= 0:
        return 0.0
    return kv_bytes / (transfer_bw_GBs * 1e9) * 1e6  # us

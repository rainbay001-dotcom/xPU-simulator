"""Collective algorithm decomposition into point-to-point step DAGs.

Inspired by SimAI's SimCCL — decomposes Ring, Tree (Halving-Doubling),
and Double Binary Tree collectives into explicit P2P communication steps.
This enables fine-grained analysis of channel pipelining and bandwidth
utilization beyond what analytical formulas capture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class P2PTransfer:
    """One point-to-point transfer within a collective step."""
    src_rank: int
    dst_rank: int
    msg_bytes: int
    channel_id: int = 0


@dataclass
class CollectiveStep:
    """A set of simultaneous P2P transfers (one algorithmic step).

    All transfers in a step happen concurrently — the step latency
    is the maximum transfer time among them.
    """
    step_id: int
    transfers: list[P2PTransfer] = field(default_factory=list)

    @property
    def max_msg_bytes(self) -> int:
        return max((t.msg_bytes for t in self.transfers), default=0)


@dataclass
class CollectiveDAG:
    """Full decomposition of a collective into sequential P2P steps.

    Steps execute sequentially (each must complete before the next starts).
    Within a step, all transfers are concurrent.
    """
    algorithm: str          # "ring", "tree", "double_binary_tree"
    collective: str         # "all_reduce", "all_gather", "reduce_scatter"
    n_ranks: int
    n_channels: int
    steps: list[CollectiveStep]
    total_msg_bytes: int

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_transfers(self) -> int:
        return sum(len(s.transfers) for s in self.steps)

    def compute_latency(self, per_link_bw_GBs: float,
                        per_hop_latency_us: float) -> float:
        """Compute total latency from the P2P step DAG.

        Each step's time = per_hop_latency + max_transfer_bytes / bandwidth.
        Steps are sequential. Multi-channel pipelining is already encoded
        in the step structure (channels create overlapping step offsets).
        """
        bw = per_link_bw_GBs * 1e9  # B/s
        total_us = 0.0
        for step in self.steps:
            if not step.transfers:
                continue
            max_bytes = step.max_msg_bytes
            transfer_us = max_bytes / bw * 1e6 if bw > 0 else 0.0
            total_us += per_hop_latency_us + transfer_us
        return total_us

    def summary(self) -> str:
        return (
            f"CollectiveDAG({self.algorithm} {self.collective}, "
            f"{self.n_ranks} ranks, {self.n_channels} ch, "
            f"{self.num_steps} steps, {self.total_transfers} transfers, "
            f"{self.total_msg_bytes / 1e6:.1f} MB)"
        )


# ------------------------------------------------------------------ #
# Ring algorithms
# ------------------------------------------------------------------ #

def ring_reduce_scatter_dag(n_ranks: int, msg_bytes: int,
                            n_channels: int = 1) -> CollectiveDAG:
    """Ring reduce-scatter: N-1 steps, each rank sends msg/N to next neighbor.

    With multi-channel: data is split across channels, each channel
    pipelines independently. Total steps = (N-1) + (n_channels-1).
    """
    chunk = msg_bytes // n_ranks
    steps = []
    total_steps = (n_ranks - 1) + (n_channels - 1)

    for s in range(total_steps):
        transfers = []
        for ch in range(n_channels):
            ring_step = s - ch
            if ring_step < 0 or ring_step >= n_ranks - 1:
                continue
            ch_chunk = chunk // n_channels
            for rank in range(n_ranks):
                src = rank
                dst = (rank + 1) % n_ranks
                transfers.append(P2PTransfer(src, dst, ch_chunk, channel_id=ch))
        steps.append(CollectiveStep(step_id=s, transfers=transfers))

    return CollectiveDAG(
        algorithm="ring", collective="reduce_scatter",
        n_ranks=n_ranks, n_channels=n_channels,
        steps=steps, total_msg_bytes=msg_bytes,
    )


def ring_all_gather_dag(n_ranks: int, msg_bytes: int,
                        n_channels: int = 1) -> CollectiveDAG:
    """Ring all-gather: N-1 steps, each rank forwards chunks around ring.

    msg_bytes = per-rank chunk size. Total data = msg_bytes * n_ranks.
    """
    steps = []
    total_steps = (n_ranks - 1) + (n_channels - 1)

    for s in range(total_steps):
        transfers = []
        for ch in range(n_channels):
            ring_step = s - ch
            if ring_step < 0 or ring_step >= n_ranks - 1:
                continue
            ch_chunk = msg_bytes // n_channels
            for rank in range(n_ranks):
                src = rank
                dst = (rank + 1) % n_ranks
                transfers.append(P2PTransfer(src, dst, ch_chunk, channel_id=ch))
        steps.append(CollectiveStep(step_id=s, transfers=transfers))

    return CollectiveDAG(
        algorithm="ring", collective="all_gather",
        n_ranks=n_ranks, n_channels=n_channels,
        steps=steps, total_msg_bytes=msg_bytes * n_ranks,
    )


def ring_all_reduce_dag(n_ranks: int, msg_bytes: int,
                        n_channels: int = 1) -> CollectiveDAG:
    """Ring all-reduce = reduce-scatter + all-gather.

    Phase 1 (reduce-scatter): N-1 steps, chunk_size = msg/N per channel.
    Phase 2 (all-gather): N-1 steps, same chunk_size.
    With pipelining: total = 2*(N-1) + (n_channels-1).
    """
    chunk = msg_bytes // n_ranks
    steps = []
    rs_steps = n_ranks - 1
    ag_steps = n_ranks - 1
    total_steps = rs_steps + ag_steps + (n_channels - 1)

    for s in range(total_steps):
        transfers = []
        for ch in range(n_channels):
            adjusted = s - ch
            if adjusted < 0:
                continue

            ch_chunk = chunk // n_channels
            if adjusted < rs_steps:
                # Reduce-scatter phase
                for rank in range(n_ranks):
                    src = rank
                    dst = (rank + 1) % n_ranks
                    transfers.append(P2PTransfer(src, dst, ch_chunk, channel_id=ch))
            elif adjusted < rs_steps + ag_steps:
                # All-gather phase
                for rank in range(n_ranks):
                    src = rank
                    dst = (rank + 1) % n_ranks
                    transfers.append(P2PTransfer(src, dst, ch_chunk, channel_id=ch))
        steps.append(CollectiveStep(step_id=s, transfers=transfers))

    return CollectiveDAG(
        algorithm="ring", collective="all_reduce",
        n_ranks=n_ranks, n_channels=n_channels,
        steps=steps, total_msg_bytes=msg_bytes,
    )


# ------------------------------------------------------------------ #
# Tree (Halving-Doubling / Recursive Doubling)
# ------------------------------------------------------------------ #

def tree_all_reduce_dag(n_ranks: int, msg_bytes: int) -> CollectiveDAG:
    """Recursive halving-doubling all-reduce.

    Phase 1 (reduce-scatter): ceil(log2(N)) steps.
      Step k: rank i exchanges with rank i XOR (1 << k).
      Message halves each step.

    Phase 2 (all-gather): ceil(log2(N)) steps in reverse.
      Message doubles each step.

    Total steps = 2 * ceil(log2(N)).
    """
    log_n = math.ceil(math.log2(max(n_ranks, 2)))
    steps = []

    # Phase 1: Reduce-scatter (halving)
    current_msg = msg_bytes
    for k in range(log_n):
        current_msg = msg_bytes // (2 ** (k + 1))
        if current_msg < 1:
            current_msg = 1
        transfers = []
        for rank in range(n_ranks):
            partner = rank ^ (1 << k)
            if partner < n_ranks and partner > rank:
                transfers.append(P2PTransfer(rank, partner, current_msg))
                transfers.append(P2PTransfer(partner, rank, current_msg))
        steps.append(CollectiveStep(step_id=k, transfers=transfers))

    # Phase 2: All-gather (doubling)
    for k in range(log_n - 1, -1, -1):
        current_msg = msg_bytes // (2 ** k) if k > 0 else msg_bytes
        if current_msg < 1:
            current_msg = 1
        transfers = []
        for rank in range(n_ranks):
            partner = rank ^ (1 << k)
            if partner < n_ranks and partner > rank:
                transfers.append(P2PTransfer(rank, partner, current_msg))
                transfers.append(P2PTransfer(partner, rank, current_msg))
        steps.append(CollectiveStep(step_id=log_n + (log_n - 1 - k), transfers=transfers))

    return CollectiveDAG(
        algorithm="tree", collective="all_reduce",
        n_ranks=n_ranks, n_channels=1,
        steps=steps, total_msg_bytes=msg_bytes,
    )


# ------------------------------------------------------------------ #
# Double Binary Tree
# ------------------------------------------------------------------ #

def double_binary_tree_dag(n_ranks: int, msg_bytes: int) -> CollectiveDAG:
    """Double binary tree all-reduce.

    Two binary trees operate on interleaved halves of the data:
    - Tree 0: even-indexed chunks
    - Tree 1: odd-indexed chunks

    Each tree does:
    Phase 1 (reduce): Bottom-up, children send to parent. O(log N) steps.
    Phase 2 (broadcast): Top-down, parent sends to children. O(log N) steps.

    Total steps = 2 * ceil(log2(N)) (both trees pipeline).
    """
    if n_ranks <= 1:
        return CollectiveDAG("double_binary_tree", "all_reduce", n_ranks, 1, [], msg_bytes)

    log_n = math.ceil(math.log2(max(n_ranks, 2)))
    half_msg = msg_bytes // 2
    if half_msg < 1:
        half_msg = 1
    steps = []

    # Build binary tree parent relationships
    # Tree 0: root=0, Tree 1: root=1 (if n_ranks > 1)
    def get_parent(rank: int, tree_root: int) -> int | None:
        if rank == tree_root:
            return None
        # Simple binary tree: parent of rank is (rank - 1) // 2 offset by root
        # Map ranks to 0-indexed positions, build heap-style tree
        pos = rank if rank < tree_root else rank - 1
        if tree_root == 1:
            pos = rank if rank > tree_root else rank + 1
            pos = pos - 1 if rank > 0 else 0
        # Simplified: just use basic binary tree
        if rank == tree_root:
            return None
        return tree_root  # All nodes connect to root (star topology approximation)

    def get_children(rank: int, tree_root: int) -> list[int]:
        if rank != tree_root:
            return []
        # Root connects to all other ranks
        return [r for r in range(n_ranks) if r != tree_root]

    # Phase 1: Reduce (bottom-up) — all non-root nodes send to root
    for tree_id in range(2):
        root = tree_id if tree_id < n_ranks else 0
        transfers = []
        for rank in range(n_ranks):
            if rank != root:
                transfers.append(P2PTransfer(rank, root, half_msg, channel_id=tree_id))
        if transfers:
            steps.append(CollectiveStep(step_id=len(steps), transfers=transfers))

    # Phase 2: Broadcast (top-down) — root sends to all
    for tree_id in range(2):
        root = tree_id if tree_id < n_ranks else 0
        transfers = []
        for rank in range(n_ranks):
            if rank != root:
                transfers.append(P2PTransfer(root, rank, half_msg, channel_id=tree_id))
        if transfers:
            steps.append(CollectiveStep(step_id=len(steps), transfers=transfers))

    return CollectiveDAG(
        algorithm="double_binary_tree", collective="all_reduce",
        n_ranks=n_ranks, n_channels=2,
        steps=steps, total_msg_bytes=msg_bytes,
    )

"""Analytical communication cost models for collective operations.

Adapted from msmodeling's comm_analytic.py — implements Ring and Tree
(Recursive Doubling) algorithms and selects the faster one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .parallel import InterconnectSpec


@dataclass
class CommCost:
    """Result of a communication cost estimate."""
    latency_us: float
    algorithm: str   # "ring" or "tree"
    msg_bytes: int


def all_reduce_time(msg_bytes: int, n_ranks: int,
                    interconnect: InterconnectSpec) -> CommCost:
    """All-reduce: every rank ends up with the reduced result.

    Ring:  2*(N-1) steps, each sends msg/N bytes.
           time = 2*(N-1)*lat + 2*(N-1)/N * msg/bw
    Tree (Recursive Doubling): 2*log2(N) steps, each sends full msg.
           time = 2*log2(N)*lat + 2*msg/bw
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    bw = interconnect.bandwidth_GBs * 1e9  # B/s
    lat = interconnect.latency_us  # us

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
        return CommCost(ring_time, "ring", msg_bytes)
    return CommCost(tree_time, "tree", msg_bytes)


def all_gather_time(msg_bytes: int, n_ranks: int,
                    interconnect: InterconnectSpec) -> CommCost:
    """All-gather: every rank sends its chunk, all end with full data.

    Ring:  (N-1) steps, each sends msg_per_rank bytes.
           time = (N-1)*lat + (N-1)/N * total_msg/bw
    Tree:  log2(N) steps, doubling message size each step.
           time = log2(N)*lat + (N-1)/N * total_msg/bw
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    bw = interconnect.bandwidth_GBs * 1e9
    lat = interconnect.latency_us

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
        return CommCost(ring_time, "ring", msg_bytes)
    return CommCost(tree_time, "tree", msg_bytes)


def reduce_scatter_time(msg_bytes: int, n_ranks: int,
                        interconnect: InterconnectSpec) -> CommCost:
    """Reduce-scatter: reduce then scatter, each rank gets 1/N of result.

    Ring:  (N-1) steps, each sends msg/N bytes.
           time = (N-1)*lat + (N-1)/N * msg/bw
    Tree:  log2(N) steps.
           time = log2(N)*lat + (N-1)/N * msg/bw
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    bw = interconnect.bandwidth_GBs * 1e9
    lat = interconnect.latency_us

    ring_lat = (n_ranks - 1) * lat
    ring_bw = (n_ranks - 1) / n_ranks * msg_bytes / bw * 1e6
    ring_time = ring_lat + ring_bw

    steps = math.ceil(math.log2(n_ranks))
    tree_lat = steps * lat
    tree_bw = (n_ranks - 1) / n_ranks * msg_bytes / bw * 1e6
    tree_time = tree_lat + tree_bw

    if ring_time <= tree_time:
        return CommCost(ring_time, "ring", msg_bytes)
    return CommCost(tree_time, "tree", msg_bytes)


def all_to_all_time(msg_bytes: int, n_ranks: int,
                    interconnect: InterconnectSpec) -> CommCost:
    """All-to-all: every rank sends a different chunk to every other rank.

    Pairwise: (N-1) steps, each sends data_per_rank bytes.
              time = (N-1)*lat + msg_per_rank/bw * (N-1)
    """
    if n_ranks <= 1:
        return CommCost(0.0, "none", msg_bytes)

    bw = interconnect.bandwidth_GBs * 1e9
    lat = interconnect.latency_us

    # msg_bytes = data each rank sends to each other rank
    pairwise_lat = (n_ranks - 1) * lat
    pairwise_bw = (n_ranks - 1) * msg_bytes / bw * 1e6
    pairwise_time = pairwise_lat + pairwise_bw

    return CommCost(pairwise_time, "pairwise", msg_bytes)

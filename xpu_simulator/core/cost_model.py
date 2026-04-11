"""Cost model abstractions and roofline implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .hardware import HardwareSpec
from .operator import OpSpec, OpType
from .parallel import InterconnectSpec, ParallelConfig

if TYPE_CHECKING:
    from .profiling_db import ProfilingDB


@dataclass
class OpCost:
    """Cost estimate for a single operation."""
    compute_us: float       # Compute-limited latency (microseconds)
    memory_us: float        # Memory-limited latency (microseconds)
    latency_us: float       # Actual estimated latency = max(compute, memory)
    bound: str              # "compute" or "memory"
    flops: int              # Total FLOPs
    bytes_accessed: int     # Total bytes read + written
    utilization: float      # Fraction of peak utilized (0-1)

    # Optional per-pipe microarchitectural breakdown (NPU / CA-model style)
    # Populated by backends that model pipe-level behavior. Values are the
    # cycles each hardware pipe is busy, expressed in microseconds. They
    # need not sum to latency_us — pipes overlap via double-buffering.
    mte2_us: float = 0.0     # GM/L2 → L1/UB transfer engine
    mte3_us: float = 0.0     # L1/UB → GM/L2 transfer engine
    vec_us: float = 0.0      # VECTOR unit compute (elementwise/reduction)
    cube_us: float = 0.0     # CUBE unit compute (matrix/conv)
    scalar_us: float = 0.0   # SCALAR unit (orchestration, address calc)

    @property
    def arithmetic_intensity(self) -> float:
        if self.bytes_accessed == 0:
            return float("inf")
        return self.flops / self.bytes_accessed


class CostModel(ABC):
    """Abstract base class for cost models."""

    def __init__(self, hw: HardwareSpec):
        self.hw = hw

    @abstractmethod
    def estimate(self, op: OpSpec) -> OpCost:
        """Estimate cost of executing an operation on this hardware."""
        ...


class RooflineCostModel(CostModel):
    """Basic roofline cost model — works for any hardware."""

    def estimate(self, op: OpSpec) -> OpCost:
        dtype = op.inputs[0].dtype.value if op.inputs else "fp16"
        flops = op.flops
        mem_bytes = op.memory_bytes

        peak = self.hw.peak_flops_for(dtype)
        bw = self.hw.effective_bandwidth(mem_bytes) * 1e9  # B/s

        compute_us = (flops / peak * 1e6) if peak > 0 else 0.0
        memory_us = (mem_bytes / bw * 1e6) if bw > 0 else 0.0
        latency_us = max(compute_us, memory_us)

        bound = "compute" if compute_us >= memory_us else "memory"

        # Utilization: how much of peak we'd use if running at this latency
        if latency_us > 0 and peak > 0:
            utilization = flops / (latency_us * 1e-6 * peak)
        else:
            utilization = 0.0

        return OpCost(
            compute_us=compute_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=min(utilization, 1.0),
        )


class CalibratedCostModel(CostModel):
    """Wraps a cost model and overrides with measured latencies when available.

    On hit: uses measured latency_us from ProfilingDB.
    On miss: falls back to analytical base model.
    """

    def __init__(self, base: CostModel, db: "ProfilingDB"):
        super().__init__(base.hw)
        self.base = base
        self.db = db
        self._hits = 0
        self._misses = 0

    def estimate(self, op: OpSpec) -> OpCost:
        measured = self.db.lookup(op)
        if measured is not None:
            self._hits += 1
            # Use measured latency but keep analytical breakdown for info
            analytical = self.base.estimate(op)
            return OpCost(
                compute_us=analytical.compute_us,
                memory_us=analytical.memory_us,
                latency_us=measured,
                bound=analytical.bound,
                flops=analytical.flops,
                bytes_accessed=analytical.bytes_accessed,
                utilization=analytical.utilization,
            )
        self._misses += 1
        return self.base.estimate(op)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def total_queries(self) -> int:
        return self._hits + self._misses


_COMM_OPS = {OpType.ALL_REDUCE, OpType.ALL_GATHER,
             OpType.REDUCE_SCATTER, OpType.ALL_TO_ALL}


class CommAwareCostModel(CostModel):
    """Wraps a compute cost model and handles communication ops via interconnect."""

    def __init__(self, base: CostModel, interconnect: InterconnectSpec,
                 parallel: ParallelConfig):
        super().__init__(base.hw)
        self.base = base
        self.interconnect = interconnect
        self.parallel = parallel

    def estimate(self, op: OpSpec) -> OpCost:
        if op.op_type in _COMM_OPS:
            return self._estimate_comm(op)
        return self.base.estimate(op)

    def _estimate_comm(self, op: OpSpec) -> OpCost:
        from .communication import (
            all_reduce_time, all_gather_time,
            reduce_scatter_time, all_to_all_time,
        )

        msg_bytes = op.memory_bytes
        n_ranks = op.attrs.get("n_ranks", self.parallel.tp_size)

        if op.op_type == OpType.ALL_REDUCE:
            cc = all_reduce_time(msg_bytes, n_ranks, self.interconnect)
        elif op.op_type == OpType.ALL_GATHER:
            cc = all_gather_time(msg_bytes, n_ranks, self.interconnect)
        elif op.op_type == OpType.REDUCE_SCATTER:
            cc = reduce_scatter_time(msg_bytes, n_ranks, self.interconnect)
        elif op.op_type == OpType.ALL_TO_ALL:
            cc = all_to_all_time(msg_bytes, n_ranks, self.interconnect)
        else:
            cc = None

        latency = cc.latency_us if cc else 0.0
        return OpCost(
            compute_us=0.0,
            memory_us=latency,
            latency_us=latency,
            bound="communication",
            flops=0,
            bytes_accessed=msg_bytes,
            utilization=0.0,
        )

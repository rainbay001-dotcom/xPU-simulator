"""GPU-specific cost model."""
from __future__ import annotations

from ...core.cost_model import RooflineCostModel, OpCost
from ...core.operator import OpSpec, OpType


class GPUCostModel(RooflineCostModel):
    """GPU cost model with roofline + GPU-specific adjustments."""

    def estimate(self, op: OpSpec) -> OpCost:
        cost = super().estimate(op)

        # Apply GPU-specific kernel launch overhead (~5us typical)
        if cost.latency_us > 0:
            launch_overhead_us = 5.0
            cost = OpCost(
                compute_us=cost.compute_us,
                memory_us=cost.memory_us,
                latency_us=cost.latency_us + launch_overhead_us,
                bound=cost.bound,
                flops=cost.flops,
                bytes_accessed=cost.bytes_accessed,
                utilization=cost.utilization,
            )

        return cost

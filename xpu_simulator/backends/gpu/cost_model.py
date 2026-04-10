"""GPU-specific cost model."""
from __future__ import annotations

from ...core.cost_model import CostModel, OpCost
from ...core.operator import OpSpec, OpType
from .hardware import GPUSpec


# Ops that use Tensor Cores
_TENSOR_CORE_OPS = {OpType.MATMUL, OpType.CONV2D}


class GPUCostModel(CostModel):
    """GPU cost model with Tensor Core vs CUDA Core distinction,
    memory-hierarchy-aware bandwidth, and efficiency factors."""

    hw: GPUSpec

    def __init__(self, hw: GPUSpec):
        super().__init__(hw)
        self.hw: GPUSpec = hw

    def estimate(self, op: OpSpec) -> OpCost:
        dtype = op.inputs[0].dtype.value if op.inputs else "fp16"
        flops = op.flops
        mem_bytes = op.memory_bytes

        # Select peak FLOPS: Tensor Core ops vs CUDA Core (SIMT) ops
        is_tc_op = op.op_type in _TENSOR_CORE_OPS
        if is_tc_op:
            peak = self.hw.peak_flops_for(dtype)
            compute_efficiency = self.hw.get_efficiency(f"matmul_{dtype}")
        else:
            peak = self.hw.cuda_core_flops_for(dtype)
            compute_efficiency = self.hw.get_efficiency(f"elementwise_{dtype}")

        # Apply compute efficiency
        effective_peak = peak * compute_efficiency

        # Memory-hierarchy-aware bandwidth
        bw_GBs = self.hw.effective_bandwidth(mem_bytes)
        mem_efficiency = self.hw.get_efficiency("memory")
        effective_bw = bw_GBs * mem_efficiency * 1e9  # -> B/s

        compute_us = (flops / effective_peak * 1e6) if effective_peak > 0 else 0.0
        memory_us = (mem_bytes / effective_bw * 1e6) if effective_bw > 0 else 0.0

        # Per-op static overhead: Tensor Core ops ~5us, CUDA Core ops ~2us
        if is_tc_op:
            static_overhead_us = self.hw.get_efficiency("static_tc_us")
        else:
            static_overhead_us = self.hw.get_efficiency("static_cuda_us")
        latency_us = max(compute_us, memory_us) + static_overhead_us
        bound = "compute" if compute_us >= memory_us else "memory"

        if latency_us > static_overhead_us and effective_peak > 0:
            utilization = flops / ((latency_us - static_overhead_us) * 1e-6 * effective_peak)
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

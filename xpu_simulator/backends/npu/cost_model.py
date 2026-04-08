"""Ascend NPU cost model with CUBE/VECTOR pipeline awareness."""
from __future__ import annotations

import math

from ...core.cost_model import CostModel, OpCost
from ...core.operator import OpSpec, OpType
from .hardware import AscendSpec


# Ops that run on the CUBE unit (matrix engine)
_CUBE_OPS = {OpType.MATMUL, OpType.CONV2D}

# Ops that run on the VECTOR unit
_VECTOR_OPS = {OpType.RELU, OpType.ADD, OpType.GELU, OpType.LAYER_NORM, OpType.SOFTMAX}

# Data layout preferences per op type
_LAYOUT_MAP = {
    OpType.MATMUL: "Fractal_NZ",
    OpType.CONV2D: "NC1HWC0",
    OpType.RELU: "NC1HWC0",
    OpType.ADD: "NC1HWC0",
    OpType.GELU: "ND",
    OpType.LAYER_NORM: "ND",
    OpType.SOFTMAX: "ND",
}


class NPUCostModel(CostModel):
    """Cost model for Ascend NPU with CUBE/VECTOR pipeline modeling."""

    hw: AscendSpec

    def __init__(self, hw: AscendSpec):
        super().__init__(hw)
        self.hw: AscendSpec = hw

    def estimate(self, op: OpSpec) -> OpCost:
        if op.op_type in _CUBE_OPS:
            return self._estimate_cube(op)
        elif op.op_type in _VECTOR_OPS:
            return self._estimate_vector(op)
        else:
            return self._estimate_fallback(op)

    def _estimate_cube(self, op: OpSpec) -> OpCost:
        """Estimate cost for CUBE (matrix) operations.

        CUBE operates on tiles of size (tile x tile). Misaligned shapes
        cause padding waste and reduced utilization.
        """
        dtype = op.inputs[0].dtype.value if op.inputs else "fp16"
        flops = op.flops
        mem_bytes = op.memory_bytes
        tile = self.hw.cube_tile_size

        # Compute CUBE utilization based on tile alignment
        utilization = self._cube_utilization(op, tile)

        # Effective peak = peak * utilization
        cube_peak = self.hw.cube_peak_for(dtype)
        effective_peak = cube_peak * utilization

        # GM bandwidth for data movement
        gm_bw = self.hw.main_memory_bandwidth() * 1e9  # B/s

        compute_us = (flops / effective_peak * 1e6) if effective_peak > 0 else 0.0
        memory_us = (mem_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0

        # Format conversion overhead (ND -> NZ or NC1HWC0)
        format_overhead_us = self._format_conversion_cost(op, dtype)

        latency_us = max(compute_us, memory_us) + format_overhead_us
        bound = "compute (PIPE_M)" if compute_us >= memory_us else "memory"

        return OpCost(
            compute_us=compute_us + format_overhead_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=utilization,
        )

    def _estimate_vector(self, op: OpSpec) -> OpCost:
        """Estimate cost for VECTOR (elementwise/reduction) operations.

        VECTOR ops are typically memory-bound, limited by UB <-> GM bandwidth.
        """
        dtype = op.inputs[0].dtype.value if op.inputs else "fp16"
        flops = op.flops
        mem_bytes = op.memory_bytes

        vector_peak = self.hw.vector_peak_for(dtype)
        gm_bw = self.hw.main_memory_bandwidth() * 1e9

        compute_us = (flops / vector_peak * 1e6) if vector_peak > 0 else 0.0
        memory_us = (mem_bytes / gm_bw * 1e6) if gm_bw > 0 else 0.0

        # Task scheduling overhead on NPU (~3us typical)
        task_overhead_us = 3.0

        latency_us = max(compute_us, memory_us) + task_overhead_us
        bound = "compute (PIPE_V)" if compute_us >= memory_us else "memory"

        if latency_us > 0 and vector_peak > 0:
            util = flops / ((latency_us - task_overhead_us) * 1e-6 * vector_peak)
        else:
            util = 0.0

        return OpCost(
            compute_us=compute_us,
            memory_us=memory_us,
            latency_us=latency_us,
            bound=bound,
            flops=flops,
            bytes_accessed=mem_bytes,
            utilization=min(util, 1.0),
        )

    def _estimate_fallback(self, op: OpSpec) -> OpCost:
        """Fallback for unsupported ops — use basic roofline on VECTOR."""
        return self._estimate_vector(op)

    def _cube_utilization(self, op: OpSpec, tile: int) -> float:
        """Compute CUBE utilization based on tile alignment.

        For matmul [M,K] x [K,N], CUBE processes (tile x tile) blocks.
        Misalignment causes wasted computation.
        """
        if op.op_type == OpType.MATMUL and len(op.inputs) >= 2:
            a, b = op.inputs[0], op.inputs[1]
            M = a.shape[-2] if len(a.shape) >= 2 else 1
            K = a.shape[-1] if len(a.shape) >= 1 else 1
            N = b.shape[-1] if len(b.shape) >= 2 else 1

            # Padded dimensions
            M_pad = math.ceil(M / tile) * tile
            K_pad = math.ceil(K / tile) * tile
            N_pad = math.ceil(N / tile) * tile

            actual = M * K * N
            padded = M_pad * K_pad * N_pad

            return actual / padded if padded > 0 else 1.0

        elif op.op_type == OpType.CONV2D and len(op.inputs) >= 2:
            kernel = op.inputs[1]
            C_out = kernel.shape[0]
            C_in = kernel.shape[1]

            C_out_pad = math.ceil(C_out / tile) * tile
            C_in_pad = math.ceil(C_in / tile) * tile

            actual = C_out * C_in
            padded = C_out_pad * C_in_pad

            return actual / padded if padded > 0 else 1.0

        return 1.0

    def _format_conversion_cost(self, op: OpSpec, dtype: str) -> float:
        """Estimate cost of data format conversion (e.g., ND -> Fractal_NZ).

        Format conversion is essentially a memory copy with reshape,
        costing ~1.5x a simple memory copy.
        """
        target_layout = _LAYOUT_MAP.get(op.op_type, "ND")
        if target_layout == "ND":
            return 0.0

        # Estimate: conversion reads input + writes in new format
        # ~1.5x overhead compared to simple copy
        total_input_bytes = sum(t.size_bytes for t in op.inputs)
        gm_bw = self.hw.main_memory_bandwidth() * 1e9
        if gm_bw == 0:
            return 0.0

        conversion_bytes = total_input_bytes * 1.5
        return conversion_bytes / gm_bw * 1e6  # microseconds

    @staticmethod
    def preferred_layout(op_type: OpType) -> str:
        """Return the preferred data layout for an op type on Ascend."""
        return _LAYOUT_MAP.get(op_type, "ND")

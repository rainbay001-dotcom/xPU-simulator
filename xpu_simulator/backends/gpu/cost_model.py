"""GPU-specific cost model with wave quantization."""
from __future__ import annotations

import math

from ...core.cost_model import CostModel, OpCost
from ...core.operator import OpSpec, OpType
from .hardware import GPUSpec


# Ops that use Tensor Cores
_TENSOR_CORE_OPS = {OpType.MATMUL, OpType.CONV2D}


class GPUCostModel(CostModel):
    """GPU cost model with Tensor Core vs CUDA Core distinction,
    memory-hierarchy-aware bandwidth, wave quantization, and efficiency factors."""

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

        # Wave quantization for MATMUL: partial last wave wastes SM slots
        wave_eff = self._wave_efficiency(op) if is_tc_op else 1.0

        # Apply compute efficiency and wave quantization
        effective_peak = peak * compute_efficiency * wave_eff

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

    def _wave_efficiency(self, op: OpSpec) -> float:
        """Compute wave quantization efficiency for GEMM/Conv ops.

        When output tiles don't fill all SMs evenly, the last wave has
        idle SMs. For example, 117 tiles on 108 SMs → 2 waves, but the
        second wave only uses 9/108 = 8.3% of SMs.

        wave_efficiency = num_tiles / (num_waves * num_SMs)

        Returns 1.0 for non-MATMUL ops or when dimensions are unavailable.
        """
        M, N = self._extract_output_dims(op)
        if M <= 0 or N <= 0:
            return 1.0

        tile_M, tile_N = self.hw.cta_tile
        num_tiles = math.ceil(M / tile_M) * math.ceil(N / tile_N)
        num_sms = self.hw.sm_count
        num_waves = math.ceil(num_tiles / num_sms)

        return num_tiles / (num_waves * num_sms)

    @staticmethod
    def _extract_output_dims(op: OpSpec) -> tuple[int, int]:
        """Extract (M, N) from a MATMUL or Conv2D op for wave quantization."""
        if op.op_type == OpType.MATMUL and len(op.inputs) >= 2:
            a, b = op.inputs[0], op.inputs[1]
            M = a.shape[-2] if len(a.shape) >= 2 else 1
            N = b.shape[-1] if len(b.shape) >= 2 else 1
            # Batched matmul: batch dims contribute to M
            batch = math.prod(a.shape[:-2]) if len(a.shape) > 2 else 1
            M = M * batch
            return M, N
        elif op.op_type == OpType.CONV2D and len(op.outputs) >= 1:
            out = op.outputs[0]
            # Output [N, C_out, H, W] → M = N*H*W, N_dim = C_out
            N_batch = out.shape[0] if len(out.shape) > 0 else 1
            C_out = out.shape[1] if len(out.shape) > 1 else 1
            H = out.shape[2] if len(out.shape) > 2 else 1
            W = out.shape[3] if len(out.shape) > 3 else 1
            return N_batch * H * W, C_out
        return 0, 0

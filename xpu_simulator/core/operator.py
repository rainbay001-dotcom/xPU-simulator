"""Operator and tensor specifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import math


class OpType(Enum):
    MATMUL = auto()
    CONV2D = auto()
    RELU = auto()
    ADD = auto()
    LAYER_NORM = auto()
    SOFTMAX = auto()
    GELU = auto()
    SILU = auto()
    MUL = auto()
    ROPE = auto()
    EMBEDDING = auto()
    GATHER = auto()
    ALL_REDUCE = auto()
    ALL_TO_ALL = auto()
    TRANSPOSE = auto()
    RESHAPE = auto()
    TOP_K = auto()
    DEQUANT = auto()
    ALL_GATHER = auto()
    REDUCE_SCATTER = auto()
    UNKNOWN = auto()


class Phase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class Dtype(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"

    @property
    def bytes(self) -> float:
        return {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1, "int4": 0.5}[self.value]


@dataclass
class QuantConfig:
    """Quantization configuration for linear layers.

    weight_dtype / activation_dtype control the dtypes of weight and activation
    tensors in the matmul.  group_size is used for INT4 group quantization.
    """
    weight_dtype: Dtype = Dtype.FP16
    activation_dtype: Dtype = Dtype.FP16
    group_size: Optional[int] = None


@dataclass
class TensorSpec:
    shape: tuple[int, ...]
    dtype: Dtype = Dtype.FP16
    layout: str = "NCHW"

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def size_bytes(self) -> int:
        return int(self.numel * self.dtype.bytes)


@dataclass
class OpSpec:
    op_type: OpType
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    attrs: dict = field(default_factory=dict)
    name: Optional[str] = None

    @property
    def flops(self) -> int:
        """Compute FLOPs for this operation."""
        # Fused ops store pre-computed FLOPs that account for all merged ops
        fused_flops = self.attrs.get("_fused_flops")
        if fused_flops is not None:
            return fused_flops
        if self.op_type == OpType.MATMUL:
            return self._matmul_flops()
        elif self.op_type == OpType.CONV2D:
            return self._conv2d_flops()
        elif self.op_type in (OpType.RELU, OpType.ADD, OpType.GELU, OpType.SILU, OpType.MUL,
                              OpType.TOP_K):
            return self._elementwise_flops()
        elif self.op_type in (OpType.LAYER_NORM, OpType.SOFTMAX):
            return self._reduction_flops()
        elif self.op_type == OpType.ROPE:
            # RoPE applies sin/cos rotation per head dim pair: ~6 ops per element
            return 6 * self.inputs[0].numel
        elif self.op_type == OpType.EMBEDDING:
            # Embedding lookup: negligible compute, dominated by memory
            return 0
        elif self.op_type == OpType.GATHER:
            # Gather/scatter: negligible compute, memory-bound
            return 0
        elif self.op_type == OpType.DEQUANT:
            # Dequantize: ~2 ops per element (scale + zero-point)
            return 2 * self.inputs[0].numel if self.inputs else 0
        elif self.op_type in (OpType.ALL_REDUCE, OpType.ALL_TO_ALL):
            # Collective ops: FLOPs from reduction (e.g., sum), ~1 op per element
            return self.inputs[0].numel if self.inputs else 0
        elif self.op_type in (OpType.ALL_GATHER, OpType.REDUCE_SCATTER):
            # Communication ops: no compute, bandwidth-dominated
            return 0
        elif self.op_type in (OpType.TRANSPOSE, OpType.RESHAPE):
            return 0
        return 0

    @property
    def memory_bytes(self) -> int:
        """Total bytes read + written."""
        read = sum(t.size_bytes for t in self.inputs)
        written = sum(t.size_bytes for t in self.outputs)
        return read + written

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs / byte — higher means more compute-bound."""
        if self.memory_bytes == 0:
            return float("inf")
        return self.flops / self.memory_bytes

    def _matmul_flops(self) -> int:
        # [M, K] x [K, N] -> [M, N]: 2*M*K*N
        a = self.inputs[0]
        b = self.inputs[1]
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            _, N = b.shape
            return 2 * M * K * N
        # Batched matmul: [..., M, K] x [..., K, N]
        M, K = a.shape[-2], a.shape[-1]
        N = b.shape[-1]
        batch = math.prod(a.shape[:-2])
        return 2 * batch * M * K * N

    def _conv2d_flops(self) -> int:
        # Input: [N, C_in, H, W], Kernel: [C_out, C_in, Kh, Kw]
        inp = self.inputs[0]
        kernel = self.inputs[1]
        out = self.outputs[0]
        N = inp.shape[0]
        C_in = kernel.shape[1]
        C_out = kernel.shape[0]
        Kh, Kw = kernel.shape[2], kernel.shape[3]
        groups = self.attrs.get("groups", 1)
        H_out, W_out = out.shape[2], out.shape[3]
        return 2 * N * C_out * H_out * W_out * (C_in // groups) * Kh * Kw

    def _elementwise_flops(self) -> int:
        return self.outputs[0].numel

    def _reduction_flops(self) -> int:
        # ~5 ops per element for layernorm/softmax (mean, var, sub, div, etc.)
        return 5 * self.inputs[0].numel

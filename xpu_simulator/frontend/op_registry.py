"""Registry mapping operation names to OpSpec constructors."""
from __future__ import annotations

from typing import Optional
from ..core.operator import OpSpec, TensorSpec, OpType, Dtype


# Map common op names to OpType
_OP_NAME_MAP = {
    "aten::mm": OpType.MATMUL,
    "aten::matmul": OpType.MATMUL,
    "aten::bmm": OpType.MATMUL,
    "aten::linear": OpType.MATMUL,
    "aten::conv2d": OpType.CONV2D,
    "aten::relu": OpType.RELU,
    "aten::relu_": OpType.RELU,
    "aten::gelu": OpType.GELU,
    "aten::add": OpType.ADD,
    "aten::add_": OpType.ADD,
    "aten::layer_norm": OpType.LAYER_NORM,
    "aten::softmax": OpType.SOFTMAX,
    "aten::transpose": OpType.TRANSPOSE,
    "aten::reshape": OpType.RESHAPE,
    "aten::view": OpType.RESHAPE,
}


class OpRegistry:
    """Maps operation names to OpType and builds OpSpec from tensor metadata."""

    def __init__(self):
        self._custom_ops: dict[str, OpType] = {}

    def register(self, op_name: str, op_type: OpType):
        """Register a custom op name mapping."""
        self._custom_ops[op_name] = op_type

    def resolve_op_type(self, op_name: str) -> OpType:
        """Resolve an operation name to an OpType."""
        if op_name in self._custom_ops:
            return self._custom_ops[op_name]
        if op_name in _OP_NAME_MAP:
            return _OP_NAME_MAP[op_name]
        return OpType.UNKNOWN

    def build_op(
        self,
        op_name: str,
        input_specs: list[TensorSpec],
        output_specs: list[TensorSpec],
        attrs: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> OpSpec:
        """Build an OpSpec from an operation name and tensor specs."""
        op_type = self.resolve_op_type(op_name)
        return OpSpec(
            op_type=op_type,
            inputs=input_specs,
            outputs=output_specs,
            attrs=attrs or {},
            name=name or op_name,
        )

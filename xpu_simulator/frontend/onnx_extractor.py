"""Extract a ComputeGraph from an ONNX model."""
from __future__ import annotations

from typing import Union

from ..core.graph import ComputeGraph
from ..core.operator import Dtype, TensorSpec
from .base import GraphExtractor

try:
    import onnx
    from onnx import TensorProto
except ImportError as e:
    raise ImportError(
        "ONNX is required for ONNXExtractor. Install it with: "
        "pip install 'xpu-simulator[onnx]'"
    ) from e


_ONNX_DTYPE_MAP = {
    TensorProto.FLOAT: Dtype.FP32,
    TensorProto.FLOAT16: Dtype.FP16,
    TensorProto.BFLOAT16: Dtype.BF16,
    int(TensorProto.INT8): Dtype.INT8,
}


def _tensor_spec_from_type_proto(
    type_proto: onnx.TypeProto, fallback_dtype: Dtype
) -> TensorSpec | None:
    """Build a TensorSpec from an ONNX TypeProto, or None if shape is unknown."""
    tensor_type = type_proto.tensor_type
    if tensor_type.elem_type == 0:
        return None
    dtype = _ONNX_DTYPE_MAP.get(tensor_type.elem_type, fallback_dtype)
    shape_proto = tensor_type.shape
    if shape_proto is None:
        return None
    dims: list[int] = []
    for d in shape_proto.dim:
        dims.append(d.dim_value if d.dim_value > 0 else 1)
    if not dims:
        return None
    return TensorSpec(shape=tuple(dims), dtype=dtype)


def _tensor_spec_from_initializer(
    init: onnx.TensorProto, fallback_dtype: Dtype
) -> TensorSpec:
    dtype = _ONNX_DTYPE_MAP.get(init.data_type, fallback_dtype)
    shape = tuple(init.dims) if init.dims else (1,)
    return TensorSpec(shape=shape, dtype=dtype)


class ONNXExtractor(GraphExtractor):
    """Extract a ComputeGraph from an ONNX ModelProto or .onnx file."""

    def extract(
        self,
        source: Union[str, onnx.ModelProto],
        graph_name: str = "model",
    ) -> ComputeGraph:
        if isinstance(source, str):
            model = onnx.load(source)
        else:
            model = source

        model = onnx.shape_inference.infer_shapes(model)
        g = model.graph

        # Build tensor_name -> TensorSpec map
        tensor_map: dict[str, TensorSpec] = {}
        initializer_names: set[str] = set()

        for init in g.initializer:
            initializer_names.add(init.name)
            tensor_map[init.name] = _tensor_spec_from_initializer(init, self.dtype)

        for vi in list(g.input) + list(g.value_info) + list(g.output):
            spec = _tensor_spec_from_type_proto(vi.type, self.dtype)
            if spec is not None:
                tensor_map[vi.name] = spec

        # Track which tensor is produced by which node
        tensor_producer: dict[str, object] = {}  # tensor_name -> Node
        cg = ComputeGraph(name=graph_name)

        for node in g.node:
            # Skip nodes whose inputs are all initializers (constant folding)
            non_init_inputs = [
                i for i in node.input if i and i not in initializer_names
            ]
            if not non_init_inputs and node.output:
                # Still register outputs so downstream nodes can find them
                for out_name in node.output:
                    if out_name not in tensor_map:
                        tensor_map[out_name] = TensorSpec(shape=(1,), dtype=self.dtype)
                continue

            op_name = f"onnx::{node.op_type}"

            input_specs = []
            for inp in node.input:
                if inp and inp in tensor_map:
                    input_specs.append(tensor_map[inp])

            output_specs = []
            for out in node.output:
                if out and out in tensor_map:
                    output_specs.append(tensor_map[out])

            if not output_specs:
                output_specs = [
                    input_specs[0] if input_specs
                    else TensorSpec(shape=(1,), dtype=self.dtype)
                ]

            # Extract relevant attributes
            attrs: dict = {}
            for attr in node.attribute:
                if attr.name == "group":
                    attrs["groups"] = attr.i
                elif attr.name in ("transA", "transB"):
                    attrs[attr.name] = attr.i

            op_spec = self.registry.build_op(
                op_name=op_name,
                input_specs=input_specs,
                output_specs=output_specs,
                attrs=attrs if attrs else None,
                name=node.name or op_name,
            )

            graph_node = cg.add_node(op_spec, name=node.name or None)

            # Register this node as producer of its outputs
            for out_name in node.output:
                if out_name:
                    tensor_producer[out_name] = graph_node

            # Add edges from producer nodes
            for inp in node.input:
                if inp and inp in tensor_producer:
                    src_node = tensor_producer[inp]
                    cg.add_edge(src_node, graph_node, tensor=tensor_map.get(inp))

        return cg

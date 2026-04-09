"""Extract computation graphs from PyTorch models using torch.fx."""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.fx

from ..core.graph import ComputeGraph
from ..core.operator import OpSpec, TensorSpec, OpType, Dtype
from .base import GraphExtractor
from ._torch_utils import torch_dtype_to_dtype, shape_from_meta, dtype_from_meta


# Map torch functions/methods to aten op names
_FUNC_MAP = {
    torch.relu: "aten::relu",
    torch.nn.functional.relu: "aten::relu",
    torch.nn.functional.gelu: "aten::gelu",
    torch.matmul: "aten::matmul",
    torch.mm: "aten::mm",
    torch.bmm: "aten::bmm",
    torch.add: "aten::add",
    torch.nn.functional.softmax: "aten::softmax",
    torch.nn.functional.layer_norm: "aten::layer_norm",
    torch.nn.functional.conv2d: "aten::conv2d",
}

_MODULE_MAP = {
    nn.Linear: "aten::linear",
    nn.Conv2d: "aten::conv2d",
    nn.ReLU: "aten::relu",
    nn.GELU: "aten::gelu",
    nn.LayerNorm: "aten::layer_norm",
    nn.Softmax: "aten::softmax",
}


class TorchGraphExtractor(GraphExtractor):
    """Extract a ComputeGraph from a PyTorch model using torch.fx tracing."""

    def extract(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        graph_name: str = "model",
    ) -> ComputeGraph:
        """
        Trace a PyTorch model and build a ComputeGraph.

        Args:
            model: PyTorch module to trace
            example_inputs: Example input tensors for shape inference
            graph_name: Name for the resulting graph
        """
        # Use torch.fx symbolic trace
        traced = torch.fx.symbolic_trace(model)
        fx_graph = traced.graph

        # Run shape propagation using example inputs
        shapes = self._propagate_shapes(traced, example_inputs)

        # Build compute graph
        graph = ComputeGraph(graph_name)
        node_map: dict[str, any] = {}  # fx node name -> ComputeGraph Node

        for fx_node in fx_graph.nodes:
            if fx_node.op == "placeholder":
                # Input node — no op, just record shape
                shape = self._get_node_shape(fx_node, shapes, example_inputs)
                node_map[fx_node.name] = ("input", shape)

            elif fx_node.op == "call_function":
                op_name = _FUNC_MAP.get(fx_node.target, f"aten::{fx_node.target.__name__}")
                g_node = self._make_node(graph, fx_node, op_name, node_map, shapes)
                if g_node:
                    node_map[fx_node.name] = g_node

            elif fx_node.op == "call_module":
                submod = dict(traced.named_modules()).get(fx_node.target)
                op_name = _MODULE_MAP.get(type(submod), f"module::{type(submod).__name__}")
                g_node = self._make_node(
                    graph, fx_node, op_name, node_map, shapes,
                    module=submod,
                )
                if g_node:
                    node_map[fx_node.name] = g_node

            elif fx_node.op == "call_method":
                op_name = f"aten::{fx_node.target}"
                g_node = self._make_node(graph, fx_node, op_name, node_map, shapes)
                if g_node:
                    node_map[fx_node.name] = g_node

            elif fx_node.op == "output":
                pass  # terminal node

        # Add edges based on FX graph data deps
        for fx_node in fx_graph.nodes:
            if fx_node.name not in node_map:
                continue
            dst_entry = node_map[fx_node.name]
            if isinstance(dst_entry, tuple):
                continue  # input placeholder
            dst_node = dst_entry

            for arg in fx_node.args:
                if isinstance(arg, torch.fx.Node) and arg.name in node_map:
                    src_entry = node_map[arg.name]
                    if isinstance(src_entry, tuple):
                        continue  # skip input->first_op edges (no src Node)
                    graph.add_edge(src_entry, dst_node)

        return graph

    def _propagate_shapes(
        self, traced: torch.fx.GraphModule, example_inputs: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Run the model with example inputs to capture intermediate shapes."""
        shapes = {}

        # Use interpreter to capture shapes
        class ShapeCapture(torch.fx.Interpreter):
            def run_node(self, n):
                result = super().run_node(n)
                if isinstance(result, torch.Tensor):
                    shapes[n.name] = result
                return result

        interp = ShapeCapture(traced)
        interp.run(*example_inputs)
        return shapes

    def _get_node_shape(
        self,
        fx_node: torch.fx.Node,
        shapes: dict,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> tuple[int, ...]:
        if fx_node.name in shapes:
            return tuple(shapes[fx_node.name].shape)
        # For placeholders, use example input shapes
        for i, node in enumerate(
            n for n in fx_node.graph.nodes if n.op == "placeholder"
        ):
            if node.name == fx_node.name and i < len(example_inputs):
                return tuple(example_inputs[i].shape)
        return (1,)

    def _make_node(
        self,
        graph: ComputeGraph,
        fx_node: torch.fx.Node,
        op_name: str,
        node_map: dict,
        shapes: dict,
        module: Optional[nn.Module] = None,
    ):
        """Create a ComputeGraph node from an FX node."""
        # Gather input shapes
        input_specs = []
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node):
                if arg.name in shapes:
                    t = shapes[arg.name]
                    input_specs.append(TensorSpec(
                        tuple(t.shape), torch_dtype_to_dtype(t.dtype),
                    ))
                elif arg.name in node_map:
                    entry = node_map[arg.name]
                    if isinstance(entry, tuple):
                        input_specs.append(TensorSpec(entry[1], self.dtype))

        # Get output shape
        if fx_node.name in shapes:
            out_t = shapes[fx_node.name]
            output_specs = [TensorSpec(tuple(out_t.shape), torch_dtype_to_dtype(out_t.dtype))]
        else:
            # Fallback: same shape as first input
            if input_specs:
                output_specs = [TensorSpec(input_specs[0].shape, self.dtype)]
            else:
                return None

        # Module-specific attrs
        attrs = {}
        if module is not None:
            if isinstance(module, nn.Conv2d):
                attrs["groups"] = module.groups
                # Add weight as input
                weight_shape = tuple(module.weight.shape)
                input_specs.append(TensorSpec(weight_shape, self.dtype))
            elif isinstance(module, nn.Linear):
                weight_shape = (module.out_features, module.in_features)
                input_specs.append(TensorSpec(weight_shape, self.dtype))

        if not input_specs:
            return None

        op = self.registry.build_op(op_name, input_specs, output_specs, attrs, fx_node.name)
        return graph.add_node(op, fx_node.name)

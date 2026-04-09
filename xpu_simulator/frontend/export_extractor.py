"""Extract computation graphs from PyTorch models using torch.export."""
from __future__ import annotations

import sys
from typing import Optional

import torch
import torch.nn as nn

from ..core.graph import ComputeGraph
from ..core.operator import OpSpec, TensorSpec, OpType, Dtype
from .base import GraphExtractor
from ._torch_utils import torch_dtype_to_dtype, shape_from_meta, dtype_from_meta


def _normalize_target(target) -> str:
    """Normalize an OpOverload/OpOverloadPacket target to 'namespace::opname' format.

    torch.export call_function targets are typically OpOverload objects like
    ``torch.ops.aten.mm.default``.  We extract the namespace and op name and
    return a string such as ``"aten::mm"``.
    """
    # OpOverload: torch.ops.aten.mm.default -> namespace()="aten", name()="aten::mm"
    if hasattr(target, "namespace") and callable(target.namespace):
        ns = target.namespace()
        # .name() returns e.g. "aten::mm.default"; strip the overload suffix
        full = target.name() if callable(getattr(target, "name", None)) else str(target)
        op = full.split("::")[1].split(".")[0] if "::" in full else full.split(".")[0]
        return f"{ns}::{op}"

    # OpOverloadPacket: torch.ops.aten.mm
    if hasattr(target, "_qualified_op_name"):
        qname = target._qualified_op_name  # e.g. "aten::mm"
        return qname.split(".")[0]

    # Fallback: use __name__ or str representation
    name = getattr(target, "__name__", None) or str(target)
    return f"aten::{name}"


class ExportExtractor(GraphExtractor):
    """Extract a ComputeGraph from a PyTorch model using ``torch.export``."""

    def extract(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        graph_name: str = "model",
        dynamic_shapes: Optional[dict] = None,
    ) -> ComputeGraph:
        """Export a PyTorch model and build a ComputeGraph.

        Args:
            model: PyTorch module to export.
            example_inputs: Example input tensors for tracing.
            graph_name: Name for the resulting graph.
            dynamic_shapes: Optional dynamic shape constraints passed to
                ``torch.export.export``.

        Returns:
            A ``ComputeGraph`` representing the model's operations.

        Raises:
            RuntimeError: If ``torch.export`` is not available (PyTorch < 2.1).
        """
        if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
            raise RuntimeError(
                "torch.export is not available. ExportExtractor requires PyTorch >= 2.1. "
                f"Current version: {torch.__version__}"
            )

        exported = torch.export.export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

        return self._build_graph(exported.graph_module, graph_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_graph(self, gm: torch.fx.GraphModule, graph_name: str) -> ComputeGraph:
        """Walk the FX graph inside an ExportedProgram's graph_module."""
        graph = ComputeGraph(graph_name)
        node_map: dict[str, object] = {}  # fx node name -> Node | ("input", shape)

        for fx_node in gm.graph.nodes:
            if fx_node.op == "placeholder":
                shape = shape_from_meta(fx_node.meta)
                if shape is None:
                    shape = (1,)
                dtype = dtype_from_meta(fx_node.meta)
                node_map[fx_node.name] = ("input", shape, dtype)

            elif fx_node.op == "call_function":
                g_node = self._handle_call_function(graph, fx_node, node_map)
                if g_node is not None:
                    node_map[fx_node.name] = g_node

            elif fx_node.op == "get_attr":
                # Constant / parameter tensor embedded in the graph module
                param = _get_nested_attr(gm, fx_node.target)
                if isinstance(param, torch.Tensor):
                    node_map[fx_node.name] = (
                        "input",
                        tuple(param.shape),
                        torch_dtype_to_dtype(param.dtype),
                    )

            elif fx_node.op == "output":
                pass  # terminal

        # Add edges based on data dependencies
        for fx_node in gm.graph.nodes:
            if fx_node.name not in node_map:
                continue
            dst_entry = node_map[fx_node.name]
            if isinstance(dst_entry, tuple):
                continue  # input / placeholder

            for arg in self._flatten_args(fx_node.args):
                if isinstance(arg, torch.fx.Node) and arg.name in node_map:
                    src_entry = node_map[arg.name]
                    if isinstance(src_entry, tuple):
                        continue  # skip edges from inputs
                    graph.add_edge(src_entry, dst_entry)

        return graph

    def _handle_call_function(
        self,
        graph: ComputeGraph,
        fx_node: torch.fx.Node,
        node_map: dict,
    ):
        """Create a ComputeGraph node from a call_function FX node."""
        op_name = _normalize_target(fx_node.target)

        # Collect input tensor specs
        input_specs = self._collect_input_specs(fx_node, node_map)

        # Derive output shape/dtype from node metadata
        output_specs = self._output_specs_from_meta(fx_node)
        if output_specs is None:
            # Fallback: mirror first input
            if input_specs:
                output_specs = [TensorSpec(input_specs[0].shape, input_specs[0].dtype)]
            else:
                return None

        if not input_specs:
            return None

        attrs: dict = {}
        op = self.registry.build_op(op_name, input_specs, output_specs, attrs, fx_node.name)
        return graph.add_node(op, fx_node.name)

    def _collect_input_specs(
        self, fx_node: torch.fx.Node, node_map: dict
    ) -> list[TensorSpec]:
        specs: list[TensorSpec] = []
        for arg in self._flatten_args(fx_node.args):
            if not isinstance(arg, torch.fx.Node):
                continue
            if arg.name not in node_map:
                continue

            entry = node_map[arg.name]
            if isinstance(entry, tuple):
                # ("input", shape, dtype)
                _, shape, dtype = entry
                specs.append(TensorSpec(shape, dtype))
            else:
                # It's a Node — pull shape/dtype from the node's op outputs
                if entry.op.outputs:
                    out = entry.op.outputs[0]
                    specs.append(TensorSpec(out.shape, out.dtype))
        return specs

    def _output_specs_from_meta(self, fx_node: torch.fx.Node) -> Optional[list[TensorSpec]]:
        val = fx_node.meta.get("val")
        if val is None:
            return None

        if isinstance(val, torch.Tensor):
            return [TensorSpec(tuple(val.shape), torch_dtype_to_dtype(val.dtype))]

        if hasattr(val, "shape") and hasattr(val, "dtype"):
            return [TensorSpec(tuple(val.shape), torch_dtype_to_dtype(val.dtype))]

        # val may be a tuple/list of tensors (multiple outputs)
        if isinstance(val, (tuple, list)):
            specs = []
            for v in val:
                if isinstance(v, torch.Tensor):
                    specs.append(TensorSpec(tuple(v.shape), torch_dtype_to_dtype(v.dtype)))
                elif hasattr(v, "shape") and hasattr(v, "dtype"):
                    specs.append(TensorSpec(tuple(v.shape), torch_dtype_to_dtype(v.dtype)))
            if specs:
                return specs

        return None

    @staticmethod
    def _flatten_args(args) -> list:
        """Flatten nested tuples/lists of FX node arguments."""
        flat: list = []
        for a in args:
            if isinstance(a, (tuple, list)):
                flat.extend(ExportExtractor._flatten_args(a))
            else:
                flat.append(a)
        return flat


def _get_nested_attr(module: nn.Module, target: str):
    """Retrieve a nested attribute from a module by dot-separated path."""
    parts = target.split(".")
    obj = module
    for p in parts:
        obj = getattr(obj, p)
    return obj

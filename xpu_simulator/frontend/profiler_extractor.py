"""Extract computation graphs from Chrome trace JSON (PyTorch profiler output)."""
from __future__ import annotations

import json
from typing import Optional

from ..core.graph import ComputeGraph
from ..core.operator import TensorSpec, OpType, Dtype
from .base import GraphExtractor

# Categories that indicate compute-related events
_COMPUTE_CATEGORIES = {"cpu_op", "kernel", "Kernel", "operator"}

# Minimum event duration (us) to keep — shorter events are framework overhead
_MIN_DURATION_US = 0.1


class ProfilerExtractor(GraphExtractor):
    """Extract a ComputeGraph from a Chrome trace JSON file.

    Works with traces produced by ``torch.profiler`` with
    ``record_shapes=True``.  Edge inference is best-effort: edges are
    added when a preceding op's output shape matches a subsequent op's
    input shape.
    """

    def extract(
        self,
        trace_path: str,
        graph_name: str = "model",
    ) -> ComputeGraph:
        """Load a Chrome trace JSON and build a ComputeGraph.

        Args:
            trace_path: Path to the Chrome trace JSON file.
            graph_name: Name for the resulting graph.
        """
        with open(trace_path) as f:
            trace = json.load(f)

        events = trace.get("traceEvents", [])

        # Filter to complete-duration compute events
        ops = []
        for ev in events:
            if ev.get("ph") != "X":
                continue
            cat = ev.get("cat", "")
            if cat not in _COMPUTE_CATEGORIES:
                continue
            dur = ev.get("dur", 0)
            if dur < _MIN_DURATION_US:
                continue
            ops.append(ev)

        # Sort by start timestamp for deterministic ordering and edge inference
        ops.sort(key=lambda e: e.get("ts", 0))

        graph = ComputeGraph(graph_name)
        # Track (node, output_specs) for edge inference
        node_records: list[tuple] = []  # (graph_node, input_specs, output_specs)

        for ev in ops:
            op_name = ev.get("name", "unknown")
            args = ev.get("args", {})
            input_shapes = args.get("Input Shapes")

            op_type = self.registry.resolve_op_type(op_name)

            # Build input TensorSpecs
            input_specs = self._build_input_specs(input_shapes)

            # If we have no shape info and can't resolve the op, mark UNKNOWN
            if not input_specs and op_type == OpType.UNKNOWN:
                input_specs = [TensorSpec((1,), self.dtype)]

            # Infer output shapes
            output_specs = self._infer_output_specs(op_type, input_specs)

            # Profiler-specific attributes
            attrs = {
                "profiled_latency_us": ev.get("dur", 0),
                "profiled_start_us": ev.get("ts", 0),
            }

            op = self.registry.build_op(op_name, input_specs, output_specs, attrs, op_name)
            node = graph.add_node(op, op_name)
            node_records.append((node, input_specs, output_specs))

        # Best-effort edge inference: match output shapes to input shapes
        self._infer_edges(graph, node_records)

        return graph

    def _build_input_specs(
        self, input_shapes: Optional[list] = None,
    ) -> list[TensorSpec]:
        if not input_shapes:
            return []
        specs = []
        for shape in input_shapes:
            if isinstance(shape, list) and len(shape) > 0:
                specs.append(TensorSpec(tuple(shape), self.dtype))
        return specs

    def _infer_output_specs(
        self,
        op_type: OpType,
        input_specs: list[TensorSpec],
    ) -> list[TensorSpec]:
        if not input_specs:
            return [TensorSpec((1,), self.dtype)]

        if op_type == OpType.MATMUL and len(input_specs) >= 2:
            a, b = input_specs[0], input_specs[1]
            if len(a.shape) >= 2 and len(b.shape) >= 2:
                out_shape = a.shape[:-1] + (b.shape[-1],)
                return [TensorSpec(out_shape, self.dtype)]

        # Elementwise / default: output matches first input
        return [TensorSpec(input_specs[0].shape, self.dtype)]

    def _infer_edges(
        self,
        graph: ComputeGraph,
        node_records: list[tuple],
    ) -> None:
        """Add edges by matching preceding output shapes to current input shapes.

        For each op, scan backwards through earlier ops and connect the most
        recent op whose output shape matches each required input shape.
        """
        for idx, (node, input_specs, _output_specs) in enumerate(node_records):
            if not input_specs:
                continue
            matched: set[int] = set()
            for inp_idx, inp_spec in enumerate(input_specs):
                # Scan backwards for a matching producer
                for prev_idx in range(idx - 1, -1, -1):
                    if prev_idx in matched:
                        continue
                    prev_node, _prev_in, prev_out = node_records[prev_idx]
                    for pout in prev_out:
                        if pout.shape == inp_spec.shape:
                            graph.add_edge(prev_node, node, pout)
                            matched.add(prev_idx)
                            break
                    else:
                        continue
                    break

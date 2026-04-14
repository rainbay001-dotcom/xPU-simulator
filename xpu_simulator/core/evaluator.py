"""Performance evaluator — runs cost model over a computation graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .graph import ComputeGraph, Node
from .cost_model import CostModel, OpCost
from .operator import OpSpec, OpType, TensorSpec, Dtype
from .parallel import ParallelConfig


@dataclass
class OpResult:
    """Result for a single operation."""
    node: Node
    cost: OpCost
    start_us: float = 0.0  # start time in the schedule
    end_us: float = 0.0    # end time in the schedule


@dataclass
class SimResult:
    """Overall simulation result."""
    total_latency_us: float
    per_op: list[OpResult]
    bottleneck_op: OpResult
    critical_path: list[OpResult] = field(default_factory=list)
    phase: str | None = None  # "prefill" or "decode"

    @property
    def total_flops(self) -> int:
        return sum(r.cost.flops for r in self.per_op)

    @property
    def total_bytes(self) -> int:
        return sum(r.cost.bytes_accessed for r in self.per_op)

    @property
    def compute_bound_count(self) -> int:
        return sum(1 for r in self.per_op if "compute" in r.cost.bound)

    @property
    def memory_bound_count(self) -> int:
        return sum(1 for r in self.per_op if "memory" in r.cost.bound)

    @property
    def sequential_latency_us(self) -> float:
        """What the latency would be without any overlap."""
        return sum(r.cost.latency_us for r in self.per_op)

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token (prefill latency) in ms."""
        if self.phase == "prefill":
            return self.total_latency_us / 1000
        return None

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token (decode latency) in ms."""
        if self.phase == "decode":
            return self.total_latency_us / 1000
        return None

    @property
    def speedup_from_overlap(self) -> float:
        seq = self.sequential_latency_us
        if self.total_latency_us > 0:
            return seq / self.total_latency_us
        return 1.0

    def summary(self) -> str:
        lines = [
            f"Total latency:    {self.total_latency_us:.2f} us ({self.total_latency_us / 1000:.3f} ms)",
            f"Total FLOPs:      {self.total_flops:,.0f}",
            f"Total memory:     {self.total_bytes:,.0f} bytes",
            f"Ops:              {len(self.per_op)} ({self.compute_bound_count} compute-bound, {self.memory_bound_count} memory-bound)",
            f"Bottleneck:       {self.bottleneck_op.node} ({self.bottleneck_op.cost.latency_us:.2f} us, {self.bottleneck_op.cost.bound})",
        ]
        if self.speedup_from_overlap > 1.01:
            lines.append(f"Overlap speedup:  {self.speedup_from_overlap:.2f}x (sequential: {self.sequential_latency_us:.2f} us)")
        return "\n".join(lines)


class PerformanceEvaluator:
    """Evaluates a computation graph using a cost model."""

    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def run(self, graph: ComputeGraph, overlap: bool = False) -> SimResult:
        """Evaluate graph performance.

        Args:
            graph: The computation graph to evaluate
            overlap: If True, model parallel execution of independent ops
                     (ASAP scheduling based on data dependencies)
        """
        # Estimate cost for each node
        node_costs: dict[int, OpCost] = {}
        for node in graph.nodes:
            node_costs[node.id] = self.cost_model.estimate(node.op)

        if overlap:
            return self._run_overlap(graph, node_costs)
        else:
            return self._run_sequential(graph, node_costs)

    def _run_sequential(self, graph: ComputeGraph, node_costs: dict[int, OpCost]) -> SimResult:
        """Sequential execution — no overlap."""
        per_op = []
        time_cursor = 0.0
        for node in graph.topo_order():
            cost = node_costs[node.id]
            per_op.append(OpResult(
                node=node, cost=cost,
                start_us=time_cursor, end_us=time_cursor + cost.latency_us,
            ))
            time_cursor += cost.latency_us

        bottleneck = max(per_op, key=lambda r: r.cost.latency_us)
        return SimResult(
            total_latency_us=time_cursor,
            per_op=per_op,
            bottleneck_op=bottleneck,
        )

    @staticmethod
    def _is_compute_saturating(cost: OpCost) -> bool:
        """Check if an op saturates the compute unit (compute-bound and large)."""
        return "compute" in cost.bound and cost.compute_us > 10.0

    @staticmethod
    def _get_resource_type(cost: OpCost) -> str:
        """Determine which hardware resource an op uses.

        On Ascend NPU, CUBE (PIPE_M) and VECTOR (PIPE_V) are independent
        pipelines that can run concurrently. Communication ops use a separate
        resource. On GPU, all compute ops share one resource.
        """
        if cost.bound == "communication":
            return "comm"
        if "PIPE_M" in cost.bound:
            return "cube"
        elif "PIPE_V" in cost.bound:
            return "vector"
        return "shared"

    def _run_overlap(self, graph: ComputeGraph, node_costs: dict[int, OpCost]) -> SimResult:
        """ASAP scheduling with per-resource constraints.

        Models dual-pipeline overlap: CUBE and VECTOR ops can run concurrently
        on Ascend NPU (they use separate hardware units). On GPU, all compute
        ops fall to "shared" resource and serialize as before.
        """
        earliest_end: dict[int, float] = {}
        per_op = []
        # Track when each resource becomes free
        resource_busy = {"cube": 0.0, "vector": 0.0, "shared": 0.0, "comm": 0.0}

        for node in graph.topo_order():
            cost = node_costs[node.id]

            # Start time = max end time of all predecessors
            preds = graph.predecessors(node)
            if preds:
                start = max(earliest_end[p.id] for p in preds)
            else:
                start = 0.0

            # Resource constraint: saturating ops block their own resource
            if self._is_compute_saturating(cost):
                res = self._get_resource_type(cost)
                start = max(start, resource_busy[res])

            end = start + cost.latency_us
            earliest_end[node.id] = end

            # Block the resource until this op finishes
            if self._is_compute_saturating(cost):
                res = self._get_resource_type(cost)
                resource_busy[res] = end

            per_op.append(OpResult(
                node=node, cost=cost,
                start_us=start, end_us=end,
            ))

        total_latency = max(r.end_us for r in per_op) if per_op else 0.0
        bottleneck = max(per_op, key=lambda r: r.cost.latency_us)

        # Identify critical path
        critical_path = self._find_critical_path(graph, node_costs, per_op)

        return SimResult(
            total_latency_us=total_latency,
            per_op=per_op,
            bottleneck_op=bottleneck,
            critical_path=critical_path,
        )

    def _find_critical_path(
        self,
        graph: ComputeGraph,
        node_costs: dict[int, OpCost],
        per_op: list[OpResult],
    ) -> list[OpResult]:
        """Find the critical path — longest path through the graph."""
        # Build weighted graph for longest path
        G = graph.nx_graph
        if G.number_of_nodes() == 0:
            return []

        # Build weighted copy for longest path
        weighted = nx.DiGraph()
        for node in G.nodes():
            weighted.add_node(node)
        for u, v in G.edges():
            weighted.add_edge(u, v, weight=node_costs[u.id].latency_us)

        try:
            path = nx.dag_longest_path(weighted, weight="weight")
        except nx.NetworkXError:
            return []

        op_map = {r.node.id: r for r in per_op}
        return [op_map[n.id] for n in path if n.id in op_map]


# ------------------------------------------------------------------ #
# Pipeline parallelism
# ------------------------------------------------------------------ #

@dataclass
class PipelineResult:
    """Result of a pipeline-parallel simulation across multiple stages."""
    per_stage: list[SimResult]
    bubble_fraction: float
    total_latency_us: float          # With bubble overhead
    ideal_latency_us: float          # Without bubble (max stage)
    pp_send_recv_us: float = 0.0     # Inter-stage transfer overhead

    @property
    def bubble_overhead_us(self) -> float:
        return self.total_latency_us - self.ideal_latency_us

    def summary(self) -> str:
        lines = [
            f"Pipeline stages:  {len(self.per_stage)}",
            f"Bubble fraction:  {self.bubble_fraction:.1%}",
            f"Ideal latency:    {self.ideal_latency_us:.2f} us ({self.ideal_latency_us / 1000:.3f} ms)",
            f"Total latency:    {self.total_latency_us:.2f} us ({self.total_latency_us / 1000:.3f} ms)",
            f"Bubble overhead:  {self.bubble_overhead_us:.2f} us",
            f"PP send/recv:     {self.pp_send_recv_us:.2f} us",
        ]
        for i, stage in enumerate(self.per_stage):
            lines.append(f"  Stage {i}: {stage.total_latency_us:.2f} us "
                          f"({stage.compute_bound_count} compute, "
                          f"{stage.memory_bound_count} memory)")
        return "\n".join(lines)


def run_pipeline(stages: list[ComputeGraph],
                 cost_model: CostModel,
                 parallel: ParallelConfig,
                 overlap: bool = False,
                 pp_transfer_bytes: int = 0) -> PipelineResult:
    """Simulate pipeline-parallel execution across stages.

    Args:
        stages: Per-stage computation graphs (one per PP rank).
        cost_model: The cost model to use (should be CommAwareCostModel
                    for PP_SEND_RECV support).
        parallel: ParallelConfig with pp_size, vpp, gradient_accumulation.
        overlap: Whether to model intra-stage op overlap.
        pp_transfer_bytes: Activation bytes transferred between stages.
                           If 0, estimated from the last op of each stage.
    """
    evaluator = PerformanceEvaluator(cost_model)

    per_stage = []
    for graph in stages:
        result = evaluator.run(graph, overlap=overlap)
        per_stage.append(result)

    # Max stage determines the pipeline steady-state
    ideal_latency = max(s.total_latency_us for s in per_stage) if per_stage else 0.0

    # PP send/recv overhead (per micro-batch, amortized)
    pp_send_recv = 0.0
    if pp_transfer_bytes > 0 and hasattr(cost_model, '_estimate_pp'):
        pp_op = OpSpec(
            OpType.PP_SEND_RECV,
            [TensorSpec((pp_transfer_bytes,), dtype=Dtype.INT8)],
            [TensorSpec((pp_transfer_bytes,), dtype=Dtype.INT8)],
            name="pp_transfer",
        )
        pp_cost = cost_model.estimate(pp_op)
        pp_send_recv = pp_cost.latency_us

    bubble = parallel.bubble_fraction
    # Total = ideal * (1 + bubble) + pp_overhead * (pp_size - 1)
    total = ideal_latency * (1.0 + bubble) + pp_send_recv * max(0, parallel.pp_size - 1)

    return PipelineResult(
        per_stage=per_stage,
        bubble_fraction=bubble,
        total_latency_us=total,
        ideal_latency_us=ideal_latency,
        pp_send_recv_us=pp_send_recv,
    )

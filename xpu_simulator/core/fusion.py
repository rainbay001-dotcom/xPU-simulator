"""Kernel fusion pass — rewrites a ComputeGraph by merging fusible ops.

Fusion eliminates intermediate memory traffic between ops that can run
as a single kernel. This is one of the largest sources of real vs. simulated
performance divergence.

Usage:
    fused_graph = FusionPass(rules).apply(graph)
    result = evaluator.run(fused_graph)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from .graph import ComputeGraph, Node
from .operator import OpSpec, TensorSpec, OpType


@dataclass
class FusionResult:
    """Result of applying fusion to a graph."""
    original_nodes: int
    fused_nodes: int
    fusions_applied: list[str]  # descriptions of each fusion

    @property
    def nodes_eliminated(self) -> int:
        return self.original_nodes - self.fused_nodes

    def summary(self) -> str:
        lines = [
            f"Fusion: {self.original_nodes} ops -> {self.fused_nodes} ops ({self.nodes_eliminated} eliminated)",
            f"Fusions applied: {len(self.fusions_applied)}",
        ]
        for f in self.fusions_applied[:20]:
            lines.append(f"  - {f}")
        if len(self.fusions_applied) > 20:
            lines.append(f"  ... and {len(self.fusions_applied) - 20} more")
        return "\n".join(lines)


class FusionRule(ABC):
    """Base class for a fusion rule that matches and rewrites patterns."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        """Try to match a fusion pattern starting at `node`.

        Returns list of nodes to fuse (including `node`), or None if no match.
        """
        ...

    @abstractmethod
    def fuse(self, nodes: list[Node]) -> OpSpec:
        """Create a fused OpSpec from the matched nodes."""
        ...


# ============================================================
# Fusion Rules
# ============================================================

class MatMulEpilogueFusion(FusionRule):
    """Fuse MatMul + elementwise epilogue (bias, activation, residual add).

    Pattern: MatMul -> {ReLU, GELU, ADD} (single consumer)
    Result: Single op with MatMul FLOPs, but only reads MatMul inputs + writes final output.
    """

    @property
    def name(self) -> str:
        return "matmul_epilogue"

    _EPILOGUE_OPS = {OpType.RELU, OpType.GELU, OpType.SILU, OpType.ADD, OpType.MUL}

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if node.op.op_type != OpType.MATMUL:
            return None
        succs = graph.successors(node)
        if len(succs) != 1:
            return None
        succ = succs[0]
        if succ.op.op_type not in self._EPILOGUE_OPS:
            return None
        # Epilogue must have only this matmul as predecessor (or one other for ADD/MUL)
        preds = graph.predecessors(succ)
        if succ.op.op_type in (OpType.ADD, OpType.MUL):
            if len(preds) > 2:
                return None
        else:
            if len(preds) != 1:
                return None
        return [node, succ]

    def fuse(self, nodes: list[Node]) -> OpSpec:
        matmul_op = nodes[0].op
        epilogue_op = nodes[1].op
        # Fused op: MatMul FLOPs + epilogue FLOPs, but memory = matmul inputs + final output only
        fused_inputs = list(matmul_op.inputs)
        fused_outputs = list(epilogue_op.outputs)
        return OpSpec(
            op_type=OpType.MATMUL,
            inputs=fused_inputs,
            outputs=fused_outputs,
            attrs={"fused": f"matmul+{epilogue_op.op_type.name.lower()}"},
            name=f"{nodes[0].op.name or nodes[0].name}_fused",
        )


class ElementwiseChainFusion(FusionRule):
    """Fuse consecutive elementwise ops into a single kernel.

    Pattern: Elem -> Elem -> ... (each with single consumer)
    Result: Single op reading first input, writing last output.
    """

    @property
    def name(self) -> str:
        return "elementwise_chain"

    _ELEM_OPS = {OpType.RELU, OpType.GELU, OpType.SILU, OpType.ADD, OpType.MUL,
                  OpType.LAYER_NORM, OpType.SOFTMAX, OpType.ROPE}

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if node.op.op_type not in self._ELEM_OPS:
            return None
        chain = [node]
        current = node
        while True:
            succs = graph.successors(current)
            if len(succs) != 1:
                break
            succ = succs[0]
            if succ.op.op_type not in self._ELEM_OPS:
                break
            preds = graph.predecessors(succ)
            if len(preds) != 1:
                break
            chain.append(succ)
            current = succ
        if len(chain) < 2:
            return None
        return chain

    def fuse(self, nodes: list[Node]) -> OpSpec:
        first_op = nodes[0].op
        last_op = nodes[-1].op
        total_flops_estimate = sum(n.op.flops for n in nodes)
        fused_inputs = list(first_op.inputs)
        fused_outputs = list(last_op.outputs)
        names = "+".join(n.op.op_type.name.lower() for n in nodes)
        return OpSpec(
            op_type=first_op.op_type,
            inputs=fused_inputs,
            outputs=fused_outputs,
            attrs={"fused": names, "_fused_flops": total_flops_estimate},
            name=f"{nodes[0].name}_chain_fused",
        )


class SwiGLUFusion(FusionRule):
    """Fuse SwiGLU pattern: SiLU(w1(x)) * w3(x) -> w2.

    Pattern: Two parallel matmuls (w1, w3) -> activation -> elementwise mul
    Result: Fuse the activation + mul into one kernel (still 3 matmuls,
            but eliminate intermediate memory for SiLU output and gate output).
    """

    @property
    def name(self) -> str:
        return "swiglu"

    # Match SiLU by OpType or by name fallback (for legacy graphs using GELU/RELU as proxy)
    _SILU_TYPES = {OpType.GELU, OpType.RELU}
    # Match MUL by OpType or by name fallback (for legacy graphs using ADD as proxy)
    _MUL_TYPES = {OpType.ADD}

    def _is_silu(self, node: Node) -> bool:
        if hasattr(OpType, "SILU") and node.op.op_type == OpType.SILU:
            return True
        return node.op.op_type in self._SILU_TYPES and bool(node.name) and "silu" in node.name

    def _is_mul(self, node: Node) -> bool:
        if hasattr(OpType, "MUL") and node.op.op_type == OpType.MUL:
            return True
        return node.op.op_type in self._MUL_TYPES and bool(node.name) and "mul" in node.name

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if not self._is_silu(node):
            return None
        succs = graph.successors(node)
        if len(succs) != 1:
            return None
        mul_node = succs[0]
        if not self._is_mul(mul_node):
            return None
        return [node, mul_node]

    def fuse(self, nodes: list[Node]) -> OpSpec:
        silu_node = nodes[0]
        mul_node = nodes[1]
        # Fused: reads silu input + gate input, writes mul output
        # Eliminates silu output intermediate
        return OpSpec(
            op_type=OpType.SILU,
            inputs=list(silu_node.op.inputs) + list(mul_node.op.inputs),
            outputs=list(mul_node.op.outputs),
            attrs={"fused": "swiglu"},
            name=f"{silu_node.name}_swiglu_fused",
        )


class FlashAttentionFusion(FusionRule):
    """Fuse attention: QK^T -> softmax -> V into a single FlashAttention kernel.

    Pattern: attn_score (MatMul) -> attn_softmax (Softmax) -> attn_v (MatMul)
    Result: Single op with combined FLOPs but O(N) memory instead of O(N^2).
    """

    @property
    def name(self) -> str:
        return "flash_attention"

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if node.op.op_type != OpType.MATMUL:
            return None
        if not node.name or "attn_score" not in node.name:
            return None
        succs = graph.successors(node)
        if len(succs) != 1:
            return None
        softmax = succs[0]
        if softmax.op.op_type != OpType.SOFTMAX:
            return None
        succs2 = graph.successors(softmax)
        if len(succs2) != 1:
            return None
        attn_v = succs2[0]
        if attn_v.op.op_type != OpType.MATMUL:
            return None
        return [node, softmax, attn_v]

    def fuse(self, nodes: list[Node]) -> OpSpec:
        qk_node = nodes[0]
        attn_v_node = nodes[2]

        # FlashAttention: total FLOPs = QK FLOPs + softmax FLOPs + V FLOPs
        total_flops = sum(n.op.flops for n in nodes)

        # Memory: Q + K + V inputs + output. NO O(N^2) score matrix.
        # Q,K from qk_node inputs, V from attn_v_node second input
        flash_inputs = list(qk_node.op.inputs)  # Q, K^T
        # Add V tensor from the attn_v matmul (second input, the value matrix)
        if len(attn_v_node.op.inputs) > 1:
            flash_inputs.append(attn_v_node.op.inputs[1])  # V
        flash_outputs = list(attn_v_node.op.outputs)  # attention output

        # Memory savings: eliminate the N^2 score matrix
        return OpSpec(
            op_type=OpType.MATMUL,
            inputs=flash_inputs,
            outputs=flash_outputs,
            attrs={
                "fused": "flash_attention",
                "_fused_flops": total_flops,
            },
            name=f"{qk_node.name}_flash_fused",
        )


class NPUFormatFusion(FusionRule):
    """NPU-specific: fuse format conversion into the compute kernel.

    On Ascend, data format conversion (ND->NZ, NCHW->NC1HWC0) can be
    absorbed into the CUBE/VECTOR kernel, eliminating a separate pass.

    This rule doesn't match graph patterns — it's applied as a cost
    reduction on NPU backends (tag nodes to skip format conversion cost).
    """

    @property
    def name(self) -> str:
        return "npu_format_fusion"

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        # Match any MatMul/Conv2d followed by an elementwise op
        if node.op.op_type not in (OpType.MATMUL, OpType.CONV2D):
            return None
        succs = graph.successors(node)
        if len(succs) != 1:
            return None
        succ = succs[0]
        if succ.op.op_type in (OpType.RELU, OpType.GELU, OpType.SILU, OpType.ADD, OpType.MUL):
            preds = graph.predecessors(succ)
            if len(preds) == 1:
                return [node, succ]
        return None

    def fuse(self, nodes: list[Node]) -> OpSpec:
        compute_op = nodes[0].op
        epilogue_op = nodes[1].op
        return OpSpec(
            op_type=compute_op.op_type,
            inputs=list(compute_op.inputs),
            outputs=list(epilogue_op.outputs),
            attrs={
                "fused": f"{compute_op.op_type.name.lower()}+{epilogue_op.op_type.name.lower()}",
                "skip_format_conversion": True,
            },
            name=f"{nodes[0].name}_npu_fused",
        )


# ============================================================
# Fusion Pass Engine
# ============================================================

# Predefined rule sets
GPU_FUSION_RULES = [
    FlashAttentionFusion(),
    SwiGLUFusion(),
    MatMulEpilogueFusion(),
    ElementwiseChainFusion(),
]

NPU_FUSION_RULES = [
    FlashAttentionFusion(),
    SwiGLUFusion(),
    MatMulEpilogueFusion(),
    ElementwiseChainFusion(),
    NPUFormatFusion(),
]


class FusionPass:
    """Applies fusion rules to a ComputeGraph, producing a new fused graph."""

    def __init__(self, rules: list[FusionRule]):
        self.rules = rules

    def apply(self, graph: ComputeGraph) -> tuple[ComputeGraph, FusionResult]:
        """Apply all fusion rules to the graph.

        Returns (fused_graph, fusion_result).
        The original graph is not modified.
        """
        original_count = graph.num_nodes
        fusions_applied = []

        # Track which nodes have been fused (by id)
        fused_ids: set[int] = set()

        # Collect all fusions first (match phase)
        pending_fusions: list[tuple[FusionRule, list[Node]]] = []

        for rule in self.rules:
            for node in graph.topo_order():
                if node.id in fused_ids:
                    continue
                match = rule.match(graph, node)
                if match is None:
                    continue
                # Check no node in match is already claimed
                if any(n.id in fused_ids for n in match):
                    continue
                pending_fusions.append((rule, match))
                for n in match:
                    fused_ids.add(n.id)

        # Build new graph (rewrite phase)
        new_graph = ComputeGraph(graph.name + "_fused")

        # Map old node id -> new node (for unfused nodes and fused replacement nodes)
        node_map: dict[int, Node] = {}

        # First, create fused replacement nodes
        fused_replacement: dict[int, Node] = {}  # old_id -> new fused node
        fused_groups: dict[int, tuple[FusionRule, list[Node]]] = {}

        for rule, match_nodes in pending_fusions:
            fused_op = rule.fuse(match_nodes)
            new_node = new_graph.add_node(fused_op, fused_op.name)

            names = [n.name or n.op.op_type.name for n in match_nodes]
            fusions_applied.append(f"{rule.name}: {' + '.join(names)}")

            # Map all nodes in the match group to the fused node
            for n in match_nodes:
                fused_replacement[n.id] = new_node

        # Then, copy unfused nodes
        for node in graph.topo_order():
            if node.id in fused_ids:
                continue
            new_node = new_graph.add_node(node.op, node.name)
            node_map[node.id] = new_node

        # Merge fused replacements into node_map
        for old_id, new_node in fused_replacement.items():
            node_map[old_id] = new_node

        # Rebuild edges
        seen_edges: set[tuple[int, int]] = set()
        for node in graph.topo_order():
            src_new = node_map.get(node.id)
            if src_new is None:
                continue
            for succ in graph.successors(node):
                dst_new = node_map.get(succ.id)
                if dst_new is None:
                    continue
                if src_new.id == dst_new.id:
                    continue  # internal edge within fused group
                edge_key = (src_new.id, dst_new.id)
                if edge_key not in seen_edges:
                    new_graph.add_edge(src_new, dst_new)
                    seen_edges.add(edge_key)

        fusion_result = FusionResult(
            original_nodes=original_count,
            fused_nodes=new_graph.num_nodes,
            fusions_applied=fusions_applied,
        )

        return new_graph, fusion_result

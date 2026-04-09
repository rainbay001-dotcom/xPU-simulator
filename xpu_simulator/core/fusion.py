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
# Aten-level fusion rules (for DispatchExtractor graphs)
# Inspired by msmodeling's compilation/patterns/
# ============================================================

class RMSNormFusion(FusionRule):
    """Fuse decomposed RMSNorm: pow -> mean -> add -> rsqrt -> mul -> [to_copy ->] mul.

    msmodeling insight: RMSNorm decomposes into 5-6 aten ops that should be
    fused into a single LAYER_NORM kernel. This eliminates 4-5 intermediate
    tensors and their memory traffic.

    Pattern: MUL(pow) -> ADD(mean) -> MUL(rsqrt) -> MUL(weight)
    with optional to_copy (dtype casts) between stages.
    """

    @property
    def name(self) -> str:
        return "rms_norm"

    _NORM_COMPONENT_OPS = {OpType.MUL, OpType.ADD}

    @staticmethod
    def _aten_name(node: Node) -> str:
        """Get the aten op short name from node attrs or name."""
        aten = node.op.attrs.get("aten", "")
        if aten:
            return aten
        return node.name or ""

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        # Look for pow (mapped as MUL)
        if node.op.op_type != OpType.MUL:
            return None
        # Check for pow via aten attr or node name
        aten = self._aten_name(node)
        if "pow" not in aten and "pow" not in (node.name or ""):
            return None

        # Walk the chain: pow -> mean -> (add for eps) -> rsqrt -> mul -> (to_copy) -> mul
        chain = [node]
        current = node
        for _ in range(8):  # max 8 steps to find full RMSNorm
            succs = graph.successors(current)
            if len(succs) != 1:
                break
            succ = succs[0]
            # Accept MUL, ADD (for mean/rsqrt/weight mul) and any dtype cast
            if succ.op.op_type in self._NORM_COMPONENT_OPS:
                chain.append(succ)
                current = succ
            else:
                break

        # Minimum: pow + mean + rsqrt + mul = 4 ops
        if len(chain) < 4:
            return None

        # Verify the chain looks like RMSNorm by checking aten names and node names
        aten_names = " ".join(self._aten_name(n) for n in chain).lower()
        node_names = " ".join(n.name or "" for n in chain).lower()
        all_names = aten_names + " " + node_names
        # Must have pow and (mean or rsqrt) pattern
        if "pow" not in all_names:
            return None
        if "mean" not in all_names and "rsqrt" not in all_names:
            return None

        return chain

    def fuse(self, nodes: list[Node]) -> OpSpec:
        first = nodes[0]
        last = nodes[-1]
        # RMSNorm: read input + weight, write output
        # Find the actual input (predecessor of pow that isn't in the chain)
        fused_inputs = list(first.op.inputs[:1])  # input tensor
        fused_outputs = list(last.op.outputs)
        return OpSpec(
            op_type=OpType.LAYER_NORM,
            inputs=fused_inputs,
            outputs=fused_outputs,
            attrs={"fused": "rms_norm"},
            name=f"{first.name}_rms_norm_fused",
        )


class ResidualAddRMSNormFusion(FusionRule):
    """Fuse residual ADD + RMSNorm into a single kernel.

    msmodeling pattern: add_rms_norm / add_rms_norm2
    Eliminates the intermediate residual tensor between add and norm.

    Pattern: ADD (residual) -> fused_rms_norm (from RMSNormFusion)
    """

    @property
    def name(self) -> str:
        return "add_rms_norm"

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if node.op.op_type != OpType.ADD:
            return None
        succs = graph.successors(node)
        if len(succs) != 1:
            return None
        succ = succs[0]
        # Match fused rms_norm or raw LAYER_NORM
        if succ.op.op_type != OpType.LAYER_NORM:
            return None
        if succ.op.attrs.get("fused") == "rms_norm" or "norm" in (succ.name or "").lower():
            return [node, succ]
        return None

    def fuse(self, nodes: list[Node]) -> OpSpec:
        add_node = nodes[0]
        norm_node = nodes[1]
        # Read both residual inputs + norm weight, write norm output
        fused_inputs = list(add_node.op.inputs)
        fused_outputs = list(norm_node.op.outputs)
        return OpSpec(
            op_type=OpType.LAYER_NORM,
            inputs=fused_inputs,
            outputs=fused_outputs,
            attrs={"fused": "add_rms_norm"},
            name=f"{add_node.name}_add_rms_norm_fused",
        )


class RoPEFusion(FusionRule):
    """Fuse rotary position embedding: cos/sin -> mul -> cat -> mul -> add.

    msmodeling pattern: apply_rope
    RoPE decomposes into cos, sin, neg, mul, cat, mul, add ops.
    Fuse into a single kernel that reads (Q/K, cos, sin) and writes embedded output.

    Pattern: ROPE ops in sequence (cos, sin, neg, mul, add sequences)
    """

    @property
    def name(self) -> str:
        return "rope"

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if node.op.op_type != OpType.ROPE:
            return None
        # Check for cos via aten attr or node name
        aten = node.op.attrs.get("aten", "")
        if "cos" not in aten and "cos" not in (node.name or ""):
            return None

        # Collect all ROPE/MUL/ADD ops reachable in a short chain
        chain = [node]
        visited = {node.id}
        frontier = [node]

        for _ in range(15):  # max steps
            next_frontier = []
            for n in frontier:
                for succ in graph.successors(n):
                    if succ.id in visited:
                        continue
                    if succ.op.op_type in (OpType.ROPE, OpType.MUL, OpType.ADD, OpType.RESHAPE):
                        # Check it's part of RoPE (not some other mul/add)
                        if succ.op.op_type in (OpType.MUL, OpType.ADD):
                            # Only include if it has a ROPE predecessor or is in our chain
                            preds = graph.predecessors(succ)
                            if not any(p.id in visited for p in preds):
                                continue
                        chain.append(succ)
                        visited.add(succ.id)
                        next_frontier.append(succ)
                    elif succ.op.op_type == OpType.RESHAPE and len(graph.successors(succ)) == 1:
                        # Skip through reshapes within RoPE
                        chain.append(succ)
                        visited.add(succ.id)
                        next_frontier.append(succ)
            frontier = next_frontier
            if not frontier:
                break

        if len(chain) < 3:
            return None

        return chain

    def fuse(self, nodes: list[Node]) -> OpSpec:
        first = nodes[0]
        last = nodes[-1]
        fused_inputs = list(first.op.inputs[:1])  # Q or K tensor
        fused_outputs = list(last.op.outputs)
        total_flops = sum(n.op.flops for n in nodes)
        return OpSpec(
            op_type=OpType.ROPE,
            inputs=fused_inputs,
            outputs=fused_outputs,
            attrs={"fused": "rope", "_fused_flops": total_flops},
            name=f"{first.name}_rope_fused",
        )


class GroupedMatMulSwiGLUFusion(FusionRule):
    """Fuse grouped_matmul -> split -> swiglu for MoE experts.

    msmodeling pattern: grouped_matmul_swiglu
    In MoE layers, the gate/up projection is a single grouped matmul that gets
    split into gate+up, then SwiGLU is applied. Fusing eliminates the split buffer.

    Pattern: MATMUL (grouped) -> SILU -> MUL (already SwiGLU-like in same batch)
    """

    @property
    def name(self) -> str:
        return "grouped_matmul_swiglu"

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if node.op.op_type != OpType.MATMUL:
            return None
        # Look for grouped matmul pattern (3D weight tensor)
        if len(node.op.inputs) < 2:
            return None
        weight = node.op.inputs[1]
        if len(weight.shape) != 3:  # [num_experts, hidden, intermediate]
            return None

        # Check for SILU in successors (may go through reshapes)
        for succ in graph.successors(node):
            if succ.op.op_type == OpType.SILU:
                silu_succs = graph.successors(succ)
                if len(silu_succs) == 1 and silu_succs[0].op.op_type == OpType.MUL:
                    return [node, succ, silu_succs[0]]
        return None

    def fuse(self, nodes: list[Node]) -> OpSpec:
        matmul = nodes[0]
        mul_out = nodes[-1]
        total_flops = sum(n.op.flops for n in nodes)
        return OpSpec(
            op_type=OpType.MATMUL,
            inputs=list(matmul.op.inputs),
            outputs=list(mul_out.op.outputs),
            attrs={"fused": "grouped_matmul_swiglu", "_fused_flops": total_flops},
            name=f"{matmul.name}_gm_swiglu_fused",
        )


class DispatchFlashAttentionFusion(FusionRule):
    """Fuse attention from aten-level ops: bmm (scores) -> softmax -> bmm (values).

    Unlike FlashAttentionFusion which relies on node names like "attn_score",
    this rule works on DispatchExtractor graphs using structural matching:
    BMM -> Softmax -> BMM pattern with compatible shapes.
    """

    @property
    def name(self) -> str:
        return "dispatch_flash_attention"

    def match(self, graph: ComputeGraph, node: Node) -> Optional[list[Node]]:
        if node.op.op_type != OpType.MATMUL:
            return None
        # First matmul should have batch dims (3D+ output = attention scores)
        if not node.op.outputs or len(node.op.outputs[0].shape) < 3:
            return None
        # Output should be square-ish in last two dims (S x S attention matrix)
        out_shape = node.op.outputs[0].shape
        if out_shape[-2] != out_shape[-1] and out_shape[-1] > 1:
            # Not necessarily square, but the last dim should match seq_len
            pass

        succs = graph.successors(node)
        # May have scaling (MUL) or ADD (mask) between bmm and softmax
        chain = [node]
        current = node
        for _ in range(3):  # allow up to 3 intermediate ops (scale, mask, etc.)
            succs = graph.successors(current)
            if len(succs) != 1:
                return None
            succ = succs[0]
            if succ.op.op_type == OpType.SOFTMAX:
                chain.append(succ)
                current = succ
                break
            elif succ.op.op_type in (OpType.MUL, OpType.ADD):
                chain.append(succ)
                current = succ
            else:
                return None
        else:
            return None

        # After softmax, expect matmul (attention * V)
        softmax_succs = graph.successors(current)
        if len(softmax_succs) != 1:
            return None
        attn_v = softmax_succs[0]
        if attn_v.op.op_type != OpType.MATMUL:
            return None
        chain.append(attn_v)

        return chain

    def fuse(self, nodes: list[Node]) -> OpSpec:
        qk_node = nodes[0]
        attn_v_node = nodes[-1]
        total_flops = sum(n.op.flops for n in nodes)

        flash_inputs = list(qk_node.op.inputs)
        if len(attn_v_node.op.inputs) > 1:
            flash_inputs.append(attn_v_node.op.inputs[1])
        flash_outputs = list(attn_v_node.op.outputs)

        return OpSpec(
            op_type=OpType.MATMUL,
            inputs=flash_inputs,
            outputs=flash_outputs,
            attrs={"fused": "flash_attention", "_fused_flops": total_flops},
            name=f"{qk_node.name}_flash_fused",
        )


# ============================================================
# Fusion Pass Engine
# ============================================================

# Predefined rule sets (for ConfigExtractor graphs — named logical ops)
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

# Dispatch-level fusion rules (for DispatchExtractor graphs — aten ops)
# Applied in order: RMSNorm first, then compound patterns, then general
DISPATCH_FUSION_RULES = [
    # Level 0: primitive fusions
    RMSNormFusion(),
    RoPEFusion(),
    # Level 1: compound fusions (depend on level 0)
    ResidualAddRMSNormFusion(),
    # Level 2: attention + MoE fusions
    DispatchFlashAttentionFusion(),
    GroupedMatMulSwiGLUFusion(),
    SwiGLUFusion(),
    # Level 3: general epilogue fusions
    MatMulEpilogueFusion(),
    ElementwiseChainFusion(),
]

DISPATCH_NPU_FUSION_RULES = DISPATCH_FUSION_RULES + [NPUFormatFusion()]


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

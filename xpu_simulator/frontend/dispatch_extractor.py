"""Extract computation graphs by intercepting PyTorch ops via TorchDispatchMode.

Unlike TorchGraphExtractor (FX tracing), this works with dynamic control flow,
MoE routing, custom attention patterns, and any model that runs in PyTorch —
it records ops as they actually execute on meta tensors.

Usage:
    extractor = DispatchExtractor()
    graph = extractor.extract(model, example_inputs)

    # Or with a model factory (for models that can't be moved to meta after init):
    graph = extractor.extract_from_factory(
        factory=lambda: AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V3", torch_dtype=torch.float16
        ),
        example_inputs_factory=lambda model: {
            "input_ids": torch.randint(0, 1000, (1, 1024)),
        },
    )
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from ..core.graph import ComputeGraph
from ..core.operator import OpSpec, TensorSpec, OpType, Dtype
from .base import GraphExtractor
from ._torch_utils import torch_dtype_to_dtype

logger = logging.getLogger(__name__)


# ---- aten op -> OpType mapping ----
# Keys are the str(func) representation of aten ops from TorchDispatchMode
_ATEN_OP_MAP: Dict[str, OpType] = {}


def _register_aten(names: List[str], op_type: OpType):
    for n in names:
        _ATEN_OP_MAP[n] = op_type


_register_aten([
    "aten.mm.default", "aten.bmm.default", "aten.matmul.default",
    "aten.addmm.default", "aten.linear.default",
], OpType.MATMUL)

_register_aten([
    "aten.conv2d.default", "aten.convolution.default",
    "aten._convolution.default",
], OpType.CONV2D)

_register_aten(["aten.relu.default", "aten.relu_.default"], OpType.RELU)
_register_aten(["aten.gelu.default"], OpType.GELU)
_register_aten(["aten.silu.default", "aten.silu_.default"], OpType.SILU)
_register_aten([
    "aten.add.Tensor", "aten.add_.Tensor", "aten.add.default",
    "aten.sub.Tensor",
], OpType.ADD)
_register_aten([
    "aten.mul.Tensor", "aten.mul_.Tensor", "aten.mul.default",
    "aten.mul.Scalar", "aten.div.Tensor", "aten.div.Scalar",
], OpType.MUL)

_register_aten([
    "aten.layer_norm.default", "aten.native_layer_norm.default",
    "aten.group_norm.default",
], OpType.LAYER_NORM)

# RMSNorm decomposes to pow+mean+rsqrt+mul — classify components appropriately
_register_aten(["aten.rms_norm.default"], OpType.LAYER_NORM)
_register_aten([
    "aten.pow.Tensor_Scalar", "aten.pow.Tensor_Tensor",
], OpType.MUL)  # elementwise power
_register_aten([
    "aten.mean.dim", "aten.sum.dim_IntList", "aten.sum.default",
], OpType.ADD)   # reductions
_register_aten(["aten.rsqrt.default", "aten.sqrt.default"], OpType.MUL)

_register_aten([
    "aten.softmax.int", "aten.softmax.default",
    "aten._softmax.default", "aten._safe_softmax.default",
], OpType.SOFTMAX)

# dtype conversion (memory-bound copy)
_register_aten([
    "aten._to_copy.default", "aten.to.dtype", "aten.to.device",
], OpType.ADD)  # memory-bandwidth bound, same as elementwise

_register_aten(["aten.embedding.default"], OpType.EMBEDDING)
_register_aten([
    "aten.gather.default", "aten.index_select.default",
    "aten.index.Tensor",
], OpType.GATHER)

_register_aten(["aten.topk.default"], OpType.TOP_K)

# RoPE components
_register_aten([
    "aten.cos.default", "aten.sin.default",
    "aten.neg.default",
], OpType.ROPE)

# Grouped matmul (MoE experts)
_register_aten(["aten._grouped_mm.default"], OpType.MATMUL)

# Mask creation & comparison (minimal cost)
_register_aten([
    "aten.arange.default", "aten.arange.start",
    "aten.ones.default", "aten.zeros.default",
    "aten.full.default", "aten.scalar_tensor.default",
    "aten.tril.default", "aten.triu.default",
    "aten.where.self", "aten.where.ScalarSelf",
    "aten.ge.Scalar", "aten.gt.Scalar", "aten.lt.Scalar", "aten.le.Scalar",
    "aten.eq.Scalar", "aten.ne.Scalar",
    "aten.masked_fill.Scalar", "aten.masked_fill_.Scalar",
    # MoE routing ops (minimal cost)
    "aten.sort.default", "aten.sort.stable",
    "aten.index_put_.default", "aten.index_put.default",
    "aten.scatter_.value", "aten.scatter_.src",
    "aten.empty_like.default", "aten.zeros_like.default",
    "aten.histc.default", "aten.cumsum.default",
    "aten.bitwise_not.default",
], OpType.RESHAPE)  # negligible cost, treat as zero-cost

# Sigmoid (MoE gating)
_register_aten(["aten.sigmoid.default"], OpType.SILU)  # similar elementwise
# In-place div
_register_aten(["aten.div_.Tensor", "aten.div_.Scalar"], OpType.MUL)

_register_aten([
    "aten.transpose.int", "aten.permute.default", "aten.t.default",
], OpType.TRANSPOSE)

_register_aten([
    "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
    "aten.expand.default", "aten.contiguous.default",
    "aten.slice.Tensor", "aten.select.int",
    "aten.unsqueeze.default", "aten.squeeze.dim",
    "aten.split.Tensor", "aten.split_with_sizes.default",
    "aten.cat.default", "aten.stack.default",
    "aten.detach.default", "aten.alias.default",
    "aten.clone.default",
], OpType.RESHAPE)

# Collective communication
_register_aten(["c10d.allreduce_.default"], OpType.ALL_REDUCE)
_register_aten(["c10d.allgather_.default"], OpType.ALL_GATHER)
_register_aten(["c10d.alltoall_.default"], OpType.ALL_TO_ALL)
_register_aten(["c10d.reduce_scatter_.default"], OpType.REDUCE_SCATTER)

# Ops to skip (zero cost, no meaningful compute)
_SKIP_OPS = {
    "aten.detach.default", "aten.alias.default", "aten.t.default",
    "aten.view.default", "aten._unsafe_view.default",
    "aten.expand.default", "aten.contiguous.default",
    "aten.slice.Tensor", "aten.select.int",
    "aten.unsqueeze.default", "aten.squeeze.dim",
    "aten.split.Tensor", "aten.split_with_sizes.default",
    "aten.permute.default", "aten.reshape.default",
    "aten.clone.default",
    # Mask/index creation (zero compute)
    "aten.arange.default", "aten.arange.start",
    "aten.ones.default", "aten.zeros.default",
    "aten.full.default", "aten.scalar_tensor.default",
    "aten.tril.default", "aten.triu.default",
    # MoE routing bookkeeping (negligible cost)
    "aten.empty_like.default", "aten.zeros_like.default",
    "aten.index_put_.default", "aten.index_put.default",
    "aten.scatter_.value", "aten.scatter_.src",
    "aten.histc.default", "aten.cumsum.default",
    "aten.bitwise_not.default",
    "aten.sort.default", "aten.sort.stable",
}


def _tensor_spec(t: torch.Tensor) -> TensorSpec:
    """Build a TensorSpec from a torch.Tensor (or meta tensor)."""
    shape = tuple(t.shape)
    dtype = torch_dtype_to_dtype(t.dtype)
    return TensorSpec(shape, dtype)


def _collect_tensors(args: tuple, kwargs: dict) -> List[torch.Tensor]:
    """Recursively collect all tensor arguments."""
    tensors = []
    for a in args:
        if isinstance(a, torch.Tensor):
            tensors.append(a)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    return tensors


def _collect_output_tensors(out: Any) -> List[torch.Tensor]:
    """Collect tensors from op output (may be tensor, tuple, or list)."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        tensors = []
        for item in out:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
        return tensors
    return []


class _ModuleTracker:
    """Track which nn.Module is active using forward hooks.

    Maintains a stack of module names to attribute aten ops
    to their source module (e.g., "model.layers.3.self_attn.q_a_proj").
    """

    def __init__(self, root: nn.Module):
        self._stack: List[str] = []
        self._handles: List[Any] = []
        self._install(root, "")

    def _install(self, module: nn.Module, prefix: str):
        """Register pre/post forward hooks on all submodules."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            def _pre_hook(m, inp, _fn=full_name):
                self._stack.append(_fn)
                return None  # Don't modify inputs

            def _post_hook(m, inp, out, _fn=full_name):
                if self._stack and self._stack[-1] == _fn:
                    self._stack.pop()
                return None  # Don't modify output

            h1 = child.register_forward_pre_hook(_pre_hook)
            h2 = child.register_forward_hook(_post_hook)
            self._handles.extend([h1, h2])
            self._install(child, full_name)

    @property
    def current_module(self) -> str:
        """Return the deepest active module name."""
        return self._stack[-1] if self._stack else ""

    def remove(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


class _RecordingDispatch(TorchDispatchMode):
    """TorchDispatchMode that records every aten op with shapes."""

    def __init__(self, module_tracker: Optional[_ModuleTracker] = None):
        super().__init__()
        self.recorded_ops: List[Dict[str, Any]] = []
        # Track tensor data_ptr -> (op_index, output_index) for edge building
        self._tensor_source: Dict[int, Tuple[int, int]] = {}
        self._module_tracker = module_tracker

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        func_name = str(func.overloadpacket) + "." + func._overloadname
        # Normalize: "aten.mm.default"
        if not func_name.startswith("aten.") and hasattr(func, '__module__'):
            func_name = f"{func.__module__}.{func.__name__}"

        input_tensors = _collect_tensors(args, kwargs)
        output_tensors = _collect_output_tensors(out)

        # Record input edges (which previous op produced each input tensor)
        input_sources = []
        for t in input_tensors:
            ptr = t.data_ptr()
            if ptr in self._tensor_source:
                input_sources.append(self._tensor_source[ptr])
            else:
                input_sources.append(None)

        op_idx = len(self.recorded_ops)

        # Capture source module path
        module_path = ""
        if self._module_tracker:
            module_path = self._module_tracker.current_module

        self.recorded_ops.append({
            "func_name": func_name,
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
            "input_sources": input_sources,
            "op_idx": op_idx,
            "module_path": module_path,
        })

        # Register output tensors as sources for future ops
        for i, t in enumerate(output_tensors):
            self._tensor_source[t.data_ptr()] = (op_idx, i)

        return out


class DispatchExtractor(GraphExtractor):
    """Extract a ComputeGraph by intercepting aten ops via TorchDispatchMode.

    This works with any PyTorch model — including dynamic control flow,
    MoE routing, and custom attention — because it records ops at execution
    time rather than tracing the code statically.
    """

    def __init__(self, dtype: Dtype = Dtype.FP16, skip_reshapes: bool = True):
        """
        Args:
            dtype: Default dtype for tensors without dtype info.
            skip_reshapes: If True, skip zero-cost reshape/view ops in the graph.
        """
        super().__init__(dtype)
        self.skip_reshapes = skip_reshapes

    def extract(
        self,
        model: nn.Module,
        example_inputs: Union[
            Tuple[torch.Tensor, ...],
            Dict[str, torch.Tensor],
        ],
        graph_name: str = "model",
    ) -> ComputeGraph:
        """
        Run the model with TorchDispatchMode interception and build a ComputeGraph.

        The model should be on meta device or CPU. If on CPU, tensors will be
        moved to meta device automatically to avoid actual computation.

        Args:
            model: PyTorch module.
            example_inputs: Tuple of tensors or dict of keyword args.
            graph_name: Name for the resulting graph.
        """
        # Move model to meta device for shape-only execution
        model = self._to_meta(model)
        if isinstance(example_inputs, dict):
            example_inputs = {
                k: v.to("meta") if isinstance(v, torch.Tensor) else v
                for k, v in example_inputs.items()
            }
        else:
            example_inputs = tuple(
                t.to("meta") if isinstance(t, torch.Tensor) else t
                for t in example_inputs
            )

        # Record ops with module tracking
        tracker = _ModuleTracker(model)
        recorder = _RecordingDispatch(module_tracker=tracker)
        try:
            with recorder, torch.no_grad():
                if isinstance(example_inputs, dict):
                    model(**example_inputs)
                else:
                    model(*example_inputs)
        finally:
            tracker.remove()

        # Build ComputeGraph from recorded ops
        return self._build_graph(recorder.recorded_ops, graph_name)

    def extract_from_config(
        self,
        model_id: str,
        batch_size: int = 1,
        seq_len: int = 1024,
        graph_name: Optional[str] = None,
        model_dtype: torch.dtype = torch.bfloat16,
        num_hidden_layers: Optional[int] = None,
    ) -> ComputeGraph:
        """
        Load a HuggingFace model and extract its graph.

        Requires `transformers` to be installed.

        Args:
            model_id: HuggingFace model ID (e.g., "deepseek-ai/DeepSeek-V3").
            batch_size: Batch size for example inputs.
            seq_len: Sequence length for example inputs.
            graph_name: Name for the graph (defaults to model_id).
            model_dtype: dtype for model weights (default BF16).
            num_hidden_layers: Override layer count (for faster testing).
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=model_dtype)
        model.eval()

        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device="meta"
        )

        return self.extract(
            model,
            {"input_ids": input_ids},
            graph_name=graph_name or model_id.split("/")[-1],
        )

    def _to_meta(self, model: nn.Module) -> nn.Module:
        """Move model to meta device if not already there."""
        # Check if already on meta
        try:
            p = next(model.parameters())
            if p.device.type == "meta":
                return model
        except StopIteration:
            pass
        return model.to("meta")

    def _build_graph(
        self,
        recorded_ops: List[Dict[str, Any]],
        graph_name: str,
    ) -> ComputeGraph:
        """Convert recorded aten ops into a ComputeGraph."""
        graph = ComputeGraph(graph_name)
        self._seen_names: Dict[str, int] = {}

        # Map op_idx -> ComputeGraph Node (only for non-skipped ops)
        idx_to_node = {}
        # For skipped ops, track passthrough: skipped op -> the source op it forwards
        passthrough: Dict[int, Optional[Tuple[int, int]]] = {}

        for record in recorded_ops:
            func_name = record["func_name"]
            op_idx = record["op_idx"]

            # Skip zero-cost reshape ops
            if self.skip_reshapes and func_name in _SKIP_OPS:
                # Pass through: this op's output inherits its input's source
                if record["input_sources"] and record["input_sources"][0] is not None:
                    passthrough[op_idx] = record["input_sources"][0]
                else:
                    passthrough[op_idx] = None
                continue

            # Resolve op type
            op_type = _ATEN_OP_MAP.get(func_name, OpType.UNKNOWN)
            if op_type == OpType.UNKNOWN:
                logger.debug("Unknown aten op: %s", func_name)

            # Build input/output TensorSpecs
            input_tensors = record["input_tensors"]

            # aten.addmm(bias, input, weight) -> reorder to (input, weight) for MATMUL
            if "addmm" in func_name and len(input_tensors) == 3:
                input_tensors = [input_tensors[1], input_tensors[2]]

            input_specs = [_tensor_spec(t) for t in input_tensors]
            output_specs = [_tensor_spec(t) for t in record["output_tensors"]]

            if not input_specs and not output_specs:
                continue

            if not output_specs:
                # Some ops (like in-place) may not have separate output tensors
                if input_specs:
                    output_specs = [input_specs[0]]

            # Name the op (with layer prefix from module tracker)
            module_path = record.get("module_path", "")
            name = self._name_op(func_name, op_idx, module_path)
            # Disambiguate duplicate names within a graph
            if name in self._seen_names:
                self._seen_names[name] += 1
                name = f"{name}_{self._seen_names[name]}"
            else:
                self._seen_names[name] = 0

            # Store the aten op short name for fusion pattern matching
            aten_parts = func_name.split(".")
            aten_short = aten_parts[1] if len(aten_parts) >= 2 else func_name

            op = OpSpec(
                op_type=op_type,
                inputs=input_specs,
                outputs=output_specs,
                attrs={"aten": aten_short},
                name=name,
            )
            node = graph.add_node(op, name)
            idx_to_node[op_idx] = node

        # Add edges
        for record in recorded_ops:
            op_idx = record["op_idx"]
            if op_idx not in idx_to_node:
                continue

            dst_node = idx_to_node[op_idx]

            for source in record["input_sources"]:
                if source is None:
                    continue

                # Resolve through passthrough chain
                src_op_idx = self._resolve_source(source[0], passthrough)
                if src_op_idx is not None and src_op_idx in idx_to_node:
                    src_node = idx_to_node[src_op_idx]
                    if src_node != dst_node:
                        graph.add_edge(src_node, dst_node)

        return graph

    def _resolve_source(
        self, op_idx: int, passthrough: Dict[int, Optional[Tuple[int, int]]]
    ) -> Optional[int]:
        """Follow passthrough chain to find the real source op."""
        visited = set()
        while op_idx in passthrough:
            if op_idx in visited:
                return None  # cycle guard
            visited.add(op_idx)
            entry = passthrough[op_idx]
            if entry is None:
                return None
            op_idx = entry[0]
        return op_idx

    @staticmethod
    def _name_op(func_name: str, idx: int, module_path: str = "") -> str:
        """Generate a human-readable name with layer prefix.

        Maps module paths like "model.layers.3.self_attn.q_a_proj" to
        names like "L3.q_a_proj.mm_42" so the HTML report can group
        ops by layer and subcomponent.
        """
        # "aten.mm.default" -> "mm"
        parts = func_name.split(".")
        short = parts[1] if len(parts) >= 2 else func_name

        if not module_path:
            return f"{short}_{idx}"

        # Extract layer index and subcomponent from module path
        # e.g., "model.layers.3.self_attn.q_a_proj" -> layer=3, sub="self_attn.q_a_proj"
        layer_idx = None
        sub_path = module_path

        mp_parts = module_path.split(".")
        for i, p in enumerate(mp_parts):
            if p == "layers" and i + 1 < len(mp_parts):
                try:
                    layer_idx = int(mp_parts[i + 1])
                    # Everything after "layers.N." is the subcomponent
                    sub_path = ".".join(mp_parts[i + 2:])
                    break
                except ValueError:
                    pass

        if layer_idx is not None:
            # Map common subcomponent names for better HTML categorization
            sub = _classify_dispatch_sub(sub_path, short)
            return f"L{layer_idx}.{sub}"

        # Non-layer ops (embedding, final norm, lm_head)
        return f"{short}_{idx}"


def _classify_dispatch_sub(sub_path: str, op_short: str) -> str:
    """Classify a dispatch subcomponent for HTML architecture overview.

    Maps module subpaths to names compatible with the HTML report's
    _classify_sub() function (e.g., "attn_score", "ffn.w1", "attn_norm").
    """
    s = sub_path.lower()

    # --- Attention norms ---
    if "input_layernorm" in s or "norm1" in s:
        return "attn_norm"
    if "post_attention_layernorm" in s or "norm2" in s:
        return "ffn_norm"

    # --- MLA attention components ---
    if "q_a_proj" in s or "q_a_layernorm" in s:
        return "wq_a"
    if "q_b_proj" in s:
        return "wq_b"
    if "q_proj" in s:
        return "wq_a"
    if "q_norm" in s or "q_a_layernorm" in s:
        return "q_norm"
    if "kv_a_proj" in s:
        return "wkv_a"
    if "kv_a_layernorm" in s or "kv_norm" in s:
        return "kv_norm"
    if "kv_b_proj" in s:
        return "wkv_b"
    if "k_proj" in s:
        return "wkv_a"
    if "v_proj" in s:
        return "wkv_b"
    if "o_proj" in s:
        return "wo"

    # --- Attention compute ---
    if "self_attn" in s or "attn" in s:
        if op_short in ("bmm", "mm", "matmul"):
            return "attn_score"
        if op_short in ("_safe_softmax", "softmax"):
            return "attn_softmax"
        if "rope" in s or op_short in ("cos", "sin"):
            return "rope"
        return f"attn.{op_short}"

    # --- MoE ---
    if "mlp" in s or "moe" in s:
        if "gate" in s and "experts" not in s and "up" not in s:
            # MoE router gate (softmax + topk)
            return "moe.gate_softmax"
        if "shared_expert" in s or "shared" in s:
            if "gate" in s:
                return "moe.shared.w1"
            if "up" in s:
                return "moe.shared.w3"
            if "down" in s:
                return "moe.shared.w2"
            return f"moe.shared.{op_short}"
        if "experts" in s:
            return f"moe.experts.{op_short}"
        # Dense FFN
        if "gate_proj" in s or "w1" in s:
            return "ffn.w1"
        if "up_proj" in s or "w3" in s:
            return "ffn.w3"
        if "down_proj" in s or "w2" in s:
            return "ffn.w2"
        if op_short == "silu":
            return "ffn.silu"
        if op_short == "mul":
            return "ffn.mul"
        return f"ffn.{op_short}"

    # --- Indexer (DSA) ---
    if "indexer" in s:
        return f"indexer_{op_short}"

    # Fallback
    return f"{sub_path}.{op_short}" if sub_path else op_short

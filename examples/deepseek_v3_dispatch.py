"""
DeepSeek V3 671B — DispatchExtractor simulation with TP=1 and TP=3.

Extracts the computation graph via TorchDispatchMode (aten-level ops),
applies dispatch-level fusion, then evaluates on GPU and NPU targets.

For TP=3, applies post-extraction tensor parallelism by sharding matmul
dimensions and inserting ALL_REDUCE communication ops.

Usage:
    source .venv312/bin/activate
    python examples/deepseek_v3_dispatch.py [batch_size] [seq_len]
"""

import sys
import math
from copy import deepcopy
from collections import Counter

sys.path.insert(0, ".")

import torch

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph, Node
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.parallel import ParallelConfig, InterconnectSpec
from xpu_simulator.core.cost_model import CommAwareCostModel
from xpu_simulator.core.fusion import (
    FusionPass, DISPATCH_FUSION_RULES, DISPATCH_NPU_FUSION_RULES,
)
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.frontend.dispatch_extractor import DispatchExtractor
from xpu_simulator.utils.html_report import export_html_report


def apply_tp_to_dispatch_graph(
    graph: ComputeGraph, tp_size: int, dtype: Dtype = Dtype.BF16,
) -> ComputeGraph:
    """Apply tensor parallelism to a dispatch-extracted graph.

    Strategy (mirrors how ConfigExtractor/GraphBuilder handles TP):
    - Each MATMUL: shard one dimension by tp_size
      - For attention Q/K/V projections and FFN gate/up: column-parallel (shard N)
      - For attention output proj and FFN down: row-parallel (shard K), + ALL_REDUCE
    - Non-MATMUL ops: keep as-is (element-wise ops work on sharded tensors)

    Heuristic to distinguish column vs row parallel:
    - In a transformer, linear layers alternate: column-parallel, then row-parallel
    - Attention: QKV are column-parallel, output is row-parallel
    - FFN: gate/up are column-parallel, down is row-parallel
    - We detect row-parallel as: output dim < input dim (projecting down to hidden_size)
    """
    tp_graph = ComputeGraph(f"{graph.name}_tp{tp_size}")
    node_map: dict[int, Node] = {}

    for node in graph.topo_order():
        if node.op.op_type == OpType.MATMUL and len(node.op.inputs) >= 2:
            inp = node.op.inputs[0]
            weight = node.op.inputs[1]
            out = node.op.outputs[0] if node.op.outputs else None

            # Determine column vs row parallel
            # Column parallel: shard output dim (N)
            # Row parallel: shard input dim (K), needs ALL_REDUCE after
            is_row_parallel = False
            if len(inp.shape) == 2 and len(weight.shape) == 2:
                M, K = inp.shape
                _, N = weight.shape
                # Row-parallel heuristic: output dim <= input K dim
                # (output projection, FFN down projection)
                is_row_parallel = N <= K and K > N
            elif len(inp.shape) >= 3 and len(weight.shape) >= 2:
                K = inp.shape[-1]
                N = weight.shape[-1]
                is_row_parallel = N <= K and K > N

            if is_row_parallel:
                # Row-parallel: shard K dimension
                new_inputs = list(node.op.inputs)
                if len(inp.shape) == 2:
                    M, K = inp.shape
                    K_shard = K // tp_size
                    new_inputs[0] = TensorSpec((M, K_shard), inp.dtype)
                    new_inputs[1] = TensorSpec((K_shard, weight.shape[-1]), weight.dtype)
                else:
                    # Batched
                    new_shape = list(inp.shape)
                    new_shape[-1] = inp.shape[-1] // tp_size
                    new_inputs[0] = TensorSpec(tuple(new_shape), inp.dtype)
                    w_shape = list(weight.shape)
                    w_shape[-2] = weight.shape[-2] // tp_size
                    new_inputs[1] = TensorSpec(tuple(w_shape), weight.dtype)

                new_op = OpSpec(OpType.MATMUL, new_inputs, list(node.op.outputs),
                                attrs=node.op.attrs, name=node.op.name)
                mm_node = tp_graph.add_node(new_op, node.name)
                node_map[node.id] = mm_node

                # Add ALL_REDUCE after row-parallel
                out_spec = node.op.outputs[0] if node.op.outputs else new_inputs[0]
                ar_op = OpSpec(OpType.ALL_REDUCE,
                               [TensorSpec(out_spec.shape, dtype)],
                               [TensorSpec(out_spec.shape, dtype)],
                               attrs={"n_ranks": tp_size},
                               name=f"{node.name or 'mm'}_all_reduce")
                ar_node = tp_graph.add_node(ar_op, ar_op.name)
                tp_graph.add_edge(mm_node, ar_node)
                # Successors should connect to the ALL_REDUCE node
                node_map[node.id] = ar_node
                # Keep a separate ref so we can connect predecessors to mm_node
                node_map[(-1, node.id)] = mm_node

            else:
                # Column-parallel: shard N dimension
                new_inputs = list(node.op.inputs)
                new_outputs = list(node.op.outputs)
                if len(weight.shape) == 2:
                    K_w, N = weight.shape
                    N_shard = N // tp_size
                    new_inputs[1] = TensorSpec((K_w, N_shard), weight.dtype)
                    if new_outputs and len(new_outputs[0].shape) == 2:
                        M_o = new_outputs[0].shape[0]
                        new_outputs[0] = TensorSpec((M_o, N_shard), new_outputs[0].dtype)
                elif len(weight.shape) >= 3:
                    w_shape = list(weight.shape)
                    w_shape[-1] = weight.shape[-1] // tp_size
                    new_inputs[1] = TensorSpec(tuple(w_shape), weight.dtype)
                    if new_outputs:
                        o_shape = list(new_outputs[0].shape)
                        o_shape[-1] = new_outputs[0].shape[-1] // tp_size
                        new_outputs[0] = TensorSpec(tuple(o_shape), new_outputs[0].dtype)

                new_op = OpSpec(OpType.MATMUL, new_inputs, new_outputs,
                                attrs=node.op.attrs, name=node.op.name)
                new_node = tp_graph.add_node(new_op, node.name)
                node_map[node.id] = new_node
        else:
            # Non-matmul ops: copy as-is
            new_node = tp_graph.add_node(node.op, node.name)
            node_map[node.id] = new_node

    # Rebuild edges
    seen_edges = set()
    for node in graph.topo_order():
        # For row-parallel matmuls, predecessors connect to the matmul (not the ALL_REDUCE)
        src_new = node_map.get((-1, node.id)) or node_map.get(node.id)
        if src_new is None:
            continue
        for succ in graph.successors(node):
            dst_new = node_map.get(succ.id)
            if dst_new is None:
                continue
            if src_new.id == dst_new.id:
                continue
            # For row-parallel src, the ALL_REDUCE is the real output
            actual_src = node_map.get(node.id)
            edge_key = (actual_src.id, dst_new.id)
            if edge_key not in seen_edges:
                tp_graph.add_edge(actual_src, dst_new)
                seen_edges.add(edge_key)

    return tp_graph


def main():
    batch_size = 1
    seq_len = 4096

    # Parse args
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) > 0:
        batch_size = int(args[0])
    if len(args) > 1:
        seq_len = int(args[1])

    tp_size = 3

    print(f"\n{'='*70}")
    print(f"  DeepSeek V3 671B — DispatchExtractor Simulation")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, TP=1 vs TP={tp_size}")
    print(f"{'='*70}\n")

    # ---- Step 1: Extract graph with DispatchExtractor ----
    print("Extracting graph with DispatchExtractor (meta tensors, no GPU needed)...")
    ext = DispatchExtractor(skip_reshapes=True)
    graph = ext.extract_from_config(
        "deepseek-ai/DeepSeek-V3",
        batch_size=batch_size,
        seq_len=seq_len,
        graph_name="DeepSeek-V3-671B",
    )
    print(f"  TP=1 graph: {graph.num_nodes} aten ops, {graph.num_edges} edges")

    # Op type breakdown
    op_counts = Counter(n.op.op_type.name for n in graph.topo_order())
    print(f"  Top ops: {', '.join(f'{k}={v}' for k, v in op_counts.most_common(8))}")

    # ---- Step 2: Apply dispatch-level fusion ----
    print("\nApplying dispatch-level fusion...")
    fused_graph, fusion_result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
    print(f"  {fusion_result.original_nodes} -> {fusion_result.fused_nodes} ops "
          f"({fusion_result.nodes_eliminated} eliminated)")

    rule_counts = Counter()
    for desc in fusion_result.fusions_applied:
        rule = desc.split(":")[0].strip()
        rule_counts[rule] += 1
    for rule, count in rule_counts.most_common():
        print(f"    {rule}: {count}")

    # NPU fusion (adds NPUFormatFusion on top)
    npu_fused, npu_fusion_result = FusionPass(DISPATCH_NPU_FUSION_RULES).apply(graph)

    # ---- Step 3: Build TP=3 graph ----
    print(f"\nBuilding TP={tp_size} graph...")
    tp_graph = apply_tp_to_dispatch_graph(fused_graph, tp_size)
    comm_ops = [n for n in tp_graph.topo_order()
                if n.op.op_type in (OpType.ALL_REDUCE, OpType.ALL_GATHER,
                                     OpType.REDUCE_SCATTER, OpType.ALL_TO_ALL)]
    print(f"  TP={tp_size} graph: {tp_graph.num_nodes} ops ({len(comm_ops)} communication ops)")

    # ---- Step 4: Evaluate on all devices ----
    parallel = ParallelConfig(tp_size=tp_size)

    hw_configs = [
        ("NVIDIA A100 80GB", GPUCostModel(A100_80GB), A100_80GB),
        ("NVIDIA H100 80GB", GPUCostModel(H100_80GB), H100_80GB),
        ("Ascend 910B",      NPUCostModel(ASCEND_910B), ASCEND_910B),
        ("Ascend 910C",      NPUCostModel(ASCEND_910C), ASCEND_910C),
    ]

    results_tp1 = {}
    results_tp3 = {}
    tp_comparison_devices = []

    print(f"\n{'='*70}")
    print(f"  {'Device':<25} {'TP=1':>12} {'TP='+str(tp_size):>12} {'Speedup':>10} {'Comm%':>8}")
    print(f"  {'-'*67}")

    for name, base_model, hw_spec in hw_configs:
        # TP=1: use fused graph with appropriate cost model
        if "NPU" in name or "Ascend" in name:
            eval_graph_tp1 = npu_fused
        else:
            eval_graph_tp1 = fused_graph

        result_tp1 = PerformanceEvaluator(base_model).run(eval_graph_tp1, overlap=True)
        results_tp1[name] = result_tp1

        # TP=3: use comm-aware cost model
        if hw_spec.interconnect:
            comm_model = CommAwareCostModel(base_model, hw_spec.interconnect, parallel)
        else:
            comm_model = base_model
        result_tp3 = PerformanceEvaluator(comm_model).run(tp_graph, overlap=True)
        results_tp3[name] = result_tp3

        speedup = result_tp1.total_latency_us / result_tp3.total_latency_us
        comm_lat = sum(r.cost.latency_us for r in result_tp3.per_op
                       if r.cost.bound == "communication")
        comm_pct = comm_lat / result_tp3.total_latency_us * 100 if result_tp3.total_latency_us > 0 else 0

        tp_comparison_devices.append({
            "name": name,
            "single_us": round(result_tp1.total_latency_us, 1),
            "tp_us": round(result_tp3.total_latency_us, 1),
            "comm_us": round(comm_lat, 1),
        })

        def fmt_lat(us):
            if us >= 1e6:
                return f"{us / 1e6:.3f} s"
            return f"{us / 1e3:.1f} ms"

        print(f"  {name:<25} {fmt_lat(result_tp1.total_latency_us):>12} "
              f"{fmt_lat(result_tp3.total_latency_us):>12} "
              f"{speedup:>9.2f}x {comm_pct:>7.1f}%")

    print()

    # ---- Step 5: Generate HTML Report ----
    print("Generating HTML report...")

    # Fusion info for HTML
    fusion_info = {
        "original_nodes": fusion_result.original_nodes,
        "fused_nodes": fusion_result.fused_nodes,
        "rules_applied": dict(rule_counts),
        "extractor": "DispatchExtractor",
    }

    # TP comparison data
    tp_comparison_data = {
        "tp_size": tp_size,
        "single_ops": fused_graph.num_nodes,
        "tp_ops": tp_graph.num_nodes,
        "comm_ops": len(comm_ops),
        "devices": tp_comparison_devices,
    }

    # Combine TP=1 and TP=N results for the report
    all_results = {}
    for name in results_tp1:
        all_results[f"{name} (TP=1)"] = results_tp1[name]
    for name in results_tp3:
        all_results[f"{name} (TP={tp_size})"] = results_tp3[name]

    config = {
        "model_type": "deepseek_v3",
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "vocab_size": 129280,
        "hidden_act": "silu",
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "phase": "prefill",
        "dtype": "bf16",
        "tokens": batch_size * seq_len,
        "extractor": "DispatchExtractor",
        "tp_size": tp_size,
    }

    fname = export_html_report(
        tp_graph, all_results,
        "deepseek_v3_dispatch_report.html",
        model_name=f"DeepSeek V3 671B — DispatchExtractor (batch={batch_size}, seq={seq_len})",
        config=config,
        n_dense=3,
        fusion_info=fusion_info,
        tp_comparison=tp_comparison_data,
    )
    print(f"Exported: {fname}")

    # Also print FLOP summary
    print(f"\n--- Summary ---")
    print(f"Total FLOPs (TP=1): {results_tp1['NVIDIA A100 80GB'].total_flops / 1e12:.2f} TFLOPS")
    print(f"Graph extraction: DispatchExtractor (TorchDispatchMode)")
    print(f"Fusion: {fusion_result.nodes_eliminated} ops eliminated ({len(fusion_result.fusions_applied)} fusions)")
    print(f"TP={tp_size}: {len(comm_ops)} communication ops added")


if __name__ == "__main__":
    main()

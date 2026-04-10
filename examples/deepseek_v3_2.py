"""
DeepSeek V3.2 671B — Simulation on GPU and NPU using ConfigExtractor.

Architecture (from HuggingFace config):
  - 61 transformer layers (3 dense + 58 MoE)
  - dim=7168, inter_dim=18432 (dense FFN), moe_inter_dim=2048
  - 128 attention heads, MLA (Multi-head Latent Attention)
  - 256 routed experts, 8 activated per token, 1 shared expert
  - q_lora_rank=1536, kv_lora_rank=512
  - qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128

We model a single forward pass (prefill) for a given batch_size and seq_len.
"""

import sys
import os
sys.path.insert(0, ".")

from xpu_simulator.core.operator import Dtype
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.parallel import ParallelConfig
from xpu_simulator.core.cost_model import CommAwareCostModel
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.frontend.config_extractor import ConfigExtractor


# ---- DeepSeek V3.2 671B Config (HuggingFace format) ----
DEEPSEEK_CONFIG = {
    "model_type": "deepseek_v2",
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
}

N_DENSE_LAYERS = DEEPSEEK_CONFIG["first_k_dense_replace"]

# ---- DeepSeek V3.2 with DSA (Sparse Attention) ----
DEEPSEEK_DSA_CONFIG = {
    **DEEPSEEK_CONFIG,
    "dsa_num_indexer_heads": 8,
    "dsa_k": 2048,
    "dsa_indexer_dim": 128,
}


def build_deepseek_graph(batch_size: int, seq_len: int, dtype: Dtype = Dtype.FP16,
                         parallel: ParallelConfig = None):
    """Build computation graph for DeepSeek V3.2 forward pass (prefill)."""
    extractor = ConfigExtractor(dtype=dtype)
    return extractor.extract(DEEPSEEK_CONFIG, batch_size=batch_size,
                             seq_len=seq_len, graph_name="DeepSeek-V3.2-671B",
                             parallel_config=parallel)


def run_simulation(batch_size: int, seq_len: int, dtype: Dtype = Dtype.FP16):
    """Run DeepSeek V3.2 simulation across all devices."""

    print(f"\n{'='*70}")
    print(f"  DeepSeek V3.2 671B Simulation")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, Dtype={dtype.value}")
    print(f"{'='*70}\n")

    from xpu_simulator.core.fusion import FusionPass, GPU_FUSION_RULES, NPU_FUSION_RULES

    graph = build_deepseek_graph(batch_size, seq_len, dtype)
    print(f"Graph: {graph.num_nodes} ops, {graph.num_edges} edges\n")

    # Apply fusion
    gpu_fused, gpu_fusion_result = FusionPass(GPU_FUSION_RULES).apply(graph)
    npu_fused, npu_fusion_result = FusionPass(NPU_FUSION_RULES).apply(graph)
    print(f"GPU fusion: {gpu_fusion_result.original_nodes} -> {gpu_fusion_result.fused_nodes} ops ({gpu_fusion_result.nodes_eliminated} eliminated)")
    print(f"NPU fusion: {npu_fusion_result.original_nodes} -> {npu_fusion_result.fused_nodes} ops ({npu_fusion_result.nodes_eliminated} eliminated)")
    print()

    devices = [
        ("NVIDIA A100 80GB", GPUCostModel(A100_80GB), graph, gpu_fused),
        ("NVIDIA H100 80GB", GPUCostModel(H100_80GB), graph, gpu_fused),
        ("Ascend 910B",      NPUCostModel(ASCEND_910B), graph, npu_fused),
        ("Ascend 910C",      NPUCostModel(ASCEND_910C), graph, npu_fused),
    ]

    results = {}
    for name, cost_model, orig_g, fused_g in devices:
        evaluator = PerformanceEvaluator(cost_model)
        orig_result = evaluator.run(orig_g, overlap=True)
        fused_result = evaluator.run(fused_g, overlap=True)
        results[name] = fused_result
        speedup = orig_result.total_latency_us / fused_result.total_latency_us

        print(f"--- {name} ---")
        print(f"  Without fusion:  {orig_result.total_latency_us:,.0f} us ({orig_result.total_latency_us / 1e6:.3f} s)")
        print(f"  With fusion:     {fused_result.total_latency_us:,.0f} us ({fused_result.total_latency_us / 1e6:.3f} s)")
        print(f"  Fusion speedup:  {speedup:.2f}x")
        print(f"  Total FLOPs:     {fused_result.total_flops / 1e12:.2f} TFLOPS")
        print(f"  Total memory:    {fused_result.total_bytes / 1e9:.2f} GB")
        print(f"  Compute-bound:   {fused_result.compute_bound_count} ops")
        print(f"  Memory-bound:    {fused_result.memory_bound_count} ops")
        print(f"  Bottleneck:      {fused_result.bottleneck_op.node} ({fused_result.bottleneck_op.cost.latency_us:,.0f} us)")
        print()

    # Comparison table
    print(f"{'='*70}")
    print(f"  Comparison Summary")
    print(f"{'='*70}")
    print(f"  {'Device':<25} {'Latency':>12} {'vs A100':>10}")
    print(f"  {'-'*47}")
    a100_lat = results["NVIDIA A100 80GB"].total_latency_us
    for name, result in results.items():
        lat = result.total_latency_us
        speedup = a100_lat / lat if lat > 0 else 0
        if lat >= 1e6:
            lat_str = f"{lat / 1e6:.3f} s"
        else:
            lat_str = f"{lat / 1e3:.1f} ms"
        print(f"  {name:<25} {lat_str:>12} {speedup:>9.2f}x")
    print()

    # Top 10 most expensive ops on A100
    a100_result = results["NVIDIA A100 80GB"]
    sorted_ops = sorted(a100_result.per_op, key=lambda r: r.cost.latency_us, reverse=True)
    print(f"Top 10 most expensive ops (A100):")
    print(f"  {'Op':<40} {'Latency':>12} {'Bound':>20} {'FLOPs':>14}")
    print(f"  {'-'*86}")
    for r in sorted_ops[:10]:
        name = (r.node.name or r.node.op.op_type.name)[:38]
        lat = f"{r.cost.latency_us:,.0f} us"
        flops = f"{r.cost.flops / 1e9:.1f} GFLOPS"
        print(f"  {name:<40} {lat:>12} {r.cost.bound:>20} {flops:>14}")
    print()

    # Layer type breakdown on A100
    print(f"Layer type breakdown (A100):")
    type_stats = {}
    for r in a100_result.per_op:
        op_name = r.node.name or ""
        if ".attn_score" in op_name or ".attn_v" in op_name or ".attn_softmax" in op_name:
            cat = "Attention (QKV compute)"
        elif ".indexer_" in op_name or ".top_k" in op_name:
            cat = "Attention (QKV compute)"
        elif ".wq" in op_name or ".wkv" in op_name or ".wo" in op_name:
            cat = "Attention (projections)"
        elif ".moe.experts" in op_name:
            cat = "MoE Experts"
        elif ".moe.shared" in op_name:
            cat = "MoE Shared Expert"
        elif ".moe.gate" in op_name:
            cat = "MoE Gate"
        elif ".ffn." in op_name:
            cat = "Dense FFN"
        elif "norm" in op_name:
            cat = "Norms"
        elif "rope" in op_name:
            cat = "RoPE"
        else:
            cat = "Other"

        if cat not in type_stats:
            type_stats[cat] = {"latency_us": 0, "flops": 0}
        type_stats[cat]["latency_us"] += r.cost.latency_us
        type_stats[cat]["flops"] += r.cost.flops

    total_lat = sum(v["latency_us"] for v in type_stats.values())
    print(f"  {'Category':<30} {'Latency':>12} {'% of Total':>12} {'TFLOPS':>10}")
    print(f"  {'-'*64}")
    for cat, stats in sorted(type_stats.items(), key=lambda x: -x[1]["latency_us"]):
        lat = stats["latency_us"]
        pct = lat / total_lat * 100 if total_lat > 0 else 0
        tflops = stats["flops"] / 1e12
        if lat >= 1e6:
            lat_str = f"{lat / 1e6:.3f} s"
        else:
            lat_str = f"{lat / 1e3:.1f} ms"
        print(f"  {cat:<30} {lat_str:>12} {pct:>10.1f}% {tflops:>10.2f}")
    print()


def run_dsa_comparison(batch_size: int, seq_len: int, dtype: Dtype = Dtype.FP16):
    """Compare dense MLA vs DSA at the given sequence length."""

    print(f"\n{'='*70}")
    print(f"  Dense MLA vs DSA Comparison")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, Dtype={dtype.value}")
    print(f"{'='*70}\n")

    from xpu_simulator.core.fusion import FusionPass, GPU_FUSION_RULES

    extractor = ConfigExtractor(dtype=dtype)
    dense_graph = extractor.extract(DEEPSEEK_CONFIG, batch_size=batch_size,
                                    seq_len=seq_len, graph_name="DeepSeek-V3.2-Dense-MLA")
    dsa_graph = extractor.extract(DEEPSEEK_DSA_CONFIG, batch_size=batch_size,
                                  seq_len=seq_len, graph_name="DeepSeek-V3.2-DSA")

    print(f"Dense MLA graph: {dense_graph.num_nodes} ops")
    print(f"DSA graph:       {dsa_graph.num_nodes} ops\n")

    cost_model = GPUCostModel(H100_80GB)
    evaluator = PerformanceEvaluator(cost_model)

    dense_result = evaluator.run(dense_graph, overlap=True)
    dsa_result = evaluator.run(dsa_graph, overlap=True)

    print(f"  {'Metric':<25} {'Dense MLA':>15} {'DSA':>15} {'Speedup':>10}")
    print(f"  {'-'*65}")

    d_lat = dense_result.total_latency_us
    s_lat = dsa_result.total_latency_us
    speedup = d_lat / s_lat if s_lat > 0 else 0

    def fmt_lat(us):
        if us >= 1e6:
            return f"{us / 1e6:.3f} s"
        return f"{us / 1e3:.1f} ms"

    print(f"  {'Latency':<25} {fmt_lat(d_lat):>15} {fmt_lat(s_lat):>15} {speedup:>9.2f}x")
    print(f"  {'Total FLOPs':<25} {dense_result.total_flops / 1e12:>14.2f}T {dsa_result.total_flops / 1e12:>14.2f}T {dense_result.total_flops / dsa_result.total_flops:>9.2f}x")
    print(f"  {'Total Memory':<25} {dense_result.total_bytes / 1e9:>13.2f}GB {dsa_result.total_bytes / 1e9:>13.2f}GB")
    print()


def run_tp_comparison(batch_size: int, seq_len: int, tp_size: int = 2,
                      dtype: Dtype = Dtype.FP16):
    """Compare single-device vs TP-parallel on each device."""

    print(f"\n{'='*70}")
    print(f"  Single Device vs TP={tp_size} Comparison")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, Dtype={dtype.value}")
    print(f"{'='*70}\n")

    parallel = ParallelConfig(tp_size=tp_size)

    graph_single = build_deepseek_graph(batch_size, seq_len, dtype)
    graph_tp = build_deepseek_graph(batch_size, seq_len, dtype, parallel=parallel)

    print(f"Single-device graph: {graph_single.num_nodes} ops, {graph_single.num_edges} edges")
    print(f"TP={tp_size} graph:          {graph_tp.num_nodes} ops, {graph_tp.num_edges} edges")

    # Count comm ops
    from xpu_simulator.core.operator import OpType
    comm_ops = [n for n in graph_tp.nodes
                if n.op.op_type in (OpType.ALL_REDUCE, OpType.ALL_GATHER,
                                     OpType.REDUCE_SCATTER, OpType.ALL_TO_ALL)]
    print(f"Communication ops:   {len(comm_ops)} (ALL_REDUCE, ALL_GATHER, etc.)\n")

    hw_configs = [
        ("NVIDIA H100 80GB", GPUCostModel(H100_80GB), H100_80GB),
        ("NVIDIA A100 80GB", GPUCostModel(A100_80GB), A100_80GB),
        ("Ascend 910C",      NPUCostModel(ASCEND_910C), ASCEND_910C),
    ]

    print(f"  {'Device':<25} {'Single':>12} {'TP='+str(tp_size):>12} {'Speedup':>10} {'Comm OH':>10}")
    print(f"  {'-'*69}")

    tp_comparison_data = {
        "tp_size": tp_size,
        "single_ops": graph_single.num_nodes,
        "tp_ops": graph_tp.num_nodes,
        "comm_ops": len(comm_ops),
        "devices": [],
    }

    for name, base_model, hw_spec in hw_configs:
        evaluator_single = PerformanceEvaluator(base_model)
        result_single = evaluator_single.run(graph_single, overlap=True)

        # TP: use CommAwareCostModel with the device's interconnect
        if hw_spec.interconnect:
            comm_model = CommAwareCostModel(base_model, hw_spec.interconnect, parallel)
        else:
            comm_model = base_model
        evaluator_tp = PerformanceEvaluator(comm_model)
        result_tp = evaluator_tp.run(graph_tp, overlap=True)

        speedup = result_single.total_latency_us / result_tp.total_latency_us
        # Communication overhead: sum of comm op latencies
        comm_lat = sum(r.cost.latency_us for r in result_tp.per_op
                       if r.cost.bound == "communication")
        comm_pct = comm_lat / result_tp.total_latency_us * 100 if result_tp.total_latency_us > 0 else 0

        tp_comparison_data["devices"].append({
            "name": name,
            "single_us": round(result_single.total_latency_us, 1),
            "tp_us": round(result_tp.total_latency_us, 1),
            "comm_us": round(comm_lat, 1),
        })

        def fmt_lat(us):
            if us >= 1e6:
                return f"{us / 1e6:.3f} s"
            return f"{us / 1e3:.1f} ms"

        print(f"  {name:<25} {fmt_lat(result_single.total_latency_us):>12} "
              f"{fmt_lat(result_tp.total_latency_us):>12} {speedup:>9.2f}x {comm_pct:>8.1f}%")

    print()
    return graph_tp, parallel, tp_comparison_data


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    # Default: batch=1, seq_len=1024 (typical prefill)
    batch_size = 1
    seq_len = 1024
    dsa_mode = "--dsa" in sys.argv
    tp_mode = "--tp" in sys.argv
    tp_size = 2

    # Parse --tp N
    for i, a in enumerate(sys.argv):
        if a == "--tp" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            tp_size = int(sys.argv[i + 1])

    # Filter out flags for positional args
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")
                and not (len(sys.argv) > 1 and sys.argv[sys.argv.index(a) - 1] == "--tp"
                         if a.isdigit() else False)]
    # Simpler: just take non-flag, non-tp-value args
    clean_args = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--tp":
            skip_next = True
            continue
        if a.startswith("--"):
            continue
        clean_args.append(a)

    if len(clean_args) > 0:
        batch_size = int(clean_args[0])
    if len(clean_args) > 1:
        seq_len = int(clean_args[1])

    run_simulation(batch_size, seq_len)

    if dsa_mode:
        run_dsa_comparison(batch_size, seq_len)

    tp_graph = None
    tp_parallel = None
    tp_comparison_data = None
    if tp_mode:
        tp_graph, tp_parallel, tp_comparison_data = run_tp_comparison(batch_size, seq_len, tp_size)

    # Export interactive HTML report
    from xpu_simulator.utils.html_report import export_html_report

    graph = build_deepseek_graph(batch_size, seq_len)

    config = {
        **DEEPSEEK_CONFIG,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "phase": "prefill",
        "dtype": "fp16",
        "tokens": batch_size * seq_len,
    }
    if tp_mode:
        config["tp_size"] = tp_size

    # Build results for HTML — include both TP=1 and TP=N when --tp is used
    hw_list = [
        ("NVIDIA A100 80GB", GPUCostModel(A100_80GB), A100_80GB),
        ("NVIDIA H100 80GB", GPUCostModel(H100_80GB), H100_80GB),
        ("Ascend 910B", NPUCostModel(ASCEND_910B), ASCEND_910B),
        ("Ascend 910C", NPUCostModel(ASCEND_910C), ASCEND_910C),
    ]

    results = {}
    # Always add single-device (TP=1) results
    for device_label, base_model, hw_spec in hw_list:
        label = f"{device_label} (TP=1)" if tp_mode else device_label
        results[label] = PerformanceEvaluator(base_model).run(graph, overlap=True)

    # Add TP=N results when requested
    if tp_mode and tp_graph is not None:
        for device_label, base_model, hw_spec in hw_list:
            if hw_spec.interconnect:
                cm = CommAwareCostModel(base_model, hw_spec.interconnect, tp_parallel)
            else:
                cm = base_model
            results[f"{device_label} (TP={tp_size})"] = PerformanceEvaluator(cm).run(tp_graph, overlap=True)

    # Use the TP graph for the report if available (shows comm ops in architecture)
    report_graph = tp_graph if tp_mode and tp_graph else graph

    # Build accuracy info from hardware efficiency factors
    def _fmt_tflops(v):
        return f"{v / 1e12:.0f} TFLOPS" if v >= 1e12 else f"{v / 1e9:.0f} GFLOPS"
    def _fmt_bw(v):
        return f"{v:.0f} GB/s"

    accuracy_devices = []
    for name, base_model, hw_spec in hw_list:
        ef = getattr(hw_spec, 'efficiency_factors', None) or {}
        if ef:
            comp_eff = ef.get("matmul_fp16") or ef.get("cube_fp16") or ef.get("cube_bf16", 1.0)
            mem_eff = ef.get("memory", 1.0)
            static_mm = ef.get("static_tc_us") or ef.get("static_cube_us", 0)
            static_other = ef.get("static_cuda_us") or ef.get("static_vector_us", 0)
            peak = hw_spec.peak_flops_for("fp16") or hw_spec.peak_flops_for("bf16")
            bw = hw_spec.main_memory_bandwidth()
            accuracy_devices.append({
                "name": name,
                "compute_efficiency": comp_eff,
                "memory_efficiency": mem_eff,
                "static_matmul_us": static_mm,
                "static_other_us": static_other,
                "effective_peak": _fmt_tflops(peak * comp_eff),
                "effective_bw": _fmt_bw(bw * mem_eff),
            })
    accuracy_info = {"devices": accuracy_devices} if accuracy_devices else None

    fname = export_html_report(
        report_graph, results, "reports/deepseek_v3.2_report.html",
        model_name=f"DeepSeek V3.2 671B (batch={batch_size}, seq={seq_len})",
        config=config,
        n_dense=N_DENSE_LAYERS,
        tp_comparison=tp_comparison_data,
        accuracy_info=accuracy_info,
    )
    print(f"Exported: {fname}")

    if dsa_mode:
        # Export DSA HTML report
        extractor = ConfigExtractor()
        dsa_graph = extractor.extract(DEEPSEEK_DSA_CONFIG, batch_size=batch_size,
                                      seq_len=seq_len, graph_name="DeepSeek-V3.2-DSA")
        dsa_config = {
            **DEEPSEEK_DSA_CONFIG,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "phase": "prefill",
            "dtype": "fp16",
            "tokens": batch_size * seq_len,
        }
        dsa_results = {}
        for device_label, cost_model in [
            ("NVIDIA A100 80GB", GPUCostModel(A100_80GB)),
            ("NVIDIA H100 80GB", GPUCostModel(H100_80GB)),
            ("Ascend 910B", NPUCostModel(ASCEND_910B)),
            ("Ascend 910C", NPUCostModel(ASCEND_910C)),
        ]:
            dsa_results[device_label] = PerformanceEvaluator(cost_model).run(dsa_graph, overlap=True)

        dsa_fname = export_html_report(
            dsa_graph, dsa_results, "reports/deepseek_v3.2_dsa_report.html",
            model_name=f"DeepSeek V3.2 671B + DSA (batch={batch_size}, seq={seq_len})",
            config=dsa_config,
            n_dense=N_DENSE_LAYERS,
        )
        print(f"Exported: {dsa_fname}")

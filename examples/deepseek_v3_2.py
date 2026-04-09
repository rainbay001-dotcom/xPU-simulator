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
sys.path.insert(0, ".")

from xpu_simulator.core.operator import Dtype
from xpu_simulator.core.evaluator import PerformanceEvaluator
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


def build_deepseek_graph(batch_size: int, seq_len: int, dtype: Dtype = Dtype.FP16):
    """Build computation graph for DeepSeek V3.2 forward pass (prefill)."""
    extractor = ConfigExtractor(dtype=dtype)
    return extractor.extract(DEEPSEEK_CONFIG, batch_size=batch_size,
                             seq_len=seq_len, graph_name="DeepSeek-V3.2-671B")


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


if __name__ == "__main__":
    # Default: batch=1, seq_len=1024 (typical prefill)
    batch_size = 1
    seq_len = 1024

    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        seq_len = int(sys.argv[2])

    run_simulation(batch_size, seq_len)

    # Export interactive HTML report
    from xpu_simulator.utils.html_report import export_html_report

    graph = build_deepseek_graph(batch_size, seq_len)

    config = {
        "dim": DEEPSEEK_CONFIG["hidden_size"],
        "inter_dim": DEEPSEEK_CONFIG["intermediate_size"],
        "moe_inter_dim": DEEPSEEK_CONFIG["moe_intermediate_size"],
        "n_layers": DEEPSEEK_CONFIG["num_hidden_layers"],
        "n_dense_layers": N_DENSE_LAYERS,
        "n_heads": DEEPSEEK_CONFIG["num_attention_heads"],
        "n_routed_experts": DEEPSEEK_CONFIG["n_routed_experts"],
        "n_activated_experts": DEEPSEEK_CONFIG["num_experts_per_tok"],
        "n_shared_experts": DEEPSEEK_CONFIG["n_shared_experts"],
        "q_lora_rank": DEEPSEEK_CONFIG["q_lora_rank"],
        "kv_lora_rank": DEEPSEEK_CONFIG["kv_lora_rank"],
        "v_head_dim": DEEPSEEK_CONFIG["v_head_dim"],
        "vocab_size": DEEPSEEK_CONFIG["vocab_size"],
        "batch_size": batch_size, "seq_len": seq_len,
    }

    results = {}
    for device_label, cost_model in [
        ("NVIDIA A100 80GB", GPUCostModel(A100_80GB)),
        ("NVIDIA H100 80GB", GPUCostModel(H100_80GB)),
        ("Ascend 910B", NPUCostModel(ASCEND_910B)),
        ("Ascend 910C", NPUCostModel(ASCEND_910C)),
    ]:
        results[device_label] = PerformanceEvaluator(cost_model).run(graph, overlap=True)

    fname = export_html_report(
        graph, results, "deepseek_v3.2_report.html",
        model_name=f"DeepSeek V3.2 671B (batch={batch_size}, seq={seq_len})",
        config=config,
        n_dense=N_DENSE_LAYERS,
    )
    print(f"Exported: {fname}")

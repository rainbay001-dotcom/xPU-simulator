"""
Real HuggingFace Model Benchmark: ConfigExtractor vs DispatchExtractor.

Loads actual HuggingFace model configs and source code, runs both extractors,
and compares per-op graph structure, FLOPS, and latency estimates across devices.

Key insight: DispatchExtractor captures real PyTorch execution patterns
(e.g., FP32 attention scores for numerical stability) that ConfigExtractor
models as idealized FP16 operations.

Usage:
    source .venv312/bin/activate
    python benchmarks/run_real_models.py
"""

import sys
import os
import gc
import csv
from dataclasses import dataclass

sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from xpu_simulator.core.operator import OpType
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.fusion import FusionPass, GPU_FUSION_RULES, NPU_FUSION_RULES
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.frontend.config_extractor import ConfigExtractor
from xpu_simulator.frontend.dispatch_extractor import DispatchExtractor


# ---- Model registry ----

MODELS = {
    "LLaMA-3.1-8B":  {"config": "benchmarks/hf_models/llama3_8b/config.json",   "dtype": torch.float16},
    "LLaMA-3.1-70B": {"config": "benchmarks/hf_models/llama3_70b/config.json",  "dtype": torch.float16},
    "Mistral-7B":     {"config": "benchmarks/hf_models/mistral_7b/config.json",  "dtype": torch.float16},
    "Qwen2-7B":       {"config": "benchmarks/hf_models/qwen2_7b/config.json",   "dtype": torch.float16},
    "Qwen2-72B":      {"config": "benchmarks/hf_models/qwen2_72b/config.json",  "dtype": torch.float16},
    "Mixtral-8x7B":   {"config": "benchmarks/hf_models/mixtral_8x7b/config.json","dtype": torch.bfloat16},
}

DEVICES = {
    "A100": (GPUCostModel(A100_80GB), GPU_FUSION_RULES),
    "H100": (GPUCostModel(H100_80GB), GPU_FUSION_RULES),
    "910B": (NPUCostModel(ASCEND_910B), NPU_FUSION_RULES),
    "910C": (NPUCostModel(ASCEND_910C), NPU_FUSION_RULES),
}


def fmt_lat(us):
    if us >= 1e6:
        return f"{us/1e6:.3f} s"
    elif us >= 1000:
        return f"{us/1000:.1f} ms"
    else:
        return f"{us:.0f} us"


def graph_stats(graph, cost_model=None):
    """Extract op count, matmul count, FLOPS, and optional latency."""
    mm_count = 0
    total_flops = 0
    fp32_mm_count = 0
    fp32_mm_flops = 0

    for n in graph.topo_order():
        total_flops += n.op.flops
        if n.op.op_type == OpType.MATMUL:
            mm_count += 1
            if n.op.inputs and n.op.inputs[0].dtype.value == "fp32":
                fp32_mm_count += 1
                fp32_mm_flops += n.op.flops

    lat_us = None
    if cost_model:
        cm, fusion_rules = cost_model
        fused, _ = FusionPass(fusion_rules).apply(graph)
        result = PerformanceEvaluator(cm).run(fused, overlap=True)
        lat_us = result.total_latency_us

    return {
        "ops": graph.num_nodes,
        "edges": graph.num_edges,
        "matmuls": mm_count,
        "flops_T": total_flops / 1e12,
        "fp32_mms": fp32_mm_count,
        "fp32_mm_flops_T": fp32_mm_flops / 1e12,
        "lat_us": lat_us,
    }


def run_benchmark(batch_size=1, seq_len=1024):
    os.makedirs("reports", exist_ok=True)
    config_ext = ConfigExtractor()
    dispatch_ext = DispatchExtractor()

    print(f"\n{'='*110}")
    print(f"  Real HuggingFace Model Benchmark: ConfigExtractor vs DispatchExtractor")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, Prefill phase")
    print(f"  Models: {len(MODELS)}, Devices: {len(DEVICES)}")
    print(f"{'='*110}")

    all_results = []

    for model_name, model_info in MODELS.items():
        cfg_path = model_info["config"]
        model_dtype = model_info["dtype"]

        print(f"\n\n--- {model_name} ---")

        # 1) ConfigExtractor graph
        try:
            cg = config_ext.extract(cfg_path, batch_size=batch_size, seq_len=seq_len)
        except Exception as e:
            print(f"  ConfigExtractor SKIP: {e}")
            continue

        # 2) DispatchExtractor graph
        try:
            hf_config = AutoConfig.from_pretrained(cfg_path)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(hf_config, dtype=model_dtype)
            model.eval()
            input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device="meta")
            dg = dispatch_ext.extract(model, {"input_ids": input_ids}, graph_name=model_name)
            del model
            gc.collect()
        except Exception as e:
            print(f"  DispatchExtractor ERROR: {e}")
            dg = None

        # Graph structure comparison
        cs = graph_stats(cg)
        print(f"  ConfigExtractor:   {cs['ops']:>5} ops, {cs['matmuls']:>4} MMs, {cs['flops_T']:.2f}T FLOPS")

        if dg:
            ds = graph_stats(dg)
            flops_match = "MATCH" if abs(cs["flops_T"] - ds["flops_T"]) / max(cs["flops_T"], 0.001) < 0.02 else "DIFF"
            print(f"  DispatchExtractor: {ds['ops']:>5} ops, {ds['matmuls']:>4} MMs, {ds['flops_T']:.2f}T FLOPS [{flops_match}]")
            if ds["fp32_mms"] > 0:
                print(f"    FP32 attention MATMULs: {ds['fp32_mms']} ({ds['fp32_mm_flops_T']:.3f}T)")

        # Per-device latency comparison
        for dev_key, (cost_model, fusion_rules) in DEVICES.items():
            c_fused, _ = FusionPass(fusion_rules).apply(cg)
            c_result = PerformanceEvaluator(cost_model).run(c_fused, overlap=True)
            c_us = c_result.total_latency_us

            d_us = None
            ratio = None
            if dg:
                d_fused, _ = FusionPass(fusion_rules).apply(dg)
                d_result = PerformanceEvaluator(cost_model).run(d_fused, overlap=True)
                d_us = d_result.total_latency_us
                ratio = d_us / c_us if c_us > 0 else 0

            all_results.append({
                "model": model_name,
                "device": dev_key,
                "config_ops": cs["ops"],
                "dispatch_ops": ds["ops"] if dg else 0,
                "config_matmuls": cs["matmuls"],
                "dispatch_matmuls": ds["matmuls"] if dg else 0,
                "flops_T": round(cs["flops_T"], 2),
                "fp32_mms": ds["fp32_mms"] if dg else 0,
                "config_us": round(c_us, 1),
                "dispatch_us": round(d_us, 1) if d_us else 0,
                "ratio": round(ratio, 3) if ratio else 0,
            })

    # ---- Summary Table ----
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY: ConfigExtractor vs DispatchExtractor Latency")
    print(f"{'='*110}")
    print(f"\n  {'Model':<18} {'Dev':<6} {'Config':>12} {'Dispatch':>12} {'Ratio':>8}  {'Config':>6} {'Disp':>6} {'FLOPs':>8} {'FP32 MMs':>8}")
    print(f"  {'':18} {'':6} {'(latency)':>12} {'(latency)':>12} {'':>8}  {'(ops)':>6} {'(ops)':>6} {'(T)':>8} {'':>8}")
    print(f"  {'-'*98}")

    for r in all_results:
        d_lat = fmt_lat(r["dispatch_us"]) if r["dispatch_us"] else "N/A"
        ratio_s = f"{r['ratio']:.3f}x" if r["ratio"] else "N/A"
        print(f"  {r['model']:<18} {r['device']:<6} "
              f"{fmt_lat(r['config_us']):>12} {d_lat:>12} {ratio_s:>8}  "
              f"{r['config_ops']:>6} {r['dispatch_ops']:>6} {r['flops_T']:>8.2f} {r['fp32_mms']:>8}")

    # ---- Analysis: FP32 attention overhead ----
    print(f"\n\n{'='*110}")
    print(f"  ANALYSIS: FP32 Attention Score Overhead")
    print(f"{'='*110}")
    print(f"""
  Key finding: HuggingFace transformers upcasts attention scores to FP32 for
  numerical stability (softmax precision). This is captured by DispatchExtractor
  but not by ConfigExtractor, which models everything as FP16.

  Impact on A100: FP32 peak = 19.5 TFLOPS vs FP16 peak = 312 TFLOPS (16x slower).
  For LLaMA-8B: 64 FP32 attention BMMs add ~40ms overhead on A100.

  In production: Flash Attention fuses Q*K^T and softmax in FP16/BF16,
  eliminating this FP32 overhead. ConfigExtractor's estimate is closer to
  Flash Attention reality; DispatchExtractor shows vanilla PyTorch overhead.
""")

    # ---- Per-device aggregate ----
    print(f"  Per-device average ratio (Dispatch/Config):")
    for dev_key in ["A100", "H100", "910B", "910C"]:
        dev_results = [r for r in all_results if r["device"] == dev_key and r["ratio"] > 0]
        if dev_results:
            avg = sum(r["ratio"] for r in dev_results) / len(dev_results)
            print(f"    {dev_key}: {avg:.3f}x")

    # ---- Save CSV ----
    csv_path = "reports/real_model_benchmark.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Results saved to {csv_path}")

    return all_results


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    run_benchmark(batch, seq)

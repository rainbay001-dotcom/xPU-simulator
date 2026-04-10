"""
Multi-model benchmark: xPU-simulator vs msmodeling analytical formulas.

Runs 7 LLM architectures across 4 hardware targets, comparing per-op and
total latency between our simulator and msmodeling's roofline formulas.

Usage:
    source .venv312/bin/activate
    python benchmarks/run_benchmark.py
"""

import sys
import os
import json
import csv
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpType, Dtype
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.fusion import FusionPass, GPU_FUSION_RULES, NPU_FUSION_RULES
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.frontend.config_extractor import ConfigExtractor


# ---- msmodeling device profiles (exact specs from their codebase) ----

@dataclass
class MsDevice:
    name: str
    mma_ops: float          # peak matmul FLOPS
    gp_ops: float           # peak general-purpose FLOPS
    mem_bw: float           # bytes/s
    compute_eff: float      # 0.7 typical
    memory_eff: float       # 0.6 typical
    mma_static_s: float     # 5e-6
    gp_static_s: float      # 2e-6


# msmodeling specs: use OUR hardware specs + msmodeling's formula for fair comparison.
# This isolates modeling methodology differences from hardware spec differences.
# Original msmodeling used: 910B=376T/22T, 910C=376T/22T, A100=312T/78T, H100=989T/247T
def _build_ms_devices():
    """Build msmodeling device profiles from our hardware presets for fair comparison."""
    return {
        "A100": MsDevice(
            "A100 (ms-formula)", A100_80GB.peak_flops_for("fp16"),
            A100_80GB.cuda_core_flops_for("fp16"), A100_80GB.main_memory_bandwidth() * 1e9,
            A100_80GB.get_efficiency("matmul_fp16"), A100_80GB.get_efficiency("memory"),
            A100_80GB.get_efficiency("static_tc_us") * 1e-6, A100_80GB.get_efficiency("static_cuda_us") * 1e-6,
        ),
        "H100": MsDevice(
            "H100 (ms-formula)", H100_80GB.peak_flops_for("fp16"),
            H100_80GB.cuda_core_flops_for("fp16"), H100_80GB.main_memory_bandwidth() * 1e9,
            H100_80GB.get_efficiency("matmul_fp16"), H100_80GB.get_efficiency("memory"),
            H100_80GB.get_efficiency("static_tc_us") * 1e-6, H100_80GB.get_efficiency("static_cuda_us") * 1e-6,
        ),
        "910B": MsDevice(
            "910B (ms-formula)", ASCEND_910B.cube_peak_for("fp16"),
            ASCEND_910B.vector_peak_for("fp16"), ASCEND_910B.main_memory_bandwidth() * 1e9,
            ASCEND_910B.get_efficiency("cube_fp16"), ASCEND_910B.get_efficiency("memory"),
            ASCEND_910B.get_efficiency("static_cube_us") * 1e-6, ASCEND_910B.get_efficiency("static_vector_us") * 1e-6,
        ),
        "910C": MsDevice(
            "910C (ms-formula)", ASCEND_910C.cube_peak_for("fp16"),
            ASCEND_910C.vector_peak_for("fp16"), ASCEND_910C.main_memory_bandwidth() * 1e9,
            ASCEND_910C.get_efficiency("cube_fp16"), ASCEND_910C.get_efficiency("memory"),
            ASCEND_910C.get_efficiency("static_cube_us") * 1e-6, ASCEND_910C.get_efficiency("static_vector_us") * 1e-6,
        ),
    }

MS_DEVICES = _build_ms_devices()

# xPU-simulator hardware configs
XPU_DEVICES = {
    "A100": ("NVIDIA A100 80GB", GPUCostModel(A100_80GB), A100_80GB, GPU_FUSION_RULES),
    "H100": ("NVIDIA H100 80GB", GPUCostModel(H100_80GB), H100_80GB, GPU_FUSION_RULES),
    "910B": ("Ascend 910B", NPUCostModel(ASCEND_910B), ASCEND_910B, NPU_FUSION_RULES),
    "910C": ("Ascend 910C", NPUCostModel(ASCEND_910C), ASCEND_910C, NPU_FUSION_RULES),
}


def msmodeling_estimate(graph, device: MsDevice, hw_spec=None) -> tuple[float, dict]:
    """Estimate total latency using msmodeling's roofline formula.
    Returns (total_us, {op_type_breakdown}).
    If hw_spec is provided, uses dtype-aware peak FLOPS selection.
    """
    total_us = 0.0
    mma_us = 0.0
    gp_us = 0.0
    mma_count = 0
    gp_count = 0

    for node in graph.topo_order():
        op = node.op
        flops = op.flops
        mem_bytes = sum(t.size_bytes for t in op.inputs) + sum(t.size_bytes for t in op.outputs)
        is_mma = op.op_type in (OpType.MATMUL, OpType.CONV2D)

        if is_mma:
            # Use dtype-aware peak if hardware spec is available
            if hw_spec and op.inputs:
                dtype_str = op.inputs[0].dtype.value
                mma_peak = hw_spec.peak_flops_for(dtype_str)
            else:
                mma_peak = device.mma_ops
            compute_t = flops / (mma_peak * device.compute_eff) if flops > 0 else 0
            static = device.mma_static_s
            mma_count += 1
        else:
            compute_t = flops / (device.gp_ops * device.compute_eff) if flops > 0 else 0
            static = device.gp_static_s
            gp_count += 1

        memory_t = mem_bytes / (device.mem_bw * device.memory_eff) if mem_bytes > 0 else 0
        lat = max(compute_t, memory_t) + static
        lat_us = lat * 1e6
        total_us += lat_us

        if is_mma:
            mma_us += lat_us
        else:
            gp_us += lat_us

    return total_us, {
        "mma_us": mma_us, "gp_us": gp_us,
        "mma_count": mma_count, "gp_count": gp_count,
        "static_us": (mma_count * device.mma_static_s + gp_count * device.gp_static_s) * 1e6,
    }


# ---- Model configs ----

MODELS = {
    "LLaMA-3.1-8B":    {"config": "benchmarks/configs/llama3_8b.json",   "params": "8B"},
    "LLaMA-3.1-70B":   {"config": "benchmarks/configs/llama3_70b.json",  "params": "70B"},
    "Mistral-7B":      {"config": "benchmarks/configs/mistral_7b.json",  "params": "7B"},
    "Qwen2-7B":        {"config": "benchmarks/configs/qwen2_7b.json",    "params": "7B"},
    "Qwen2-72B":       {"config": "benchmarks/configs/qwen2_72b.json",   "params": "72B"},
    "Mixtral-8x7B":    {"config": "benchmarks/configs/mixtral_8x7b.json","params": "47B"},
    "DeepSeek-V3-671B":{"config": "benchmarks/configs/deepseek_v3.json", "params": "671B"},
}


def fmt_lat(us):
    if us >= 1e6:
        return f"{us/1e6:.3f} s"
    elif us >= 1000:
        return f"{us/1000:.1f} ms"
    else:
        return f"{us:.0f} us"


def run_benchmark(batch_size=1, seq_len=1024):
    os.makedirs("reports", exist_ok=True)
    extractor = ConfigExtractor()

    print(f"\n{'='*100}")
    print(f"  xPU-Simulator vs msmodeling Benchmark")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, Dtype=fp16, Prefill phase")
    print(f"  Models: {len(MODELS)}, Devices: {len(XPU_DEVICES)}")
    print(f"{'='*100}\n")

    # Collect all results
    all_results = []

    for model_name, model_info in MODELS.items():
        print(f"\n--- {model_name} ({model_info['params']}) ---")

        # Extract graph
        try:
            graph = extractor.extract(model_info["config"],
                                      batch_size=batch_size, seq_len=seq_len)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Graph: {graph.num_nodes} ops, {graph.num_edges} edges")

        # Count MATMULs and other ops
        matmul_count = sum(1 for n in graph.topo_order() if n.op.op_type == OpType.MATMUL)
        other_count = graph.num_nodes - matmul_count
        total_flops = sum(n.op.flops for n in graph.topo_order())
        print(f"  FLOPs: {total_flops/1e12:.2f} T, MATMULs: {matmul_count}, Other: {other_count}")

        for dev_key in ["A100", "H100", "910B", "910C"]:
            dev_name, cost_model, hw_spec, fusion_rules = XPU_DEVICES[dev_key]
            ms_dev = MS_DEVICES[dev_key]

            # xPU-simulator: apply fusion then evaluate
            fused, fuse_result = FusionPass(fusion_rules).apply(graph)
            xpu_result = PerformanceEvaluator(cost_model).run(fused, overlap=True)
            xpu_us = xpu_result.total_latency_us

            # msmodeling: unfused graph, simple roofline
            ms_us, ms_breakdown = msmodeling_estimate(graph, ms_dev, hw_spec)

            ratio = xpu_us / ms_us if ms_us > 0 else 0
            diff_pct = (xpu_us - ms_us) / ms_us * 100 if ms_us > 0 else 0

            all_results.append({
                "model": model_name,
                "params": model_info["params"],
                "device": dev_name,
                "dev_key": dev_key,
                "ops": graph.num_nodes,
                "fused_ops": fused.num_nodes,
                "matmul_count": matmul_count,
                "total_flops_T": round(total_flops / 1e12, 2),
                "xpu_us": round(xpu_us, 1),
                "ms_us": round(ms_us, 1),
                "ratio": round(ratio, 3),
                "diff_pct": round(diff_pct, 1),
                "ms_static_us": round(ms_breakdown["static_us"], 1),
                "ms_mma_us": round(ms_breakdown["mma_us"], 1),
                "ms_gp_us": round(ms_breakdown["gp_us"], 1),
                "xpu_compute_bound": xpu_result.compute_bound_count,
                "xpu_memory_bound": xpu_result.memory_bound_count,
            })

    # ---- Print Summary Table ----
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY: xPU-simulator vs msmodeling")
    print(f"{'='*100}")
    print(f"\n  {'Model':<20} {'Device':<15} {'xPU-sim':>12} {'msmodeling':>12} {'Ratio':>8} {'Diff%':>8}")
    print(f"  {'-'*75}")

    for r in all_results:
        print(f"  {r['model']:<20} {r['dev_key']:<15} "
              f"{fmt_lat(r['xpu_us']):>12} {fmt_lat(r['ms_us']):>12} "
              f"{r['ratio']:>7.3f}x {r['diff_pct']:>+7.1f}%")

    # ---- Per-device aggregate analysis ----
    print(f"\n\n{'='*100}")
    print(f"  PER-DEVICE AGGREGATE ANALYSIS")
    print(f"{'='*100}")

    for dev_key in ["A100", "H100", "910B", "910C"]:
        dev_results = [r for r in all_results if r["dev_key"] == dev_key]
        if not dev_results:
            continue

        ratios = [r["ratio"] for r in dev_results]
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        min_model = next(r["model"] for r in dev_results if r["ratio"] == min_ratio)
        max_model = next(r["model"] for r in dev_results if r["ratio"] == max_ratio)

        print(f"\n  {dev_key}:")
        print(f"    Avg ratio (xPU/ms): {avg_ratio:.3f}x")
        print(f"    Range: {min_ratio:.3f}x ({min_model}) — {max_ratio:.3f}x ({max_model})")

        # Analyze what drives the difference
        total_xpu = sum(r["xpu_us"] for r in dev_results)
        total_ms = sum(r["ms_us"] for r in dev_results)
        total_static = sum(r["ms_static_us"] for r in dev_results)
        print(f"    Total xPU: {fmt_lat(total_xpu)}, Total ms: {fmt_lat(total_ms)}")
        print(f"    msmodeling static overhead: {fmt_lat(total_static)} ({total_static/total_ms*100:.1f}% of ms total)")

    # ---- Breakdown: where does xPU diverge? ----
    print(f"\n\n{'='*100}")
    print(f"  DIVERGENCE ANALYSIS (xPU/ms ratio by op type)")
    print(f"{'='*100}")

    # For each device, compare matmul-heavy vs vector-heavy models
    for dev_key in ["A100", "910B"]:
        dev_results = [r for r in all_results if r["dev_key"] == dev_key]
        print(f"\n  {dev_key}:")
        print(f"    {'Model':<20} {'Ratio':>8} {'MATMULs':>8} {'CB%':>6} {'MB%':>6} {'Static%':>8}")
        print(f"    {'-'*56}")
        for r in sorted(dev_results, key=lambda x: x["ratio"]):
            total = r["xpu_compute_bound"] + r["xpu_memory_bound"]
            cb_pct = r["xpu_compute_bound"] / total * 100 if total > 0 else 0
            mb_pct = r["xpu_memory_bound"] / total * 100 if total > 0 else 0
            static_pct = r["ms_static_us"] / r["ms_us"] * 100 if r["ms_us"] > 0 else 0
            print(f"    {r['model']:<20} {r['ratio']:>7.3f}x {r['matmul_count']:>8} "
                  f"{cb_pct:>5.0f}% {mb_pct:>5.0f}% {static_pct:>7.1f}%")

    # ---- Save CSV ----
    csv_path = "reports/benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    return all_results


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    run_benchmark(batch, seq)

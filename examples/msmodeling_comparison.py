"""
Side-by-side comparison: xPU-simulator vs msmodeling analytical formulas.

msmodeling uses the same roofline model (latency = max(compute_time, memory_time))
but applies efficiency factors and static op costs that xPU-simulator does not.

This script:
1. Runs xPU-simulator's ConfigExtractor on DeepSeek V3.2 671B
2. Replays the same ops through msmodeling's exact roofline formulas
3. Compares per-layer and total latency side by side
"""

import sys
sys.path.insert(0, ".")

from dataclasses import dataclass
from typing import Dict, List, Tuple

from xpu_simulator.core.operator import OpType, Dtype
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.frontend.config_extractor import ConfigExtractor
from xpu_simulator.core.fusion import FusionPass, GPU_FUSION_RULES, NPU_FUSION_RULES


# ---- DeepSeek V3.2 671B Config ----
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


# ---- msmodeling Device Profiles (from device.py) ----
# These are the EXACT specs from msmodeling's codebase
@dataclass
class MsModelingDevice:
    name: str
    # Peak compute ops/s by dtype
    mma_ops_fp16: float      # tensor core / CUBE peak (ops/s)
    gp_ops_fp16: float       # general purpose / VECTOR peak (ops/s)
    # Memory
    memory_bandwidth: float   # bytes/s
    memory_size_gb: float
    # Efficiency factors (msmodeling's key differentiator)
    compute_efficiency: float  # applied to peak compute
    memory_efficiency: float   # applied to peak bandwidth
    # Static per-op overhead
    mma_op_cost_s: float      # fixed cost per MMA op
    gp_op_cost_s: float       # fixed cost per GP op


# Ascend 910B equivalent: ATLAS_800_A2_376T_64G
MSMODEL_910B = MsModelingDevice(
    name="ATLAS_800_A2_376T (910B)",
    mma_ops_fp16=376e12,       # 376 TFLOPS FP16
    gp_ops_fp16=22e12,         # 22 TFLOPS FP16 (vector)
    memory_bandwidth=1.6 * (1024**4),  # 1.6 TiB/s = 1759 GB/s
    memory_size_gb=64,
    compute_efficiency=0.7,
    memory_efficiency=0.6,
    mma_op_cost_s=5e-6,        # 5 us per MMA op
    gp_op_cost_s=2e-6,         # 2 us per GP op
)

# Ascend 910C equivalent: ATLAS_800_A3_752T_128G_DIE (one die)
MSMODEL_910C = MsModelingDevice(
    name="ATLAS_800_A3_752T (910C die)",
    mma_ops_fp16=376e12,       # same as 910B per die
    gp_ops_fp16=22e12,
    memory_bandwidth=1.6 * (1024**4),  # same as 910B per die
    memory_size_gb=64,          # per die
    compute_efficiency=0.7,
    memory_efficiency=0.6,
    mma_op_cost_s=5e-6,
    gp_op_cost_s=2e-6,
)


# xPU-simulator device specs for reference
@dataclass
class XpuDevice:
    name: str
    peak_fp16_tflops: float
    memory_bandwidth_GBs: float
    memory_size_gb: float


XPU_910B = XpuDevice("Ascend 910B", 320, 1600, 64)
XPU_910C = XpuDevice("Ascend 910C", 400, 1800, 64)


def msmodeling_roofline(flops: float, mem_bytes: float, device: MsModelingDevice,
                        is_mma: bool = True) -> float:
    """
    msmodeling's roofline formula:
      execution_time = max(compute_time, memory_time) + static_cost

    Key differences from xPU-simulator:
      1. Applies compute_efficiency (0.7) — effective peak = peak * 0.7
      2. Applies memory_efficiency (0.6) — effective BW = BW * 0.6
      3. Adds static per-op cost (5us for MMA, 2us for GP)
    """
    if is_mma:
        compute_time = flops / (device.mma_ops_fp16 * device.compute_efficiency) if flops > 0 else 0
        static_cost = device.mma_op_cost_s
    else:
        compute_time = flops / (device.gp_ops_fp16 * device.compute_efficiency) if flops > 0 else 0
        static_cost = device.gp_op_cost_s

    memory_time = mem_bytes / (device.memory_bandwidth * device.memory_efficiency) if mem_bytes > 0 else 0

    return max(compute_time, memory_time) + static_cost


def estimate_ops_msmodeling(graph, device: MsModelingDevice) -> Tuple[float, List[Dict]]:
    """Walk xPU-simulator graph and estimate each op using msmodeling formulas."""
    total_us = 0.0
    op_details = []

    for node in graph.topo_order():
        op = node.op

        # Compute FLOPs
        flops = op.flops

        # Compute memory bytes (read inputs + write outputs)
        mem_bytes = sum(t.size_bytes for t in op.inputs) + sum(t.size_bytes for t in op.outputs)

        # Classify: MMA ops (matmul, attention) vs GP ops (elementwise, norms)
        # MMA ops: matmul (linear, attention projections), CONV2D
        # GP ops: everything else (elementwise, norms, softmax, etc.)
        is_mma = op.op_type in (OpType.MATMUL, OpType.CONV2D)

        latency_s = msmodeling_roofline(flops, mem_bytes, device, is_mma=is_mma)
        latency_us = latency_s * 1e6

        total_us += latency_us
        op_details.append({
            "name": node.name,
            "op_type": op.op_type.name,
            "flops": flops,
            "mem_bytes": mem_bytes,
            "latency_us": latency_us,
            "is_mma": is_mma,
        })

    return total_us, op_details


def run_comparison(batch_size=1, seq_len=1024):
    print(f"\n{'='*80}")
    print(f"  DeepSeek V3.2 671B — xPU-simulator vs msmodeling Comparison")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, Dtype=fp16")
    print(f"{'='*80}\n")

    # Build graph
    extractor = ConfigExtractor()
    graph = extractor.extract(DEEPSEEK_CONFIG, batch_size=batch_size, seq_len=seq_len)
    print(f"Graph: {graph.num_nodes} ops, {graph.num_edges} edges\n")

    # --- xPU-simulator results ---
    npu_910b_model = NPUCostModel(ASCEND_910B)
    npu_910c_model = NPUCostModel(ASCEND_910C)

    # Apply NPU fusion
    npu_fused, _ = FusionPass(NPU_FUSION_RULES).apply(graph)

    xpu_910b = PerformanceEvaluator(npu_910b_model).run(npu_fused, overlap=True)
    xpu_910c = PerformanceEvaluator(npu_910c_model).run(npu_fused, overlap=True)

    # --- msmodeling results (using same graph, their roofline formulas) ---
    ms_910b_us, ms_910b_ops = estimate_ops_msmodeling(graph, MSMODEL_910B)
    ms_910c_us, ms_910c_ops = estimate_ops_msmodeling(graph, MSMODEL_910C)

    # Print comparison
    print(f"{'Device':<30} {'xPU-sim (ms)':>15} {'msmodeling (ms)':>15} {'Ratio':>10}")
    print(f"{'-'*70}")

    devices = [
        ("Ascend 910B", xpu_910b.total_latency_us, ms_910b_us),
        ("Ascend 910C", xpu_910c.total_latency_us, ms_910c_us),
    ]

    for name, xpu_us, ms_us in devices:
        ratio = ms_us / xpu_us if xpu_us > 0 else 0
        print(f"{name:<30} {xpu_us/1000:>12.1f} ms {ms_us/1000:>12.1f} ms {ratio:>9.2f}x")

    # Explain the differences
    print(f"\n{'='*80}")
    print("  Key Differences Explained")
    print(f"{'='*80}")
    print("""
  msmodeling applies two major corrections that xPU-simulator does not:

  1. EFFICIENCY FACTORS:
     - compute_efficiency = 0.7  (only 70% of peak TFLOPS achievable)
     - memory_efficiency  = 0.6  (only 60% of peak bandwidth achievable)
     → This alone makes estimates ~1.4-1.7x slower

  2. STATIC PER-OP COST:
     - MMA ops: +5 us each  (matmul, attention)
     - GP ops:  +2 us each  (elementwise, norms)
     → With ~1600 ops, this adds ~6-8 ms

  3. DEVICE SPECS DIFFER:
     msmodeling 910B (A2_376T):  376 TFLOPS FP16, 1759 GB/s (1.6 TiB/s)
     xPU-simulator 910B:         320 TFLOPS FP16, 1600 GB/s
     → msmodeling uses higher raw peaks but scales down with efficiency
""")

    # Top 10 ops comparison for 910B
    print(f"\n{'='*80}")
    print("  Top 10 Most Expensive Ops — 910B Comparison")
    print(f"{'='*80}")

    # Sort msmodeling ops by latency
    ms_910b_ops.sort(key=lambda x: x["latency_us"], reverse=True)

    # Get xPU-simulator per-op costs
    xpu_op_costs = {}
    for node in npu_fused.topo_order():
        cost = npu_910b_model.estimate(node.op)
        xpu_op_costs[node.name] = cost.latency_us

    print(f"\n  {'Op':<40} {'msmodeling':>12} {'xPU-sim':>12} {'Ratio':>8}")
    print(f"  {'-'*72}")

    for op in ms_910b_ops[:10]:
        name = op["name"]
        ms_lat = op["latency_us"]
        xpu_lat = xpu_op_costs.get(name, 0)
        ratio = ms_lat / xpu_lat if xpu_lat > 0 else float('inf')
        print(f"  {name:<40} {ms_lat:>9.0f} us {xpu_lat:>9.0f} us {ratio:>7.2f}x")

    # Breakdown: what fraction of overhead comes from efficiency vs static cost
    print(f"\n{'='*80}")
    print("  Overhead Breakdown (910B)")
    print(f"{'='*80}")

    total_static = sum(
        MSMODEL_910B.mma_op_cost_s * 1e6 if op["is_mma"] else MSMODEL_910B.gp_op_cost_s * 1e6
        for op in ms_910b_ops
    )
    total_without_static = ms_910b_us - total_static
    xpu_total = xpu_910b.total_latency_us

    print(f"\n  xPU-simulator total:           {xpu_total/1000:>10.1f} ms")
    print(f"  msmodeling total:              {ms_910b_us/1000:>10.1f} ms")
    print(f"    - Roofline w/ efficiency:    {total_without_static/1000:>10.1f} ms")
    print(f"    - Static op costs:           {total_static/1000:>10.1f} ms  ({len(ms_910b_ops)} ops)")
    print(f"  Overall ratio:                 {ms_910b_us/xpu_total:>10.2f}x")
    print()


    # Effective specs comparison
    print(f"{'='*80}")
    print("  Effective Device Specs Comparison")
    print(f"{'='*80}")
    print(f"""
  {'Metric':<35} {'xPU-sim 910B':>15} {'msmodel 910B':>15}
  {'-'*65}
  Raw FP16 TFLOPS                    {'320':>15} {'376':>15}
  Compute efficiency                 {'1.0 (100%)':>15} {'0.7 (70%)':>15}
  Effective TFLOPS                   {'320':>15} {'263':>15}
  Raw HBM bandwidth (GB/s)           {'1600':>15} {'1759':>15}
  Memory efficiency                  {'1.0 (100%)':>15} {'0.6 (60%)':>15}
  Effective bandwidth (GB/s)         {'1600':>15} {'1055':>15}
  Static cost per MMA op             {'0 us':>15} {'5 us':>15}
  Static cost per GP op              {'0 us':>15} {'2 us':>15}

  Note: xPU-simulator's NPU backend uses a CA-style tiled pipeline model
  with format conversion overhead, alignment waste, and double-buffering
  simulation — these add latency beyond simple roofline for memory-bound ops.
  For compute-bound ops, xPU-simulator's lower raw peak (320 vs 376*0.7=263)
  actually makes it ~22% slower than msmodeling's effective peak.
""")


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    run_comparison(batch, seq)

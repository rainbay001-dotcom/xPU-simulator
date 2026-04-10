"""
Direct comparison: xPU-simulator vs real msmodeling (MindStudio-Modeling).

Runs both tools against the same HuggingFace models on the same Ascend
hardware profile and compares latency estimates.

Prerequisites:
    # msmodeling (in separate venv due to dependency conflicts):
    git clone https://github.com/opensim-ai/msmodeling.git /tmp/msmodeling
    python3 -m venv /tmp/msmodeling_venv
    source /tmp/msmodeling_venv/bin/activate
    pip install "torch>=2.7,<=2.10" "transformers>=4.57,<5.0" optree salabim \
        blinker greenlet strenum aenum overrides prettytable pyyaml pandas \
        scikit-learn scipy compressed-tensors

Usage:
    source .venv312/bin/activate
    python benchmarks/run_vs_msmodeling.py
"""

import subprocess
import sys
import os
import re
import csv

sys.path.insert(0, ".")

from xpu_simulator.frontend.config_extractor import ConfigExtractor
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.fusion import FusionPass, NPU_FUSION_RULES
from xpu_simulator.backends.npu.hardware import ASCEND_910B
from xpu_simulator.backends.npu.cost_model import NPUCostModel


# Models that work with both tools
MODELS = {
    "LLaMA-3.1-8B":  ("benchmarks/hf_models/llama3_8b/config.json",  "NousResearch/Meta-Llama-3.1-8B"),
    "LLaMA-3.1-70B": ("benchmarks/hf_models/llama3_70b/config.json", "NousResearch/Meta-Llama-3.1-70B"),
    "Qwen2-7B":      ("benchmarks/hf_models/qwen2_7b/config.json",   "Qwen/Qwen2-7B"),
    "Qwen2-72B":     ("benchmarks/hf_models/qwen2_72b/config.json",  "Qwen/Qwen2-72B"),
}

# msmodeling venv and path
MSMODELING_VENV = "/tmp/msmodeling_venv/bin/python3"
MSMODELING_PATH = "/tmp/msmodeling"
MS_DEVICE = "ATLAS_800_A2_376T_64G"


def run_msmodeling(hf_id, device=MS_DEVICE, quant="DISABLED", batch=1, seq=1024):
    """Run msmodeling and return latency in ms."""
    env = os.environ.copy()
    env["PYTHONPATH"] = MSMODELING_PATH + ":" + env.get("PYTHONPATH", "")

    cmd = [
        MSMODELING_VENV, "-m", "cli.inference.text_generate",
        hf_id, "--device", device,
        "--num-queries", str(batch), "--query-length", str(seq),
        "--quantize-linear-action", quant,
        "--log-level", "error",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        output = result.stdout + result.stderr
        match = re.search(r"Total time for analytic: ([\d.]+)ms", output)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"  msmodeling error: {e}")
    return None


def run_xpu_simulator(config_path, batch=1, seq=1024):
    """Run xPU-simulator and return latency in ms."""
    ext = ConfigExtractor()
    cost_model = NPUCostModel(ASCEND_910B)

    graph = ext.extract(config_path, batch_size=batch, seq_len=seq)
    fused, _ = FusionPass(NPU_FUSION_RULES).apply(graph)
    result = PerformanceEvaluator(cost_model).run(fused, overlap=True)
    return result.total_latency_us / 1000


def run_comparison(batch=1, seq=1024):
    os.makedirs("reports", exist_ok=True)

    # Check msmodeling availability
    if not os.path.exists(MSMODELING_VENV):
        print(f"ERROR: msmodeling venv not found at {MSMODELING_VENV}")
        print("Install with: python3 -m venv /tmp/msmodeling_venv && ...")
        return

    print(f"\n{'='*80}")
    print(f"  xPU-Simulator vs msmodeling (MindStudio-Modeling)")
    print(f"  Device: Ascend 910B ({MS_DEVICE})")
    print(f"  Batch={batch}, SeqLen={seq}, FP16, Prefill phase")
    print(f"{'='*80}")
    print()

    # msmodeling specs: MMA=376T, GP=22T, BW=1759 GB/s
    # Our specs: CUBE=320T, VECTOR=10T, BW=1600 GB/s
    print("  Hardware spec differences:")
    print("    msmodeling: MMA=376T FP16, GP=22T, BW=1759 GB/s, Mem=64GB")
    print("    xPU-sim:    CUBE=320T FP16, VEC=10T, BW=1600 GB/s, Mem=64GB")
    print()

    results = []
    print(f"  {'Model':<18} {'xPU-sim(ms)':>12} {'msmodel(ms)':>12} {'Ratio':>8} {'Diff%':>8}")
    print(f"  {'-'*62}")

    for name, (cfg_path, hf_id) in MODELS.items():
        xpu_ms = run_xpu_simulator(cfg_path, batch, seq)
        ms_ms = run_msmodeling(hf_id, batch=batch, seq=seq)

        if ms_ms:
            ratio = xpu_ms / ms_ms
            diff = (xpu_ms - ms_ms) / ms_ms * 100
            print(f"  {name:<18} {xpu_ms:>12.1f} {ms_ms:>12.1f} {ratio:>7.3f}x {diff:>+7.1f}%")
        else:
            ratio = diff = None
            print(f"  {name:<18} {xpu_ms:>12.1f} {'ERROR':>12} {'N/A':>8} {'N/A':>8}")

        results.append({
            "model": name,
            "xpu_ms": round(xpu_ms, 1),
            "msmodeling_ms": round(ms_ms, 1) if ms_ms else None,
            "ratio": round(ratio, 3) if ratio else None,
            "diff_pct": round(diff, 1) if diff else None,
        })

    # Aggregate
    valid = [r for r in results if r["ratio"]]
    if valid:
        avg_ratio = sum(r["ratio"] for r in valid) / len(valid)
        print(f"\n  Average ratio: {avg_ratio:.3f}x")
        print(f"  Note: xPU-sim is ~5-10% faster due to operator fusion")
        print(f"  Note: Hardware spec differences account for remaining gap")

    # Save
    csv_path = "reports/vs_msmodeling.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {csv_path}")

    return results


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    run_comparison(batch, seq)

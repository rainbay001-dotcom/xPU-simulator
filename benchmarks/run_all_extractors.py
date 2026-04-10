"""
Run all applicable frontend extractors against every HF model config.

Extractors tested:
  1. ConfigExtractor     — from config.json (analytical graph construction)
  2. DispatchExtractor   — from HF model ID via TorchDispatchMode (runtime interception)
  3. TorchGraphExtractor — from nn.Module via torch.fx symbolic trace
  4. ExportExtractor     — from nn.Module via torch.export

Not applicable:
  - GraphBuilder: manual DSL, not auto-extraction
  - ONNXExtractor: requires .onnx files
  - ProfilerExtractor: requires Chrome trace JSON

Usage:
    source .venv312/bin/activate
    python benchmarks/run_all_extractors.py
"""

import sys
import os
import time
import csv
import traceback

sys.path.insert(0, ".")

import torch

from xpu_simulator.frontend.config_extractor import ConfigExtractor
from xpu_simulator.frontend.dispatch_extractor import DispatchExtractor
from xpu_simulator.frontend.torch_extractor import TorchGraphExtractor
from xpu_simulator.frontend.export_extractor import ExportExtractor
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.core.fusion import FusionPass, GPU_FUSION_RULES, NPU_FUSION_RULES, DISPATCH_FUSION_RULES
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C


MODELS = {
    "LLaMA-3.1-8B": {
        "config": "benchmarks/hf_models/llama3_8b/config.json",
        "hf_id": "NousResearch/Meta-Llama-3.1-8B",
        "layers_limit": 2,  # limit layers for FX/Export (full model too complex)
    },
    "LLaMA-3.1-70B": {
        "config": "benchmarks/hf_models/llama3_70b/config.json",
        "hf_id": "NousResearch/Meta-Llama-3.1-70B",
        "layers_limit": 2,
    },
    "Mistral-7B": {
        "config": "benchmarks/hf_models/mistral_7b/config.json",
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "layers_limit": 2,
    },
    "Qwen2-7B": {
        "config": "benchmarks/hf_models/qwen2_7b/config.json",
        "hf_id": "Qwen/Qwen2-7B",
        "layers_limit": 2,
    },
    "Qwen2-72B": {
        "config": "benchmarks/hf_models/qwen2_72b/config.json",
        "hf_id": "Qwen/Qwen2-72B",
        "layers_limit": 2,
    },
    "Mixtral-8x7B": {
        "config": "benchmarks/hf_models/mixtral_8x7b/config.json",
        "hf_id": "mistralai/Mixtral-8x7B-v0.1",
        "layers_limit": 2,
    },
    "DeepSeek-V3": {
        "config": "benchmarks/hf_models/deepseek_v3/config.json",
        "hf_id": "deepseek-ai/DeepSeek-V3-0324",
        "layers_limit": 2,
    },
}

DEVICES = {
    "A100": (GPUCostModel(A100_80GB), GPU_FUSION_RULES),
    "H100": (GPUCostModel(H100_80GB), GPU_FUSION_RULES),
    "910B": (NPUCostModel(ASCEND_910B), NPU_FUSION_RULES),
    "910C": (NPUCostModel(ASCEND_910C), NPU_FUSION_RULES),
}

BATCH, SEQ = 1, 1024


def _load_hf_model(hf_id, num_layers=2, dtype=torch.bfloat16):
    """Load HF model on meta device with limited layers."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config.num_hidden_layers = num_layers
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.eval()
    return model, config


def run_config_extractor(model_info):
    """Run ConfigExtractor on config.json."""
    ext = ConfigExtractor()
    t0 = time.time()
    graph = ext.extract(model_info["config"], batch_size=BATCH, seq_len=SEQ)
    elapsed = time.time() - t0
    return graph, elapsed, GPU_FUSION_RULES


def run_dispatch_extractor(model_info):
    """Run DispatchExtractor via extract_from_config (full model)."""
    ext = DispatchExtractor(skip_reshapes=True)
    t0 = time.time()
    graph = ext.extract_from_config(
        model_info["hf_id"],
        batch_size=BATCH, seq_len=SEQ,
    )
    elapsed = time.time() - t0
    return graph, elapsed, DISPATCH_FUSION_RULES


def run_dispatch_extractor_limited(model_info):
    """Run DispatchExtractor with limited layers for faster comparison."""
    ext = DispatchExtractor(skip_reshapes=True)
    t0 = time.time()
    graph = ext.extract_from_config(
        model_info["hf_id"],
        batch_size=BATCH, seq_len=SEQ,
        num_hidden_layers=model_info["layers_limit"],
    )
    elapsed = time.time() - t0
    return graph, elapsed, DISPATCH_FUSION_RULES


def run_fx_extractor(model_info):
    """Run TorchGraphExtractor (FX trace) on HF model."""
    model, config = _load_hf_model(
        model_info["hf_id"], num_layers=model_info["layers_limit"]
    )
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ))

    ext = TorchGraphExtractor()
    t0 = time.time()
    graph = ext.extract(model, (input_ids,), graph_name="fx_model")
    elapsed = time.time() - t0
    return graph, elapsed, GPU_FUSION_RULES


def run_export_extractor(model_info):
    """Run ExportExtractor (torch.export) on HF model."""
    model, config = _load_hf_model(
        model_info["hf_id"], num_layers=model_info["layers_limit"]
    )
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ))

    ext = ExportExtractor()
    t0 = time.time()
    graph = ext.extract(model, (input_ids,), graph_name="export_model")
    elapsed = time.time() - t0
    return graph, elapsed, GPU_FUSION_RULES


EXTRACTORS = {
    "ConfigExtractor": run_config_extractor,
    "DispatchExtractor": run_dispatch_extractor,
    "Dispatch(2-layer)": run_dispatch_extractor_limited,
    "FX(2-layer)": run_fx_extractor,
    "Export(2-layer)": run_export_extractor,
}


def evaluate(graph, fusion_rules, cost_model):
    """Fuse and evaluate graph on a device."""
    fused, _ = FusionPass(fusion_rules).apply(graph)
    result = PerformanceEvaluator(cost_model).run(fused, overlap=True)
    return result


def main():
    os.makedirs("reports", exist_ok=True)

    print(f"\n{'='*100}")
    print(f"  All-Extractor Benchmark: 7 models x 5 extractors x 4 devices")
    print(f"  Batch={BATCH}, SeqLen={SEQ}, FP16/BF16, Prefill phase")
    print(f"{'='*100}\n")

    results = []

    for model_name, model_info in MODELS.items():
        print(f"\n{'─'*80}")
        print(f"  {model_name}")
        print(f"{'─'*80}")

        for ext_name, ext_fn in EXTRACTORS.items():
            print(f"\n  [{ext_name}]", end=" ", flush=True)

            try:
                graph, extract_time, fusion_rules = ext_fn(model_info)
                nodes_before = graph.num_nodes

                print(f"OK ({nodes_before} nodes, {extract_time:.1f}s)")

                # Evaluate on each device
                for dev_name, (cost_model, dev_fusion_rules) in DEVICES.items():
                    # Use dispatch fusion rules for dispatch extractor graphs
                    rules = fusion_rules if "Dispatch" in ext_name else dev_fusion_rules
                    try:
                        result = evaluate(graph, rules, cost_model)
                        latency_ms = result.total_latency_us / 1000
                        fused_ops = len(result.per_op)
                        compute_bound = result.compute_bound_count
                        memory_bound = result.memory_bound_count
                        total_flops_T = result.total_flops / 1e12

                        row = {
                            "model": model_name,
                            "extractor": ext_name,
                            "device": dev_name,
                            "nodes_before": nodes_before,
                            "nodes_after": fused_ops,
                            "extract_time_s": round(extract_time, 2),
                            "latency_ms": round(latency_ms, 1),
                            "flops_T": round(total_flops_T, 2),
                            "compute_bound": compute_bound,
                            "memory_bound": memory_bound,
                            "status": "OK",
                            "error": "",
                        }
                        results.append(row)

                        print(f"    {dev_name:>5}: {latency_ms:>10.1f} ms  "
                              f"({fused_ops} ops, {total_flops_T:.2f} TFLOPS, "
                              f"C={compute_bound}/M={memory_bound})")

                    except Exception as e:
                        row = {
                            "model": model_name, "extractor": ext_name,
                            "device": dev_name, "nodes_before": nodes_before,
                            "nodes_after": 0, "extract_time_s": round(extract_time, 2),
                            "latency_ms": 0, "flops_T": 0,
                            "compute_bound": 0, "memory_bound": 0,
                            "status": "EVAL_ERROR", "error": str(e)[:80],
                        }
                        results.append(row)
                        print(f"    {dev_name:>5}: EVAL ERROR — {str(e)[:60]}")

            except Exception as e:
                print(f"FAILED — {str(e)[:80]}")
                tb = traceback.format_exc()
                # Print last 3 lines of traceback for debugging
                for line in tb.strip().split("\n")[-3:]:
                    print(f"    {line}")

                for dev_name in DEVICES:
                    results.append({
                        "model": model_name, "extractor": ext_name,
                        "device": dev_name, "nodes_before": 0,
                        "nodes_after": 0, "extract_time_s": 0,
                        "latency_ms": 0, "flops_T": 0,
                        "compute_bound": 0, "memory_bound": 0,
                        "status": "EXTRACT_ERROR", "error": str(e)[:80],
                    })

    # Summary table
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY: A100 Latency Comparison (ms)")
    print(f"{'='*100}")
    print(f"  {'Model':<18} {'ConfigExt':>12} {'Dispatch':>12} {'Disp(2L)':>12} {'FX(2L)':>12} {'Export(2L)':>12}")
    print(f"  {'-'*78}")

    for model_name in MODELS:
        row = f"  {model_name:<18}"
        for ext_name in EXTRACTORS:
            matches = [r for r in results
                       if r["model"] == model_name and r["extractor"] == ext_name
                       and r["device"] == "A100"]
            if matches and matches[0]["status"] == "OK":
                row += f" {matches[0]['latency_ms']:>11.1f}"
            else:
                err = matches[0]["status"] if matches else "N/A"
                row += f" {'FAIL':>11}"
        print(row)

    # Extractor capability summary
    print(f"\n  Extractor Success Rate:")
    for ext_name in EXTRACTORS:
        ok = sum(1 for r in results if r["extractor"] == ext_name and r["status"] == "OK")
        total = sum(1 for r in results if r["extractor"] == ext_name)
        models_ok = len(set(r["model"] for r in results
                           if r["extractor"] == ext_name and r["status"] == "OK"))
        print(f"    {ext_name:<22} {models_ok}/7 models  ({ok}/{total} device evaluations)")

    # Save CSV
    csv_path = "reports/all_extractors_benchmark.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to {csv_path}")

    return results


if __name__ == "__main__":
    main()

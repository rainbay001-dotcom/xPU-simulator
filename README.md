# xPU-Simulator
A modular performance simulator for NVIDIA GPU and Huawei Ascend NPU.

Given a model architecture, xPU-Simulator builds a computation graph, estimates per-operator costs using roofline analysis, and evaluates end-to-end latency across different hardware targets.

## Features

- **Roofline-based cost model** with compute/memory bound classification, hardware efficiency factors, and per-op static overhead
- **NVIDIA GPU backend** (A100, H100) with Tensor Core / CUDA Core distinction, compute/memory efficiency factors, and op-class-aware static overhead (5µs TC, 2µs CUDA)
- **Huawei Ascend NPU backend** (910B, 910C) with hybrid roofline cost model (chip-level cube peak with utilization + aggregate memory bandwidth), format conversion overhead, compute/memory efficiency factors (0.70/0.60), per-op static overhead, **per-pipe microarchitectural breakdown** (MTE2/MTE3/VEC/CUBE/SCALAR), and **warm-L2 / cold-HBM regime switch** — validated against Huawei's [msmodeling](https://github.com/opensim-ai/msmodeling) (0.999x–1.077x accuracy) and the CANN CA cycle-accurate simulator (0.98–1.24x geomean across 40 kernels)
- **7 graph extraction methods**: FX trace, torch.export, ONNX, profiler trace, GraphBuilder DSL, ConfigExtractor, DispatchExtractor
- **Config-driven LLM simulation** from HuggingFace `config.json` — supports 8 architectures (LLaMA, Mistral, Qwen2, Mixtral, DeepSeek, GPT-2, Falcon, GPT-NeoX)
- **Quantization-aware modeling**: INT4/INT8/FP8 weight and activation quantization with dequantization overhead, HuggingFace `quantization_config` parsing (GPTQ, AWQ, FP8)
- **Multi-device parallelism**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP) with analytical communication costs (NVLink, HCCS)
- **Prefill and decode phases**: KV cache modeling, autoregressive decode with TTFT/TPOT metrics
- **Serving simulation**: continuous batching scheduler, block-based KV cache allocator, throughput optimizer with SLA-aware batch sizing
- **Empirical calibration**: profiling database with CSV persistence, calibrated cost model (measured latency on hit, analytical fallback on miss)
- **Sparse attention patterns**: dense, top-k (DeepSeek DSA), sliding window — extensible via `AttentionPattern`
- **Kernel fusion**: GPU (FlashAttention, SwiGLU, matmul epilogue), NPU (format conversion), and dispatch-level (RMSNorm, RoPE, FlashAttention, GroupedMatMulSwiGLU for aten-op graphs)
- **Overlap (ASAP) scheduling** for parallel op execution with dual-pipeline support (NPU CUBE/VECTOR)
- **Interactive HTML reports** with architecture overview, per-layer latency breakdown, serving metrics, and top-ops analysis
- **Chrome tracing export** for visualization
- **CLI** with device comparison

## Installation

```bash
pip install networkx torch
```

## Quick Start

### Config-Driven LLM Simulation (Recommended)

The simplest way to simulate an LLM is from its HuggingFace `config.json`:

```python
from xpu_simulator.frontend.config_extractor import ConfigExtractor
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.gpu.hardware import H100_80GB

# Build graph from HuggingFace config
extractor = ConfigExtractor()
graph = extractor.extract("path/to/config.json", batch_size=1, seq_len=1024)

# Or pass a dict directly
graph = extractor.extract({
    "model_type": "llama",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "vocab_size": 128256,
    "hidden_act": "silu",
}, batch_size=1, seq_len=1024)

# Evaluate
result = PerformanceEvaluator(GPUCostModel(H100_80GB)).run(graph, overlap=True)
print(result.summary())
```

### Supported Model Architectures

| model_type | Attention | FFN | Examples |
|---|---|---|---|
| `llama` | GQA/MHA | SwiGLU | LLaMA-2/3, Code Llama |
| `mistral` | GQA | SwiGLU | Mistral-7B |
| `qwen2` | GQA | SwiGLU | Qwen2 |
| `mixtral` | GQA | MoE(SwiGLU) | Mixtral-8x7B |
| `deepseek_v2` | MLA | Dense + MoE(SwiGLU) | DeepSeek-V2/V3 |
| `gpt2` | MHA (no RoPE) | Dense(GELU) | GPT-2 |
| `falcon` | MQA/GQA | Dense(GELU) | Falcon-7B/40B |
| `gpt_neox` | MHA | Dense(GELU) | GPT-NeoX, Pythia |

Custom architectures can be added via `ConfigExtractor.register_handler()`.

### Sparse Attention Patterns

The simulator supports different attention scoring strategies via `AttentionPattern`. These change the effective context length in attention matmuls, directly affecting FLOP counts and latency estimates.

**From config fields** (auto-detected):
```python
# DeepSeek Sparse Attention (top-k)
config = {
    "model_type": "deepseek_v2",
    # ... standard fields ...
    "dsa_num_indexer_heads": 8,
    "dsa_k": 2048,
    "dsa_indexer_dim": 128,
}

# Sliding window (Mistral-style)
config = {
    "model_type": "mistral",
    # ... standard fields ...
    "sliding_window": 4096,
}
```

**Explicit pattern** (for any architecture):
```python
config = {
    "model_type": "llama",
    # ... standard fields ...
    "attention_pattern": {"kind": "sliding_window", "window_size": 512},
}
```

Available patterns: `"dense"` (default), `"top_k"`, `"sliding_window"`.

### Quantization

Simulate quantized inference with INT4/INT8/FP8 precision:

```python
# From HuggingFace quantization_config (auto-detected)
graph = extractor.extract({
    "model_type": "llama",
    # ... standard fields ...
    "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128},
}, batch_size=1, seq_len=1024)

# Or explicit QuantConfig
from xpu_simulator.core.operator import Dtype, QuantConfig
graph = extractor.extract(config, batch_size=1, seq_len=1024)
# QuantConfig(weight_dtype=Dtype.INT4, activation_dtype=Dtype.INT8, group_size=128)
```

Supported: W8A8 (INT8 weights + activations), FP8, W4A8 (INT4 weights + INT8 activations with dequantization).

### Multi-Device Parallelism

Model tensor, data, and expert parallelism with communication costs:

```python
from xpu_simulator.core.parallel import ParallelConfig

# TP=4 on H100 NVLink
graph = extractor.extract(config, batch_size=1, seq_len=1024,
                          parallel_config=ParallelConfig(tp_size=4))

# EP=8 for MoE models
graph = extractor.extract(config, batch_size=1, seq_len=1024,
                          parallel_config=ParallelConfig(ep_size=8))
```

Communication ops (ALL_REDUCE, ALL_GATHER, ALL_TO_ALL) are analytically modeled using ring/tree algorithms and overlap with compute in the scheduler.

### Prefill and Decode Phases

Simulate both prefill (prompt processing) and decode (token generation):

```python
# Prefill
graph_prefill = extractor.extract(config, batch_size=1, seq_len=1024)
result = evaluator.run(graph_prefill, overlap=True)
print(f"TTFT: {result.ttft_ms:.2f} ms")

# Decode (autoregressive, 1 new token, reading KV cache)
graph_decode = extractor.extract(config, batch_size=1, seq_len=1024,
                                  phase="decode", kv_seq_len=1024)
result = evaluator.run(graph_decode, overlap=True)
print(f"TPOT: {result.tpot_ms:.3f} ms")
```

### Serving Simulation

End-to-end inference serving with continuous batching:

```python
from xpu_simulator.serving import ServingSimulator, ServingConfig, Request

cfg = ServingConfig(max_batch_size=32, max_seq_len=4096,
                    max_tokens_budget=4096, block_size=16, num_kv_blocks=1024)

sim = ServingSimulator(model_config=config, cost_model=model, serving_config=cfg)

requests = [Request(id=i, prompt_len=128, output_len=64) for i in range(100)]
metrics = sim.run(requests)
print(metrics.summary())
# Requests, throughput (tok/s, req/s), TTFT (avg/p50/p99), TPOT (avg/p50/p99)
```

Find the maximum batch size that meets a TPOT SLA:

```python
from xpu_simulator.serving.simulator import find_max_throughput

best_batch = find_max_throughput(
    model_config=config, cost_model=model, serving_config=cfg,
    requests=requests, sla_tpot_ms=50.0, min_batch=1, max_batch=128,
)
```

### Profiling Calibration

Override analytical estimates with measured latencies:

```python
from xpu_simulator.core.profiling_db import ProfilingDB
from xpu_simulator.core.cost_model import CalibratedCostModel

db = ProfilingDB()
db.store(op, measured_latency_us=42.5)
db.save("profile.csv")  # Persist to CSV

calibrated = CalibratedCostModel(base_model, db)
cost = calibrated.estimate(op)  # Uses measured value on hit, analytical on miss
print(f"Hit rate: {calibrated.hit_rate:.1%}")
```

### DeepSeek V3.2 671B Example

```bash
# ConfigExtractor: batch=1, seq_len=1024
python examples/deepseek_v3_2.py

# Custom batch size and sequence length
python examples/deepseek_v3_2.py 4 2048

# Compare dense MLA vs DSA (sparse attention)
python examples/deepseek_v3_2.py --dsa 1 4096

# DispatchExtractor: full aten-level graph with TP comparison
python examples/deepseek_v3_dispatch.py --model=deepseek-ai/DeepSeek-V3-0324 --tp=2
python examples/deepseek_v3_dispatch.py --model=deepseek-ai/DeepSeek-V3 --tp=4
```

### CLI

```bash
# Simulate an MLP on A100
python -m xpu_simulator.cli --model mlp --backend gpu --device a100

# With overlap modeling and ASCII timeline
python -m xpu_simulator.cli --model mlp --backend gpu --device a100 --overlap --timeline

# Compare GPU vs NPU
python -m xpu_simulator.cli --model mlp --backend gpu --device a100 --compare npu:910b

# Config-driven extraction
python -m xpu_simulator.cli --extractor config --config-path config.json --batch-size 1 --seq-len 1024

# Export Chrome trace (open with chrome://tracing or https://ui.perfetto.dev)
python -m xpu_simulator.cli --model mlp --backend npu --device 910c --trace trace.json
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--model` | Model name: `mlp`, `resnet18`, `resnet50` |
| `--backend` | Hardware backend: `gpu`, `npu` |
| `--device` | Device: `a100`, `h100` (GPU) or `910b`, `910c` (NPU) |
| `--dtype` | Data type: `fp16`, `fp32`, `bf16` |
| `--extractor` | Graph extraction: `fx`, `export`, `onnx`, `profiler`, `config` |
| `--config-path` | Path to HuggingFace `config.json` (with `--extractor config`) |
| `--batch-size` | Batch size (with `--extractor config`, default: 1) |
| `--seq-len` | Sequence length (with `--extractor config`, default: 1024) |
| `--overlap` | Enable parallel execution of independent ops |
| `--timeline` | Print ASCII execution timeline |
| `--trace FILE` | Export Chrome tracing JSON |
| `--compare BACKEND:DEVICE` | Compare with another device (e.g., `npu:910b`) |

### Python API

```python
from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel

# Estimate cost of a single matmul
op = OpSpec(
    op_type=OpType.MATMUL,
    inputs=[TensorSpec((2048, 2048), Dtype.FP16), TensorSpec((2048, 2048), Dtype.FP16)],
    outputs=[TensorSpec((2048, 2048), Dtype.FP16)],
)
cost = GPUCostModel(A100_80GB).estimate(op)
print(f"Latency: {cost.latency_us:.2f} us, Bound: {cost.bound}")

# Build and evaluate a graph
graph = ComputeGraph("example")
n1 = graph.add_node(op, "matmul_1")
n2 = graph.add_node(op, "matmul_2")
graph.add_edge(n1, n2)

result = PerformanceEvaluator(GPUCostModel(A100_80GB)).run(graph, overlap=True)
print(result.summary())
```

### Graph Extraction Methods

| Method | Use Case |
|--------|----------|
| **ConfigExtractor** | LLMs from HuggingFace `config.json` (recommended for LLMs) |
| **DispatchExtractor** | Any PyTorch model via `TorchDispatchMode` — handles dynamic control flow, MoE, custom ops |
| **GraphBuilder** | Manual construction for custom/hypothetical architectures |
| **TorchGraphExtractor** | PyTorch models via `torch.fx` (simple models only) |
| **ExportExtractor** | PyTorch models via `torch.export` (handles more control flow) |
| **ONNXExtractor** | ONNX model files |
| **ProfilerExtractor** | Reconstruct from `torch.profiler` Chrome trace |

### DispatchExtractor

The `DispatchExtractor` intercepts every aten op at runtime via PyTorch's `TorchDispatchMode`. Unlike FX tracing, it handles dynamic control flow, data-dependent branching, and custom ops.

```python
from xpu_simulator.frontend.dispatch_extractor import DispatchExtractor

# From a PyTorch model
extractor = DispatchExtractor(skip_reshapes=True)
graph = extractor.extract(model, (input_tensor,), "my_model")

# From a HuggingFace model ID (loads on meta device, no GPU needed)
graph = extractor.extract_from_config(
    "deepseek-ai/DeepSeek-V3",
    batch_size=1, seq_len=4096,
    num_hidden_layers=2,  # optional: limit layers for quick testing
)
```

The dispatch graph captures fine-grained aten ops (6,513 for DeepSeek V3). Apply dispatch-level fusion to consolidate them before evaluation:

```python
from xpu_simulator.core.fusion import FusionPass, DISPATCH_FUSION_RULES

fused_graph, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
print(result.summary())
# Fusion: 6513 ops -> 3425 ops (3088 eliminated)
# rms_norm: 245, flash_attention: 61, grouped_matmul_swiglu: 58, ...
```

### Kernel Fusion

The simulator applies hardware-specific fusion passes before evaluation. There are two fusion levels:

**Config-level fusion** (for ConfigExtractor graphs with named logical ops):
```python
from xpu_simulator.core.fusion import FusionPass, GPU_FUSION_RULES, NPU_FUSION_RULES

fused_graph, result = FusionPass(GPU_FUSION_RULES).apply(graph)   # FlashAttention, SwiGLU, epilogue
fused_graph, result = FusionPass(NPU_FUSION_RULES).apply(graph)   # + format conversion
```

**Dispatch-level fusion** (for DispatchExtractor graphs with fine-grained aten ops):
```python
from xpu_simulator.core.fusion import FusionPass, DISPATCH_FUSION_RULES, DISPATCH_NPU_FUSION_RULES

fused_graph, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
fused_graph, result = FusionPass(DISPATCH_NPU_FUSION_RULES).apply(graph)
print(result.summary())
```

Dispatch fusion rules (inspired by [msmodeling](https://github.com/opensim-ai/msmodeling)):

| Rule | Pattern | Fused Result |
|------|---------|-------------|
| `RMSNormFusion` | pow → mean → add → rsqrt → mul → mul | 1 LAYER_NORM |
| `ResidualAddRMSNormFusion` | ADD + fused RMSNorm | 1 LAYER_NORM |
| `RoPEFusion` | cos → sin → neg → mul → add | 1 ROPE |
| `DispatchFlashAttentionFusion` | BMM → (scale/mask) → Softmax → BMM | 1 fused MATMUL |
| `GroupedMatMulSwiGLUFusion` | grouped_mm → SILU → MUL | 1 fused MATMUL |

### HTML Reports

Generate interactive HTML reports with architecture diagrams and latency breakdowns:

```python
from xpu_simulator.utils.html_report import export_html_report

export_html_report(graph, results, "report.html",
                   model_name="DeepSeek V3.2 671B",
                   config=config_dict, n_dense=3)
```

## Supported Hardware

| Device | Type | Peak FP16 | Peak INT8/FP8 | HBM Bandwidth | Interconnect |
|--------|------|-----------|---------------|----------------|--------------|
| NVIDIA A100 80GB | GPU | 312 TFLOPS | 624 TOPS (INT8) | 2039 GB/s | NVLink 600 GB/s |
| NVIDIA H100 80GB | GPU | 989 TFLOPS | 1979 TOPS (FP8/INT8) | 3350 GB/s | NVLink 900 GB/s |
| Ascend 910B | NPU | 320 TFLOPS (CUBE) | 640 TOPS (FP8) | 1600 GB/s | HCCS 392 GB/s |
| Ascend 910C | NPU | 400 TFLOPS (CUBE) | 800 TOPS (FP8) | 1800 GB/s | HCCS 600 GB/s |

### Efficiency Factors

Raw peak specs are unachievable in practice. The simulator applies hardware-calibrated efficiency factors (inspired by [msmodeling](https://github.com/opensim-ai/msmodeling), re-calibrated 2026-04-11 against 910B msprof + CA simulator):

| Factor | GPU (A100/H100) | NPU 910B | NPU 910C | Effect |
|--------|-----------------|-----------|-----------|--------|
| Compute (matmul) | 0.70 | 0.70 | 0.59 | Effective TFLOPS = peak × factor |
| Compute (elementwise) | 0.85 | 0.80 (VECTOR) | 0.80 | Elementwise/norm/activation |
| Memory bandwidth | 0.80 | 0.60 | 0.71 | Effective BW = peak × factor |
| Static (matmul) | 5 µs/op | 5 µs/op | 5 µs/op | Kernel dispatch, sync |
| Static (vector floor) | 2 µs/op | 1.7 µs/op | 2 µs/op | CA sim: ~3000 cycle floor |
| LayerNorm / Softmax extra | — | +2 / +3 µs | +2 / +3 µs | Reduction-tree init |

The 910B memory-bandwidth factor of 0.60 is independently corroborated by the CA simulator: an ADD 2048×4096 kernel on cold HBM completes at 940 GB/s effective BW, exactly 0.59 × 1600 GB/s peak.

### Microarchitectural Modeling (NPU)

The NPU cost model populates per-pipe busy-time fields on every `OpCost`,
exposing the DaVinci AI core pipeline breakdown without running a
cycle-accurate simulator:

| Pipe | What it models |
|---|---|
| `mte2_us` | GM/L2 → L1/UB load engine (scaled by memory-pass count for LN/Softmax/RoPE) |
| `mte3_us` | L1/UB → GM store engine (fixpipe for CUBE outputs) |
| `vec_us` | VECTOR unit compute time (elementwise, reduction, activation) |
| `cube_us` | CUBE unit compute time (matrix / conv) |
| `scalar_us` | SCALAR unit dispatch + sync floor |

Pipe values do **not** sum to `latency_us` — the pipes overlap via
double-buffering. Latency stays `max(compute, memory) + overhead`, matching
observed kernel times. Pipe breakdown feeds downstream bottleneck analysis
and pipeline visualization.

**Multi-pass memory multipliers** (`_VECTOR_MEM_PASSES`) for reduction ops:

| Op | Passes | Rationale |
|---|---|---|
| LayerNorm | 2.8× | mean + variance + normalize |
| Softmax | 1.7× | max + exp-sum + normalize (inner passes hit UB) |
| RoPE | 2.5× | trig unit serialization (CA sim) |
| ADD / MUL / RELU / GELU / SILU | 1.0× | single streaming pass |

**Warm vs cold execution regime** (`NPUCostModel(hw, warm_l2=True|False)`):

- `warm_l2=True` (default) — simple elementwise ops on warm activations
  (≤ 48 MB unique input) hit L2 at 3200 GB/s × 0.60 instead of HBM. Matches
  steady-state LLM inference and microbenchmark iter-2+ measurements.
- `warm_l2=False` — cold HBM everywhere. Matches first-iteration execution
  and the CANN CA simulator.

The 48 MB cap was derived from the observed msprof warm→cold transition
(Mul 2048×4096 warm at 11 µs; Mul 4096×4096 cold at 75 µs).

### Validated Against msmodeling

The NPU backend has been validated against Huawei's open-source [msmodeling](https://github.com/opensim-ai/msmodeling) (MindStudio-Modeling) simulator on real HuggingFace model configs running on the Ascend 910B (ATLAS_800_A2_376T_64G) device profile:

| Model | xPU-sim (ms) | msmodeling (ms) | Ratio | Diff |
|-------|-------------|-----------------|-------|------|
| LLaMA-3.1-8B | 79.6 | 79.7 | 0.999x | -0.1% |
| Qwen2-7B | 74.8 | 72.3 | 1.034x | +3.4% |
| LLaMA-3.1-70B | 692.2 | 645.7 | 1.072x | +7.2% |
| Qwen2-72B | 711.5 | 660.4 | 1.077x | +7.7% |

**Average ratio: 1.046x** (within 5% for 7B models, within 8% for 70B+ models). Differences are mainly due to hardware spec differences — msmodeling uses 376T MMA / 22T GP / 1759 GB/s while xPU-sim uses 320T CUBE / 10T VECTOR / 1600 GB/s for the 910B.

### Validated Against the CANN CA Simulator

`scripts/validate_against_ca_sim.py` runs the NPU cost model in cold-path
mode (`warm_l2=False`) against 40 kernels traced from the CANN cycle-
accurate simulator (`reports/ca_pipe_analysis.json`):

| Op family | Kernels | Geomean ratio | Range |
|-----------|---------|---------------|-------|
| ADD | 10 | 0.98× | 0.86–1.11 |
| MUL | 10 | 0.98× | 0.86–1.10 |
| RELU | 10 | 1.06× | 0.95–1.21 |
| RoPE | 10 | 1.24× | 1.02–1.66 |

The script fails CI if any op family drifts outside [0.80, 1.25], giving us
a regression harness every future cost-model change must satisfy.

### ConfigExtractor vs DispatchExtractor

Both extraction methods were benchmarked on real HuggingFace models (LLaMA 8B/70B, Mistral 7B, Qwen2 7B/72B, Mixtral 8x7B) across 4 devices:

- **ConfigExtractor**: Produces logical ops (~515 for 8B, ~1283 for 70B). Closer to Flash Attention production behavior.
- **DispatchExtractor**: Captures aten-level ops (~1813 for 8B, ~4501 for 70B) including FP32 attention score upcasting for softmax numerical stability — adds ~40ms overhead on GPU for 8B models.
- **NPU agreement**: ConfigExtractor and DispatchExtractor agree within 4–8% on NPU (no FP32 attention overhead since NPU doesn't upcast).
- **GPU divergence**: DispatchExtractor is 25–50% slower on GPU due to FP32 attention matmuls (FP32 peak is ~16x lower than FP16 on A100).

See `benchmarks/run_real_models.py` and `benchmarks/run_vs_msmodeling.py` for reproducible comparisons.

## Project Structure

```
xpu_simulator/
  core/           # Graph, operators, cost model, evaluator, fusion, parallelism, KV cache, profiling DB
  backends/       # Device-specific: GPU (A100/H100), NPU (910B/910C) with interconnect specs
  frontend/       # Graph extraction: FX, export, ONNX, profiler, config, builder, dispatch (quant/TP/EP/decode-aware)
  serving/        # Serving simulation: scheduler, KV cache allocator, request state, metrics, throughput optimizer
  utils/          # HTML reports, Chrome tracing, op categorization
  cli.py          # Command-line interface
benchmarks/       # Comparison benchmarks: run_real_models.py, run_vs_msmodeling.py, HF model configs
examples/         # DeepSeek V3.2 671B simulation scripts
reports/          # Generated HTML reports, Perfetto traces, and benchmark CSVs
tests/            # 173 tests across 17 test files
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Limitations

- Cost estimates are analytical (roofline + efficiency factors), not cycle-accurate — use profiling calibration for higher accuracy
- NPU format conversion costs are approximate
- MoE expert routing assumes uniform token distribution
- Communication costs use analytical ring/tree models (no hierarchical topology routing)
- Serving simulation uses simplified continuous batching (no preemption or speculative decoding)

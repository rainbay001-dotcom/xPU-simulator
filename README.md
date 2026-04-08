# xPU-Simulator
A modular performance simulator for NVIDIA GPU and Huawei Ascend NPU.

Given a model architecture, xPU-Simulator builds a computation graph, estimates per-operator costs using roofline analysis, and evaluates end-to-end latency across different hardware targets.

## Features

- Roofline-based cost model with compute/memory bound classification
- NVIDIA GPU backend (A100, H100) with kernel launch overhead
- Huawei Ascend NPU backend (910B, 910C) with CUBE/VECTOR pipeline modeling, tile alignment, and format conversion costs
- PyTorch FX graph extraction for simple models
- Manual graph construction for complex models (MoE, MLA, custom kernels)
- Overlap (ASAP) scheduling for parallel op execution
- Chrome tracing export for visualization
- CLI with device comparison

## Installation

```bash
pip install networkx torch
```

## Quick Start

### CLI

```bash
# Simulate an MLP on A100
python -m xpu_simulator.cli --model mlp --backend gpu --device a100

# With overlap modeling and ASCII timeline
python -m xpu_simulator.cli --model mlp --backend gpu --device a100 --overlap --timeline

# Compare GPU vs NPU
python -m xpu_simulator.cli --model mlp --backend gpu --device a100 --compare npu:910b

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

### PyTorch Model Extraction

For simple models that `torch.fx` can trace:

```python
import torch.nn as nn
from xpu_simulator.frontend.torch_extractor import TorchGraphExtractor

model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 512))
graph = TorchGraphExtractor().extract(model, (torch.randn(32, 1024),))
```

For complex models (MoE, custom kernels, dynamic control flow), build the graph manually from the model config. See `examples/deepseek_v3_2.py` for a full example.

### DeepSeek V3.2 671B Example

```bash
# Default: batch=1, seq_len=1024
python examples/deepseek_v3_2.py

# Custom batch size and sequence length
python examples/deepseek_v3_2.py 4 2048
```

## Supported Hardware

| Device | Type | Peak FP16 | HBM Bandwidth |
|--------|------|-----------|----------------|
| NVIDIA A100 80GB | GPU | 312 TFLOPS | 2039 GB/s |
| NVIDIA H100 80GB | GPU | 989 TFLOPS | 3350 GB/s |
| Ascend 910B | NPU | 320 TFLOPS (CUBE) | 1600 GB/s |
| Ascend 910C | NPU | 400 TFLOPS (CUBE) | 1800 GB/s |

## Project Structure

```
xpu_simulator/
  core/           # Hardware-agnostic: graph, operators, cost model, evaluator
  backends/       # Device-specific: GPU (A100/H100), NPU (910B/910C)
  frontend/       # PyTorch FX graph extraction, op registry
  utils/          # Roofline analysis, Chrome tracing export
  cli.py          # Command-line interface
examples/         # DeepSeek V3.2 671B simulation
tests/            # Phase 1-5 verification tests
```

## Running Tests

```bash
python -m pytest tests/ -v
# or run individually:
python tests/test_phase1.py   # Roofline cost model
python tests/test_phase2.py   # Graph + evaluator
python tests/test_phase3.py   # PyTorch frontend
python tests/test_phase4.py   # Ascend NPU backend
python tests/test_phase5.py   # Overlap + profiling
```

## Limitations

- Cost estimates are analytical (roofline-based), not cycle-accurate
- Does not model memory capacity constraints or OOM scenarios
- NPU format conversion costs are approximate
- MoE expert routing assumes uniform token distribution
- No multi-GPU/multi-NPU distributed simulation yet

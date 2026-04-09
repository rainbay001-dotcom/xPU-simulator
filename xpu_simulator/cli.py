"""CLI entry point for xPU-Simulator."""
from __future__ import annotations

import argparse
import sys


def get_hardware(backend: str, device: str):
    """Get hardware spec by backend and device name."""
    if backend == "gpu":
        from .backends.gpu.hardware import A100_80GB, H100_80GB
        devices = {"a100": A100_80GB, "h100": H100_80GB}
    elif backend == "npu":
        from .backends.npu.hardware import ASCEND_910B, ASCEND_910C
        devices = {"910b": ASCEND_910B, "910c": ASCEND_910C}
    else:
        print(f"Unknown backend: {backend}")
        sys.exit(1)

    device = device.lower()
    if device not in devices:
        print(f"Unknown device '{device}' for {backend}. Available: {list(devices.keys())}")
        sys.exit(1)

    return devices[device]


def get_cost_model(backend: str, hw):
    """Get cost model for a backend."""
    if backend == "gpu":
        from .backends.gpu.cost_model import GPUCostModel
        return GPUCostModel(hw)
    elif backend == "npu":
        from .backends.npu.cost_model import NPUCostModel
        return NPUCostModel(hw)


def get_model(model_name: str):
    """Get a PyTorch model by name."""
    import torch
    import torch.nn as nn

    models = {}

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 4096)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4096, 1024)
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    models["mlp"] = (SimpleMLP(), (torch.randn(32, 1024),))

    try:
        import torchvision.models as tv
        models["resnet18"] = (tv.resnet18(), (torch.randn(1, 3, 224, 224),))
        models["resnet50"] = (tv.resnet50(), (torch.randn(1, 3, 224, 224),))
    except ImportError:
        pass

    model_name = model_name.lower()
    if model_name not in models:
        print(f"Unknown model '{model_name}'. Available: {list(models.keys())}")
        sys.exit(1)

    return models[model_name]


def main():
    parser = argparse.ArgumentParser(
        description="xPU-Simulator: Performance estimation for GPU and NPU",
        prog="xpu-sim",
    )
    parser.add_argument("--model", type=str, default="mlp",
                        help="Model name (mlp, resnet18, resnet50)")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "npu"],
                        help="Hardware backend")
    parser.add_argument("--device", type=str, default="a100",
                        help="Device name (gpu: a100/h100, npu: 910b/910c)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"],
                        help="Data type")
    parser.add_argument("--extractor", type=str, default="fx",
                        choices=["fx", "export", "onnx", "profiler", "config"],
                        help="Graph extraction method (default: fx)")
    parser.add_argument("--onnx-path", type=str, default=None,
                        help="Path to ONNX model file (required for --extractor onnx)")
    parser.add_argument("--trace-path", type=str, default=None,
                        help="Path to profiler Chrome trace JSON (required for --extractor profiler)")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to HuggingFace config.json (required for --extractor config)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (used with --extractor config)")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length (used with --extractor config)")
    parser.add_argument("--overlap", action="store_true",
                        help="Enable overlap modeling (parallel execution of independent ops)")
    parser.add_argument("--trace", type=str, default=None,
                        help="Export Chrome trace to file (e.g., trace.json)")
    parser.add_argument("--timeline", action="store_true",
                        help="Print ASCII timeline")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare with another backend:device (e.g., npu:910b)")

    args = parser.parse_args()

    # Setup
    hw = get_hardware(args.backend, args.device)
    cost_model = get_cost_model(args.backend, hw)

    # Extract model graph
    from .core.operator import Dtype

    dtype_map = {"fp16": Dtype.FP16, "fp32": Dtype.FP32, "bf16": Dtype.BF16}
    dtype = dtype_map[args.dtype]

    if args.extractor == "fx":
        from .frontend.torch_extractor import TorchGraphExtractor
        extractor = TorchGraphExtractor(dtype=dtype)
        model, example_inputs = get_model(args.model)
        graph = extractor.extract(model, example_inputs, args.model)
    elif args.extractor == "export":
        from .frontend.export_extractor import ExportExtractor
        extractor = ExportExtractor(dtype=dtype)
        model, example_inputs = get_model(args.model)
        graph = extractor.extract(model, example_inputs, args.model)
    elif args.extractor == "onnx":
        if not args.onnx_path:
            print("Error: --onnx-path is required when using --extractor onnx")
            sys.exit(1)
        from .frontend.onnx_extractor import ONNXExtractor
        extractor = ONNXExtractor(dtype=dtype)
        graph = extractor.extract(args.onnx_path, args.model)
    elif args.extractor == "profiler":
        if not args.trace_path:
            print("Error: --trace-path is required when using --extractor profiler")
            sys.exit(1)
        from .frontend.profiler_extractor import ProfilerExtractor
        extractor = ProfilerExtractor(dtype=dtype)
        graph = extractor.extract(args.trace_path, args.model)
    elif args.extractor == "config":
        if not args.config_path:
            print("Error: --config-path is required when using --extractor config")
            sys.exit(1)
        from .frontend.config_extractor import ConfigExtractor
        extractor = ConfigExtractor(dtype=dtype)
        graph = extractor.extract(args.config_path, batch_size=args.batch_size,
                                  seq_len=args.seq_len, graph_name=args.model)

    # Evaluate
    from .core.evaluator import PerformanceEvaluator

    evaluator = PerformanceEvaluator(cost_model)
    result = evaluator.run(graph, overlap=args.overlap)

    # Output
    print(f"\n{'='*60}")
    print(f"xPU-Simulator: {args.model} on {hw.name}")
    print(f"{'='*60}")
    print(f"Graph: {graph.num_nodes} ops, {graph.num_edges} edges")
    print(f"Dtype: {args.dtype}")
    print(f"Overlap: {'enabled' if args.overlap else 'disabled'}")
    print(f"{'='*60}")
    print(result.summary())
    print()

    # Per-op breakdown
    print("Per-op breakdown:")
    for r in result.per_op:
        name = (r.node.name or r.node.op.op_type.name)[:20].ljust(20)
        print(f"  {name}  {r.cost.latency_us:8.2f} us  {r.cost.bound:20s}  util={r.cost.utilization:.0%}")
    print()

    # Timeline
    if args.timeline:
        from .utils.profiling import print_timeline
        print_timeline(result)
        print()

    # Chrome trace
    if args.trace:
        from .utils.profiling import to_chrome_trace
        to_chrome_trace(result, args.trace)
        print(f"Trace exported to: {args.trace}")
        print()

    # Comparison
    if args.compare:
        parts = args.compare.split(":")
        if len(parts) != 2:
            print("--compare format: backend:device (e.g., npu:910b)")
            sys.exit(1)
        cmp_backend, cmp_device = parts
        cmp_hw = get_hardware(cmp_backend, cmp_device)
        cmp_cost = get_cost_model(cmp_backend, cmp_hw)
        cmp_eval = PerformanceEvaluator(cmp_cost)
        cmp_result = cmp_eval.run(graph, overlap=args.overlap)

        print(f"{'='*60}")
        print(f"Comparison: {hw.name} vs {cmp_hw.name}")
        print(f"{'='*60}")
        print(f"  {hw.name:25s}: {result.total_latency_us:.2f} us")
        print(f"  {cmp_hw.name:25s}: {cmp_result.total_latency_us:.2f} us")
        ratio = result.total_latency_us / cmp_result.total_latency_us if cmp_result.total_latency_us > 0 else 0
        if ratio > 1:
            print(f"  {cmp_hw.name} is {ratio:.2f}x faster")
        else:
            print(f"  {hw.name} is {1/ratio:.2f}x faster")
        print()


if __name__ == "__main__":
    main()

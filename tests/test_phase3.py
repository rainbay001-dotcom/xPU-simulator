"""Phase 3 verification: PyTorch frontend extractor."""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn

from xpu_simulator.frontend.torch_extractor import TorchGraphExtractor
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


def test_simple_mlp():
    """Extract graph from a simple MLP and evaluate."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 2048)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(2048, 512)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = MLP()
    x = torch.randn(32, 1024)

    extractor = TorchGraphExtractor()
    graph = extractor.extract(model, (x,), "mlp")

    print(f"Graph: {graph}")
    print(f"Nodes: {graph.num_nodes}")
    for node in graph.topo_order():
        print(f"  {node}: {node.op.op_type.name} flops={node.op.flops:,} bytes={node.op.memory_bytes:,}")
    print()

    # Evaluate on A100
    model_cost = GPUCostModel(A100_80GB)
    evaluator = PerformanceEvaluator(model_cost)
    result = evaluator.run(graph)
    print(result.summary())
    print()

    assert graph.num_nodes >= 3, f"Expected >= 3 nodes, got {graph.num_nodes}"
    assert result.total_latency_us > 0


def test_resnet_block():
    """Extract from a residual block pattern."""

    class ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            residual = x
            out = self.relu1(self.conv1(x))
            out = self.conv2(out)
            out = self.relu2(out + residual)
            return out

    model = ResBlock()
    x = torch.randn(1, 64, 56, 56)

    extractor = TorchGraphExtractor()
    graph = extractor.extract(model, (x,), "resblock")

    print(f"Graph: {graph}")
    for node in graph.topo_order():
        print(f"  {node}: {node.op.op_type.name} flops={node.op.flops:,}")
    print()

    model_cost = GPUCostModel(A100_80GB)
    result = PerformanceEvaluator(model_cost).run(graph)
    print(result.summary())

    assert graph.num_nodes >= 4


if __name__ == "__main__":
    test_simple_mlp()
    print("---")
    test_resnet_block()
    print("\nAll Phase 3 tests passed!")

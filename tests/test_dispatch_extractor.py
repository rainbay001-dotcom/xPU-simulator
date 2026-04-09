"""Tests for DispatchExtractor — TorchDispatchMode-based graph extraction."""
import pytest
import torch
import torch.nn as nn

from xpu_simulator.frontend.dispatch_extractor import DispatchExtractor
from xpu_simulator.core.operator import OpType, Dtype
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


class SimpleMLP(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d * 4, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MLPWithBranch(nn.Module):
    """Model with data-dependent control flow — FX trace would fail."""
    def __init__(self, d=512):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        out = self.fc1(x)
        # Dynamic branch: FX can't handle this
        if out.shape[0] > 0:
            out = self.fc2(out)
        return out


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d=256, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn_qkv = nn.Linear(d, d * 3)
        self.attn_out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn1 = nn.Linear(d, d * 4)
        self.act = nn.GELU()
        self.ffn2 = nn.Linear(d * 4, d)
        self.heads = heads
        self.head_dim = d // heads

    def forward(self, x):
        B, S, D = x.shape
        # Attention
        h = self.norm1(x)
        qkv = self.attn_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.attn_out(out)
        x = x + out
        # FFN
        h = self.norm2(x)
        h = self.ffn2(self.act(self.ffn1(h)))
        return x + h


class TestDispatchExtractorBasic:
    def test_simple_mlp(self):
        model = SimpleMLP(512)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(2, 512),), "mlp")

        assert graph.num_nodes > 0
        op_types = [n.op.op_type for n in graph.topo_order()]
        assert OpType.MATMUL in op_types, f"Expected MATMUL, got {op_types}"
        assert OpType.SILU in op_types, f"Expected SILU, got {op_types}"

    def test_mlp_has_edges(self):
        model = SimpleMLP(512)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(2, 512),), "mlp")

        assert graph.num_edges > 0

    def test_dynamic_control_flow(self):
        """DispatchExtractor handles dynamic branches that FX can't trace."""
        model = MLPWithBranch(512)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(2, 512),), "branch")

        op_types = [n.op.op_type for n in graph.topo_order()]
        # Both fc1 and fc2 should be captured (branch was taken)
        matmul_count = sum(1 for t in op_types if t == OpType.MATMUL)
        assert matmul_count >= 2, f"Expected 2+ matmuls, got {matmul_count}"

    def test_skip_reshapes(self):
        """With skip_reshapes=True, view/reshape ops are excluded."""
        model = SimpleMLP(512)
        ext_skip = DispatchExtractor(skip_reshapes=True)
        ext_keep = DispatchExtractor(skip_reshapes=False)

        graph_skip = ext_skip.extract(model, (torch.randn(2, 512),))
        graph_keep = ext_keep.extract(model, (torch.randn(2, 512),))

        assert graph_skip.num_nodes <= graph_keep.num_nodes

    def test_dict_inputs(self):
        """Can pass inputs as a dict (kwargs)."""
        model = SimpleMLP(512)
        extractor = DispatchExtractor()
        # Wrap so model accepts 'x' kwarg
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)

        graph = extractor.extract(Wrapper(model), {"x": torch.randn(2, 512)})
        assert graph.num_nodes > 0


class TestDispatchExtractorTransformer:
    def test_transformer_block(self):
        model = SimpleTransformerBlock(d=256, heads=4)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(1, 32, 256),), "transformer")

        op_types = [n.op.op_type for n in graph.topo_order()]
        assert OpType.MATMUL in op_types
        assert OpType.SOFTMAX in op_types
        assert OpType.LAYER_NORM in op_types
        assert OpType.GELU in op_types

    def test_transformer_matmul_count(self):
        """QKV proj + attn scores + attn*V + attn_out + ffn1 + ffn2 = 6 matmuls."""
        model = SimpleTransformerBlock(d=256, heads=4)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(1, 32, 256),), "transformer")

        matmul_count = sum(1 for n in graph.topo_order() if n.op.op_type == OpType.MATMUL)
        # qkv(1) + scores(1) + attn*v(1) + out(1) + ffn1(1) + ffn2(1) = 6
        assert matmul_count == 6, f"Expected 6 matmuls, got {matmul_count}"

    def test_evaluator_runs(self):
        """Extracted graph can be evaluated by the cost model."""
        model = SimpleTransformerBlock(d=256, heads=4)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(1, 32, 256),))

        result = PerformanceEvaluator(GPUCostModel(A100_80GB)).run(graph, overlap=True)
        assert result.total_latency_us > 0
        assert result.total_flops > 0


class TestDispatchExtractorShapes:
    def test_matmul_shapes_correct(self):
        """Verify the captured matmul has correct input shapes."""
        model = nn.Linear(512, 1024, bias=False)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(2, 512),))

        matmuls = [n for n in graph.topo_order() if n.op.op_type == OpType.MATMUL]
        assert len(matmuls) == 1
        op = matmuls[0].op
        # mm: input [2, 512] x weight [512, 1024] -> output [2, 1024]
        assert op.inputs[0].shape == (2, 512)
        assert op.outputs[0].shape == (2, 1024)

    def test_dtype_preserved(self):
        """Tensor dtypes are captured correctly."""
        model = nn.Linear(64, 128, bias=False)
        extractor = DispatchExtractor()
        graph = extractor.extract(
            model.to(torch.float16),
            (torch.randn(1, 64, dtype=torch.float16),),
        )
        matmuls = [n for n in graph.topo_order() if n.op.op_type == OpType.MATMUL]
        assert len(matmuls) == 1
        assert matmuls[0].op.inputs[0].dtype == Dtype.FP16


class TestDispatchExtractorFlops:
    def test_matmul_flops(self):
        """FLOPs computed correctly from captured shapes."""
        model = nn.Linear(512, 1024, bias=False)
        extractor = DispatchExtractor()
        graph = extractor.extract(model, (torch.randn(4, 512),))

        matmuls = [n for n in graph.topo_order() if n.op.op_type == OpType.MATMUL]
        flops = matmuls[0].op.flops
        # [4, 512] x [512, 1024]: 2 * 4 * 512 * 1024 = 4,194,304
        assert flops == 2 * 4 * 512 * 1024

"""Tests for dispatch-level fusion rules (aten op consolidation)."""
import pytest
import torch
import torch.nn as nn

from xpu_simulator.frontend.dispatch_extractor import DispatchExtractor
from xpu_simulator.core.fusion import (
    FusionPass,
    DISPATCH_FUSION_RULES,
    DISPATCH_NPU_FUSION_RULES,
    RMSNormFusion,
    ResidualAddRMSNormFusion,
    RoPEFusion,
    GroupedMatMulSwiGLUFusion,
    DispatchFlashAttentionFusion,
)
from xpu_simulator.core.operator import OpType
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B
from xpu_simulator.backends.npu.cost_model import NPUCostModel


class ManualRMSNorm(nn.Module):
    """RMSNorm that decomposes into pow/mean/rsqrt/mul at aten level."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # This decomposes into: pow -> mean -> add(eps) -> rsqrt -> mul -> mul(weight)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RMSNormModel(nn.Module):
    """Model using manual RMSNorm (decomposes into aten ops)."""
    def __init__(self, d=256):
        super().__init__()
        self.norm = ManualRMSNorm(d)
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return self.fc(self.norm(x))


class ResidualNormModel(nn.Module):
    """Model with residual add + LayerNorm."""
    def __init__(self, d=256):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        h = self.fc1(x)
        x = x + h  # residual
        return self.fc2(self.norm(x))


class AttentionModel(nn.Module):
    """Simple multi-head attention with explicit BMM+softmax+BMM pattern."""
    def __init__(self, d=256, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class SwiGLUModel(nn.Module):
    """Model with SwiGLU activation (gate + up + silu + mul)."""
    def __init__(self, d=256):
        super().__init__()
        self.gate = nn.Linear(d, d * 4, bias=False)
        self.up = nn.Linear(d, d * 4, bias=False)
        self.down = nn.Linear(d * 4, d, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class TestRMSNormFusion:
    def test_rmsnorm_detected(self):
        """RMSNorm decomposition (pow/mean/rsqrt/mul) gets fused."""
        model = RMSNormModel(256)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        fused, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        rms_fusions = [f for f in result.fusions_applied if "rms_norm" in f]
        assert len(rms_fusions) >= 1, f"Expected RMSNorm fusion, got: {result.fusions_applied}"

    def test_rmsnorm_reduces_nodes(self):
        """Fusing RMSNorm eliminates at least 3 intermediate nodes."""
        model = RMSNormModel(256)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        fused, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        assert result.nodes_eliminated >= 3

    def test_rmsnorm_produces_layer_norm_op(self):
        """Fused RMSNorm becomes a LAYER_NORM op."""
        model = RMSNormModel(256)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        fused, _ = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        layer_norms = [n for n in fused.topo_order() if n.op.op_type == OpType.LAYER_NORM]
        assert len(layer_norms) >= 1


class TestResidualAddRMSNormFusion:
    def test_residual_add_norm_detected(self):
        """Residual ADD + LayerNorm gets fused after RMSNorm fusion."""
        model = ResidualNormModel(256)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        fused, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        add_norm_fusions = [f for f in result.fusions_applied if "add_rms_norm" in f]
        assert len(add_norm_fusions) >= 1, f"Expected add_rms_norm fusion, got: {result.fusions_applied}"


class TestDispatchFlashAttentionFusion:
    def test_attention_pattern_fused(self):
        """BMM + softmax + BMM pattern gets fused into flash attention."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        fused, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        flash_fusions = [f for f in result.fusions_applied if "flash_attention" in f]
        assert len(flash_fusions) == 1, f"Expected 1 flash attention fusion, got {len(flash_fusions)}: {result.fusions_applied}"

    def test_flash_attention_eliminates_softmax(self):
        """After flash fusion, no standalone SOFTMAX node remains."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        # Before: should have SOFTMAX
        has_softmax = any(n.op.op_type == OpType.SOFTMAX for n in graph.topo_order())
        assert has_softmax, "Pre-condition: graph should have SOFTMAX"

        fused, _ = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        softmaxes = [n for n in fused.topo_order() if n.op.op_type == OpType.SOFTMAX]
        assert len(softmaxes) == 0, f"Expected no SOFTMAX after fusion, got {len(softmaxes)}"

    def test_flash_attention_reduces_matmuls(self):
        """Flash attention merges 2 MATMULs (QK + AV) into 1 fused op."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        before_matmuls = sum(1 for n in graph.topo_order() if n.op.op_type == OpType.MATMUL)
        fused, _ = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        after_matmuls = sum(1 for n in fused.topo_order() if n.op.op_type == OpType.MATMUL)
        # Should have 1 fewer MATMUL (2 attention matmuls -> 1 fused)
        assert after_matmuls < before_matmuls


class TestSwiGLUFusion:
    def test_swiglu_fused(self):
        """SwiGLU (silu + mul) gets fused via matmul epilogue or swiglu rule."""
        model = SwiGLUModel(256)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        fused, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        assert result.nodes_eliminated > 0
        # SILU should be fused into matmul epilogue
        silus = [n for n in fused.topo_order() if n.op.op_type == OpType.SILU]
        assert len(silus) == 0, f"Expected SILU to be fused away, got {len(silus)}"


class TestFusionPassIntegration:
    def test_full_transformer_fusion(self):
        """End-to-end: transformer block gets all applicable fusions."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 32, 256),))

        fused, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        assert result.nodes_eliminated > 0
        assert fused.num_nodes < graph.num_nodes

    def test_fused_graph_evaluates(self):
        """Fused dispatch graph can be evaluated by cost model."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 32, 256),))

        fused, _ = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        result = PerformanceEvaluator(GPUCostModel(A100_80GB)).run(fused, overlap=True)
        assert result.total_latency_us > 0
        assert result.total_flops > 0

    def test_npu_fused_graph_evaluates(self):
        """Fused dispatch graph evaluates correctly on NPU."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 32, 256),))

        fused, _ = FusionPass(DISPATCH_NPU_FUSION_RULES).apply(graph)
        result = PerformanceEvaluator(NPUCostModel(ASCEND_910B)).run(fused, overlap=True)
        assert result.total_latency_us > 0

    def test_npu_fusion_improves_latency(self):
        """Fusion should reduce NPU latency (fewer vector ops to schedule)."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 32, 256),))

        unfused_result = PerformanceEvaluator(NPUCostModel(ASCEND_910B)).run(graph, overlap=True)
        fused, _ = FusionPass(DISPATCH_NPU_FUSION_RULES).apply(graph)
        fused_result = PerformanceEvaluator(NPUCostModel(ASCEND_910B)).run(fused, overlap=True)
        assert fused_result.total_latency_us <= unfused_result.total_latency_us

    def test_dispatch_rules_dont_break_existing(self):
        """Dispatch fusion rules applied to a simple MLP don't crash."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(2, 128),))
        fused, result = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        assert fused.num_nodes <= graph.num_nodes

    def test_idempotent(self):
        """Applying fusion twice gives the same result as once."""
        model = AttentionModel(256, heads=4)
        ext = DispatchExtractor()
        graph = ext.extract(model, (torch.randn(1, 16, 256),))

        fused1, r1 = FusionPass(DISPATCH_FUSION_RULES).apply(graph)
        fused2, r2 = FusionPass(DISPATCH_FUSION_RULES).apply(fused1)
        assert fused2.num_nodes == fused1.num_nodes

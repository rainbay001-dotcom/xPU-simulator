"""Tests for quantization-aware cost modeling."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import (
    Dtype, QuantConfig, OpSpec, OpType, TensorSpec,
)
from xpu_simulator.frontend.graph_builder import GraphBuilder
from xpu_simulator.frontend.config_normalizer import normalize_config
from xpu_simulator.backends.gpu.hardware import H100_80GB, A100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B
from xpu_simulator.backends.npu.cost_model import NPUCostModel


# ------------------------------------------------------------------ #
# Dtype tests
# ------------------------------------------------------------------ #

def test_int4_dtype_bytes():
    """INT4 dtype should be 0.5 bytes per element."""
    assert Dtype.INT4.bytes == 0.5


def test_int4_tensor_size():
    """INT4 tensor size_bytes should be numel * 0.5."""
    t = TensorSpec((1024, 1024), Dtype.INT4)
    assert t.size_bytes == 1024 * 1024 // 2


# ------------------------------------------------------------------ #
# QuantConfig tests
# ------------------------------------------------------------------ #

def test_quant_config_creation():
    """Create various QuantConfig combinations."""
    w8a8 = QuantConfig(Dtype.INT8, Dtype.INT8)
    assert w8a8.weight_dtype == Dtype.INT8
    assert w8a8.activation_dtype == Dtype.INT8
    assert w8a8.group_size is None

    fp8 = QuantConfig(Dtype.FP8, Dtype.FP8)
    assert fp8.weight_dtype == Dtype.FP8

    w4a8 = QuantConfig(Dtype.INT4, Dtype.INT8, group_size=128)
    assert w4a8.weight_dtype == Dtype.INT4
    assert w4a8.group_size == 128


# ------------------------------------------------------------------ #
# GraphBuilder linear() tests
# ------------------------------------------------------------------ #

def test_linear_no_quant():
    """linear() without quant should produce same graph as matmul()."""
    gb = GraphBuilder("test")
    node = gb.linear("fc", 1024, 1024, 1024)
    g = gb.build()
    assert len(g.nodes) == 1
    assert g.nodes[0].op.op_type == OpType.MATMUL
    assert g.nodes[0].op.inputs[0].dtype == Dtype.FP16


def test_linear_w8a8():
    """W8A8 linear should produce matmul with INT8 typed tensors."""
    gb = GraphBuilder("test")
    quant = QuantConfig(Dtype.INT8, Dtype.INT8)
    node = gb.linear("fc", 1024, 1024, 1024, quant=quant)
    g = gb.build()

    assert len(g.nodes) == 1
    mm = g.nodes[0]
    assert mm.op.op_type == OpType.MATMUL
    # Activation input should be INT8
    assert mm.op.inputs[0].dtype == Dtype.INT8
    # Weight should be INT8
    assert mm.op.inputs[1].dtype == Dtype.INT8


def test_linear_fp8():
    """FP8 linear should produce matmul with FP8 typed tensors."""
    gb = GraphBuilder("test")
    quant = QuantConfig(Dtype.FP8, Dtype.FP8)
    node = gb.linear("fc", 1024, 1024, 1024, quant=quant)
    g = gb.build()

    assert len(g.nodes) == 1
    mm = g.nodes[0]
    assert mm.op.inputs[0].dtype == Dtype.FP8
    assert mm.op.inputs[1].dtype == Dtype.FP8


def test_linear_w4a8_has_dequant():
    """W4A8 linear should have a DEQUANT node before matmul."""
    gb = GraphBuilder("test")
    quant = QuantConfig(Dtype.INT4, Dtype.INT8, group_size=128)
    node = gb.linear("fc", 1024, 1024, 1024, quant=quant)
    g = gb.build()

    # Should have 2 nodes: dequant + matmul
    assert len(g.nodes) == 2
    dequant_nodes = [n for n in g.nodes if n.op.op_type == OpType.DEQUANT]
    matmul_nodes = [n for n in g.nodes if n.op.op_type == OpType.MATMUL]
    assert len(dequant_nodes) == 1
    assert len(matmul_nodes) == 1

    # Dequant: INT4 input -> INT8 output
    dq = dequant_nodes[0]
    assert dq.op.inputs[0].dtype == Dtype.INT4
    assert dq.op.outputs[0].dtype == Dtype.INT8

    # MatMul inputs should be INT8
    mm = matmul_nodes[0]
    assert mm.op.inputs[0].dtype == Dtype.INT8
    assert mm.op.inputs[1].dtype == Dtype.INT8


def test_builder_level_quant():
    """QuantConfig set on builder should apply to all linear() calls."""
    quant = QuantConfig(Dtype.INT8, Dtype.INT8)
    gb = GraphBuilder("test", quant=quant)
    gb.linear("fc1", 1024, 1024, 1024)
    gb.linear("fc2", 1024, 1024, 512)
    g = gb.build()

    for node in g.nodes:
        assert node.op.inputs[0].dtype == Dtype.INT8


def test_swiglu_with_quant():
    """SwiGLU MLP with quant should produce quantized linear ops."""
    quant = QuantConfig(Dtype.INT8, Dtype.INT8)
    gb = GraphBuilder("test", quant=quant)
    prev = gb.norm("norm", (1024, 4096))
    gb.swiglu_mlp("ffn", 1024, 4096, 11008, prev)
    g = gb.build()

    matmul_nodes = [n for n in g.nodes if n.op.op_type == OpType.MATMUL]
    assert len(matmul_nodes) == 3  # w1, w3, w2
    for mm in matmul_nodes:
        assert mm.op.inputs[0].dtype == Dtype.INT8, f"{mm.name}: {mm.op.inputs[0].dtype}"


# ------------------------------------------------------------------ #
# Cost model tests
# ------------------------------------------------------------------ #

def test_gpu_int8_peak():
    """INT8 matmul on H100 should be ~2x faster than FP16."""
    model = GPUCostModel(H100_80GB)

    def t(shape, dtype=Dtype.FP16):
        return TensorSpec(shape, dtype)

    op_fp16 = OpSpec(OpType.MATMUL,
                     [t((2048, 2048)), t((2048, 2048))],
                     [t((2048, 2048))], name="fp16")
    op_int8 = OpSpec(OpType.MATMUL,
                     [t((2048, 2048), Dtype.INT8), t((2048, 2048), Dtype.INT8)],
                     [t((2048, 2048))], name="int8")

    cost_fp16 = model.estimate(op_fp16)
    cost_int8 = model.estimate(op_int8)

    # INT8 should be faster (higher peak)
    print(f"  FP16: {cost_fp16.latency_us:.1f} us")
    print(f"  INT8: {cost_int8.latency_us:.1f} us")

    # INT8 peak is 2x FP16, so compute portion should be roughly half
    assert cost_int8.compute_us < cost_fp16.compute_us


def test_npu_fp8_peak():
    """FP8 matmul on 910B should use FP8 peak (same as INT8 = 2x FP16)."""
    model = NPUCostModel(ASCEND_910B)

    def t(shape, dtype=Dtype.FP16):
        return TensorSpec(shape, dtype)

    op_fp16 = OpSpec(OpType.MATMUL,
                     [t((2048, 2048)), t((2048, 2048))],
                     [t((2048, 2048))], name="fp16")
    op_fp8 = OpSpec(OpType.MATMUL,
                    [t((2048, 2048), Dtype.FP8), t((2048, 2048), Dtype.FP8)],
                    [t((2048, 2048), Dtype.FP8)], name="fp8")

    cost_fp16 = model.estimate(op_fp16)
    cost_fp8 = model.estimate(op_fp8)

    print(f"  FP16: {cost_fp16.latency_us:.1f} us")
    print(f"  FP8:  {cost_fp8.latency_us:.1f} us")

    # FP8 has 2x peak, so should be faster or equal
    assert cost_fp8.latency_us <= cost_fp16.latency_us


def test_dequant_flops():
    """DEQUANT op should have 2 * numel flops."""
    op = OpSpec(OpType.DEQUANT,
                [TensorSpec((1024, 1024), Dtype.INT4)],
                [TensorSpec((1024, 1024), Dtype.INT8)],
                name="dequant")
    assert op.flops == 2 * 1024 * 1024


# ------------------------------------------------------------------ #
# ConfigExtractor quantization parsing
# ------------------------------------------------------------------ #

def test_config_extractor_gptq():
    """ConfigExtractor should parse GPTQ quantization_config."""
    raw = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 2,
        "vocab_size": 32000,
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
        },
    }
    cfg = normalize_config(raw)
    assert cfg.quant_config is not None
    assert cfg.quant_config.weight_dtype == Dtype.INT4
    assert cfg.quant_config.activation_dtype == Dtype.INT8
    assert cfg.quant_config.group_size == 128


def test_config_extractor_fp8():
    """ConfigExtractor should parse FP8 quantization_config."""
    raw = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 2,
        "vocab_size": 32000,
        "quantization_config": {
            "quant_method": "fp8",
        },
    }
    cfg = normalize_config(raw)
    assert cfg.quant_config is not None
    assert cfg.quant_config.weight_dtype == Dtype.FP8
    assert cfg.quant_config.activation_dtype == Dtype.FP8


def test_config_no_quant():
    """Config without quantization should have quant_config=None."""
    raw = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 2,
        "vocab_size": 32000,
    }
    cfg = normalize_config(raw)
    assert cfg.quant_config is None


if __name__ == "__main__":
    print("=== Quantization Tests ===\n")

    for name, fn in [
        ("INT4 dtype bytes", test_int4_dtype_bytes),
        ("INT4 tensor size", test_int4_tensor_size),
        ("QuantConfig creation", test_quant_config_creation),
        ("Linear no quant", test_linear_no_quant),
        ("Linear W8A8", test_linear_w8a8),
        ("Linear FP8", test_linear_fp8),
        ("Linear W4A8 dequant", test_linear_w4a8_has_dequant),
        ("Builder-level quant", test_builder_level_quant),
        ("SwiGLU with quant", test_swiglu_with_quant),
        ("GPU INT8 peak", test_gpu_int8_peak),
        ("NPU FP8 peak", test_npu_fp8_peak),
        ("DEQUANT flops", test_dequant_flops),
        ("Config GPTQ", test_config_extractor_gptq),
        ("Config FP8", test_config_extractor_fp8),
        ("Config no quant", test_config_no_quant),
    ]:
        print(f"--- {name} ---")
        fn()
        print()

    print("All quantization tests passed!")

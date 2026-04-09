"""Tests for profiling database and calibrated cost model."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from xpu_simulator.core.operator import Dtype, OpType, OpSpec, TensorSpec
from xpu_simulator.core.profiling_db import ProfilingDB, shape_key
from xpu_simulator.core.cost_model import CalibratedCostModel
from xpu_simulator.backends.gpu.hardware import H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


def _make_matmul(M, K, N, dtype=Dtype.FP16):
    """Helper to build a matmul OpSpec."""
    return OpSpec(
        op_type=OpType.MATMUL,
        inputs=[TensorSpec((M, K), dtype), TensorSpec((K, N), dtype)],
        outputs=[TensorSpec((M, N), dtype)],
    )


def _make_layernorm(B, D, dtype=Dtype.FP16):
    return OpSpec(
        op_type=OpType.LAYER_NORM,
        inputs=[TensorSpec((B, D), dtype)],
        outputs=[TensorSpec((B, D), dtype)],
    )


# ------------------------------------------------------------------ #
# shape_key tests
# ------------------------------------------------------------------ #

def test_shape_key_matmul():
    """Matmul shape key should be matmul_MxKxN_dtype."""
    op = _make_matmul(2048, 2048, 2048)
    key = shape_key(op)
    assert key == "matmul_2048x2048x2048_fp16"


def test_shape_key_layernorm():
    """Non-matmul ops use first input shape."""
    op = _make_layernorm(1, 1024)
    key = shape_key(op)
    assert key == "layer_norm_1x1024_fp16"


def test_shape_key_dtype():
    """Key includes dtype."""
    op = _make_matmul(512, 512, 512, Dtype.INT8)
    key = shape_key(op)
    assert "int8" in key


# ------------------------------------------------------------------ #
# ProfilingDB tests
# ------------------------------------------------------------------ #

def test_store_and_lookup():
    """Round-trip store and lookup."""
    db = ProfilingDB()
    op = _make_matmul(1024, 1024, 1024)
    db.store(op, 42.5)
    assert db.lookup(op) == 42.5
    assert len(db) == 1


def test_lookup_miss():
    """Lookup on unknown op returns None."""
    db = ProfilingDB()
    op = _make_matmul(1024, 1024, 1024)
    assert db.lookup(op) is None


def test_contains():
    """__contains__ works."""
    db = ProfilingDB()
    op = _make_matmul(256, 256, 256)
    assert op not in db
    db.store(op, 1.0)
    assert op in db


def test_save_load_csv():
    """Save and load round-trip via CSV."""
    db = ProfilingDB()
    op1 = _make_matmul(1024, 1024, 1024)
    op2 = _make_matmul(2048, 4096, 2048)
    db.store(op1, 10.0)
    db.store(op2, 50.0)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    db.save(path)

    db2 = ProfilingDB()
    db2.load(path)
    assert db2.lookup(op1) == 10.0
    assert db2.lookup(op2) == 50.0
    assert len(db2) == 2

    Path(path).unlink()


# ------------------------------------------------------------------ #
# CalibratedCostModel tests
# ------------------------------------------------------------------ #

def test_calibrated_hit():
    """On hit, CalibratedCostModel uses measured latency."""
    base = GPUCostModel(H100_80GB)
    db = ProfilingDB()
    op = _make_matmul(2048, 2048, 2048)
    db.store(op, 99.9)

    cal = CalibratedCostModel(base, db)
    cost = cal.estimate(op)
    assert cost.latency_us == 99.9


def test_calibrated_miss():
    """On miss, CalibratedCostModel falls back to analytical."""
    base = GPUCostModel(H100_80GB)
    db = ProfilingDB()
    op = _make_matmul(2048, 2048, 2048)

    cal = CalibratedCostModel(base, db)
    cost_cal = cal.estimate(op)
    cost_base = base.estimate(op)
    assert cost_cal.latency_us == cost_base.latency_us


def test_calibrated_hit_rate():
    """Hit rate tracks correctly."""
    base = GPUCostModel(H100_80GB)
    db = ProfilingDB()
    op_known = _make_matmul(1024, 1024, 1024)
    op_unknown = _make_matmul(2048, 2048, 2048)
    db.store(op_known, 5.0)

    cal = CalibratedCostModel(base, db)
    cal.estimate(op_known)   # hit
    cal.estimate(op_unknown)  # miss
    cal.estimate(op_known)   # hit

    assert cal._hits == 2
    assert cal._misses == 1
    assert abs(cal.hit_rate - 2 / 3) < 1e-9
    assert cal.total_queries == 3


def test_empty_db_transparent():
    """Empty DB makes calibrated model identical to base."""
    base = GPUCostModel(H100_80GB)
    db = ProfilingDB()
    cal = CalibratedCostModel(base, db)

    op = _make_matmul(512, 512, 512)
    cost_cal = cal.estimate(op)
    cost_base = base.estimate(op)
    assert cost_cal.latency_us == cost_base.latency_us
    assert cal.hit_rate == 0.0


if __name__ == "__main__":
    print("=== Profiling DB Tests ===\n")

    for name, fn in [
        ("Shape key matmul", test_shape_key_matmul),
        ("Shape key layernorm", test_shape_key_layernorm),
        ("Shape key dtype", test_shape_key_dtype),
        ("Store and lookup", test_store_and_lookup),
        ("Lookup miss", test_lookup_miss),
        ("Contains", test_contains),
        ("Save/load CSV", test_save_load_csv),
        ("Calibrated hit", test_calibrated_hit),
        ("Calibrated miss", test_calibrated_miss),
        ("Calibrated hit rate", test_calibrated_hit_rate),
        ("Empty DB transparent", test_empty_db_transparent),
    ]:
        print(f"--- {name} ---")
        fn()
        print()

    print("All profiling DB tests passed!")

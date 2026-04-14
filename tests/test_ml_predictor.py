"""Tests for ML-based execution time predictor."""
import sys
sys.path.insert(0, ".")

import pytest
from xpu_simulator.core.operator import OpType, Dtype, TensorSpec, OpSpec
from xpu_simulator.backends.gpu.hardware import H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def test_import_and_fallback():
    """MLExecutionTimePredictor should import and fall back to base model."""
    from xpu_simulator.backends.ml_predictor import MLExecutionTimePredictor

    base = GPUCostModel(H100_80GB)
    predictor = MLExecutionTimePredictor(base)

    # Without training, should fall back to base
    mm = OpSpec(OpType.MATMUL,
                [TensorSpec((1024, 1024)), TensorSpec((1024, 1024))],
                [TensorSpec((1024, 1024))])
    cost = predictor.estimate(mm)
    base_cost = base.estimate(mm)
    assert abs(cost.latency_us - base_cost.latency_us) < 0.01
    assert predictor.hit_rate == 0.0
    print(f"  Fallback: {cost.latency_us:.2f} us (same as base: {base_cost.latency_us:.2f} us)")


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_train_and_predict():
    """Train on synthetic matmul data, verify predictions are reasonable."""
    from xpu_simulator.backends.ml_predictor import MLExecutionTimePredictor

    base = GPUCostModel(H100_80GB)
    predictor = MLExecutionTimePredictor(base)

    # Generate synthetic training data: larger matmuls take longer
    samples = []
    for size in [128, 256, 512, 1024, 2048, 4096]:
        op = OpSpec(OpType.MATMUL,
                    [TensorSpec((size, size)), TensorSpec((size, size))],
                    [TensorSpec((size, size))])
        # Synthetic "measured" latency: scales with size^3 / peak_flops
        latency = 2 * size**3 / (989e12) * 1e6  # H100 FP16 peak
        samples.append((op, latency))

    scores = predictor.train(samples, cv=2)
    assert "matmul" in scores
    print(f"  Train R^2: {scores}")

    # Predict for a new size
    test_op = OpSpec(OpType.MATMUL,
                     [TensorSpec((768, 768)), TensorSpec((768, 768))],
                     [TensorSpec((768, 768))])
    cost = predictor.estimate(test_op)
    assert cost.latency_us > 0
    assert predictor.hit_rate > 0
    print(f"  Predicted 768x768: {cost.latency_us:.4f} us, hit_rate: {predictor.hit_rate:.1%}")


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_categories_trained():
    """Only trained categories should be listed."""
    from xpu_simulator.backends.ml_predictor import MLExecutionTimePredictor

    base = GPUCostModel(H100_80GB)
    predictor = MLExecutionTimePredictor(base)
    assert predictor.categories_trained == []

    samples = []
    for size in [128, 256, 512, 1024]:
        op = OpSpec(OpType.MATMUL,
                    [TensorSpec((size, size)), TensorSpec((size, size))],
                    [TensorSpec((size, size))])
        samples.append((op, float(size)))

    predictor.train(samples, cv=2)
    assert "matmul" in predictor.categories_trained
    assert "elementwise" not in predictor.categories_trained
    print(f"  Trained categories: {predictor.categories_trained}")


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_save_load_roundtrip(tmp_path):
    """Save and reload models, verify identical predictions."""
    from xpu_simulator.backends.ml_predictor import MLExecutionTimePredictor

    base = GPUCostModel(H100_80GB)
    predictor = MLExecutionTimePredictor(base)

    samples = []
    for size in [128, 256, 512, 1024, 2048]:
        op = OpSpec(OpType.MATMUL,
                    [TensorSpec((size, size)), TensorSpec((size, size))],
                    [TensorSpec((size, size))])
        samples.append((op, float(size * size)))

    predictor.train(samples, cv=2)

    test_op = OpSpec(OpType.MATMUL,
                     [TensorSpec((384, 384)), TensorSpec((384, 384))],
                     [TensorSpec((384, 384))])
    cost_before = predictor.estimate(test_op)

    # Save and reload
    model_path = tmp_path / "test_models.joblib"
    predictor.save(model_path)

    predictor2 = MLExecutionTimePredictor(base)
    predictor2.load(model_path)
    cost_after = predictor2.estimate(test_op)

    assert abs(cost_before.latency_us - cost_after.latency_us) < 0.001
    print(f"  Before: {cost_before.latency_us:.4f}, After: {cost_after.latency_us:.4f}")


def test_elementwise_fallback():
    """Elementwise ops without training should fall back to base model."""
    from xpu_simulator.backends.ml_predictor import MLExecutionTimePredictor

    base = GPUCostModel(H100_80GB)
    predictor = MLExecutionTimePredictor(base)

    relu = OpSpec(OpType.RELU,
                  [TensorSpec((1024, 4096))],
                  [TensorSpec((1024, 4096))])
    cost = predictor.estimate(relu)
    base_cost = base.estimate(relu)
    assert abs(cost.latency_us - base_cost.latency_us) < 0.01
    print(f"  ReLU fallback: {cost.latency_us:.4f} us")


if __name__ == "__main__":
    print("=== ML Predictor Tests ===\n")
    for name, fn in [
        ("Import and fallback", test_import_and_fallback),
        ("Elementwise fallback", test_elementwise_fallback),
    ]:
        print(f"--- {name} ---")
        fn()
        print()

    if HAS_SKLEARN:
        import tempfile, pathlib
        for name, fn in [
            ("Train and predict", test_train_and_predict),
            ("Categories trained", test_categories_trained),
        ]:
            print(f"--- {name} ---")
            fn()
            print()

        print(f"--- Save/load roundtrip ---")
        with tempfile.TemporaryDirectory() as td:
            test_save_load_roundtrip(pathlib.Path(td))
        print()
    else:
        print("(Skipping sklearn-dependent tests — scikit-learn not installed)\n")

    print("All ML predictor tests passed!")

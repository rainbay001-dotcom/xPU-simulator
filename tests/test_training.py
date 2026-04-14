"""Tests for training iteration modeling."""
import sys
sys.path.insert(0, ".")

from xpu_simulator.core.training import (
    TrainingConfig, TrainingPhase, LayerTiming, TrainingResult,
    extract_layer_timings, estimate_training_iteration,
)
from xpu_simulator.core.parallel import ParallelConfig


def test_single_layer_no_overlap():
    """Single layer, no WG overlap — total = fwd + IG + WG + comm."""
    timings = [LayerTiming(
        layer_id=0,
        fwd_compute_us=100.0,
        input_grad_compute_us=100.0,  # ratio=2 → 200 total bwd, split 100/100
        weight_grad_compute_us=100.0,
        fwd_comm_us=10.0,
        weight_grad_comm_us=50.0,
    )]
    config = TrainingConfig(micro_batches=1, weight_grad_overlap=False)
    parallel = ParallelConfig()

    result = estimate_training_iteration(timings, parallel, config)
    # fwd = 100 + 10 = 110
    # bwd = 100 + 100 = 200
    # comm = 50 (no overlap)
    # bubble = 0 (pp_size=1)
    assert abs(result.fwd_us - 110.0) < 0.1
    assert abs(result.bwd_us - 200.0) < 0.1
    assert abs(result.comm_us - 50.0) < 0.1
    assert abs(result.bubble_us) < 0.1
    expected_total = 110 + 200 + 50
    assert abs(result.total_iteration_us - expected_total) < 0.1
    print(f"  Single layer no overlap: {result.total_iteration_us:.1f} us")


def test_weight_grad_overlap():
    """3 layers, WG comm of layer i overlaps with IG compute of layer i-1."""
    timings = [
        LayerTiming(0, fwd_compute_us=100, input_grad_compute_us=100,
                    weight_grad_compute_us=100, fwd_comm_us=0, weight_grad_comm_us=80),
        LayerTiming(1, fwd_compute_us=100, input_grad_compute_us=100,
                    weight_grad_compute_us=100, fwd_comm_us=0, weight_grad_comm_us=80),
        LayerTiming(2, fwd_compute_us=100, input_grad_compute_us=100,
                    weight_grad_compute_us=100, fwd_comm_us=0, weight_grad_comm_us=80),
    ]
    config = TrainingConfig(micro_batches=1, weight_grad_overlap=True)
    parallel = ParallelConfig()

    result = estimate_training_iteration(timings, parallel, config)
    # Backward (reversed: layer 2, 1, 0):
    # Layer 2: WG comm=80, overlaps with layer 1's IG=100 → exposed=0
    # Layer 1: WG comm=80, overlaps with layer 0's IG=100 → exposed=0
    # Layer 0: WG comm=80, no next layer → exposed=80
    assert abs(result.comm_us - 80.0) < 0.1, f"Expected 80, got {result.comm_us}"

    # Without overlap
    config_no = TrainingConfig(micro_batches=1, weight_grad_overlap=False)
    result_no = estimate_training_iteration(timings, parallel, config_no)
    assert abs(result_no.comm_us - 240.0) < 0.1  # 3 * 80

    print(f"  Overlap: {result.comm_us:.1f} us exposed, No overlap: {result_no.comm_us:.1f} us")
    print(f"  Overlap efficiency: {result.overlap_efficiency:.1%}")


def test_pipeline_bubble():
    """PP=4 should add bubble overhead."""
    timings = [LayerTiming(
        layer_id=0, fwd_compute_us=100, input_grad_compute_us=100,
        weight_grad_compute_us=100, fwd_comm_us=0, weight_grad_comm_us=0,
    )]
    config = TrainingConfig(micro_batches=1)
    parallel = ParallelConfig(pp_size=4, gradient_accumulation=1)

    result = estimate_training_iteration(timings, parallel, config)
    # bubble = (100 + 200) * (4-1)/(1*1) = 300 * 3 = 900
    # But bubble_fraction is clamped to 1.0, so: 300 * 1.0 = 300
    assert parallel.bubble_fraction == 1.0  # (4-1)/(1*1) = 3.0 → clamped to 1.0
    assert abs(result.bubble_us - 300.0) < 0.1
    print(f"  PP=4: bubble={result.bubble_us:.1f} us, total={result.total_iteration_us:.1f} us")

    # With gradient accumulation = 8
    parallel_ga = ParallelConfig(pp_size=4, gradient_accumulation=8)
    result_ga = estimate_training_iteration(timings, parallel_ga, config)
    # bubble_fraction = (4-1)/(8*1) = 0.375
    assert abs(parallel_ga.bubble_fraction - 0.375) < 0.01
    expected_bubble = 300 * 0.375
    assert abs(result_ga.bubble_us - expected_bubble) < 0.1
    print(f"  PP=4 GA=8: bubble={result_ga.bubble_us:.1f} us ({parallel_ga.bubble_fraction:.1%})")


def test_micro_batches():
    """Multiple micro-batches multiply the total."""
    timings = [LayerTiming(
        layer_id=0, fwd_compute_us=100, input_grad_compute_us=100,
        weight_grad_compute_us=100, fwd_comm_us=0, weight_grad_comm_us=0,
    )]
    config_1 = TrainingConfig(micro_batches=1)
    config_4 = TrainingConfig(micro_batches=4)
    parallel = ParallelConfig()

    r1 = estimate_training_iteration(timings, parallel, config_1)
    r4 = estimate_training_iteration(timings, parallel, config_4)
    assert abs(r4.total_iteration_us - r1.total_iteration_us * 4) < 0.1
    print(f"  1 micro-batch: {r1.total_iteration_us:.1f} us, 4: {r4.total_iteration_us:.1f} us")


def test_backward_ratio():
    """backward_compute_ratio=3.0 should scale backward times."""
    timings_r2 = [LayerTiming(0, 100, 100, 100, 0, 0)]  # ratio=2: IG=100, WG=100
    timings_r3 = [LayerTiming(0, 100, 150, 150, 0, 0)]  # ratio=3: IG=150, WG=150
    config = TrainingConfig(micro_batches=1)
    parallel = ParallelConfig()

    r2 = estimate_training_iteration(timings_r2, parallel, config)
    r3 = estimate_training_iteration(timings_r3, parallel, config)
    assert r3.bwd_us > r2.bwd_us
    print(f"  ratio=2: bwd={r2.bwd_us:.1f}, ratio=3: bwd={r3.bwd_us:.1f}")


def test_summary():
    """TrainingResult.summary() should produce readable output."""
    timings = [LayerTiming(0, 100, 100, 100, 10, 50)]
    config = TrainingConfig(micro_batches=2)
    parallel = ParallelConfig()
    result = estimate_training_iteration(timings, parallel, config)
    s = result.summary()
    assert "Training iteration" in s
    assert "Forward" in s
    print(f"  Summary:\n{s}")


if __name__ == "__main__":
    print("=== Training Tests ===\n")
    for name, fn in [
        ("Single layer no overlap", test_single_layer_no_overlap),
        ("Weight grad overlap", test_weight_grad_overlap),
        ("Pipeline bubble", test_pipeline_bubble),
        ("Micro-batches", test_micro_batches),
        ("Backward ratio", test_backward_ratio),
        ("Summary", test_summary),
    ]:
        print(f"--- {name} ---")
        fn()
        print()
    print("All training tests passed!")

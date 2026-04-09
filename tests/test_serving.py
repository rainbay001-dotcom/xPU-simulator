"""Tests for serving-level simulation."""

import sys
sys.path.insert(0, ".")

from xpu_simulator.serving.config import ServingConfig
from xpu_simulator.serving.request import Request, RequestState
from xpu_simulator.serving.kv_cache import KVCacheAllocator
from xpu_simulator.serving.scheduler import BatchScheduler
from xpu_simulator.serving.metrics import ServingMetrics
from xpu_simulator.serving.simulator import ServingSimulator, find_max_throughput
from xpu_simulator.backends.gpu.hardware import H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel


LLAMA_SMALL = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "intermediate_size": 11008,
    "vocab_size": 32000,
}


# ------------------------------------------------------------------ #
# KV cache allocator tests
# ------------------------------------------------------------------ #

def test_kv_cache_allocator_basic():
    """Allocate and free blocks."""
    alloc = KVCacheAllocator(num_blocks=100, block_size=16)
    assert alloc.free_blocks == 100

    assert alloc.allocate(0, 10) is True
    assert alloc.free_blocks == 90
    assert alloc.blocks_for(0) == 10

    # Grow allocation
    assert alloc.allocate(0, 15) is True
    assert alloc.free_blocks == 85
    assert alloc.blocks_for(0) == 15

    freed = alloc.free(0)
    assert freed == 15
    assert alloc.free_blocks == 100


def test_kv_cache_allocator_overflow():
    """Allocation fails when not enough blocks."""
    alloc = KVCacheAllocator(num_blocks=10, block_size=16)
    assert alloc.allocate(0, 8) is True
    assert alloc.allocate(1, 5) is False  # Only 2 free
    assert alloc.free_blocks == 2


# ------------------------------------------------------------------ #
# Request state machine tests
# ------------------------------------------------------------------ #

def test_request_state_machine():
    """Request state transitions."""
    req = Request(id=0, prompt_len=128, output_len=10)
    assert req.state == RequestState.WAITING

    req.state = RequestState.PREFILLING
    assert req.state == RequestState.PREFILLING

    req.state = RequestState.DECODING
    assert req.state == RequestState.DECODING

    req.state = RequestState.DONE
    assert req.state == RequestState.DONE


def test_request_ttft_tpot():
    """Request timing metrics."""
    req = Request(id=0, prompt_len=128, output_len=10, arrival_time_us=0.0)
    req.first_token_us = 5000.0
    req.decode_latencies_us = [100.0, 110.0, 90.0]
    req.finish_us = 5300.0

    assert req.ttft_us == 5000.0
    assert req.ttft_ms == 5.0
    assert abs(req.avg_tpot_us - 100.0) < 1e-6
    assert abs(req.avg_tpot_ms - 0.1) < 1e-6
    assert req.total_time_us == 5300.0


def test_request_blocks_needed():
    """Block calculation rounds up."""
    req = Request(id=0, prompt_len=100, output_len=10)
    req.generated_tokens = 5
    # 100 + 5 = 105 tokens, block_size=16 -> ceil(105/16) = 7
    assert req.blocks_needed(16) == 7


# ------------------------------------------------------------------ #
# Scheduler tests
# ------------------------------------------------------------------ #

def test_scheduler_continuous_batching():
    """Scheduler admits new prefills while decoding."""
    cfg = ServingConfig(max_batch_size=8, max_tokens_budget=2048,
                        block_size=16, num_kv_blocks=1000)
    alloc = KVCacheAllocator(cfg.num_kv_blocks, cfg.block_size)
    sched = BatchScheduler(cfg, alloc)

    waiting = [Request(id=i, prompt_len=128, output_len=10) for i in range(3)]
    decoding = [Request(id=10, prompt_len=128, output_len=10)]
    decoding[0].state = RequestState.DECODING

    batch = sched.schedule(waiting, decoding)
    # Should have 1 decode + some prefills
    assert len(batch.decode_requests) == 1
    assert len(batch.prefill_requests) > 0
    assert batch.total_sequences <= cfg.max_batch_size


def test_scheduler_budget_limit():
    """Scheduler respects max_tokens_budget."""
    cfg = ServingConfig(max_batch_size=100, max_tokens_budget=256,
                        block_size=16, num_kv_blocks=1000)
    alloc = KVCacheAllocator(cfg.num_kv_blocks, cfg.block_size)
    sched = BatchScheduler(cfg, alloc)

    # 3 requests with prompt_len=128, budget=256 -> can fit 2
    waiting = [Request(id=i, prompt_len=128, output_len=10) for i in range(3)]

    batch = sched.schedule(waiting, [])
    assert len(batch.prefill_requests) == 2
    assert batch.total_tokens == 256


# ------------------------------------------------------------------ #
# Metrics tests
# ------------------------------------------------------------------ #

def test_serving_metrics_percentiles():
    """ServingMetrics computes percentiles correctly."""
    reqs = []
    for i in range(100):
        r = Request(id=i, prompt_len=128, output_len=10, arrival_time_us=0.0)
        r.first_token_us = float(1000 + i * 10)  # 1000 to 1990
        r.generated_tokens = 10
        r.decode_latencies_us = [float(50 + i)]  # 50 to 149
        r.finish_us = float(2000 + i * 10)
        r.state = RequestState.DONE
        reqs.append(r)

    metrics = ServingMetrics(requests=reqs, total_time_us=3000.0)
    assert metrics.num_requests == 100
    assert metrics.total_generated_tokens == 1000
    assert metrics.throughput_tok_per_s > 0
    assert metrics.avg_ttft_ms > 0
    assert metrics.p50_ttft_ms > 0
    assert metrics.p99_ttft_ms >= metrics.p50_ttft_ms


# ------------------------------------------------------------------ #
# Simulator tests
# ------------------------------------------------------------------ #

def test_simulator_single_request():
    """Single request should produce valid metrics."""
    model = GPUCostModel(H100_80GB)
    cfg = ServingConfig(max_batch_size=1, max_seq_len=4096,
                        max_tokens_budget=4096, block_size=16,
                        num_kv_blocks=1000)

    sim = ServingSimulator(
        model_config=LLAMA_SMALL,
        cost_model=model,
        serving_config=cfg,
    )

    requests = [Request(id=0, prompt_len=64, output_len=5)]
    metrics = sim.run(requests)

    assert metrics.num_requests == 1
    assert metrics.total_generated_tokens == 5
    assert metrics.avg_ttft_ms > 0
    assert metrics.avg_tpot_ms > 0
    assert metrics.total_time_us > 0
    print(f"  Single request: {metrics.summary()}")


def test_simulator_multiple_requests():
    """Multiple requests with continuous batching."""
    model = GPUCostModel(H100_80GB)
    cfg = ServingConfig(max_batch_size=4, max_seq_len=4096,
                        max_tokens_budget=4096, block_size=16,
                        num_kv_blocks=1000)

    sim = ServingSimulator(
        model_config=LLAMA_SMALL,
        cost_model=model,
        serving_config=cfg,
    )

    requests = [
        Request(id=0, prompt_len=64, output_len=3, arrival_time_us=0),
        Request(id=1, prompt_len=32, output_len=3, arrival_time_us=0),
    ]
    metrics = sim.run(requests)

    assert metrics.num_requests == 2
    assert metrics.total_generated_tokens == 6
    assert metrics.throughput_tok_per_s > 0
    print(f"  Multi request: {metrics.summary()}")


def test_throughput_optimizer():
    """find_max_throughput returns a valid batch size."""
    model = GPUCostModel(H100_80GB)
    cfg = ServingConfig(max_seq_len=4096, max_tokens_budget=4096,
                        block_size=16, num_kv_blocks=1000)

    requests = [
        Request(id=i, prompt_len=64, output_len=3, arrival_time_us=0)
        for i in range(4)
    ]

    best = find_max_throughput(
        model_config=LLAMA_SMALL,
        cost_model=model,
        serving_config=cfg,
        requests=requests,
        sla_tpot_ms=10.0,  # generous SLA
        min_batch=1,
        max_batch=8,
    )

    assert 1 <= best <= 8
    print(f"  Max throughput batch size: {best}")


if __name__ == "__main__":
    print("=== Serving Tests ===\n")

    for name, fn in [
        ("KV cache allocator basic", test_kv_cache_allocator_basic),
        ("KV cache allocator overflow", test_kv_cache_allocator_overflow),
        ("Request state machine", test_request_state_machine),
        ("Request TTFT/TPOT", test_request_ttft_tpot),
        ("Request blocks needed", test_request_blocks_needed),
        ("Scheduler continuous batching", test_scheduler_continuous_batching),
        ("Scheduler budget limit", test_scheduler_budget_limit),
        ("Serving metrics percentiles", test_serving_metrics_percentiles),
        ("Simulator single request", test_simulator_single_request),
        ("Simulator multiple requests", test_simulator_multiple_requests),
        ("Throughput optimizer", test_throughput_optimizer),
    ]:
        print(f"--- {name} ---")
        fn()
        print()

    print("All serving tests passed!")

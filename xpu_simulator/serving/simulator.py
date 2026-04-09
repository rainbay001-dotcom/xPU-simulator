"""Serving simulator — end-to-end inference serving with continuous batching."""
from __future__ import annotations

from typing import Optional

from ..core.operator import Dtype
from ..core.cost_model import CostModel
from ..core.evaluator import PerformanceEvaluator
from ..core.parallel import ParallelConfig
from ..frontend.config_extractor import ConfigExtractor

from .config import ServingConfig
from .request import Request, RequestState
from .kv_cache import KVCacheAllocator
from .scheduler import BatchScheduler, ScheduledBatch
from .metrics import ServingMetrics


class ServingSimulator:
    """Simulates serving a stream of requests with continuous batching.

    Main loop per iteration:
    1. Schedule: pick prefill + decode requests for this batch.
    2. Estimate prefill latency (if any prefill requests).
    3. Estimate decode latency (if any decode requests).
    4. Advance time by max(prefill, decode) — they run in the same iteration.
    5. Update request states, collect metrics.
    """

    def __init__(
        self,
        model_config: dict,
        cost_model: CostModel,
        serving_config: ServingConfig,
        dtype: Dtype = Dtype.FP16,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        self.model_config = model_config
        self.cost_model = cost_model
        self.serving_config = serving_config
        self.dtype = dtype
        self.parallel_config = parallel_config

        self._extractor = ConfigExtractor(dtype=dtype)
        self._evaluator = PerformanceEvaluator(cost_model)

    def run(self, requests: list[Request]) -> ServingMetrics:
        """Run serving simulation on a list of requests.

        Returns ServingMetrics with per-request and aggregate stats.
        """
        cfg = self.serving_config
        kv_alloc = KVCacheAllocator(cfg.num_kv_blocks, cfg.block_size)
        scheduler = BatchScheduler(cfg, kv_alloc)

        waiting = list(requests)
        decoding: list[Request] = []
        done: list[Request] = []
        time_us = 0.0

        max_iters = len(requests) * (cfg.max_seq_len + 1)  # safety bound
        for _ in range(max_iters):
            if not waiting and not decoding:
                break

            # Filter waiting to only those that have arrived
            available = [r for r in waiting if r.arrival_time_us <= time_us]

            batch = scheduler.schedule(available, decoding)
            if batch.is_empty:
                # Fast-forward to next arrival
                future = [r for r in waiting if r.arrival_time_us > time_us]
                if future:
                    time_us = min(r.arrival_time_us for r in future)
                    continue
                break

            iter_latency = 0.0

            # Prefill
            if batch.prefill_requests:
                prefill_lat = self._estimate_prefill(batch)
                iter_latency = max(iter_latency, prefill_lat)

                for req in batch.prefill_requests:
                    req.state = RequestState.PREFILLING
                    req.prefill_start_us = time_us
                    req.prefill_end_us = time_us + prefill_lat
                    req.first_token_us = time_us + prefill_lat
                    req.kv_len = req.prompt_len
                    req.generated_tokens = 1  # First token from prefill
                    waiting.remove(req)

            # Decode
            if batch.decode_requests:
                decode_lat = self._estimate_decode(batch)
                iter_latency = max(iter_latency, decode_lat)

                for req in batch.decode_requests:
                    req.generated_tokens += 1
                    req.kv_len += 1
                    req.decode_latencies_us.append(decode_lat)

                    # Update KV cache allocation
                    blocks_needed = req.blocks_needed(cfg.block_size)
                    kv_alloc.allocate(req.id, blocks_needed)

            time_us += iter_latency

            # Transition prefilled requests to decoding
            for req in batch.prefill_requests:
                if req.generated_tokens < req.output_len:
                    req.state = RequestState.DECODING
                    decoding.append(req)
                else:
                    req.state = RequestState.DONE
                    req.finish_us = time_us
                    kv_alloc.free(req.id)
                    done.append(req)

            # Check decode completion
            finished = []
            for req in batch.decode_requests:
                if req.generated_tokens >= req.output_len:
                    req.state = RequestState.DONE
                    req.finish_us = time_us
                    kv_alloc.free(req.id)
                    done.append(req)
                    finished.append(req)

            for req in finished:
                decoding.remove(req)

        return ServingMetrics(requests=done, total_time_us=time_us)

    def _estimate_prefill(self, batch: ScheduledBatch) -> float:
        """Estimate prefill latency for batch."""
        # Use total prompt tokens as batch
        total_tokens = sum(r.prompt_len for r in batch.prefill_requests)
        batch_size = len(batch.prefill_requests)
        avg_seq = total_tokens // batch_size if batch_size > 0 else 1

        graph = self._extractor.extract(
            self.model_config,
            batch_size=batch_size,
            seq_len=avg_seq,
            phase="prefill",
            parallel_config=self.parallel_config,
        )
        result = self._evaluator.run(graph, overlap=True)
        return result.total_latency_us

    def _estimate_decode(self, batch: ScheduledBatch) -> float:
        """Estimate decode latency for batch."""
        batch_size = len(batch.decode_requests)
        # Use max KV length for conservative estimate
        max_kv = max(r.kv_len for r in batch.decode_requests)

        graph = self._extractor.extract(
            self.model_config,
            batch_size=batch_size,
            seq_len=1,
            phase="decode",
            kv_seq_len=max_kv,
            parallel_config=self.parallel_config,
        )
        result = self._evaluator.run(graph, overlap=True)
        return result.total_latency_us


def find_max_throughput(
    model_config: dict,
    cost_model: CostModel,
    serving_config: ServingConfig,
    requests: list[Request],
    sla_tpot_ms: float,
    min_batch: int = 1,
    max_batch: int = 128,
    dtype: Dtype = Dtype.FP16,
    parallel_config: Optional[ParallelConfig] = None,
) -> int:
    """Binary search for max batch size that meets TPOT SLA.

    Returns the largest max_batch_size where p99 TPOT <= sla_tpot_ms.
    """
    best = min_batch

    while min_batch <= max_batch:
        mid = (min_batch + max_batch) // 2
        cfg = ServingConfig(
            max_batch_size=mid,
            max_seq_len=serving_config.max_seq_len,
            max_tokens_budget=serving_config.max_tokens_budget,
            block_size=serving_config.block_size,
            num_kv_blocks=serving_config.num_kv_blocks,
        )
        sim = ServingSimulator(
            model_config=model_config,
            cost_model=cost_model,
            serving_config=cfg,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        # Clone requests for fresh simulation
        cloned = [
            Request(id=r.id, prompt_len=r.prompt_len, output_len=r.output_len,
                    arrival_time_us=r.arrival_time_us)
            for r in requests
        ]
        metrics = sim.run(cloned)

        if metrics.p99_tpot_ms <= sla_tpot_ms:
            best = mid
            min_batch = mid + 1
        else:
            max_batch = mid - 1

    return best

"""Serving metrics — TTFT, TPOT, throughput."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .request import Request


@dataclass
class ServingMetrics:
    """Aggregated serving metrics from a simulation run."""
    requests: List[Request]
    total_time_us: float

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def total_generated_tokens(self) -> int:
        return sum(r.generated_tokens for r in self.requests)

    @property
    def throughput_tok_per_s(self) -> float:
        """Tokens per second throughput."""
        if self.total_time_us > 0:
            return self.total_generated_tokens / (self.total_time_us / 1e6)
        return 0.0

    @property
    def throughput_req_per_s(self) -> float:
        """Requests per second throughput."""
        if self.total_time_us > 0:
            return self.num_requests / (self.total_time_us / 1e6)
        return 0.0

    def _sorted_ttfts(self) -> list[float]:
        return sorted(r.ttft_us for r in self.requests if r.ttft_us > 0)

    def _sorted_tpots(self) -> list[float]:
        return sorted(r.avg_tpot_us for r in self.requests if r.avg_tpot_us > 0)

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        idx = int(len(values) * p / 100)
        idx = min(idx, len(values) - 1)
        return values[idx]

    @property
    def avg_ttft_ms(self) -> float:
        ttfts = self._sorted_ttfts()
        return (sum(ttfts) / len(ttfts) / 1000) if ttfts else 0.0

    @property
    def p50_ttft_ms(self) -> float:
        return self._percentile(self._sorted_ttfts(), 50) / 1000

    @property
    def p99_ttft_ms(self) -> float:
        return self._percentile(self._sorted_ttfts(), 99) / 1000

    @property
    def avg_tpot_ms(self) -> float:
        tpots = self._sorted_tpots()
        return (sum(tpots) / len(tpots) / 1000) if tpots else 0.0

    @property
    def p50_tpot_ms(self) -> float:
        return self._percentile(self._sorted_tpots(), 50) / 1000

    @property
    def p99_tpot_ms(self) -> float:
        return self._percentile(self._sorted_tpots(), 99) / 1000

    def summary(self) -> str:
        lines = [
            f"Requests:     {self.num_requests}",
            f"Total tokens: {self.total_generated_tokens}",
            f"Total time:   {self.total_time_us / 1000:.1f} ms",
            f"Throughput:   {self.throughput_tok_per_s:.1f} tok/s, "
            f"{self.throughput_req_per_s:.2f} req/s",
            f"TTFT:         avg={self.avg_ttft_ms:.2f} ms, "
            f"p50={self.p50_ttft_ms:.2f} ms, p99={self.p99_ttft_ms:.2f} ms",
            f"TPOT:         avg={self.avg_tpot_ms:.3f} ms, "
            f"p50={self.p50_tpot_ms:.3f} ms, p99={self.p99_tpot_ms:.3f} ms",
        ]
        return "\n".join(lines)

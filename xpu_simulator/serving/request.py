"""Request state machine for serving simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"


@dataclass
class Request:
    """A single inference request.

    Attributes:
        id: Unique request identifier.
        prompt_len: Number of tokens in the prompt.
        output_len: Number of tokens to generate.
        arrival_time_us: When the request arrived (microseconds).
    """
    id: int
    prompt_len: int
    output_len: int
    arrival_time_us: float = 0.0

    state: RequestState = field(default=RequestState.WAITING)
    generated_tokens: int = 0
    kv_len: int = 0  # Current KV cache length

    # Timing
    prefill_start_us: float = 0.0
    prefill_end_us: float = 0.0
    first_token_us: float = 0.0
    finish_us: float = 0.0
    decode_latencies_us: list[float] = field(default_factory=list)

    @property
    def ttft_us(self) -> float:
        """Time to first token (from arrival to first token generated)."""
        if self.first_token_us > 0:
            return self.first_token_us - self.arrival_time_us
        return 0.0

    @property
    def ttft_ms(self) -> float:
        return self.ttft_us / 1000

    @property
    def avg_tpot_us(self) -> float:
        """Average time per output token."""
        if self.decode_latencies_us:
            return sum(self.decode_latencies_us) / len(self.decode_latencies_us)
        return 0.0

    @property
    def avg_tpot_ms(self) -> float:
        return self.avg_tpot_us / 1000

    @property
    def total_time_us(self) -> float:
        """Total request latency."""
        if self.finish_us > 0:
            return self.finish_us - self.arrival_time_us
        return 0.0

    def blocks_needed(self, block_size: int) -> int:
        """Number of KV cache blocks needed for current state."""
        total_tokens = self.prompt_len + self.generated_tokens
        return (total_tokens + block_size - 1) // block_size

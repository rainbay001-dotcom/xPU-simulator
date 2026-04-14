"""Training iteration modeling — Forward/Backward pass with comm overlap."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .evaluator import SimResult
from .parallel import ParallelConfig, InterconnectSpec, HierarchicalInterconnect


class TrainingPhase(Enum):
    FORWARD = "forward"
    INPUT_GRADIENT = "input_gradient"
    WEIGHT_GRADIENT = "weight_gradient"


@dataclass
class TrainingConfig:
    """Configuration for training iteration simulation.

    micro_batches: Number of micro-batches per iteration.
    backward_compute_ratio: Backward compute time as multiple of forward (typically ~2x).
    weight_grad_overlap: If True, overlap weight gradient comm with next layer's
                         input gradient compute (simulates NCCL async allreduce).
    """
    micro_batches: int = 1
    backward_compute_ratio: float = 2.0
    weight_grad_overlap: bool = True


@dataclass
class LayerTiming:
    """Timing breakdown for one transformer layer in one micro-batch."""
    layer_id: int
    fwd_compute_us: float          # Forward pass compute time
    input_grad_compute_us: float   # Backward: input gradient compute
    weight_grad_compute_us: float  # Backward: weight gradient compute
    fwd_comm_us: float = 0.0      # Forward pass communication (TP all-reduce)
    weight_grad_comm_us: float = 0.0  # Weight gradient sync (DP all-reduce/reduce-scatter)


@dataclass
class TrainingResult:
    """Result of a training iteration simulation."""
    fwd_us: float                    # Total forward pass time
    bwd_us: float                    # Total backward pass compute time
    comm_us: float                   # Exposed (non-overlapped) communication time
    bubble_us: float                 # Pipeline bubble overhead
    total_iteration_us: float        # Total wall-clock time per iteration
    layer_timings: list[LayerTiming]
    config: TrainingConfig
    parallel: ParallelConfig

    @property
    def total_ms(self) -> float:
        return self.total_iteration_us / 1000

    @property
    def overlap_efficiency(self) -> float:
        """How much of the comm was hidden behind compute (0-1)."""
        total_comm = sum(lt.weight_grad_comm_us for lt in self.layer_timings)
        if total_comm == 0:
            return 1.0
        return 1.0 - self.comm_us / total_comm

    def summary(self) -> str:
        lines = [
            f"Training iteration: {self.total_iteration_us:.1f} us ({self.total_ms:.3f} ms)",
            f"  Forward:     {self.fwd_us:.1f} us",
            f"  Backward:    {self.bwd_us:.1f} us",
            f"  Comm (exposed): {self.comm_us:.1f} us",
            f"  Bubble:      {self.bubble_us:.1f} us",
            f"  Overlap eff: {self.overlap_efficiency:.1%}",
            f"  Micro-batches: {self.config.micro_batches}",
            f"  Layers: {len(self.layer_timings)}",
        ]
        return "\n".join(lines)


def extract_layer_timings(
    per_layer_results: list[SimResult],
    parallel: ParallelConfig,
    training_config: TrainingConfig,
    param_bytes_per_layer: int = 0,
    interconnect: Optional[InterconnectSpec | HierarchicalInterconnect] = None,
) -> list[LayerTiming]:
    """Convert per-layer forward-pass SimResults into LayerTimings.

    Args:
        per_layer_results: One SimResult per layer (forward pass only).
        parallel: Parallelism configuration.
        training_config: Training configuration.
        param_bytes_per_layer: Bytes of parameters per layer (for DP gradient sync).
            If 0, weight_grad_comm_us is set to 0.
        interconnect: Interconnect for DP gradient all-reduce. If None, no comm modeled.
    """
    timings = []
    ratio = training_config.backward_compute_ratio

    for i, result in enumerate(per_layer_results):
        # Separate compute from comm in the forward result
        fwd_compute = sum(
            r.cost.latency_us for r in result.per_op
            if r.cost.bound != "communication"
        )
        fwd_comm = sum(
            r.cost.latency_us for r in result.per_op
            if r.cost.bound == "communication"
        )

        # Backward compute is forward * ratio, split evenly between IG and WG
        ig_compute = fwd_compute * ratio / 2
        wg_compute = fwd_compute * ratio / 2

        # DP gradient sync: all-reduce param_bytes across dp_size ranks
        wg_comm = 0.0
        if param_bytes_per_layer > 0 and parallel.dp_size > 1 and interconnect is not None:
            from .communication import all_reduce_time
            cc = all_reduce_time(param_bytes_per_layer, parallel.dp_size,
                                 interconnect, gpus_per_node=parallel.gpus_per_node)
            wg_comm = cc.latency_us

        timings.append(LayerTiming(
            layer_id=i,
            fwd_compute_us=fwd_compute,
            input_grad_compute_us=ig_compute,
            weight_grad_compute_us=wg_compute,
            fwd_comm_us=fwd_comm,
            weight_grad_comm_us=wg_comm,
        ))

    return timings


def estimate_training_iteration(
    layer_timings: list[LayerTiming],
    parallel: ParallelConfig,
    training_config: TrainingConfig,
) -> TrainingResult:
    """Estimate total training iteration time from per-layer timings.

    Models:
    - Forward pass: layers 0..N-1 sequentially (compute + TP comm)
    - Backward pass: layers N-1..0, each layer does IG then WG
    - Weight gradient comm overlap: WG all-reduce of layer i overlaps
      with IG compute of layer i-1 (when weight_grad_overlap=True)
    - Pipeline bubble: added as fraction of (fwd + bwd)
    - Micro-batches: total = per_microbatch * micro_batches
    """
    if not layer_timings:
        return TrainingResult(0, 0, 0, 0, 0, [], training_config, parallel)

    # Forward pass
    fwd_us = sum(lt.fwd_compute_us + lt.fwd_comm_us for lt in layer_timings)

    # Backward pass with optional WG comm overlap
    bwd_compute_us = 0.0
    exposed_comm_us = 0.0

    reversed_layers = list(reversed(layer_timings))
    for idx, lt in enumerate(reversed_layers):
        bwd_compute_us += lt.input_grad_compute_us + lt.weight_grad_compute_us

        wg_comm = lt.weight_grad_comm_us
        if wg_comm > 0:
            if training_config.weight_grad_overlap and idx + 1 < len(reversed_layers):
                # Overlap with next layer's IG compute
                next_ig = reversed_layers[idx + 1].input_grad_compute_us
                exposed = max(0.0, wg_comm - next_ig)
            else:
                exposed = wg_comm
            exposed_comm_us += exposed

    # Pipeline bubble
    compute_total = fwd_us + bwd_compute_us
    bubble_us = compute_total * parallel.bubble_fraction

    # Per micro-batch total, times micro_batches
    per_mb = compute_total + exposed_comm_us + bubble_us
    total = per_mb * training_config.micro_batches

    return TrainingResult(
        fwd_us=fwd_us * training_config.micro_batches,
        bwd_us=bwd_compute_us * training_config.micro_batches,
        comm_us=exposed_comm_us * training_config.micro_batches,
        bubble_us=bubble_us * training_config.micro_batches,
        total_iteration_us=total,
        layer_timings=layer_timings,
        config=training_config,
        parallel=parallel,
    )

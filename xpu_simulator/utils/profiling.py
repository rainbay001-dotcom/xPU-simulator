"""Profiling utilities — export simulation traces."""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..core.evaluator import SimResult, OpResult
from .categories import categorize_op

logger = logging.getLogger("xpu_simulator")


def to_perfetto_trace(result: SimResult, filename: str = "trace.json", device_name: str = "Device"):
    """Export simulation result as Perfetto-compatible trace.

    Open with https://ui.perfetto.dev
    """
    events = []

    # Process metadata
    events.append({
        "name": "process_name", "ph": "M", "pid": 1,
        "args": {"name": device_name},
    })

    # Group ops into tracks by category
    track_tids = {}
    tid_counter = 1

    for r in result.per_op:
        name = r.node.name or r.node.op.name or r.node.op.op_type.name
        category = categorize_op(name)

        if category not in track_tids:
            track_tids[category] = tid_counter
            events.append({
                "name": "thread_name", "ph": "M", "pid": 1, "tid": tid_counter,
                "args": {"name": category},
            })
            tid_counter += 1

        tid = track_tids[category]

        events.append({
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": r.start_us,
            "dur": max(r.cost.latency_us, 0.001),  # Perfetto needs dur > 0
            "pid": 1,
            "tid": tid,
            "args": {
                "op_type": r.node.op.op_type.name,
                "flops": r.cost.flops,
                "gflops": round(r.cost.flops / 1e9, 2),
                "bytes_accessed": r.cost.bytes_accessed,
                "mb_accessed": round(r.cost.bytes_accessed / 1e6, 2),
                "bound": r.cost.bound,
                "utilization": f"{r.cost.utilization:.1%}",
                "compute_us": round(r.cost.compute_us, 2),
                "memory_us": round(r.cost.memory_us, 2),
                "latency_us": round(r.cost.latency_us, 2),
            },
        })

    # Add summary counter track for cumulative latency
    events.append({
        "name": "thread_name", "ph": "M", "pid": 1, "tid": tid_counter,
        "args": {"name": "Summary"},
    })

    trace = {
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "metadata": {
            "device": device_name,
            "total_latency_us": round(result.total_latency_us, 2),
            "total_tflops": round(result.total_flops / 1e12, 2),
            "total_memory_gb": round(result.total_bytes / 1e9, 2),
            "num_ops": len(result.per_op),
            "compute_bound_ops": result.compute_bound_count,
            "memory_bound_ops": result.memory_bound_count,
        },
    }

    with open(filename, "w") as f:
        json.dump(trace, f)

    return filename


# Keep backward compatibility
to_chrome_trace = to_perfetto_trace


def print_timeline(result: SimResult, max_width: int = 60):
    """Print an ASCII timeline of the execution."""
    if not result.per_op:
        logger.info("(empty)")
        return

    total = result.total_latency_us
    if total == 0:
        return

    scale = max_width / total

    logger.info("Timeline (total: %.2f us)", total)
    logger.info("-" * (max_width + 40))

    for r in sorted(result.per_op, key=lambda r: r.start_us):
        name = (r.node.name or r.node.op.op_type.name)[:12].ljust(12)
        start_col = int(r.start_us * scale)
        width = max(1, int(r.cost.latency_us * scale))
        bar = " " * start_col + "#" * width

        bound_char = "C" if "compute" in r.cost.bound else "M"
        logger.info("  %s |%-*s| %7.2f us [%s]", name, max_width, bar, r.cost.latency_us, bound_char)

    logger.info("-" * (max_width + 40))

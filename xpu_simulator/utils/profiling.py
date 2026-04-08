"""Profiling utilities — export simulation traces."""
from __future__ import annotations

import json
from typing import Optional

from ..core.evaluator import SimResult, OpResult


def to_chrome_trace(result: SimResult, filename: str = "trace.json"):
    """Export simulation result as Chrome tracing JSON.

    Open with chrome://tracing or https://ui.perfetto.dev
    """
    events = []

    for r in result.per_op:
        name = r.node.name or r.node.op.name or r.node.op.op_type.name
        events.append({
            "name": name,
            "cat": r.cost.bound,
            "ph": "X",  # complete event
            "ts": r.start_us,
            "dur": r.cost.latency_us,
            "pid": 1,
            "tid": 1,
            "args": {
                "op_type": r.node.op.op_type.name,
                "flops": r.cost.flops,
                "bytes": r.cost.bytes_accessed,
                "bound": r.cost.bound,
                "utilization": f"{r.cost.utilization:.1%}",
                "compute_us": r.cost.compute_us,
                "memory_us": r.cost.memory_us,
            },
        })

    trace = {"traceEvents": events, "displayTimeUnit": "us"}

    with open(filename, "w") as f:
        json.dump(trace, f, indent=2)

    return filename


def print_timeline(result: SimResult, max_width: int = 60):
    """Print an ASCII timeline of the execution."""
    if not result.per_op:
        print("(empty)")
        return

    total = result.total_latency_us
    if total == 0:
        return

    scale = max_width / total

    print(f"Timeline (total: {total:.2f} us)")
    print("-" * (max_width + 40))

    for r in sorted(result.per_op, key=lambda r: r.start_us):
        name = (r.node.name or r.node.op.op_type.name)[:12].ljust(12)
        start_col = int(r.start_us * scale)
        width = max(1, int(r.cost.latency_us * scale))
        bar = " " * start_col + "#" * width

        bound_char = "C" if "compute" in r.cost.bound else "M"
        print(f"  {name} |{bar:<{max_width}}| {r.cost.latency_us:7.2f} us [{bound_char}]")

    print("-" * (max_width + 40))

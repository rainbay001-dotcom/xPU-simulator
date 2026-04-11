"""Validate NPU cost model cold-path predictions against CA simulator traces.

CA sim runs kernels with no L2 reuse (cold HBM), so the model must be
instantiated with ``warm_l2=False`` to match. Reads the analysis JSON
produced by ``/tmp/ca_pipe_analysis.py`` (stored in reports/) and reports
the geomean ratio per op family. Fails CI if any family drifts outside
[0.80, 1.25].

At 1.8 GHz: 1800 cycles = 1 us.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.backends.npu.hardware import ASCEND_910B
from xpu_simulator.backends.npu.cost_model import NPUCostModel

CLOCK_GHZ = 1.8
TOLERANCE = (0.80, 1.25)
CA_JSON = REPO / "reports" / "ca_pipe_analysis.json"

OP_MAP = {
    "add": OpType.ADD,
    "mul": OpType.MUL,
    "relu": OpType.RELU,
    "rope": OpType.ROPE,
}
SHAPE_RE = re.compile(r"^(add|mul|relu|rope)_(\d+)x(\d+)$")


def build_op(name: str) -> OpSpec | None:
    m = SHAPE_RE.match(name)
    if not m:
        return None
    op_type = OP_MAP[m.group(1)]
    rows, cols = int(m.group(2)), int(m.group(3))
    x = TensorSpec((rows, cols), Dtype.FP16)
    y = TensorSpec((rows, cols), Dtype.FP16)
    inputs = [x, y] if op_type in (OpType.ADD, OpType.MUL) else [x]
    return OpSpec(op_type, inputs, [y])


def main() -> int:
    data = json.loads(CA_JSON.read_text())
    model = NPUCostModel(ASCEND_910B, warm_l2=False)

    print(f"{'kernel':<22}{'ca_us':>10}{'model_us':>11}{'ratio':>8}"
          f"{'mte2_us':>10}{'vec_us':>9}")
    print("-" * 70)

    ratios_by_op: dict[str, list[float]] = {}
    for r in data:
        op = build_op(r["kernel"])
        if op is None or r["total_cycles"] == 0:
            continue
        ca_us = r["total_cycles"] / (CLOCK_GHZ * 1000)
        cost = model.estimate(op)
        ratio = cost.latency_us / ca_us
        family = r["kernel"].split("_")[0]
        ratios_by_op.setdefault(family, []).append(ratio)
        print(f"{r['kernel']:<22}{ca_us:>10.2f}{cost.latency_us:>11.2f}"
              f"{ratio:>8.2f}{cost.mte2_us:>10.2f}{cost.vec_us:>9.2f}")

    print()
    print("--- geomean ratio by op family (cold path, warm_l2=False) ---")
    fail = False
    for op, ratios in sorted(ratios_by_op.items()):
        g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        status = "OK" if TOLERANCE[0] <= g <= TOLERANCE[1] else "DRIFT"
        if status == "DRIFT":
            fail = True
        print(f"  {op:<8}  n={len(ratios):3}  geomean={g:.3f}  "
              f"min={min(ratios):.2f}  max={max(ratios):.2f}  [{status}]")

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export interactive HTML report with architecture visualization."""
from __future__ import annotations

import json
import base64
import io
import logging
from typing import Callable, Optional

from ..core.graph import ComputeGraph
from ..core.evaluator import SimResult, PerformanceEvaluator
from ..core.cost_model import CostModel
from .categories import categorize_op, CATEGORY_COLORS

logger = logging.getLogger("xpu_simulator")


def _build_layer_data(result: SimResult, categorize_fn: Callable[[str], str] = None) -> dict:
    """Build per-layer breakdown data."""
    _cat = categorize_fn or categorize_op
    layers = {}
    other_ops = []

    for r in result.per_op:
        name = r.node.name or ""
        cat = _cat(name)
        layer_id = None
        if name.startswith("L") and "." in name:
            try:
                layer_id = int(name.split(".")[0][1:])
            except ValueError:
                pass

        if layer_id is not None:
            if layer_id not in layers:
                layers[layer_id] = {}
            if cat not in layers[layer_id]:
                layers[layer_id][cat] = {"latency": 0, "flops": 0, "count": 0}
            layers[layer_id][cat]["latency"] += r.cost.latency_us
            layers[layer_id][cat]["flops"] += r.cost.flops
            layers[layer_id][cat]["count"] += 1
        else:
            other_ops.append({"name": name, "cat": cat, "latency": r.cost.latency_us})

    return {"layers": layers, "other": other_ops}



def _build_architecture_overview(
    graph: ComputeGraph, result: SimResult, n_dense: int,
    categorize_fn: Callable[[str], str],
) -> dict:
    """Build detailed architecture overview data.

    Returns a dict with:
      - pipeline: high-level blocks (embedding → layers → norm → lm_head)
      - dense_block: representative dense transformer block breakdown
      - moe_block: representative MoE transformer block breakdown
      - mla_block: MLA attention sub-component breakdown
    """
    _cat = categorize_fn

    # Collect per-layer and non-layer op stats
    layer_data: dict[int, dict] = {}
    pre_ops: list[dict] = []
    post_ops: list[dict] = []
    # Per-layer per-subcomponent breakdown
    layer_ops: dict[int, list[dict]] = {}

    max_layer_id = -1
    for r in result.per_op:
        name = r.node.name or ""
        cat = _cat(name)
        layer_id = None
        if name.startswith("L") and "." in name:
            try:
                layer_id = int(name.split(".")[0][1:])
            except ValueError:
                pass

        if layer_id is not None:
            max_layer_id = max(max_layer_id, layer_id)
            if layer_id not in layer_data:
                layer_data[layer_id] = {"total_lat": 0, "flops": 0, "cats": {}}
            layer_data[layer_id]["total_lat"] += r.cost.latency_us
            layer_data[layer_id]["flops"] += r.cost.flops
            if cat not in layer_data[layer_id]["cats"]:
                layer_data[layer_id]["cats"][cat] = 0
            layer_data[layer_id]["cats"][cat] += r.cost.latency_us
            # Track individual ops for sub-component breakdown
            if layer_id not in layer_ops:
                layer_ops[layer_id] = []
            layer_ops[layer_id].append({
                "name": name, "sub": name.split(".", 1)[1] if "." in name else name,
                "latency_us": round(r.cost.latency_us, 2),
                "flops": r.cost.flops, "bound": r.cost.bound,
            })
        else:
            entry = {"name": name, "cat": cat, "latency_us": r.cost.latency_us,
                     "flops": r.cost.flops, "bound": r.cost.bound}
            if not layer_data:
                pre_ops.append(entry)
            else:
                post_ops.append(entry)

    # --- Pipeline blocks ---
    pipeline = []
    if pre_ops:
        total_lat = sum(o["latency_us"] for o in pre_ops)
        pipeline.append({"label": pre_ops[0]["name"] if len(pre_ops) == 1 else "Input",
                         "type": "single", "latency_us": round(total_lat, 2),
                         "color": "#6e7681", "count": 1})

    if n_dense > 0 and layer_data:
        dense_lat = sum(layer_data[i]["total_lat"] for i in range(n_dense) if i in layer_data)
        pipeline.append({"label": f"Dense Layers (×{n_dense})", "type": "dense_group",
                         "latency_us": round(dense_lat, 2), "color": "#3fb950", "count": n_dense})
        n_moe = max_layer_id + 1 - n_dense
        if n_moe > 0:
            moe_lat = sum(layer_data[i]["total_lat"] for i in range(n_dense, max_layer_id + 1) if i in layer_data)
            pipeline.append({"label": f"MoE Layers (×{n_moe})", "type": "moe_group",
                             "latency_us": round(moe_lat, 2), "color": "#f0883e", "count": n_moe})
    elif layer_data:
        n_layers = max_layer_id + 1
        all_lat = sum(ld["total_lat"] for ld in layer_data.values())
        pipeline.append({"label": f"Layers (×{n_layers})", "type": "group",
                         "latency_us": round(all_lat, 2), "color": "#3fb950", "count": n_layers})

    for op in post_ops:
        pipeline.append({"label": op["name"], "type": "single",
                         "latency_us": round(op["latency_us"], 2), "color": "#8957e5", "count": 1})

    # --- Sub-component breakdowns for representative blocks ---
    def _classify_sub(sub_name: str) -> str:
        """Classify a sub-component name into a functional group."""
        s = sub_name.lower()
        if "attn_norm" in s:
            return "attn_norm"
        elif "ffn_norm" in s:
            return "ffn_norm"
        elif "gate_softmax" in s or (s == "gate" and "expert" not in s) or "gate_softmax" in s:
            return "gate"
        elif "gate" in s and ("expert" not in s and "softmax" not in s):
            return "gate"
        elif "shared" in s:
            return "shared_expert"
        elif "expert" in s:
            return "routed_experts"
        elif "combine" in s:
            return "combine"
        elif any(k in s for k in ["wq_a", "wq_b", "q_norm"]):
            return "q_compress"
        elif any(k in s for k in ["wkv_a", "wkv_b", "kv_norm"]):
            return "kv_compress"
        elif "rope" in s:
            return "rope"
        elif "attn_score" in s or "attn_softmax" in s or "attn_v" in s:
            return "attention"
        elif "wo" in s:
            return "output_proj"
        elif any(k in s for k in ["w1", "w3", "silu", "mul", "w2"]):
            return "ffn"
        elif "ffn" in s:
            return "ffn"
        return "other"

    def _build_block_breakdown(layer_id: int) -> list[dict]:
        """Build grouped sub-component data for a layer."""
        if layer_id not in layer_ops:
            return []
        groups: dict[str, dict] = {}
        for op in layer_ops[layer_id]:
            grp = _classify_sub(op["sub"])
            if grp not in groups:
                groups[grp] = {"label": grp, "latency_us": 0, "flops": 0, "ops": []}
            groups[grp]["latency_us"] += op["latency_us"]
            groups[grp]["flops"] += op["flops"]
            groups[grp]["ops"].append(op["sub"])
        return [{"label": k, "latency_us": round(v["latency_us"], 2),
                 "flops": v["flops"], "ops": v["ops"]} for k, v in groups.items()]

    # Representative dense block (layer 0)
    dense_block = _build_block_breakdown(0) if 0 in layer_data else []

    # Representative MoE block (first MoE layer)
    moe_layer_id = n_dense if n_dense > 0 else 3
    moe_block = _build_block_breakdown(moe_layer_id) if moe_layer_id in layer_data else []

    # MLA detail: ops from the attention portion of layer 0
    mla_ops = []
    if 0 in layer_ops:
        for op in layer_ops[0]:
            grp = _classify_sub(op["sub"])
            if grp in ("q_compress", "kv_compress", "rope", "attention", "output_proj"):
                mla_ops.append({"sub": op["sub"], "group": grp,
                                "latency_us": op["latency_us"], "flops": op["flops"]})

    return {
        "pipeline": pipeline,
        "dense_block": dense_block,
        "moe_block": moe_block,
        "mla_ops": mla_ops,
        "n_dense": n_dense,
        "n_moe": max(0, max_layer_id + 1 - n_dense) if max_layer_id >= 0 else 0,
        "n_layers": max_layer_id + 1 if max_layer_id >= 0 else 0,
    }


def export_html_report(
    graph: ComputeGraph,
    results: dict[str, SimResult],
    filename: str = "report.html",
    model_name: str = "Model",
    config: dict = None,
    categorize_fn: Callable[[str], str] = None,
    n_dense: int = 0,
):
    """Export a full interactive HTML report.

    Args:
        graph: Computation graph
        results: dict of device_name -> SimResult
        filename: Output HTML file
        model_name: Model name for title
        config: Optional model config dict
        categorize_fn: Custom categorization function. Defaults to categorize_op.
        n_dense: Number of dense layers (for Dense/MoE divider in chart). 0 to hide.
    """
    _cat = categorize_fn or categorize_op

    # Build data for each device
    device_data = {}
    for dev_name, result in results.items():
        layer_data = _build_layer_data(result, _cat)
        device_data[dev_name] = {
            "total_latency_us": round(result.total_latency_us, 2),
            "total_flops": result.total_flops,
            "total_bytes": result.total_bytes,
            "compute_bound": result.compute_bound_count,
            "memory_bound": result.memory_bound_count,
            "num_ops": len(result.per_op),
            "layers": {str(k): v for k, v in layer_data["layers"].items()},
            "other": layer_data["other"],
        }

    first_result = list(results.values())[0]

    # Top ops per device
    top_ops = {}
    for dev_name, result in results.items():
        sorted_ops = sorted(result.per_op, key=lambda r: r.cost.latency_us, reverse=True)[:15]
        top_ops[dev_name] = [{
            "name": r.node.name, "latency_us": round(r.cost.latency_us, 2),
            "bound": r.cost.bound, "flops": r.cost.flops,
            "bytes": r.cost.bytes_accessed, "utilization": round(r.cost.utilization, 3),
        } for r in sorted_ops]

    # Perfetto trace data (for first device)
    trace_events = []
    for r in first_result.per_op:
        name = r.node.name or r.node.op.op_type.name
        trace_events.append({
            "name": name, "cat": _cat(name), "ph": "X",
            "ts": r.start_us, "dur": max(r.cost.latency_us, 0.001),
            "pid": 1, "tid": 1,
        })

    # Build overall architecture overview data
    arch_overview = _build_architecture_overview(graph, first_result, n_dense, _cat)

    report_data = {
        "model_name": model_name,
        "config": config or {},
        "devices": device_data,
        "colors": CATEGORY_COLORS,
        "n_dense": n_dense,
        "top_ops": top_ops,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "arch_overview": arch_overview,
    }

    html = _HTML_TEMPLATE.replace("__REPORT_DATA__", json.dumps(report_data))

    with open(filename, "w") as f:
        f.write(html)

    return filename


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>xPU-Simulator Report</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { color: #58a6ff; font-size: 28px; margin-bottom: 8px; }
h2 { color: #58a6ff; font-size: 20px; margin: 30px 0 15px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
h3 { color: #8b949e; font-size: 16px; margin: 20px 0 10px; }
.subtitle { color: #8b949e; font-size: 14px; margin-bottom: 20px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 16px 0; }
.card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; }
.card-label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.card-value { color: #f0f6fc; font-size: 28px; font-weight: 700; margin: 4px 0; }
.card-detail { color: #8b949e; font-size: 13px; }
.comparison-table { width: 100%; border-collapse: collapse; margin: 16px 0; }
.comparison-table th, .comparison-table td { padding: 10px 16px; text-align: left; border-bottom: 1px solid #21262d; }
.comparison-table th { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; background: #161b22; }
.comparison-table td { color: #c9d1d9; font-size: 14px; }
.comparison-table tr:hover { background: #161b22; }
.speedup { color: #3fb950; font-weight: 600; }
.slowdown { color: #f85149; font-weight: 600; }
.chart-container { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; margin: 16px 0; }
canvas { width: 100% !important; }
.tabs { display: flex; gap: 0; margin: 16px 0 0; border-bottom: 1px solid #21262d; }
.tab { padding: 10px 20px; cursor: pointer; color: #8b949e; font-size: 14px; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: #c9d1d9; }
.tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
.tab-content { display: none; padding: 16px 0; }
.tab-content.active { display: block; }
.graph-container { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; margin: 16px 0; overflow: auto; }
.graph-svg { width: 100%; min-height: 500px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8b949e; }
.legend-dot { width: 12px; height: 12px; border-radius: 3px; }
.top-ops-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.top-ops-table th { color: #8b949e; font-size: 11px; text-transform: uppercase; padding: 8px 12px; border-bottom: 1px solid #21262d; text-align: left; }
.top-ops-table td { padding: 6px 12px; border-bottom: 1px solid #21262d; }
.bound-compute { color: #E8744F; }
.bound-memory { color: #4A90D9; }
.bar-bg { background: #21262d; border-radius: 4px; height: 20px; position: relative; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.config-item { font-size: 13px; }
.config-key { color: #8b949e; }
.config-val { color: #f0f6fc; font-weight: 600; }
.node { cursor: pointer; transition: opacity 0.2s; }
.node:hover { opacity: 0.8; }
.tooltip { position: absolute; background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 12px; pointer-events: none; z-index: 100; display: none; max-width: 300px; }
.tooltip .tt-title { color: #f0f6fc; font-weight: 600; margin-bottom: 4px; }
.tooltip .tt-row { color: #8b949e; }
.device-selector { display: flex; gap: 8px; margin: 10px 0; }
.device-btn { padding: 6px 14px; border: 1px solid #30363d; border-radius: 6px; background: #161b22; color: #8b949e; cursor: pointer; font-size: 13px; transition: all 0.2s; }
.device-btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
</style>
</head>
<body>
<div class="container">
<div id="app"></div>
</div>

<script>
const DATA = __REPORT_DATA__;

function fmt(n) { return n.toLocaleString(); }
function fmtLat(us) {
    if (us >= 1e6) return (us/1e6).toFixed(3) + ' s';
    if (us >= 1e3) return (us/1e3).toFixed(1) + ' ms';
    return us.toFixed(1) + ' us';
}
function fmtBytes(b) {
    if (b >= 1e9) return (b/1e9).toFixed(2) + ' GB';
    if (b >= 1e6) return (b/1e6).toFixed(1) + ' MB';
    return (b/1e3).toFixed(0) + ' KB';
}
function fmtFlops(f) {
    if (f >= 1e12) return (f/1e12).toFixed(2) + ' TFLOPS';
    if (f >= 1e9) return (f/1e9).toFixed(1) + ' GFLOPS';
    return (f/1e6).toFixed(0) + ' MFLOPS';
}

const app = document.getElementById('app');
const devices = Object.keys(DATA.devices);
const firstDev = devices[0];
const firstData = DATA.devices[firstDev];

// Header
let html = `<h1>xPU-Simulator Report</h1>`;
html += `<div class="subtitle">${DATA.model_name} &mdash; ${DATA.num_nodes} ops, ${DATA.num_edges} edges</div>`;

// Config
if (Object.keys(DATA.config).length > 0) {
    html += `<h2>Model Config</h2><div class="config-grid">`;
    for (const [k,v] of Object.entries(DATA.config)) {
        html += `<div class="config-item"><span class="config-key">${k}:</span> <span class="config-val">${v}</span></div>`;
    }
    html += `</div>`;
}

// Comparison cards
html += `<h2>Device Comparison</h2>`;
html += `<table class="comparison-table"><thead><tr>
    <th>Device</th><th>Latency</th><th>vs ${devices[0]}</th><th>Ops</th><th>Compute-bound</th><th>Memory-bound</th><th>Total FLOPs</th><th>Total Memory</th>
</tr></thead><tbody>`;

const baseLat = DATA.devices[devices[0]].total_latency_us;
for (const dev of devices) {
    const d = DATA.devices[dev];
    const speedup = baseLat / d.total_latency_us;
    const cls = speedup >= 1 ? 'speedup' : 'slowdown';
    html += `<tr>
        <td><strong>${dev}</strong></td>
        <td>${fmtLat(d.total_latency_us)}</td>
        <td class="${cls}">${speedup.toFixed(2)}x</td>
        <td>${d.num_ops}</td>
        <td>${d.compute_bound}</td>
        <td>${d.memory_bound}</td>
        <td>${fmtFlops(d.total_flops)}</td>
        <td>${fmtBytes(d.total_bytes)}</td>
    </tr>`;
}
html += `</tbody></table>`;

// Overall Architecture Overview
if (DATA.arch_overview && DATA.arch_overview.pipeline) {
    html += `<h2>Model Architecture</h2>`;
    html += `<h3>High-Level Pipeline</h3>`;
    html += `<div class="chart-container"><svg id="arch-pipeline" class="graph-svg" style="min-height:160px"></svg></div>`;

    // Detailed block views side by side
    const hasBlocks = (DATA.arch_overview.dense_block && DATA.arch_overview.dense_block.length > 0) ||
                      (DATA.arch_overview.moe_block && DATA.arch_overview.moe_block.length > 0);
    if (hasBlocks) {
        html += `<h3>Transformer Block Detail</h3>`;
        html += `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px">`;
        if (DATA.arch_overview.dense_block && DATA.arch_overview.dense_block.length > 0) {
            html += `<div class="chart-container"><h3 style="margin:0 0 10px;color:#3fb950">Dense Block <span style="font-weight:normal;font-size:0.85em;color:#8b949e">(×${DATA.arch_overview.n_dense}, showing Layer 1)</span></h3><svg id="arch-dense" class="graph-svg" style="min-height:400px"></svg></div>`;
        }
        if (DATA.arch_overview.moe_block && DATA.arch_overview.moe_block.length > 0) {
            html += `<div class="chart-container"><h3 style="margin:0 0 10px;color:#f0883e">MoE Block <span style="font-weight:normal;font-size:0.85em;color:#8b949e">(×${DATA.arch_overview.n_moe}, showing Layer ${(DATA.arch_overview.n_dense || 3) + 1})</span></h3><svg id="arch-moe" class="graph-svg" style="min-height:400px"></svg></div>`;
        }
        html += `</div>`;
    }

    // MLA detail
    if (DATA.arch_overview.mla_ops && DATA.arch_overview.mla_ops.length > 0) {
        html += `<h3>Multi-Head Latent Attention (MLA) Detail <span style="font-weight:normal;font-size:0.85em;color:#8b949e">— expanding the Attention component above</span></h3>`;
        html += `<div class="chart-container"><svg id="arch-mla" class="graph-svg" style="min-height:280px"></svg></div>`;
    }
}

// Per-layer stacked bar chart
html += `<h2>Per-Layer Latency Breakdown</h2>`;
html += `<div class="device-selector" id="layer-device-sel"></div>`;
html += `<div class="chart-container"><canvas id="layerChart" height="300"></canvas></div>`;

// Legend
html += `<div class="legend">`;
for (const [cat, color] of Object.entries(DATA.colors)) {
    html += `<div class="legend-item"><div class="legend-dot" style="background:${color}"></div>${cat}</div>`;
}
html += `</div>`;

// Tabs for graph views

// Top ops table
html += `<h2>Most Expensive Operations</h2>`;
html += `<div class="device-selector" id="topops-device-sel"></div>`;
html += `<div id="topops-table"></div>`;

// Tooltip
html += `<div class="tooltip" id="tooltip"></div>`;

app.innerHTML = html;

// --- Layer chart ---
function drawLayerChart(devName) {
    const d = DATA.devices[devName];
    const canvas = document.getElementById('layerChart');
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 40;
    canvas.height = 300;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layers = d.layers;
    const layerIds = Object.keys(layers).map(Number).sort((a,b) => a-b);
    const n = layerIds.length;
    if (n === 0) return;

    const cats = Object.keys(DATA.colors);
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const w = canvas.width - margin.left - margin.right;
    const h = canvas.height - margin.top - margin.bottom;
    const barW = Math.max(1, w / n - 1);

    // Find max
    let maxVal = 0;
    for (const lid of layerIds) {
        let total = 0;
        for (const cat of cats) {
            if (layers[lid] && layers[lid][cat]) total += layers[lid][cat].latency;
        }
        maxVal = Math.max(maxVal, total);
    }
    maxVal = maxVal / 1000; // to ms

    // Draw bars
    for (let i = 0; i < n; i++) {
        const lid = layerIds[i];
        const x = margin.left + (i / n) * w;
        let y = margin.top + h;
        for (const cat of cats) {
            if (layers[lid] && layers[lid][cat]) {
                const val = layers[lid][cat].latency / 1000;
                const barH = (val / maxVal) * h;
                ctx.fillStyle = DATA.colors[cat];
                ctx.fillRect(x, y - barH, barW, barH);
                y -= barH;
            }
        }
    }

    // Dense/MoE divider (only if n_dense > 0)
    const nDense = DATA.n_dense || 0;
    if (nDense > 0) {
        const divX = margin.left + (nDense / n) * w;
        ctx.strokeStyle = '#f85149';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);
        ctx.beginPath(); ctx.moveTo(divX, margin.top); ctx.lineTo(divX, margin.top + h); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#f85149';
        ctx.font = '12px sans-serif';
        ctx.fillText('Dense', margin.left + 5, margin.top + 14);
        ctx.fillText('MoE', divX + 8, margin.top + 14);
    }

    // Axes
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i += Math.max(1, Math.floor(n/15))) {
        const x = margin.left + (i / n) * w + barW / 2;
        ctx.fillText(layerIds[i], x, margin.top + h + 20);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const val = (maxVal * i / 5).toFixed(1);
        const y = margin.top + h - (i / 5) * h;
        ctx.fillText(val + ' ms', margin.left - 5, y + 4);
        ctx.strokeStyle = '#21262d';
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + w, y); ctx.stroke();
    }
}

// Device selectors
function makeDeviceSelector(containerId, callback) {
    const el = document.getElementById(containerId);
    devices.forEach((dev, i) => {
        const btn = document.createElement('div');
        btn.className = 'device-btn' + (i === 0 ? ' active' : '');
        btn.textContent = dev;
        btn.onclick = () => {
            el.querySelectorAll('.device-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            callback(dev);
        };
        el.appendChild(btn);
    });
}

makeDeviceSelector('layer-device-sel', drawLayerChart);
drawLayerChart(devices[0]);

// --- Dataflow graph SVG ---
// Tooltip
const tooltip = document.getElementById('tooltip');
window.showTip = function(evt, nodeId, svgId) {
    const svg = document.getElementById(svgId);
    const n = svg._nodeMap[nodeId];
    if (!n) return;
    tooltip.innerHTML = `
        <div class="tt-title">${n.label}</div>
        <div class="tt-row">Category: ${n.category}</div>
        <div class="tt-row">Latency: ${n.latency_us.toFixed(2)} us</div>
        <div class="tt-row">Bound: ${n.bound}</div>
        <div class="tt-row">FLOPs: ${fmtFlops(n.flops)}</div>
        <div class="tt-row">Memory: ${fmtBytes(n.bytes)}</div>
        <div class="tt-row">Utilization: ${(n.utilization*100).toFixed(1)}%</div>
    `;
    tooltip.style.display = 'block';
    tooltip.style.left = (evt.pageX + 12) + 'px';
    tooltip.style.top = (evt.pageY - 10) + 'px';
};
window.hideTip = function() { tooltip.style.display = 'none'; };

// --- Architecture Renderers ---
const ARCH = DATA.arch_overview || {};
const archDefs = `<defs><marker id="a-arr" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8b949e"/></marker>
<marker id="a-arr-w" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#fff"/></marker></defs>`;

function svgArrow(x1,y1,x2,y2,color){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color||'#8b949e'}" stroke-width="2" marker-end="url(#a-arr)"/>`;}

// --- Pipeline ---
function drawPipeline() {
    const svg = document.getElementById('arch-pipeline');
    if (!svg || !ARCH.pipeline) return;
    const blocks = ARCH.pipeline;
    const totalLat = blocks.reduce((s,b)=>s+b.latency_us,0);
    const pad=30, bH=80, arrW=40, minW=110, maxW=320;
    const dataW = Math.max(blocks.length*minW, 1100);
    const widths = blocks.map(b=>{
        const p = totalLat>0 ? b.latency_us/totalLat : 1/blocks.length;
        return Math.max(minW, Math.min(maxW, p*dataW));
    });
    const totW = widths.reduce((s,w)=>s+w,0) + (blocks.length-1)*arrW + pad*2;
    const svgH = bH + pad*2;
    svg.setAttribute('viewBox',`0 0 ${totW} ${svgH}`);
    let c = archDefs; let x = pad; const y = pad;
    blocks.forEach((b,i)=>{
        const w = widths[i]; const isGrp = b.type.includes('group');
        c += `<rect x="${x}" y="${y}" width="${w}" height="${bH}" rx="${isGrp?12:8}" fill="${b.color}" opacity="0.85" stroke="#fff" stroke-width="2"/>`;
        if(isGrp) c += `<rect x="${x+4}" y="${y+4}" width="${w-8}" height="${bH-8}" rx="8" fill="none" stroke="#fff" stroke-width="1" stroke-dasharray="4,3" opacity="0.35"/>`;
        const lbl = b.label.length>24 ? b.label.slice(0,22)+'..' : b.label;
        c += `<text x="${x+w/2}" y="${y+30}" text-anchor="middle" fill="#fff" font-size="13" font-weight="bold">${lbl}</text>`;
        c += `<text x="${x+w/2}" y="${y+50}" text-anchor="middle" fill="#fff" font-size="11" opacity="0.9">${fmtLat(b.latency_us)}</text>`;
        const pct = totalLat>0 ? (b.latency_us/totalLat*100).toFixed(1) : '0';
        c += `<text x="${x+w/2}" y="${y+67}" text-anchor="middle" fill="#fff" font-size="10" opacity="0.7">${pct}%</text>`;
        if(i<blocks.length-1){
            c += svgArrow(x+w+5, y+bH/2, x+w+arrW-5, y+bH/2);
        }
        x += w + arrW;
    });
    svg.innerHTML = c;
}
drawPipeline();

// --- Block detail (dense or MoE) ---
// Draws a vertical flowchart of sub-components inside a transformer block
const subColors = {
    attn_norm:'#8b949e', ffn_norm:'#8b949e',
    q_compress:'#58a6ff', kv_compress:'#79c0ff', rope:'#a371f7',
    attention:'#f0883e', output_proj:'#d2a8ff',
    ffn:'#3fb950',
    gate:'#f85149', routed_experts:'#f0883e', shared_expert:'#da7756', combine:'#8957e5',
    other:'#6e7681'
};
const subLabels = {
    attn_norm:'RMSNorm (pre-attn)', ffn_norm:'RMSNorm (pre-FFN)',
    q_compress:'Q Compression (W_DQ → norm → W_UQ)', kv_compress:'KV Compression (W_DKV → norm → W_UKV)',
    rope:'RoPE', attention:'Q@K^T → Softmax → @V', output_proj:'Output Proj (W_O)',
    ffn:'SwiGLU FFN (W₁→SiLU, W₃→Mul→W₂)',
    gate:'Gate Router → Softmax', routed_experts:'Routed Experts ×8 (SwiGLU)',
    shared_expert:'Shared Expert (SwiGLU)', combine:'Add (routed + shared)',
    other:'Other'
};

function drawBlockDetail(svgId, components, title) {
    const svg = document.getElementById(svgId);
    if (!svg || !components || components.length === 0) return;

    const pad=20, boxW=360, boxH=44, gap=8, arrH=20;
    const totalH = components.length*(boxH+gap+arrH) - arrH + pad*2;
    const svgW = boxW + pad*2 + 300; // extra for latency bars + labels
    svg.setAttribute('viewBox',`0 0 ${svgW} ${totalH}`);
    svg.style.minHeight = totalH + 'px';

    const totalLat = components.reduce((s,c)=>s+c.latency_us,0);
    let c = archDefs;
    const x0 = pad + 80; // offset for residual bracket

    components.forEach((comp,i)=>{
        const y = pad + i*(boxH+gap+arrH);
        const color = subColors[comp.label] || '#6e7681';
        const label = subLabels[comp.label] || comp.label;

        // Box
        c += `<rect x="${x0}" y="${y}" width="${boxW}" height="${boxH}" rx="8" fill="${color}" opacity="0.85" stroke="#30363d" stroke-width="1.5"/>`;

        // Label
        c += `<text x="${x0+boxW/2}" y="${y+18}" text-anchor="middle" fill="#fff" font-size="12" font-weight="bold">${label}</text>`;

        // Latency + percentage
        const pct = totalLat>0 ? (comp.latency_us/totalLat*100).toFixed(1) : '0';
        c += `<text x="${x0+boxW/2}" y="${y+34}" text-anchor="middle" fill="#fff" font-size="10" opacity="0.8">${fmtLat(comp.latency_us)} (${pct}%)</text>`;

        // Latency bar on the right
        const barX = x0+boxW+12; const barW = 100;
        const pctW = totalLat>0 ? comp.latency_us/totalLat*barW : 0;
        c += `<rect x="${barX}" y="${y+14}" width="${barW}" height="${12}" rx="3" fill="#21262d"/>`;
        c += `<rect x="${barX}" y="${y+14}" width="${pctW}" height="${12}" rx="3" fill="${color}" opacity="0.7"/>`;
        // Latency number to the right of bar
        c += `<text x="${barX+barW+6}" y="${y+24}" fill="#8b949e" font-size="10" dominant-baseline="middle">${fmtLat(comp.latency_us)} (${pct}%)</text>`;

        // Arrow to next
        if(i<components.length-1){
            const ay1 = y+boxH+2; const ay2 = y+boxH+gap+arrH-2;
            c += svgArrow(x0+boxW/2, ay1, x0+boxW/2, ay2);
        }
    });

    // Residual connection brackets
    // Find attn_norm..output_proj span and ffn_norm..end span
    const attnStart = components.findIndex(c=>c.label==='attn_norm');
    const outputEnd = components.findIndex(c=>c.label==='output_proj');
    const ffnStart = components.findIndex(c=>c.label==='ffn_norm');
    const lastIdx = components.length - 1;

    function drawResidual(startIdx, endIdx, labelText) {
        if(startIdx < 0 || endIdx < 0 || startIdx >= endIdx) return;
        const y1 = pad + startIdx*(boxH+gap+arrH) + boxH/2;
        const y2 = pad + endIdx*(boxH+gap+arrH) + boxH/2;
        const bx = x0 - 12;
        c += `<path d="M${bx} ${y1} C${bx-30} ${y1} ${bx-30} ${y2} ${bx} ${y2}" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.6"/>`;
        c += `<text x="${bx-18}" y="${(y1+y2)/2+4}" text-anchor="middle" fill="#58a6ff" font-size="9" opacity="0.7" transform="rotate(-90,${bx-18},${(y1+y2)/2})">${labelText}</text>`;
    }
    drawResidual(attnStart, outputEnd, '+ residual');
    if(ffnStart >= 0) drawResidual(ffnStart, lastIdx, '+ residual');

    svg.innerHTML = c;
}

if(ARCH.dense_block) drawBlockDetail('arch-dense', ARCH.dense_block, 'Dense Block');
if(ARCH.moe_block) drawBlockDetail('arch-moe', ARCH.moe_block, 'MoE Block');

// --- MLA Detail ---
function drawMLA() {
    const svg = document.getElementById('arch-mla');
    if (!svg || !ARCH.mla_ops || ARCH.mla_ops.length === 0) return;

    // Group ops by their functional group
    const groups = {};
    ARCH.mla_ops.forEach(op => {
        if(!groups[op.group]) groups[op.group] = {ops:[], latency:0, flops:0};
        groups[op.group].ops.push(op.sub);
        groups[op.group].latency += op.latency_us;
        groups[op.group].flops += op.flops;
    });

    // Layout: two parallel paths (Q and KV) converging at attention
    const pad=30, boxW=200, boxH=50, hGap=60, vGap=20;
    const colQ = pad; const colKV = pad + boxW + hGap;
    const mergeX = pad + (boxW*2+hGap)/2 - boxW/2;
    const svgW = pad*2 + boxW*2 + hGap + 40;

    // Vertical positions
    let row = 0;
    function yAt(r){ return pad + r*(boxH+vGap); }

    let c = archDefs;

    function drawBox(x, y, w, label, sublabel, color, lat) {
        c += `<rect x="${x}" y="${y}" width="${w}" height="${boxH}" rx="8" fill="${color}" opacity="0.85" stroke="#30363d" stroke-width="1.5"/>`;
        c += `<text x="${x+w/2}" y="${y+20}" text-anchor="middle" fill="#fff" font-size="12" font-weight="bold">${label}</text>`;
        c += `<text x="${x+w/2}" y="${y+36}" text-anchor="middle" fill="#fff" font-size="10" opacity="0.8">${sublabel}${lat ? ' — '+fmtLat(lat):''}  </text>`;
    }

    // Row 0: Input
    drawBox(mergeX, yAt(0), boxW, 'Input (h)', 'shape: [tokens, 7168]', '#6e7681', 0);
    row = 1;

    // Row 1: Q compress and KV compress (parallel)
    const qLat = groups.q_compress ? groups.q_compress.latency : 0;
    const kvLat = groups.kv_compress ? groups.kv_compress.latency : 0;
    drawBox(colQ, yAt(row), boxW, 'Q Compression', 'W_DQ → RMSNorm → W_UQ', '#58a6ff', qLat);
    drawBox(colKV, yAt(row), boxW, 'KV Compression', 'W_DKV → RMSNorm → W_UKV', '#79c0ff', kvLat);
    // Arrows from input to both
    c += svgArrow(mergeX+boxW/2-20, yAt(0)+boxH+2, colQ+boxW/2, yAt(row)-4);
    c += svgArrow(mergeX+boxW/2+20, yAt(0)+boxH+2, colKV+boxW/2, yAt(row)-4);
    row = 2;

    // Row 2: RoPE (parallel on both)
    const ropeLat = groups.rope ? groups.rope.latency : 0;
    drawBox(colQ, yAt(row), boxW, 'RoPE (Q)', 'Rotary on rope dims', '#a371f7', ropeLat/2);
    drawBox(colKV, yAt(row), boxW, 'RoPE (K)', 'Rotary on rope dims', '#a371f7', ropeLat/2);
    c += svgArrow(colQ+boxW/2, yAt(row-1)+boxH+2, colQ+boxW/2, yAt(row)-4);
    c += svgArrow(colKV+boxW/2, yAt(row-1)+boxH+2, colKV+boxW/2, yAt(row)-4);
    row = 3;

    // Row 3: Attention (Q@K^T -> softmax -> @V) — merged
    const attnLat = groups.attention ? groups.attention.latency : 0;
    drawBox(mergeX, yAt(row), boxW, 'Attention', 'Q@K^T → Softmax → Scores@V', '#f0883e', attnLat);
    c += svgArrow(colQ+boxW/2, yAt(row-1)+boxH+2, mergeX+boxW/2-20, yAt(row)-4);
    c += svgArrow(colKV+boxW/2, yAt(row-1)+boxH+2, mergeX+boxW/2+20, yAt(row)-4);
    row = 4;

    // Row 4: Output projection
    const outLat = groups.output_proj ? groups.output_proj.latency : 0;
    drawBox(mergeX, yAt(row), boxW, 'Output Projection', 'W_O → [tokens, 7168]', '#d2a8ff', outLat);
    c += svgArrow(mergeX+boxW/2, yAt(row-1)+boxH+2, mergeX+boxW/2, yAt(row)-4);

    const svgH = yAt(row) + boxH + pad;
    svg.setAttribute('viewBox',`0 0 ${svgW} ${svgH}`);
    svg.style.minHeight = svgH + 'px';
    svg.innerHTML = c;
}
drawMLA();

// --- Top ops table ---
function drawTopOps(devName) {
    const ops = DATA.top_ops[devName];
    let t = `<table class="top-ops-table"><thead><tr>
        <th>Op</th><th>Latency</th><th>Bound</th><th>FLOPs</th><th>Memory</th><th>Utilization</th>
    </tr></thead><tbody>`;
    const maxLat = ops.length > 0 ? ops[0].latency_us : 1;
    ops.forEach(op => {
        const boundCls = op.bound.includes('compute') ? 'bound-compute' : 'bound-memory';
        const pct = (op.latency_us / maxLat * 100).toFixed(0);
        const color = op.bound.includes('compute') ? '#E8744F' : '#4A90D9';
        t += `<tr>
            <td><strong>${op.name}</strong></td>
            <td>${fmtLat(op.latency_us)}<div class="bar-bg"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div></td>
            <td class="${boundCls}">${op.bound}</td>
            <td>${fmtFlops(op.flops)}</td>
            <td>${fmtBytes(op.bytes)}</td>
            <td>${(op.utilization*100).toFixed(0)}%</td>
        </tr>`;
    });
    t += `</tbody></table>`;
    document.getElementById('topops-table').innerHTML = t;
}

makeDeviceSelector('topops-device-sel', drawTopOps);
drawTopOps(devices[0]);
</script>
</body>
</html>"""

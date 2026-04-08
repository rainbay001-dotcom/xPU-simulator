"""Export interactive HTML report with architecture visualization."""
from __future__ import annotations

import json
import base64
import io
from typing import Optional

from ..core.graph import ComputeGraph
from ..core.evaluator import SimResult, PerformanceEvaluator
from ..core.cost_model import CostModel


CATEGORY_COLORS = {
    "Attention Projections": "#4A90D9",
    "Attention Compute":     "#2E5EAA",
    "MoE Experts":           "#E8744F",
    "MoE Shared Expert":     "#F5A623",
    "MoE Gate":              "#D0021B",
    "Dense FFN":             "#7ED321",
    "Norms":                 "#9B9B9B",
    "RoPE":                  "#BD10E0",
    "Embedding":             "#50E3C2",
    "LM Head":               "#50E3C2",
    "Other":                 "#C0C0C0",
}


def _categorize(name: str) -> str:
    if ".attn_score" in name or ".attn_v" in name or ".attn_softmax" in name:
        return "Attention Compute"
    elif ".wq_" in name or ".wkv_" in name or ".wo" in name:
        return "Attention Projections"
    elif ".moe.experts" in name:
        return "MoE Experts"
    elif ".moe.shared" in name:
        return "MoE Shared Expert"
    elif ".moe.gate" in name:
        return "MoE Gate"
    elif ".ffn." in name:
        return "Dense FFN"
    elif "norm" in name:
        return "Norms"
    elif "rope" in name:
        return "RoPE"
    elif "embed" in name:
        return "Embedding"
    elif "lm_head" in name:
        return "LM Head"
    return "Other"


def _build_layer_data(result: SimResult) -> dict:
    """Build per-layer breakdown data."""
    layers = {}
    other_ops = []

    for r in result.per_op:
        name = r.node.name or ""
        cat = _categorize(name)
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


def _build_graph_data(graph: ComputeGraph, result: SimResult, layer_prefix: str) -> dict:
    """Build node/edge data for a single block's dataflow graph."""
    nodes = [n for n in graph.nodes if n.name and n.name.startswith(layer_prefix + ".")]
    node_ids = {n.id for n in nodes}

    lat_map = {}
    if result:
        for r in result.per_op:
            lat_map[r.node.id] = r

    graph_nodes = []
    graph_edges = []

    for n in nodes:
        name = n.name.replace(layer_prefix + ".", "")
        cat = _categorize(n.name)
        lat = lat_map[n.id].cost.latency_us if n.id in lat_map else 0
        bound = lat_map[n.id].cost.bound if n.id in lat_map else ""
        flops = lat_map[n.id].cost.flops if n.id in lat_map else 0
        mem = lat_map[n.id].cost.bytes_accessed if n.id in lat_map else 0
        util = lat_map[n.id].cost.utilization if n.id in lat_map else 0

        graph_nodes.append({
            "id": n.id, "label": name, "category": cat,
            "color": CATEGORY_COLORS.get(cat, "#C0C0C0"),
            "latency_us": round(lat, 2), "bound": bound,
            "flops": flops, "bytes": mem, "utilization": round(util, 3),
        })

        for succ in graph.successors(n):
            if succ.id in node_ids:
                graph_edges.append({"from": n.id, "to": succ.id})

    return {"nodes": graph_nodes, "edges": graph_edges}


def export_html_report(
    graph: ComputeGraph,
    results: dict[str, SimResult],
    filename: str = "report.html",
    model_name: str = "Model",
    config: dict = None,
):
    """Export a full interactive HTML report.

    Args:
        graph: Computation graph
        results: dict of device_name -> SimResult
        filename: Output HTML file
        model_name: Model name for title
        config: Optional model config dict
    """
    # Build data for each device
    device_data = {}
    for dev_name, result in results.items():
        layer_data = _build_layer_data(result)
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

    # Build graph data for one dense block and one MoE block
    first_result = list(results.values())[0]
    dense_graph = _build_graph_data(graph, first_result, "L0")
    moe_graph = _build_graph_data(graph, first_result, "L3")

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
            "name": name, "cat": _categorize(name), "ph": "X",
            "ts": r.start_us, "dur": max(r.cost.latency_us, 0.001),
            "pid": 1, "tid": 1,
        })

    report_data = {
        "model_name": model_name,
        "config": config or {},
        "devices": device_data,
        "colors": CATEGORY_COLORS,
        "dense_graph": dense_graph,
        "moe_graph": moe_graph,
        "top_ops": top_ops,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
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
html += `<h2>Block Architecture</h2>`;
html += `<div class="tabs">
    <div class="tab active" onclick="switchTab('dense')">Dense Block (L0)</div>
    <div class="tab" onclick="switchTab('moe')">MoE Block (L3)</div>
</div>`;
html += `<div id="tab-dense" class="tab-content active"><div class="graph-container"><svg id="svg-dense" class="graph-svg"></svg></div></div>`;
html += `<div id="tab-moe" class="tab-content"><div class="graph-container"><svg id="svg-moe" class="graph-svg"></svg></div></div>`;

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

    // Dense/MoE divider
    const nDense = 3;
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
function drawGraph(svgId, graphData) {
    const svg = document.getElementById(svgId);
    const nodes = graphData.nodes;
    const edges = graphData.edges;
    if (nodes.length === 0) return;

    // Topological layout
    const adj = {};
    const inDeg = {};
    nodes.forEach(n => { adj[n.id] = []; inDeg[n.id] = 0; });
    edges.forEach(e => { adj[e.from].push(e.to); inDeg[e.to]++; });

    const generations = [];
    let queue = nodes.filter(n => inDeg[n.id] === 0).map(n => n.id);
    const visited = new Set();
    while (queue.length > 0) {
        generations.push([...queue]);
        queue.forEach(id => visited.add(id));
        const next = [];
        queue.forEach(id => {
            adj[id].forEach(to => {
                inDeg[to]--;
                if (inDeg[to] === 0 && !visited.has(to)) next.push(to);
            });
        });
        queue = next;
    }

    const nodeMap = {};
    nodes.forEach(n => nodeMap[n.id] = n);

    const colW = 160, rowH = 70;
    const maxCols = Math.max(...generations.map(g => g.length));
    const svgW = Math.max(800, maxCols * colW + 100);
    const svgH = generations.length * rowH + 80;
    svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);
    svg.style.minHeight = svgH + 'px';

    let svgContent = `<defs>
        <marker id="arrow-${svgId}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/>
        </marker>
    </defs>`;

    // Position nodes
    const pos = {};
    generations.forEach((gen, gi) => {
        const y = gi * rowH + 50;
        const totalW = gen.length * colW;
        const startX = (svgW - totalW) / 2 + colW / 2;
        gen.forEach((id, xi) => {
            pos[id] = { x: startX + xi * colW, y };
        });
    });

    // Draw edges
    edges.forEach(e => {
        if (pos[e.from] && pos[e.to]) {
            const x1 = pos[e.from].x, y1 = pos[e.from].y + 18;
            const x2 = pos[e.to].x, y2 = pos[e.to].y - 18;
            svgContent += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#444" stroke-width="1.5" marker-end="url(#arrow-${svgId})" opacity="0.6"/>`;
        }
    });

    // Draw nodes
    nodes.forEach(n => {
        if (!pos[n.id]) return;
        const {x, y} = pos[n.id];
        const r = Math.max(18, Math.min(35, 18 + n.latency_us * 0.03));
        const boundClass = n.bound.includes('compute') ? 'bound-compute' : 'bound-memory';

        svgContent += `<g class="node" onmouseover="showTip(evt, ${n.id}, '${svgId}')" onmouseout="hideTip()">`;
        svgContent += `<circle cx="${x}" cy="${y}" r="${r}" fill="${n.color}" stroke="#fff" stroke-width="2" opacity="0.9"/>`;
        svgContent += `<text x="${x}" y="${y-2}" text-anchor="middle" fill="#fff" font-size="8" font-weight="bold">${n.label.length > 16 ? n.label.slice(0,14)+'..' : n.label}</text>`;
        svgContent += `<text x="${x}" y="${y+10}" text-anchor="middle" fill="#fff" font-size="7" opacity="0.8">${n.latency_us.toFixed(0)}us</text>`;
        svgContent += `</g>`;
    });

    svg.innerHTML = svgContent;

    // Store node data for tooltips
    svg._nodeMap = nodeMap;
}

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

drawGraph('svg-dense', DATA.dense_graph);
drawGraph('svg-moe', DATA.moe_graph);

// --- Tabs ---
window.switchTab = function(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
    event.target.classList.add('active');
    // Redraw graph on tab switch
    if (tab === 'moe') drawGraph('svg-moe', DATA.moe_graph);
    else drawGraph('svg-dense', DATA.dense_graph);
};

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

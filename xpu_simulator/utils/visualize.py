"""Graph visualization utilities — export colored architecture diagrams."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from ..core.graph import ComputeGraph
from ..core.evaluator import SimResult, OpResult


# Color scheme for op categories
CATEGORY_COLORS = {
    "Attention Projections": "#4A90D9",   # blue
    "Attention Compute":     "#2E5EAA",   # dark blue
    "MoE Experts":           "#E8744F",   # orange
    "MoE Shared Expert":     "#F5A623",   # amber
    "MoE Gate":              "#D0021B",   # red
    "Dense FFN":             "#7ED321",   # green
    "Norms":                 "#9B9B9B",   # gray
    "RoPE":                  "#BD10E0",   # purple
    "Embedding":             "#50E3C2",   # teal
    "LM Head":               "#50E3C2",   # teal
    "Other":                 "#C0C0C0",   # light gray
}


def _categorize(name: str) -> str:
    """Categorize op by name."""
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


def export_block_detail(
    graph: ComputeGraph,
    layer_prefix: str,
    filename: str = "block_detail.png",
    result: SimResult = None,
    figsize: tuple = (20, 14),
):
    """Export a detailed view of a single transformer block.

    Args:
        graph: Full computation graph
        layer_prefix: Layer prefix to filter (e.g. "L3" for layer 3)
        filename: Output file path
        result: Optional SimResult for annotating latency
        figsize: Figure size
    """
    # Filter nodes for this layer
    sub_nodes = [n for n in graph.nodes if n.name and n.name.startswith(layer_prefix + ".")]
    if not sub_nodes:
        print(f"No nodes found with prefix '{layer_prefix}'")
        return

    # Build subgraph
    sub_ids = {n.id for n in sub_nodes}
    G = nx.DiGraph()
    for n in sub_nodes:
        G.add_node(n.id, label=n.name.replace(layer_prefix + ".", ""), full_name=n.name)
    for n in sub_nodes:
        for succ in graph.successors(n):
            if succ.id in sub_ids:
                G.add_edge(n.id, succ.id)

    # Build latency lookup
    latency_map = {}
    if result:
        for r in result.per_op:
            latency_map[r.node.id] = r.cost.latency_us

    # Layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if _has_graphviz() else _hierarchical_layout(G)

    # Colors and sizes
    colors = []
    labels = {}
    node_sizes = []
    for nid in G.nodes():
        full_name = G.nodes[nid]["full_name"]
        label = G.nodes[nid]["label"]
        cat = _categorize(full_name)
        colors.append(CATEGORY_COLORS.get(cat, "#C0C0C0"))

        lat_str = ""
        if nid in latency_map:
            lat_str = f"\n{latency_map[nid]:.0f}us"
        labels[nid] = label + lat_str

        # Size proportional to latency
        base = 800
        if nid in latency_map:
            base = max(600, min(3000, 600 + latency_map[nid] * 2))
        node_sizes.append(base)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888",
                           arrows=True, arrowsize=15, width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=node_sizes, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold")

    # Legend
    legend_patches = []
    used_cats = set(_categorize(G.nodes[nid]["full_name"]) for nid in G.nodes())
    for cat, color in CATEGORY_COLORS.items():
        if cat in used_cats:
            legend_patches.append(mpatches.Patch(color=color, label=cat))
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_title(f"DeepSeek V3.2 — {layer_prefix} Block Detail", fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return filename


def export_architecture_overview(
    graph: ComputeGraph,
    filename: str = "architecture_overview.png",
    result: SimResult = None,
    figsize: tuple = (24, 16),
):
    """Export a high-level architecture overview — one row per layer, collapsed by category.

    Shows the flow: Embedding → [Layer 0..N] → Norm → LM Head
    Each layer shows its component categories as colored blocks.
    """
    # Collect per-layer stats
    layer_stats = {}  # layer_id -> {category: {latency, flops, count}}
    other_ops = []    # non-layer ops

    if result:
        for r in result.per_op:
            name = r.node.name or ""
            cat = _categorize(name)

            # Extract layer id
            layer_id = None
            if name.startswith("L") and "." in name:
                try:
                    layer_id = int(name.split(".")[0][1:])
                except ValueError:
                    pass

            if layer_id is not None:
                if layer_id not in layer_stats:
                    layer_stats[layer_id] = {}
                if cat not in layer_stats[layer_id]:
                    layer_stats[layer_id][cat] = {"latency": 0, "flops": 0, "count": 0}
                layer_stats[layer_id][cat]["latency"] += r.cost.latency_us
                layer_stats[layer_id][cat]["flops"] += r.cost.flops
                layer_stats[layer_id][cat]["count"] += 1
            else:
                other_ops.append((name, cat, r.cost.latency_us))

    if not layer_stats:
        print("No layer data found in result")
        return

    n_layers = max(layer_stats.keys()) + 1

    # Determine all categories used
    all_cats = set()
    for stats in layer_stats.values():
        all_cats.update(stats.keys())
    cat_order = [c for c in CATEGORY_COLORS if c in all_cats]

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[4, 1],
                              gridspec_kw={"hspace": 0.3})

    # --- Top: Stacked bar chart per layer ---
    ax = axes[0]
    x = np.arange(n_layers)
    bar_width = 0.8
    bottoms = np.zeros(n_layers)

    for cat in cat_order:
        heights = []
        for lid in range(n_layers):
            if lid in layer_stats and cat in layer_stats[lid]:
                heights.append(layer_stats[lid][cat]["latency"] / 1000)  # ms
            else:
                heights.append(0)
        heights = np.array(heights)
        ax.bar(x, heights, bar_width, bottom=bottoms, label=cat,
               color=CATEGORY_COLORS[cat], edgecolor="white", linewidth=0.5)
        bottoms += heights

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("DeepSeek V3.2 671B — Per-Layer Latency Breakdown", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

    # Mark dense vs MoE regions
    n_dense = 3
    ax.axvline(x=n_dense - 0.5, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(n_dense / 2, ax.get_ylim()[1] * 0.95, "Dense", ha="center", fontsize=10,
            color="red", alpha=0.7, fontweight="bold")
    ax.text((n_dense + n_layers) / 2, ax.get_ylim()[1] * 0.95, "MoE", ha="center", fontsize=10,
            color="red", alpha=0.7, fontweight="bold")

    # Show every 5th tick
    tick_step = max(1, n_layers // 15)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([str(i) for i in range(0, n_layers, tick_step)])

    # --- Bottom: Category pie chart ---
    ax2 = axes[1]
    cat_totals = {}
    for stats in layer_stats.values():
        for cat, s in stats.items():
            cat_totals[cat] = cat_totals.get(cat, 0) + s["latency"]

    # Add non-layer ops
    for name, cat, lat in other_ops:
        cat_totals[cat] = cat_totals.get(cat, 0) + lat

    pie_cats = sorted(cat_totals.keys(), key=lambda c: -cat_totals[c])
    pie_vals = [cat_totals[c] / 1000 for c in pie_cats]  # ms
    pie_colors = [CATEGORY_COLORS.get(c, "#C0C0C0") for c in pie_cats]
    pie_labels = [f"{c}\n{v:.1f} ms ({v/sum(pie_vals)*100:.1f}%)" for c, v in zip(pie_cats, pie_vals)]

    ax2.pie(pie_vals, labels=pie_labels, colors=pie_colors, startangle=90,
            textprops={"fontsize": 8}, pctdistance=0.85)
    ax2.set_title("Total Latency by Category", fontsize=13, fontweight="bold")

    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return filename


def export_dataflow_graph(
    graph: ComputeGraph,
    filename: str = "dataflow.png",
    result: SimResult = None,
    max_nodes: int = 50,
    layer_filter: str = None,
    figsize: tuple = (22, 16),
):
    """Export a colored dataflow graph (node-edge diagram).

    For large graphs, use layer_filter (e.g. "L3") or max_nodes to limit scope.
    """
    nodes = graph.nodes
    if layer_filter:
        nodes = [n for n in nodes if n.name and n.name.startswith(layer_filter)]
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    node_ids = {n.id for n in nodes}
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n.id, name=n.name or n.op.op_type.name)
    for n in nodes:
        for succ in graph.successors(n):
            if succ.id in node_ids:
                G.add_edge(n.id, succ.id)

    latency_map = {}
    if result:
        for r in result.per_op:
            latency_map[r.node.id] = r

    pos = _hierarchical_layout(G)

    colors = []
    labels = {}
    sizes = []
    for nid in G.nodes():
        name = G.nodes[nid]["name"]
        cat = _categorize(name)
        colors.append(CATEGORY_COLORS.get(cat, "#C0C0C0"))

        short = name.split(".")[-1] if "." in name else name
        if nid in latency_map:
            r = latency_map[nid]
            short += f"\n{r.cost.latency_us:.0f}us"
            sizes.append(max(600, min(3000, 600 + r.cost.latency_us * 3)))
        else:
            sizes.append(800)
        labels[nid] = short

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#AAAAAA",
                           arrows=True, arrowsize=12, width=1.2, alpha=0.6,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=sizes, edgecolors="white", linewidths=2, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6, font_weight="bold")

    # Legend
    used_cats = set(_categorize(G.nodes[nid]["name"]) for nid in G.nodes())
    legend_patches = [mpatches.Patch(color=c, label=cat)
                      for cat, c in CATEGORY_COLORS.items() if cat in used_cats]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)

    title = "DeepSeek V3.2 — Dataflow Graph"
    if layer_filter:
        title += f" ({layer_filter})"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return filename


def _hierarchical_layout(G: nx.DiGraph) -> dict:
    """Simple hierarchical layout for DAGs using topological generations."""
    if G.number_of_nodes() == 0:
        return {}

    pos = {}
    for gen_idx, gen in enumerate(nx.topological_generations(G)):
        gen = sorted(gen)
        n = len(gen)
        for i, node in enumerate(gen):
            x = (i - n / 2) * 2
            y = -gen_idx * 2
            pos[node] = (x, y)
    return pos


def _has_graphviz() -> bool:
    try:
        import pygraphviz
        return True
    except ImportError:
        return False

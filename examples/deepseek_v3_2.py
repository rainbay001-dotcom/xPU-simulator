"""
DeepSeek V3.2 671B — Manual simulation on GPU and NPU.

Architecture (from config_671B_v3.2.json):
  - 61 transformer layers (3 dense + 58 MoE)
  - dim=7168, inter_dim=18432 (dense FFN), moe_inter_dim=2048
  - 128 attention heads, MLA (Multi-head Latent Attention)
  - 256 routed experts, 8 activated per token, 1 shared expert
  - q_lora_rank=1536, kv_lora_rank=512
  - qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128

We model a single forward pass (prefill) for a given batch_size and seq_len.
"""

import sys
sys.path.insert(0, ".")

from xpu_simulator.core.operator import OpSpec, TensorSpec, OpType, Dtype
from xpu_simulator.core.graph import ComputeGraph
from xpu_simulator.core.evaluator import PerformanceEvaluator
from xpu_simulator.backends.gpu.hardware import A100_80GB, H100_80GB
from xpu_simulator.backends.gpu.cost_model import GPUCostModel
from xpu_simulator.backends.npu.hardware import ASCEND_910B, ASCEND_910C
from xpu_simulator.backends.npu.cost_model import NPUCostModel
from xpu_simulator.utils.profiling import print_timeline


# ---- DeepSeek V3.2 671B Config ----
DIM = 7168
INTER_DIM = 18432        # dense FFN intermediate
MOE_INTER_DIM = 2048     # expert intermediate
N_LAYERS = 61
N_DENSE_LAYERS = 3
N_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
V_HEAD_DIM = 128
N_ROUTED_EXPERTS = 256
N_ACTIVATED_EXPERTS = 8
N_SHARED_EXPERTS = 1
VOCAB_SIZE = 129280


def build_deepseek_graph(batch_size: int, seq_len: int, dtype: Dtype = Dtype.FP16) -> ComputeGraph:
    """Build computation graph for DeepSeek V3.2 forward pass (prefill)."""

    graph = ComputeGraph("DeepSeek-V3.2-671B")
    B, S, D = batch_size, seq_len, DIM

    def t(shape):
        return TensorSpec(shape, dtype)

    def add_matmul(name, M, K, N, prev_node=None):
        op = OpSpec(OpType.MATMUL, [t((M, K)), t((K, N))], [t((M, N))], name=name)
        node = graph.add_node(op, name)
        if prev_node is not None:
            graph.add_edge(prev_node, node)
        return node

    def add_elementwise(name, shape, op_type=OpType.RELU, prev_node=None):
        op = OpSpec(op_type, [t(shape)], [t(shape)], name=name)
        node = graph.add_node(op, name)
        if prev_node is not None:
            graph.add_edge(prev_node, node)
        return node

    def add_norm(name, shape, prev_node=None):
        op = OpSpec(OpType.LAYER_NORM, [t(shape)], [t(shape)], name=name)
        node = graph.add_node(op, name)
        if prev_node is not None:
            graph.add_edge(prev_node, node)
        return node

    def add_softmax(name, shape, prev_node=None):
        op = OpSpec(OpType.SOFTMAX, [t(shape)], [t(shape)], name=name)
        node = graph.add_node(op, name)
        if prev_node is not None:
            graph.add_edge(prev_node, node)
        return node

    tokens = B * S  # flattened token count for matmuls

    # === Embedding ===
    # Embedding lookup is essentially a gather — model as memory-bound
    embed_op = OpSpec(OpType.RESHAPE, [t((B, S)), t((VOCAB_SIZE, D))], [t((B, S, D))], name="embedding")
    prev = graph.add_node(embed_op, "embedding")

    for layer_id in range(N_LAYERS):
        prefix = f"L{layer_id}"
        is_dense = layer_id < N_DENSE_LAYERS

        # === Attention Norm (RMSNorm) ===
        prev = add_norm(f"{prefix}.attn_norm", (tokens, D), prev)

        # === MLA (Multi-head Latent Attention) ===
        # Q path: x -> wq_a [D, Q_LORA_RANK] -> q_norm -> wq_b [Q_LORA_RANK, N_HEADS * QK_HEAD_DIM]
        wq_a = add_matmul(f"{prefix}.wq_a", tokens, D, Q_LORA_RANK, prev)
        q_norm = add_norm(f"{prefix}.q_norm", (tokens, Q_LORA_RANK), wq_a)
        wq_b = add_matmul(f"{prefix}.wq_b", tokens, Q_LORA_RANK, N_HEADS * QK_HEAD_DIM, q_norm)

        # KV path: x -> wkv_a [D, KV_LORA_RANK + QK_ROPE_HEAD_DIM]
        wkv_a = add_matmul(f"{prefix}.wkv_a", tokens, D, KV_LORA_RANK + QK_ROPE_HEAD_DIM, prev)
        kv_norm = add_norm(f"{prefix}.kv_norm", (tokens, KV_LORA_RANK), wkv_a)

        # wkv_b: [KV_LORA_RANK, N_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)]
        wkv_b = add_matmul(f"{prefix}.wkv_b", tokens, KV_LORA_RANK,
                           N_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), kv_norm)

        # RoPE (elementwise)
        rope_q = add_elementwise(f"{prefix}.rope_q", (tokens, N_HEADS * QK_ROPE_HEAD_DIM), OpType.RELU, wq_b)
        rope_k = add_elementwise(f"{prefix}.rope_k", (tokens, QK_ROPE_HEAD_DIM), OpType.RELU, wkv_a)

        # Attention scores: Q @ K^T  [B, N_HEADS, S, S]
        # FLOPs = B * N_HEADS * S * S * QK_HEAD_DIM * 2
        attn_flops = B * N_HEADS * S * S * QK_HEAD_DIM * 2
        attn_bytes = (B * N_HEADS * S * QK_HEAD_DIM * 2 + B * N_HEADS * S * S) * dtype.bytes
        attn_score_op = OpSpec(
            OpType.MATMUL,
            [t((B * N_HEADS, S, QK_HEAD_DIM)), t((B * N_HEADS, QK_HEAD_DIM, S))],
            [t((B * N_HEADS, S, S))],
            name=f"{prefix}.attn_score",
        )
        attn_scores = graph.add_node(attn_score_op, f"{prefix}.attn_score")
        graph.add_edge(rope_q, attn_scores)
        graph.add_edge(rope_k, attn_scores)
        graph.add_edge(wkv_b, attn_scores)

        # Softmax
        attn_softmax = add_softmax(f"{prefix}.attn_softmax", (B * N_HEADS, S, S), attn_scores)

        # Attention output: scores @ V  [B, N_HEADS, S, V_HEAD_DIM]
        attn_v = add_matmul(f"{prefix}.attn_v", B * N_HEADS * S, S, V_HEAD_DIM, attn_softmax)

        # Output projection: wo [N_HEADS * V_HEAD_DIM, D]
        wo = add_matmul(f"{prefix}.wo", tokens, N_HEADS * V_HEAD_DIM, D, attn_v)

        # === FFN Norm ===
        ffn_norm = add_norm(f"{prefix}.ffn_norm", (tokens, D), wo)

        if is_dense:
            # === Dense FFN (SwiGLU): w1, w3 [D, INTER_DIM], w2 [INTER_DIM, D] ===
            w1 = add_matmul(f"{prefix}.ffn.w1", tokens, D, INTER_DIM, ffn_norm)
            w3 = add_matmul(f"{prefix}.ffn.w3", tokens, D, INTER_DIM, ffn_norm)
            silu = add_elementwise(f"{prefix}.ffn.silu", (tokens, INTER_DIM), OpType.GELU, w1)
            mul = add_elementwise(f"{prefix}.ffn.mul", (tokens, INTER_DIM), OpType.ADD, silu)
            graph.add_edge(w3, mul)
            w2 = add_matmul(f"{prefix}.ffn.w2", tokens, INTER_DIM, D, mul)
            prev = w2
        else:
            # === MoE Layer ===
            # Gate: [D, N_ROUTED_EXPERTS]
            gate = add_matmul(f"{prefix}.moe.gate", tokens, D, N_ROUTED_EXPERTS, ffn_norm)
            gate_softmax = add_softmax(f"{prefix}.moe.gate_softmax", (tokens, N_ROUTED_EXPERTS), gate)

            # Activated experts: 8 experts, each with w1,w3 [D, MOE_INTER_DIM], w2 [MOE_INTER_DIM, D]
            # Total expert compute = N_ACTIVATED_EXPERTS * (3 matmuls per expert)
            # We model this as aggregated matmuls since all activated tokens go through experts
            expert_tokens = tokens  # each token goes through 8 experts

            # Expert w1: [expert_tokens, D] x [D, MOE_INTER_DIM] * N_ACTIVATED_EXPERTS
            expert_w1 = add_matmul(f"{prefix}.moe.experts_w1",
                                   expert_tokens * N_ACTIVATED_EXPERTS, D, MOE_INTER_DIM, gate_softmax)
            expert_w3 = add_matmul(f"{prefix}.moe.experts_w3",
                                   expert_tokens * N_ACTIVATED_EXPERTS, D, MOE_INTER_DIM, gate_softmax)
            expert_silu = add_elementwise(f"{prefix}.moe.experts_silu",
                                          (expert_tokens * N_ACTIVATED_EXPERTS, MOE_INTER_DIM), OpType.GELU, expert_w1)
            expert_mul = add_elementwise(f"{prefix}.moe.experts_mul",
                                         (expert_tokens * N_ACTIVATED_EXPERTS, MOE_INTER_DIM), OpType.ADD, expert_silu)
            graph.add_edge(expert_w3, expert_mul)
            expert_w2 = add_matmul(f"{prefix}.moe.experts_w2",
                                   expert_tokens * N_ACTIVATED_EXPERTS, MOE_INTER_DIM, D, expert_mul)

            # Shared expert: MLP with inter_dim = N_SHARED_EXPERTS * MOE_INTER_DIM
            shared_inter = N_SHARED_EXPERTS * MOE_INTER_DIM
            shared_w1 = add_matmul(f"{prefix}.moe.shared_w1", tokens, D, shared_inter, ffn_norm)
            shared_w3 = add_matmul(f"{prefix}.moe.shared_w3", tokens, D, shared_inter, ffn_norm)
            shared_silu = add_elementwise(f"{prefix}.moe.shared_silu",
                                           (tokens, shared_inter), OpType.GELU, shared_w1)
            shared_mul = add_elementwise(f"{prefix}.moe.shared_mul",
                                          (tokens, shared_inter), OpType.ADD, shared_silu)
            graph.add_edge(shared_w3, shared_mul)
            shared_w2 = add_matmul(f"{prefix}.moe.shared_w2", tokens, shared_inter, D, shared_mul)

            # Combine experts + shared
            combine = add_elementwise(f"{prefix}.moe.combine", (tokens, D), OpType.ADD, expert_w2)
            graph.add_edge(shared_w2, combine)
            prev = combine

    # === Final Norm + LM Head ===
    final_norm = add_norm("final_norm", (tokens, D), prev)
    lm_head = add_matmul("lm_head", tokens, D, VOCAB_SIZE, final_norm)

    return graph


def run_simulation(batch_size: int, seq_len: int, dtype: Dtype = Dtype.FP16):
    """Run DeepSeek V3.2 simulation across all devices."""

    print(f"\n{'='*70}")
    print(f"  DeepSeek V3.2 671B Simulation")
    print(f"  Batch={batch_size}, SeqLen={seq_len}, Dtype={dtype.value}")
    print(f"{'='*70}\n")

    graph = build_deepseek_graph(batch_size, seq_len, dtype)
    print(f"Graph: {graph.num_nodes} ops, {graph.num_edges} edges\n")

    devices = [
        ("NVIDIA A100 80GB", GPUCostModel(A100_80GB)),
        ("NVIDIA H100 80GB", GPUCostModel(H100_80GB)),
        ("Ascend 910B",      NPUCostModel(ASCEND_910B)),
        ("Ascend 910C",      NPUCostModel(ASCEND_910C)),
    ]

    results = {}
    for name, cost_model in devices:
        evaluator = PerformanceEvaluator(cost_model)
        result = evaluator.run(graph, overlap=True)
        results[name] = result

        print(f"--- {name} ---")
        print(f"  Total latency:   {result.total_latency_us:,.0f} us ({result.total_latency_us / 1e6:.3f} s)")
        print(f"  Total FLOPs:     {result.total_flops / 1e12:.2f} TFLOPS")
        print(f"  Total memory:    {result.total_bytes / 1e9:.2f} GB")
        print(f"  Compute-bound:   {result.compute_bound_count} ops")
        print(f"  Memory-bound:    {result.memory_bound_count} ops")
        print(f"  Bottleneck:      {result.bottleneck_op.node} ({result.bottleneck_op.cost.latency_us:,.0f} us)")
        print()

    # Comparison table
    print(f"{'='*70}")
    print(f"  Comparison Summary")
    print(f"{'='*70}")
    print(f"  {'Device':<25} {'Latency':>12} {'vs A100':>10}")
    print(f"  {'-'*47}")
    a100_lat = results["NVIDIA A100 80GB"].total_latency_us
    for name, result in results.items():
        lat = result.total_latency_us
        speedup = a100_lat / lat if lat > 0 else 0
        if lat >= 1e6:
            lat_str = f"{lat / 1e6:.3f} s"
        else:
            lat_str = f"{lat / 1e3:.1f} ms"
        print(f"  {name:<25} {lat_str:>12} {speedup:>9.2f}x")
    print()

    # Top 10 most expensive ops on A100
    a100_result = results["NVIDIA A100 80GB"]
    sorted_ops = sorted(a100_result.per_op, key=lambda r: r.cost.latency_us, reverse=True)
    print(f"Top 10 most expensive ops (A100):")
    print(f"  {'Op':<40} {'Latency':>12} {'Bound':>20} {'FLOPs':>14}")
    print(f"  {'-'*86}")
    for r in sorted_ops[:10]:
        name = (r.node.name or r.node.op.op_type.name)[:38]
        lat = f"{r.cost.latency_us:,.0f} us"
        flops = f"{r.cost.flops / 1e9:.1f} GFLOPS"
        print(f"  {name:<40} {lat:>12} {r.cost.bound:>20} {flops:>14}")
    print()

    # Layer type breakdown on A100
    print(f"Layer type breakdown (A100):")
    type_stats = {}
    for r in a100_result.per_op:
        op_name = r.node.name or ""
        if ".attn_score" in op_name or ".attn_v" in op_name or ".attn_softmax" in op_name:
            cat = "Attention (QKV compute)"
        elif ".wq_" in op_name or ".wkv_" in op_name or ".wo" in op_name:
            cat = "Attention (projections)"
        elif ".moe.experts" in op_name:
            cat = "MoE Experts"
        elif ".moe.shared" in op_name:
            cat = "MoE Shared Expert"
        elif ".moe.gate" in op_name:
            cat = "MoE Gate"
        elif ".ffn." in op_name:
            cat = "Dense FFN"
        elif "norm" in op_name:
            cat = "Norms"
        elif "rope" in op_name:
            cat = "RoPE"
        else:
            cat = "Other"

        if cat not in type_stats:
            type_stats[cat] = {"latency_us": 0, "flops": 0, "count": 0}
        type_stats[cat]["latency_us"] += r.cost.latency_us
        type_stats[cat]["flops"] += r.cost.flops
        type_stats[cat]["count"] += r.cost.flops  # using flops as proxy for weight

    total_lat = sum(v["latency_us"] for v in type_stats.values())
    print(f"  {'Category':<30} {'Latency':>12} {'% of Total':>12} {'TFLOPS':>10}")
    print(f"  {'-'*64}")
    for cat, stats in sorted(type_stats.items(), key=lambda x: -x[1]["latency_us"]):
        lat = stats["latency_us"]
        pct = lat / total_lat * 100 if total_lat > 0 else 0
        tflops = stats["flops"] / 1e12
        if lat >= 1e6:
            lat_str = f"{lat / 1e6:.3f} s"
        else:
            lat_str = f"{lat / 1e3:.1f} ms"
        print(f"  {cat:<30} {lat_str:>12} {pct:>10.1f}% {tflops:>10.2f}")
    print()


if __name__ == "__main__":
    # Default: batch=1, seq_len=1024 (typical prefill)
    batch_size = 1
    seq_len = 1024

    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        seq_len = int(sys.argv[2])

    run_simulation(batch_size, seq_len)

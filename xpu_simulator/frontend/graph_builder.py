"""GraphBuilder — manual construction DSL for computation graphs."""
from __future__ import annotations

from ..core.graph import ComputeGraph, Node
from ..core.operator import OpSpec, OpType, TensorSpec, Dtype


class GraphBuilder:
    """Fluent builder for manually constructing computation graphs.

    Provides primitive ops (matmul, elementwise, norm, softmax, embedding)
    and composite blocks (swiglu_mlp, mla_attention, moe_layer) that mirror
    the patterns used in examples/deepseek_v3_2.py.
    """

    def __init__(self, name: str = "graph", dtype: Dtype = Dtype.FP16):
        self.graph = ComputeGraph(name)
        self.dtype = dtype

    def t(self, shape: tuple[int, ...]) -> TensorSpec:
        """Create a TensorSpec with the builder's dtype."""
        return TensorSpec(shape, self.dtype)

    # ------------------------------------------------------------------ #
    # Primitives
    # ------------------------------------------------------------------ #

    def matmul(self, name: str, M: int, K: int, N: int, *deps: Node) -> Node:
        """[M, K] x [K, N] -> [M, N]."""
        op = OpSpec(OpType.MATMUL, [self.t((M, K)), self.t((K, N))],
                    [self.t((M, N))], name=name)
        node = self.graph.add_node(op, name)
        for dep in deps:
            self.graph.add_edge(dep, node)
        return node

    def elementwise(self, name: str, shape: tuple[int, ...],
                    op_type: OpType = OpType.RELU, *deps: Node) -> Node:
        op = OpSpec(op_type, [self.t(shape)], [self.t(shape)], name=name)
        node = self.graph.add_node(op, name)
        for dep in deps:
            self.graph.add_edge(dep, node)
        return node

    def norm(self, name: str, shape: tuple[int, ...], *deps: Node) -> Node:
        op = OpSpec(OpType.LAYER_NORM, [self.t(shape)], [self.t(shape)], name=name)
        node = self.graph.add_node(op, name)
        for dep in deps:
            self.graph.add_edge(dep, node)
        return node

    def softmax(self, name: str, shape: tuple[int, ...], *deps: Node) -> Node:
        op = OpSpec(OpType.SOFTMAX, [self.t(shape)], [self.t(shape)], name=name)
        node = self.graph.add_node(op, name)
        for dep in deps:
            self.graph.add_edge(dep, node)
        return node

    def embedding(self, name: str, vocab: int, dim: int,
                  batch: int, seq_len: int, *deps: Node) -> Node:
        """Embedding lookup: (B, S) gather from (vocab, dim) -> (B, S, dim)."""
        op = OpSpec(OpType.EMBEDDING,
                    [self.t((batch, seq_len)), self.t((vocab, dim))],
                    [self.t((batch, seq_len, dim))],
                    name=name)
        node = self.graph.add_node(op, name)
        for dep in deps:
            self.graph.add_edge(dep, node)
        return node

    # ------------------------------------------------------------------ #
    # Composites
    # ------------------------------------------------------------------ #

    def swiglu_mlp(self, prefix: str, tokens: int, dim: int,
                   inter_dim: int, prev: Node) -> Node:
        """SwiGLU MLP: w1/w3 -> silu -> mul -> w2.

        Replicates the dense FFN block from deepseek_v3_2.py.
        """
        w1 = self.matmul(f"{prefix}.w1", tokens, dim, inter_dim, prev)
        w3 = self.matmul(f"{prefix}.w3", tokens, dim, inter_dim, prev)
        silu = self.elementwise(f"{prefix}.silu",
                                (tokens, inter_dim), OpType.SILU, w1)
        mul = self.elementwise(f"{prefix}.mul",
                               (tokens, inter_dim), OpType.MUL, silu, w3)
        w2 = self.matmul(f"{prefix}.w2", tokens, inter_dim, dim, mul)
        return w2

    def gqa_attention(self, prefix: str, tokens: int, B: int, S: int,
                      dim: int, n_heads: int, n_kv_heads: int,
                      head_dim: int, rope: bool = True,
                      prev: Node = None) -> Node:
        """Grouped Query Attention (covers MHA and MQA as special cases).

        - n_kv_heads == n_heads → MHA
        - n_kv_heads == 1 → MQA
        - else → GQA
        """
        deps = (prev,) if prev else ()

        # Q/K/V projections
        wq = self.matmul(f"{prefix}.wq", tokens, dim, n_heads * head_dim, *deps)
        wk = self.matmul(f"{prefix}.wk", tokens, dim, n_kv_heads * head_dim, *deps)
        wv = self.matmul(f"{prefix}.wv", tokens, dim, n_kv_heads * head_dim, *deps)

        # Optional RoPE
        if rope:
            rope_q = self.elementwise(f"{prefix}.rope_q",
                                      (tokens, n_heads * head_dim),
                                      OpType.ROPE, wq)
            rope_k = self.elementwise(f"{prefix}.rope_k",
                                      (tokens, n_kv_heads * head_dim),
                                      OpType.ROPE, wk)
        else:
            rope_q = wq
            rope_k = wk

        # Attention score: Q @ K^T  (compute uses n_heads — GQA repeats KV)
        attn_score_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, S, head_dim)),
             self.t((B * n_heads, head_dim, S))],
            [self.t((B * n_heads, S, S))],
            name=f"{prefix}.attn_score",
        )
        attn_scores = self.graph.add_node(attn_score_op, f"{prefix}.attn_score")
        self.graph.add_edge(rope_q, attn_scores)
        self.graph.add_edge(rope_k, attn_scores)

        # Softmax
        attn_softmax = self.softmax(f"{prefix}.attn_softmax",
                                    (B * n_heads, S, S), attn_scores)

        # Attention output: scores @ V
        attn_v = self.matmul(f"{prefix}.attn_v", B * n_heads * S, S,
                             head_dim, attn_softmax)

        # Output projection
        wo = self.matmul(f"{prefix}.wo", tokens, n_heads * head_dim, dim,
                         attn_v)
        return wo

    def dense_ffn(self, prefix: str, tokens: int, dim: int,
                  inter_dim: int, activation: OpType = OpType.GELU,
                  prev: Node = None) -> Node:
        """Dense 2-matrix FFN: w1 -> activation -> w2."""
        deps = (prev,) if prev else ()
        w1 = self.matmul(f"{prefix}.w1", tokens, dim, inter_dim, *deps)
        act = self.elementwise(f"{prefix}.act", (tokens, inter_dim),
                               activation, w1)
        w2 = self.matmul(f"{prefix}.w2", tokens, inter_dim, dim, act)
        return w2

    def geglu_mlp(self, prefix: str, tokens: int, dim: int,
                  inter_dim: int, prev: Node) -> Node:
        """GeGLU MLP: w1/w3 -> gelu -> mul -> w2.

        Same structure as SwiGLU but with GELU activation.
        """
        w1 = self.matmul(f"{prefix}.w1", tokens, dim, inter_dim, prev)
        w3 = self.matmul(f"{prefix}.w3", tokens, dim, inter_dim, prev)
        gelu = self.elementwise(f"{prefix}.gelu",
                                (tokens, inter_dim), OpType.GELU, w1)
        mul = self.elementwise(f"{prefix}.mul",
                               (tokens, inter_dim), OpType.MUL, gelu, w3)
        w2 = self.matmul(f"{prefix}.w2", tokens, inter_dim, dim, mul)
        return w2

    def mla_attention(self, prefix: str, tokens: int, B: int, S: int,
                      dim: int, n_heads: int, q_lora_rank: int,
                      kv_lora_rank: int, qk_head_dim: int,
                      qk_rope_head_dim: int, v_head_dim: int,
                      prev: Node) -> Node:
        """Multi-head Latent Attention block.

        Replicates the MLA pattern from deepseek_v3_2.py and returns the
        output-projection node (wo).
        """
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim

        # Q path
        wq_a = self.matmul(f"{prefix}.wq_a", tokens, dim, q_lora_rank, prev)
        q_norm = self.norm(f"{prefix}.q_norm", (tokens, q_lora_rank), wq_a)
        wq_b = self.matmul(f"{prefix}.wq_b", tokens, q_lora_rank,
                           n_heads * qk_head_dim, q_norm)

        # KV path
        wkv_a = self.matmul(f"{prefix}.wkv_a", tokens, dim,
                            kv_lora_rank + qk_rope_head_dim, prev)
        kv_norm = self.norm(f"{prefix}.kv_norm", (tokens, kv_lora_rank), wkv_a)
        wkv_b = self.matmul(f"{prefix}.wkv_b", tokens, kv_lora_rank,
                            n_heads * (qk_nope_head_dim + v_head_dim), kv_norm)

        # RoPE
        rope_q = self.elementwise(f"{prefix}.rope_q",
                                  (tokens, n_heads * qk_rope_head_dim),
                                  OpType.ROPE, wq_b)
        rope_k = self.elementwise(f"{prefix}.rope_k",
                                  (tokens, qk_rope_head_dim),
                                  OpType.ROPE, wkv_a)

        # Attention scores: Q @ K^T
        attn_score_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, S, qk_head_dim)),
             self.t((B * n_heads, qk_head_dim, S))],
            [self.t((B * n_heads, S, S))],
            name=f"{prefix}.attn_score",
        )
        attn_scores = self.graph.add_node(attn_score_op, f"{prefix}.attn_score")
        self.graph.add_edge(rope_q, attn_scores)
        self.graph.add_edge(rope_k, attn_scores)
        self.graph.add_edge(wkv_b, attn_scores)

        # Softmax
        attn_softmax = self.softmax(f"{prefix}.attn_softmax",
                                    (B * n_heads, S, S), attn_scores)

        # Attention output: scores @ V
        attn_v = self.matmul(f"{prefix}.attn_v", B * n_heads * S, S,
                             v_head_dim, attn_softmax)

        # Output projection
        wo = self.matmul(f"{prefix}.wo", tokens, n_heads * v_head_dim, dim,
                         attn_v)
        return wo

    def moe_layer(self, prefix: str, tokens: int, dim: int,
                  inter_dim: int, n_experts: int, n_activated: int,
                  n_shared: int = 0, shared_inter: int = 0,
                  prev: Node = None) -> Node:
        """MoE layer: gate -> routed experts [+ shared expert] -> combine.

        When n_shared=0, returns routed expert output directly (e.g. Mixtral).
        When n_shared>0, adds shared expert and combine node (e.g. DeepSeek).
        """
        deps = (prev,) if prev else ()

        # Gate
        gate = self.matmul(f"{prefix}.gate", tokens, dim, n_experts, *deps)
        gate_softmax = self.softmax(f"{prefix}.gate_softmax",
                                    (tokens, n_experts), gate)

        # Routed experts (aggregated)
        expert_tokens = tokens * n_activated
        expert_w1 = self.matmul(f"{prefix}.experts_w1",
                                expert_tokens, dim, inter_dim, gate_softmax)
        expert_w3 = self.matmul(f"{prefix}.experts_w3",
                                expert_tokens, dim, inter_dim, gate_softmax)
        expert_silu = self.elementwise(f"{prefix}.experts_silu",
                                       (expert_tokens, inter_dim),
                                       OpType.SILU, expert_w1)
        expert_mul = self.elementwise(f"{prefix}.experts_mul",
                                      (expert_tokens, inter_dim),
                                      OpType.MUL, expert_silu, expert_w3)
        expert_w2 = self.matmul(f"{prefix}.experts_w2",
                                expert_tokens, inter_dim, dim, expert_mul)

        if n_shared <= 0:
            return expert_w2

        # Shared expert
        shared_w1 = self.matmul(f"{prefix}.shared_w1", tokens, dim,
                                shared_inter, *deps)
        shared_w3 = self.matmul(f"{prefix}.shared_w3", tokens, dim,
                                shared_inter, *deps)
        shared_silu = self.elementwise(f"{prefix}.shared_silu",
                                       (tokens, shared_inter),
                                       OpType.SILU, shared_w1)
        shared_mul = self.elementwise(f"{prefix}.shared_mul",
                                      (tokens, shared_inter),
                                      OpType.MUL, shared_silu, shared_w3)
        shared_w2 = self.matmul(f"{prefix}.shared_w2", tokens, shared_inter,
                                dim, shared_mul)

        # Combine routed + shared
        combine = self.elementwise(f"{prefix}.combine", (tokens, dim),
                                   OpType.ADD, expert_w2, shared_w2)
        return combine

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self) -> ComputeGraph:
        """Return the constructed graph."""
        return self.graph

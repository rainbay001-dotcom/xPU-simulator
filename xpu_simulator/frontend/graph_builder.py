"""GraphBuilder — manual construction DSL for computation graphs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.graph import ComputeGraph, Node
from ..core.operator import OpSpec, OpType, TensorSpec, Dtype, QuantConfig, Phase
from ..core.parallel import ParallelConfig

if TYPE_CHECKING:
    from .config_normalizer import AttentionPattern


class GraphBuilder:
    """Fluent builder for manually constructing computation graphs.

    Provides primitive ops (matmul, elementwise, norm, softmax, embedding)
    and composite blocks (swiglu_mlp, mla_attention, moe_layer) that mirror
    the patterns used in examples/deepseek_v3_2.py.
    """

    def __init__(self, name: str = "graph", dtype: Dtype = Dtype.FP16,
                 quant: QuantConfig | None = None,
                 parallel: ParallelConfig | None = None,
                 phase: Phase = Phase.PREFILL,
                 kv_seq_len: int = 0):
        self.graph = ComputeGraph(name)
        self.dtype = dtype
        self.quant = quant
        self.parallel = parallel or ParallelConfig()
        self.tp = self.parallel.tp_size
        self.ep = self.parallel.ep_size
        self.phase = phase
        self.kv_seq_len = kv_seq_len

    def t(self, shape: tuple[int, ...], dtype: Dtype | None = None) -> TensorSpec:
        """Create a TensorSpec with the given or builder's dtype."""
        return TensorSpec(shape, dtype or self.dtype)

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

    def linear(self, name: str, M: int, K: int, N: int,
               *deps: Node, quant: QuantConfig | None = None) -> Node:
        """Linear layer with optional quantization.

        Without quant: equivalent to matmul() at builder dtype.
        With quant: uses typed tensors and adds dequant node for W4A8.
        Falls back to builder-level quant if per-call quant not specified.
        """
        q = quant or self.quant
        if q is None:
            return self.matmul(name, M, K, N, *deps)

        w_dtype = q.weight_dtype
        a_dtype = q.activation_dtype

        # For W4A8: dequantize weight INT4 -> INT8 first
        if w_dtype == Dtype.INT4:
            compute_dtype = a_dtype
            dequant_op = OpSpec(OpType.DEQUANT,
                [TensorSpec((K, N), w_dtype)],
                [TensorSpec((K, N), a_dtype)],
                name=f"{name}.dequant")
            dequant_node = self.graph.add_node(dequant_op, f"{name}.dequant")
            for dep in deps:
                self.graph.add_edge(dep, dequant_node)
            mm_deps = (dequant_node,)
        else:
            compute_dtype = w_dtype
            mm_deps = deps

        op = OpSpec(OpType.MATMUL,
            [TensorSpec((M, K), a_dtype), TensorSpec((K, N), compute_dtype)],
            [TensorSpec((M, N), self.dtype)],
            name=name)
        node = self.graph.add_node(op, name)
        for dep in mm_deps:
            self.graph.add_edge(dep, node)
        return node

    def _comm_op(self, name: str, op_type: OpType, shape: tuple[int, ...],
                 n_ranks: int, *deps: Node) -> Node:
        """Insert a communication collective op node."""
        op = OpSpec(op_type, [self.t(shape)], [self.t(shape)],
                    attrs={"n_ranks": n_ranks}, name=name)
        node = self.graph.add_node(op, name)
        for dep in deps:
            self.graph.add_edge(dep, node)
        return node

    def tp_linear_col(self, name: str, M: int, K: int, N: int,
                      *deps: Node, quant: QuantConfig | None = None) -> Node:
        """Column-parallel linear: shards output dim by tp, adds ALL_GATHER."""
        N_shard = N // self.tp
        node = self.linear(name, M, K, N_shard, *deps, quant=quant)
        if self.tp > 1:
            node = self._comm_op(f"{name}.all_gather", OpType.ALL_GATHER,
                                 (M, N_shard), self.tp, node)
        return node

    def tp_linear_row(self, name: str, M: int, K: int, N: int,
                      *deps: Node, quant: QuantConfig | None = None) -> Node:
        """Row-parallel linear: shards input dim by tp, adds ALL_REDUCE."""
        K_shard = K // self.tp
        node = self.linear(name, M, K_shard, N, *deps, quant=quant)
        if self.tp > 1:
            node = self._comm_op(f"{name}.all_reduce", OpType.ALL_REDUCE,
                                 (M, N), self.tp, node)
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

        With TP: w1/w3 are column-parallel, w2 is row-parallel.
        """
        inter_local = inter_dim // self.tp
        w1 = self.linear(f"{prefix}.w1", tokens, dim, inter_local, prev)
        w3 = self.linear(f"{prefix}.w3", tokens, dim, inter_local, prev)
        silu = self.elementwise(f"{prefix}.silu",
                                (tokens, inter_local), OpType.SILU, w1)
        mul = self.elementwise(f"{prefix}.mul",
                               (tokens, inter_local), OpType.MUL, silu, w3)
        w2 = self.tp_linear_row(f"{prefix}.w2", tokens, inter_local, dim, mul)
        return w2

    # ------------------------------------------------------------------ #
    # Scoring strategies (private)
    # ------------------------------------------------------------------ #

    def _dense_scoring(self, prefix: str, B: int, n_heads: int, S: int,
                       qk_head_dim: int, v_head_dim: int,
                       q_node: Node, k_node: Node,
                       extra_deps: tuple[Node, ...] = ()) -> Node:
        """Dense attention scoring → attn_score, attn_softmax, attn_v.

        Prefill: Q=[B*H, S, d], K=[B*H, d, S] → score=[B*H, S, S]
        Decode:  Q=[B*H, 1, d], K=[B*H, d, kv_S] → score=[B*H, 1, kv_S]
        """
        if self.phase == Phase.DECODE:
            q_seq = 1
            kv_S = self.kv_seq_len or S
        else:
            q_seq = S
            kv_S = S

        attn_score_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, q_seq, qk_head_dim)),
             self.t((B * n_heads, qk_head_dim, kv_S))],
            [self.t((B * n_heads, q_seq, kv_S))],
            name=f"{prefix}.attn_score",
        )
        attn_scores = self.graph.add_node(attn_score_op, f"{prefix}.attn_score")
        self.graph.add_edge(q_node, attn_scores)
        self.graph.add_edge(k_node, attn_scores)
        for dep in extra_deps:
            self.graph.add_edge(dep, attn_scores)

        attn_softmax = self.softmax(f"{prefix}.attn_softmax",
                                    (B * n_heads, q_seq, kv_S), attn_scores)

        attn_v_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, q_seq, kv_S)),
             self.t((B * n_heads, kv_S, v_head_dim))],
            [self.t((B * n_heads, q_seq, v_head_dim))],
            name=f"{prefix}.attn_v",
        )
        attn_v = self.graph.add_node(attn_v_op, f"{prefix}.attn_v")
        self.graph.add_edge(attn_softmax, attn_v)
        return attn_v

    def _topk_scoring(self, prefix: str, B: int, n_heads: int, S: int,
                      qk_head_dim: int, v_head_dim: int,
                      q_node: Node, k_node: Node,
                      extra_deps: tuple[Node, ...] = (), *,
                      n_indexer_heads: int, top_k: int, indexer_dim: int,
                      tokens: int, dim: int, prev: Node) -> Node:
        """Top-k sparse scoring with lightning indexer."""
        # Indexer projections
        indexer_q = self.matmul(f"{prefix}.indexer_q", tokens, dim,
                                n_indexer_heads * indexer_dim, prev)
        indexer_k = self.matmul(f"{prefix}.indexer_k", tokens, dim,
                                indexer_dim, prev)

        indexer_score_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_indexer_heads, S, indexer_dim)),
             self.t((B * n_indexer_heads, indexer_dim, S))],
            [self.t((B * n_indexer_heads, S, S))],
            attrs={"fp8": True},
            name=f"{prefix}.indexer_score",
        )
        indexer_score = self.graph.add_node(indexer_score_op,
                                            f"{prefix}.indexer_score")
        self.graph.add_edge(indexer_q, indexer_score)
        self.graph.add_edge(indexer_k, indexer_score)

        indexer_relu = self.elementwise(f"{prefix}.indexer_relu",
                                        (B * n_indexer_heads, S, S),
                                        OpType.RELU, indexer_score)

        top_k_op = OpSpec(
            OpType.TOP_K,
            [self.t((B, S, S))],
            [self.t((B, S, top_k))],
            name=f"{prefix}.top_k",
        )
        top_k_node = self.graph.add_node(top_k_op, f"{prefix}.top_k")
        self.graph.add_edge(indexer_relu, top_k_node)

        # Sparse attention (k instead of S)
        attn_score_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, S, qk_head_dim)),
             self.t((B * n_heads, qk_head_dim, top_k))],
            [self.t((B * n_heads, S, top_k))],
            name=f"{prefix}.attn_score",
        )
        attn_scores = self.graph.add_node(attn_score_op,
                                          f"{prefix}.attn_score")
        self.graph.add_edge(q_node, attn_scores)
        self.graph.add_edge(k_node, attn_scores)
        for dep in extra_deps:
            self.graph.add_edge(dep, attn_scores)
        self.graph.add_edge(top_k_node, attn_scores)

        attn_softmax = self.softmax(f"{prefix}.attn_softmax",
                                    (B * n_heads, S, top_k), attn_scores)

        attn_v_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, S, top_k)),
             self.t((B * n_heads, top_k, v_head_dim))],
            [self.t((B * n_heads, S, v_head_dim))],
            name=f"{prefix}.attn_v",
        )
        attn_v = self.graph.add_node(attn_v_op, f"{prefix}.attn_v")
        self.graph.add_edge(attn_softmax, attn_v)
        return attn_v

    def _sliding_window_scoring(self, prefix: str, B: int, n_heads: int,
                                S: int, qk_head_dim: int, v_head_dim: int,
                                q_node: Node, k_node: Node,
                                extra_deps: tuple[Node, ...] = (), *,
                                window_size: int) -> Node:
        """Sliding window: S x w scoring (decode: 1 x w)."""
        kv_S = self.kv_seq_len or S if self.phase == Phase.DECODE else S
        w = min(window_size, kv_S)
        q_seq = 1 if self.phase == Phase.DECODE else S

        attn_score_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, q_seq, qk_head_dim)),
             self.t((B * n_heads, qk_head_dim, w))],
            [self.t((B * n_heads, q_seq, w))],
            name=f"{prefix}.attn_score",
        )
        attn_scores = self.graph.add_node(attn_score_op, f"{prefix}.attn_score")
        self.graph.add_edge(q_node, attn_scores)
        self.graph.add_edge(k_node, attn_scores)
        for dep in extra_deps:
            self.graph.add_edge(dep, attn_scores)

        attn_softmax = self.softmax(f"{prefix}.attn_softmax",
                                    (B * n_heads, q_seq, w), attn_scores)

        attn_v_op = OpSpec(
            OpType.MATMUL,
            [self.t((B * n_heads, q_seq, w)),
             self.t((B * n_heads, w, v_head_dim))],
            [self.t((B * n_heads, q_seq, v_head_dim))],
            name=f"{prefix}.attn_v",
        )
        attn_v = self.graph.add_node(attn_v_op, f"{prefix}.attn_v")
        self.graph.add_edge(attn_softmax, attn_v)
        return attn_v

    # ------------------------------------------------------------------ #
    # Attention composites
    # ------------------------------------------------------------------ #

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

        # TP: shard heads across devices
        n_heads_local = n_heads // self.tp
        n_kv_heads_local = max(1, n_kv_heads // self.tp)

        # Q/K/V projections (column-parallel: shard output by tp)
        wq = self.linear(f"{prefix}.wq", tokens, dim, n_heads_local * head_dim, *deps)
        wk = self.linear(f"{prefix}.wk", tokens, dim, n_kv_heads_local * head_dim, *deps)
        wv = self.linear(f"{prefix}.wv", tokens, dim, n_kv_heads_local * head_dim, *deps)

        # Optional RoPE
        if rope:
            rope_q = self.elementwise(f"{prefix}.rope_q",
                                      (tokens, n_heads_local * head_dim),
                                      OpType.ROPE, wq)
            rope_k = self.elementwise(f"{prefix}.rope_k",
                                      (tokens, n_kv_heads_local * head_dim),
                                      OpType.ROPE, wk)
        else:
            rope_q = wq
            rope_k = wk

        # Scoring (on local heads only)
        attn_v = self._dense_scoring(prefix, B, n_heads_local, S, head_dim,
                                     head_dim, rope_q, rope_k)

        # Output projection (row-parallel: shard input by tp, all-reduce)
        wo = self.tp_linear_row(f"{prefix}.wo", tokens,
                                n_heads_local * head_dim, dim, attn_v)
        return wo

    def dense_ffn(self, prefix: str, tokens: int, dim: int,
                  inter_dim: int, activation: OpType = OpType.GELU,
                  prev: Node = None) -> Node:
        """Dense 2-matrix FFN: w1 -> activation -> w2."""
        deps = (prev,) if prev else ()
        inter_local = inter_dim // self.tp
        w1 = self.linear(f"{prefix}.w1", tokens, dim, inter_local, *deps)
        act = self.elementwise(f"{prefix}.act", (tokens, inter_local),
                               activation, w1)
        w2 = self.tp_linear_row(f"{prefix}.w2", tokens, inter_local, dim, act)
        return w2

    def geglu_mlp(self, prefix: str, tokens: int, dim: int,
                  inter_dim: int, prev: Node) -> Node:
        """GeGLU MLP: w1/w3 -> gelu -> mul -> w2."""
        inter_local = inter_dim // self.tp
        w1 = self.linear(f"{prefix}.w1", tokens, dim, inter_local, prev)
        w3 = self.linear(f"{prefix}.w3", tokens, dim, inter_local, prev)
        gelu = self.elementwise(f"{prefix}.gelu",
                                (tokens, inter_local), OpType.GELU, w1)
        mul = self.elementwise(f"{prefix}.mul",
                               (tokens, inter_local), OpType.MUL, gelu, w3)
        w2 = self.tp_linear_row(f"{prefix}.w2", tokens, inter_local, dim, mul)
        return w2

    def _mla_projections(self, prefix: str, tokens: int, dim: int,
                         n_heads: int, q_lora_rank: int, kv_lora_rank: int,
                         qk_head_dim: int, qk_rope_head_dim: int,
                         v_head_dim: int, prev: Node):
        """MLA Q/KV projection path. Returns (rope_q, rope_k, wkv_b)."""
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim

        wq_a = self.linear(f"{prefix}.wq_a", tokens, dim, q_lora_rank, prev)
        q_norm = self.norm(f"{prefix}.q_norm", (tokens, q_lora_rank), wq_a)
        wq_b = self.linear(f"{prefix}.wq_b", tokens, q_lora_rank,
                           n_heads * qk_head_dim, q_norm)

        wkv_a = self.linear(f"{prefix}.wkv_a", tokens, dim,
                            kv_lora_rank + qk_rope_head_dim, prev)
        kv_norm = self.norm(f"{prefix}.kv_norm", (tokens, kv_lora_rank), wkv_a)
        wkv_b = self.linear(f"{prefix}.wkv_b", tokens, kv_lora_rank,
                            n_heads * (qk_nope_head_dim + v_head_dim), kv_norm)

        rope_q = self.elementwise(f"{prefix}.rope_q",
                                  (tokens, n_heads * qk_rope_head_dim),
                                  OpType.ROPE, wq_b)
        rope_k = self.elementwise(f"{prefix}.rope_k",
                                  (tokens, qk_rope_head_dim),
                                  OpType.ROPE, wkv_a)
        return rope_q, rope_k, wkv_b

    def mla_attention(self, prefix: str, tokens: int, B: int, S: int,
                      dim: int, n_heads: int, q_lora_rank: int,
                      kv_lora_rank: int, qk_head_dim: int,
                      qk_rope_head_dim: int, v_head_dim: int,
                      prev: Node) -> Node:
        """Multi-head Latent Attention block."""
        rope_q, rope_k, wkv_b = self._mla_projections(
            prefix, tokens, dim, n_heads, q_lora_rank, kv_lora_rank,
            qk_head_dim, qk_rope_head_dim, v_head_dim, prev)

        attn_v = self._dense_scoring(prefix, B, n_heads, S, qk_head_dim,
                                     v_head_dim, rope_q, rope_k,
                                     extra_deps=(wkv_b,))

        wo = self.linear(f"{prefix}.wo", tokens, n_heads * v_head_dim, dim,
                         attn_v)
        return wo

    def dsa_mla_attention(self, prefix: str, tokens: int, B: int, S: int,
                          dim: int, n_heads: int, q_lora_rank: int,
                          kv_lora_rank: int, qk_head_dim: int,
                          qk_rope_head_dim: int, v_head_dim: int,
                          n_indexer_heads: int, dsa_k: int,
                          indexer_dim: int, prev: Node) -> Node:
        """DeepSeek Sparse Attention (DSA) with MLA projections."""
        rope_q, rope_k, wkv_b = self._mla_projections(
            prefix, tokens, dim, n_heads, q_lora_rank, kv_lora_rank,
            qk_head_dim, qk_rope_head_dim, v_head_dim, prev)

        attn_v = self._topk_scoring(
            prefix, B, n_heads, S, qk_head_dim, v_head_dim,
            rope_q, rope_k, extra_deps=(wkv_b,),
            n_indexer_heads=n_indexer_heads, top_k=dsa_k,
            indexer_dim=indexer_dim, tokens=tokens, dim=dim, prev=prev)

        wo = self.linear(f"{prefix}.wo", tokens, n_heads * v_head_dim, dim,
                         attn_v)
        return wo

    def attention(self, prefix: str, tokens: int, B: int, S: int,
                  dim: int, n_heads: int, head_dim: int,
                  pattern: AttentionPattern, *,
                  n_kv_heads: int = None, rope: bool = True,
                  q_lora_rank: int = None, kv_lora_rank: int = None,
                  qk_head_dim: int = None, qk_rope_head_dim: int = None,
                  v_head_dim: int = None,
                  prev: Node = None) -> Node:
        """General attention dispatcher — selects projection and scoring strategy.

        Projection: GQA (when kv_lora_rank is None) or MLA (when set).
        Scoring: determined by pattern.kind ("dense", "top_k", "sliding_window").
        """
        from .config_normalizer import AttentionPattern as AP  # noqa: avoid circular at import time

        is_mla = kv_lora_rank is not None

        # TP: local head counts
        n_heads_local = n_heads // self.tp
        _n_kv_heads = n_kv_heads or n_heads
        n_kv_heads_local = max(1, _n_kv_heads // self.tp)

        if is_mla:
            _qk_hd = qk_head_dim or head_dim
            _qk_rope_hd = qk_rope_head_dim or 0
            _v_hd = v_head_dim or head_dim
            rope_q, rope_k, wkv_b = self._mla_projections(
                prefix, tokens, dim, n_heads_local, q_lora_rank, kv_lora_rank,
                _qk_hd, _qk_rope_hd, _v_hd, prev)
            q_node, k_node = rope_q, rope_k
            score_qk_dim = _qk_hd
            score_v_dim = _v_hd
            extra_deps = (wkv_b,)
            out_dim = n_heads_local * _v_hd
        else:
            # GQA/MHA/MQA projections (column-parallel: shard heads by tp)
            deps = (prev,) if prev else ()
            wq = self.linear(f"{prefix}.wq", tokens, dim, n_heads_local * head_dim, *deps)
            wk = self.linear(f"{prefix}.wk", tokens, dim, n_kv_heads_local * head_dim, *deps)
            wv = self.linear(f"{prefix}.wv", tokens, dim, n_kv_heads_local * head_dim, *deps)
            if rope:
                q_node = self.elementwise(f"{prefix}.rope_q",
                                          (tokens, n_heads_local * head_dim),
                                          OpType.ROPE, wq)
                k_node = self.elementwise(f"{prefix}.rope_k",
                                          (tokens, n_kv_heads_local * head_dim),
                                          OpType.ROPE, wk)
            else:
                q_node, k_node = wq, wk
            score_qk_dim = head_dim
            score_v_dim = head_dim
            extra_deps = ()
            out_dim = n_heads_local * head_dim

        # Dispatch scoring strategy (on local heads)
        if pattern.kind == "top_k":
            attn_v = self._topk_scoring(
                prefix, B, n_heads_local, S, score_qk_dim, score_v_dim,
                q_node, k_node, extra_deps,
                n_indexer_heads=pattern.num_indexer_heads,
                top_k=pattern.top_k,
                indexer_dim=pattern.indexer_dim,
                tokens=tokens, dim=dim, prev=prev)
        elif pattern.kind == "sliding_window":
            attn_v = self._sliding_window_scoring(
                prefix, B, n_heads_local, S, score_qk_dim, score_v_dim,
                q_node, k_node, extra_deps,
                window_size=pattern.window_size)
        else:
            attn_v = self._dense_scoring(
                prefix, B, n_heads_local, S, score_qk_dim, score_v_dim,
                q_node, k_node, extra_deps)

        # Output projection (row-parallel: shard input by tp, all-reduce)
        wo = self.tp_linear_row(f"{prefix}.wo", tokens, out_dim, dim, attn_v)
        return wo

    # ------------------------------------------------------------------ #
    # FFN composites
    # ------------------------------------------------------------------ #

    def moe_layer(self, prefix: str, tokens: int, dim: int,
                  inter_dim: int, n_experts: int, n_activated: int,
                  n_shared: int = 0, shared_inter: int = 0,
                  prev: Node = None) -> Node:
        """MoE layer: gate -> routed experts [+ shared expert] -> combine.

        With EP: experts are sharded across ep_size devices,
        ALL_TO_ALL dispatches tokens to expert owners and collects results.
        """
        deps = (prev,) if prev else ()

        # Gate
        gate = self.matmul(f"{prefix}.gate", tokens, dim, n_experts, *deps)
        gate_softmax = self.softmax(f"{prefix}.gate_softmax",
                                    (tokens, n_experts), gate)

        # EP: dispatch tokens to expert owners
        dispatch_input = gate_softmax
        if self.ep > 1:
            dispatch_input = self._comm_op(
                f"{prefix}.dispatch_a2a", OpType.ALL_TO_ALL,
                (tokens, dim), self.ep, gate_softmax)

        # Routed experts (local fraction with EP)
        expert_tokens = tokens * n_activated // self.ep
        expert_w1 = self.linear(f"{prefix}.experts_w1",
                                expert_tokens, dim, inter_dim, dispatch_input)
        expert_w3 = self.linear(f"{prefix}.experts_w3",
                                expert_tokens, dim, inter_dim, dispatch_input)
        expert_silu = self.elementwise(f"{prefix}.experts_silu",
                                       (expert_tokens, inter_dim),
                                       OpType.SILU, expert_w1)
        expert_mul = self.elementwise(f"{prefix}.experts_mul",
                                      (expert_tokens, inter_dim),
                                      OpType.MUL, expert_silu, expert_w3)
        expert_w2 = self.linear(f"{prefix}.experts_w2",
                                expert_tokens, inter_dim, dim, expert_mul)

        # EP: collect results back
        routed_out = expert_w2
        if self.ep > 1:
            routed_out = self._comm_op(
                f"{prefix}.combine_a2a", OpType.ALL_TO_ALL,
                (tokens * n_activated // self.ep, dim), self.ep, expert_w2)

        if n_shared <= 0:
            return routed_out

        # Shared expert (not sharded by EP, uses TP)
        shared_local = shared_inter // self.tp
        shared_w1 = self.linear(f"{prefix}.shared_w1", tokens, dim,
                                shared_local, *deps)
        shared_w3 = self.linear(f"{prefix}.shared_w3", tokens, dim,
                                shared_local, *deps)
        shared_silu = self.elementwise(f"{prefix}.shared_silu",
                                       (tokens, shared_local),
                                       OpType.SILU, shared_w1)
        shared_mul = self.elementwise(f"{prefix}.shared_mul",
                                      (tokens, shared_local),
                                      OpType.MUL, shared_silu, shared_w3)
        shared_w2 = self.tp_linear_row(f"{prefix}.shared_w2", tokens,
                                        shared_local, dim, shared_mul)

        # Combine routed + shared
        combine = self.elementwise(f"{prefix}.combine", (tokens, dim),
                                   OpType.ADD, routed_out, shared_w2)
        return combine

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self) -> ComputeGraph:
        """Return the constructed graph."""
        return self.graph

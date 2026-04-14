# LLM Serving Parallelism Analysis

A comprehensive survey of parallelism and disaggregation methods for LLM inference
serving, covering implemented features in vLLM/SGLang and recent research proposals.

*Created: 2026-04-13*

---

## 1. Traditional Parallelism (Within a Single Model)

### 1.1 Tensor Parallelism (TP)

Split weight matrices horizontally across devices. Each device holds a shard of every
layer; an all-reduce synchronizes activations after each layer.

- **Best for:** decode phase (low latency per token)
- **Limitation:** all-reduce communication scales with TP degree
- **vLLM:** `--tensor-parallel-size N` (stable, since v0.1)
- **SGLang:** `--tp N` (stable)

### 1.2 Pipeline Parallelism (PP)

Split layers vertically across devices. Device 0 runs layers 0-15, device 1 runs
layers 16-31, etc. Uses micro-batching to keep all stages busy.

- **Best for:** prefill phase (less all-reduce overhead)
- **Limitation:** pipeline bubbles, higher latency for decode
- **vLLM:** `--pipeline-parallel-size N` (stable, since v0.4)
- **SGLang:** `--pp N` (experimental, since v0.5)

### 1.3 Data Parallelism (DP)

Replicate the full model on each device group; route different requests to different
replicas. No communication during inference.

- **Best for:** throughput scaling with independent requests
- **vLLM:** `--data-parallel-size N` (stable, since v0.8)
- **SGLang:** `--dp N` (stable)

### 1.4 Expert Parallelism (EP)

For Mixture-of-Experts (MoE) models: distribute experts across devices. Each device
holds a subset of experts; all-to-all routing sends tokens to the right device.

- **Best for:** MoE models (Mixtral, DeepSeek V3, Qwen3-MoE)
- **vLLM:** `--expert-parallel-size N` (stable, since v0.7)
- **SGLang:** `--ep N` (stable, since v0.4)

### 1.5 Sequence Parallelism (SP)

Split the sequence dimension across devices. Related to Context Parallelism.

- **vLLM:** not explicitly supported (handled implicitly by TP)
- **SGLang:** not explicitly supported

### 1.6 Context Parallelism (CP) / Ring Attention

Partition input along the sequence dimension across devices using ring attention
(pass-KV or pass-Q variants). Critical for million-token inference.

- **Meta:** achieved 1M-token Llama3-405B inference in 77s on 128 H100s (93% efficiency)
- **vLLM:** `--context-parallel-size N` (stable since v0.17, decode CP in MRV2)
  - Prefill: both Q and KV sharded across GPUs
  - Decode: KV pairs distributed round-robin across CP ranks
  - Proposed variants: Prefill CP ([#25749](https://github.com/vllm-project/vllm/issues/25749)),
    Context Pipeline Parallelism ([#28912](https://github.com/vllm-project/vllm/issues/28912)),
    Dynamic CP ([#29295](https://github.com/vllm-project/vllm/issues/29295)),
    Sharded CP ([#30055](https://github.com/vllm-project/vllm/issues/30055))
- **SGLang:** `--attn-cp-size N` (stable for prefill; decode CP proposed in [#12196](https://github.com/sgl-project/sglang/issues/12196))

### 1.7 Hybrid Combinations

| Combination | Use case | vLLM | SGLang |
|---|---|---|---|
| TP + PP | Large models across many devices | `--tp M --pp N` | `--tp M --pp N` |
| DP + TP | Scale throughput with multi-GPU replicas | `--dp M --tp N` | `--dp M --tp N` |
| TP + EP | Large MoE models | `--tp M --ep N` | `--tp M --ep N` |
| DP + EP | DP for attention, EP for FFN | — | `--enable-dp-attention` |
| DP + TP + EP | Large MoE multi-node | `--dp M --tp N --enable-expert-parallel` | `--dp M --tp N --ep K` |
| CP + TP | Long context + large models | `--cp N --tp M` | `--attn-cp-size N --tp M` |

### 1.8 SGLang-Specific: DP-Attention

SGLang's `--enable-dp-attention` uses Data Parallelism for attention computation
(each replica holds its own KV cache) while using Expert Parallelism for FFN.
This is a form of lightweight AF disaggregation within the same device group.

---

## 2. Disaggregation (Separate Phases/Components)

### 2.1 Prefill-Decode (PD) Disaggregation

Run prefill (compute-bound) and decode (memory-bound) on separate instances. A
connector transfers KV caches between them.

**Motivation:** prefill and decode have fundamentally different resource profiles.
Colocation causes mutual interference.

**Architecture:**
```
Client → [Router] → Prefill Instance (KV producer)
                         ↓ KV cache transfer
                     Decode Instance (KV consumer) → tokens
```

**vLLM (since v0.6, experimental):**
```bash
# Prefill instance
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector",
    "kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'

# Decode instance
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector",
    "kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

**SGLang (since v0.4, experimental):**
```bash
# Prefill
python -m sglang.launch_server --disaggregation-mode prefill ...
# Decode
python -m sglang.launch_server --disaggregation-mode decode ...
```

**KV Transfer Connectors (vLLM):**

| Connector | Transport | Status |
|---|---|---|
| `P2pNcclConnector` | NCCL peer-to-peer | Stable (v0.7+) |
| `MooncakeConnectorV1` | Mooncake TransferEngine | v0.9+ |
| `NIXLConnector` | NVIDIA NIXL | v0.12+ |
| `LMCacheConnectorV1` | RDMA/TCP distributed cache | v0.11+ |
| `LLMDataDistCMgrConnector` | Huawei HCCL (Ascend) | vllm-ascend v0.11+ |

**Key results:**
- DistServe (OSDI'24): 4.48x goodput, 10.2x tighter SLO
- Splitwise (ISCA'24 Best Paper): 1.4x throughput at 20% lower cost

### 2.2 Attention-FFN (AF) Disaggregation

Run attention layers on one set of devices and FFN/MLP layers on another. Exploits
the fundamentally different resource profiles of attention (memory-bound, stateful KV
cache) vs FFN (compute-bound, stateless).

**Architecture:**
```
For each transformer layer:
  Attention Card → activations → FFN Card → activations → next Attention Card
```

**Challenge:** activations must transfer between cards for every single layer.
Requires sub-millisecond transfers (~272us for 20 tok/s at 50ms SLA).

**Production systems:**

| System | Org | Scale | Result |
|---|---|---|---|
| xDeepServe | Huawei | 768 Ascend NPUs (CloudMatrix384) | DeepSeek-V3 in production |
| Step-3 | StepFun | 321B MoE | 4,039 tok/s/GPU |
| MegaScale-Infer | ByteDance | SIGCOMM'25 | 1.9x throughput |

**Status:**
- vLLM: RFC stage ([#21644](https://github.com/vllm-project/vllm/issues/21644),
  [#22799](https://github.com/vllm-project/vllm/issues/22799),
  [#27584](https://github.com/vllm-project/vllm/issues/27584))
- SGLang: RFC stage ([#9347](https://github.com/sgl-project/sglang/issues/9347))
- Not available in any open-source framework yet

### 2.3 Encoder-Prefill-Decode (EPD) Disaggregation

For multimodal models: three-tier separation of encoder (ViT), prefill, and decode.

- **SGLang:** production-ready (Jan 2026), vision embedding caching, ZMQ/Mooncake transfer
- **vLLM:** experimental via vllm-ascend ([#4115](https://github.com/vllm-project/vllm-ascend/issues/4115))
- **Use case:** multi-image requests where ViT encoding is expensive

### 2.4 KV Cache Disaggregation

Decouple KV cache storage from the compute devices. KV caches live in a distributed
memory pool (host DRAM, NVMe, remote memory).

- **Infinite-LLM (OSDI'24):** DistAttention — distributed KV cache pool
- **LMCache:** 3-tier (L1 HBM, L2 host DRAM/NVMe, L3 distributed storage)
- **Mooncake:** KV-cache-centric architecture, 100B+ tokens/day at Moonshot AI
- **CacheGen (SIGCOMM'24):** KV cache compression and streaming

---

## 3. Recent Research Systems (2024-2026)

### 3.1 Phase-Aware Systems

| System | Venue | Key Innovation |
|---|---|---|
| **DistServe** | OSDI'24 | First PD disaggregation; co-optimize prefill/decode resources |
| **Splitwise** | ISCA'24 | Heterogeneous HW for prefill (high-end) vs decode (memory-opt) |
| **POD-Attention** | ASPLOS'25 | First GPU kernel for full prefill-decode overlap on same SM |
| **DuetServe** | Nov'25 | Disaggregation-level isolation within single GPU via SM partitioning |
| **Tetris** | 2024 | Phase-aware scheduling for PD |

### 3.2 Heterogeneous Hardware

| System | Venue | Key Innovation |
|---|---|---|
| **Helix** | ASPLOS'25 | Max-flow formulation for geo-distributed heterogeneous GPUs |
| **Hetis** | SC'25 | Head-granularity attention parallelism for mixed GPU clusters |
| **HeteGen** | MLSys'24 | CPU+GPU co-inference on resource-constrained devices |
| **Ghidorah** | May'25 | Hetero-core speculative decoding on edge (Jetson NX) |
| **ThunderServe** | 2025 | GPU pool management for cost-efficient heterogeneous serving |

### 3.3 Elastic/Dynamic Systems

| System | Venue | Key Innovation |
|---|---|---|
| **LoongServe** | SOSP'24 | Elastic Sequence Parallelism — dynamic SP degree per request |
| **Llumnix** | OSDI'24 | Live migration of requests across GPUs for rescheduling |
| **ElasticMM** | 2025 | Elastic multimodal parallelism |
| **Medha** | 2024 | Adaptive prefill chunking for million-token serving |
| **ServerlessLLM** | OSDI'24 | Ultra-fast checkpoint loading, GPU multiplexing |

### 3.4 SGLang Advanced EP Features

SGLang has particularly rich Expert Parallelism support:

- `--moe-a2a-backend`: deepep, mooncake, nixl, mori, flashinfer, ascend_fuseep
- `--enable-two-batch-overlap`: splits into micro-batches, interleaves attention with dispatch
- `--enable-single-batch-overlap`: dispatcher-hook overlap within single batch
- `--enable-eplb`: Expert Parallelism Load Balancer from DeepSeek
- Elastic EP for partial failure tolerance (graceful degradation when GPUs fail)

### 3.5 Speculative Decoding

| System | Key Innovation |
|---|---|
| **Mirror Speculative Decoding** | Early-exit parallel rollouts on CPU+GPU |
| **TurboSpec** | Closed-loop control system, goodput as metric |
| **Smurfs** | Collective voting from multiple small speculative models |
| **Distributed Spec Decoding** | Draft on separate device from verifier |
| vLLM support | `--speculative-model <draft_model> --num-speculative-tokens N` |

### 3.5 Multi-Token Prediction (MTP)

Predict multiple adjacent tokens per forward pass. Up to 3.6x speedup for
self-speculative decoding, 5x for code/math.

- DeepSeek-V3: native MTP heads
- Qwen3-Next-80B-A3B: MTP + hybrid attention
- L-MTP: leap mechanism predicting non-adjacent future tokens
- vLLM: MTP support for DeepSeek V3 via vllm-ascend

### 3.7 vLLM Proposed RFCs (Not Yet Implemented)

| RFC | Title | Status |
|---|---|---|
| [#20323](https://github.com/vllm-project/vllm/issues/20323) | Elastic Expert Parallelism | In progress |
| [#27584](https://github.com/vllm-project/vllm/issues/27584) | Elastic Attn-FFN Disaggregation | Active dev |
| [#28912](https://github.com/vllm-project/vllm/issues/28912) | Context Pipeline Parallelism (CPP) | Proposed |
| [#29295](https://github.com/vllm-project/vllm/issues/29295) | Dynamic Context Parallelism | Proposed |
| [#30055](https://github.com/vllm-project/vllm/issues/30055) | Sharded Context Parallelism | Proposed |
| [#33980](https://github.com/vllm-project/vllm/issues/33980) | Sparse Attention KV Cache Offloading | Proposed |
| [#34407](https://github.com/vllm-project/vllm/issues/34407) | Disaggregated Frontend | Proposed |
| [#27774](https://github.com/vllm-project/vllm/issues/27774) | Fault-Tolerant Expert Parallelism | Proposed |

### 3.8 SGLang Proposed Features

| RFC | Title | Status |
|---|---|---|
| [#12196](https://github.com/sgl-project/sglang/issues/12196) | Decode Context Parallel | In design |
| [#9347](https://github.com/sgl-project/sglang/issues/9347) | FFN-Attention Disaggregation | Experimental |
| [#19487](https://github.com/sgl-project/sglang/issues/19487) | Topology-Aware EPLB | Proposed (~35% traffic reduction) |
| [#12780](https://github.com/sgl-project/sglang/issues/12780) | Helix Parallelism (TP+CP+A2A) | Q1 2026 roadmap |

### 3.9 Ascend NPU Specific

| System | Description |
|---|---|
| **xDeepServe** | Transformerless decomposition on CloudMatrix384; attention/FFN/MoE as modular units |
| **CloudMatrix-Infer** | Phase-specific NPU allocation, EP32 for prefill |
| **DeepFlow/FlowServe** | Microkernel-inspired serverless LLM serving on NPU |
| **Pangu Ultra MoE (718B)** | 5D parallelism: Pipeline + Tensor + Expert + Data + Context |
| **Multi-core NPU study** | Dataflow + discrete memory designs, SPMD parallelism, 1.3-6x vs SOTA |

### 3.7 Infrastructure/Orchestration

| System | Description |
|---|---|
| **NVIDIA Dynamo** | Orchestration layer above inference engines; dynamic routing, multi-tier KV caching |
| **Mooncake** | KV-cache-centric disaggregated architecture |
| **StepMesh** | Communication library for AF disaggregation (open-sourced by StepFun) |
| **XCCL** | Huawei's collective communication library for CloudMatrix peer-to-peer |

---

## 4. Taxonomy Summary

```
Parallelism in LLM Serving
├── Within-model parallelism
│   ├── Tensor Parallelism (TP)      — split weights per layer
│   ├── Pipeline Parallelism (PP)    — split layers across stages
│   ├── Data Parallelism (DP)        — replicate model, split requests
│   ├── Expert Parallelism (EP)      — split MoE experts
│   ├── Sequence Parallelism (SP)    — split sequence dimension
│   ├── Context Parallelism (CP)     — ring attention for long context
│   └── Head Parallelism             — split attention heads (research)
│
├── Phase disaggregation
│   ├── Prefill-Decode (PD)          — separate P and D instances
│   ├── Attention-FFN (AF)           — separate A and F on different HW
│   └── KV Cache disaggregation      — decouple KV storage from compute
│
├── Speculative methods
│   ├── Speculative Decoding          — draft + verify
│   └── Multi-Token Prediction (MTP)  — predict N tokens per step
│
├── Dynamic/Elastic
│   ├── Elastic SP (LoongServe)       — change SP degree at runtime
│   ├── Live migration (Llumnix)      — move requests between devices
│   └── Adaptive chunking (Medha)     — dynamic prefill chunk sizes
│
└── Heterogeneous
    ├── Mixed GPU types (Helix)        — different GPUs for different parts
    ├── CPU+GPU co-inference (HeteGen) — use CPU alongside GPU
    └── NPU-specific (xDeepServe)      — Ascend CloudMatrix384
```

---

## 5. Practical Considerations for Our Setup

**Hardware:** 2x Ascend 910C (64GB HBM each), HCCS interconnect, vllm-ascend v0.7.3

| Method | Feasible? | Notes |
|---|---|---|
| TP=2 | **Yes (running)** | Qwen2.5-32B at 534 tok/s, 183ms TTFT |
| PP=2 | Possible | Less tested on Ascend with vLLM v0.7.3 |
| DP=2 | Yes (small models) | 2 replicas of 7B model, one per card |
| EP=2 | Yes | Mixtral-8x7B, 4 experts per card |
| PD disagg | Yes (small models) | Each card needs full model; 7B model only |
| AF disagg | No | Not implemented in any open-source framework |
| Spec decode | Possible | Need draft model + main model on same card |
| CP | No | Not in vLLM; SGLang only |

---

## 6. Benchmark Results (Our Setup)

**Model:** Qwen2.5-32B-Instruct, TP=2, chips 4-5, vLLM 0.7.3 + vllm-ascend 0.7.3

### Throughput (50 requests, rate=10 req/s, input=128 tok, output=128 tok)

| Metric | Value |
|---|---|
| Request throughput | 4.26 req/s |
| Output token throughput | 534.6 tok/s |
| Total token throughput | 1080.0 tok/s |

### Latency

| Metric | Mean | P50 | P90 | P99 |
|---|---|---|---|---|
| TTFT | 183 ms | 181 ms | 239 ms | 260 ms |
| TPOT | 62.5 ms | 63.7 ms | 68.9 ms | 70.8 ms |
| ITL | 62.5 ms | 51.6 ms | 74.9 ms | 224 ms |

### Streaming (single request)

| Prompt | TTFT | TPOT |
|---|---|---|
| Short | 106 ms | 38 ms |
| Medium | 97 ms | 43 ms |
| Long | 101 ms | 43 ms |

---

## References

### Production Systems
- DistServe: https://haoailab.com/blogs/distserve/ (OSDI'24)
- Splitwise: ISCA'24 Best Paper
- Mooncake: https://arxiv.org/abs/2407.00079
- xDeepServe: https://arxiv.org/abs/2508.02520
- MegaScale-Infer: https://arxiv.org/abs/2504.02263 (SIGCOMM'25)
- Step-3: https://arxiv.org/abs/2507.19427

### Research
- Helix: ASPLOS'25
- LoongServe: SOSP'24
- Llumnix: OSDI'24
- POD-Attention: ASPLOS'25
- Infinite-LLM: OSDI'24
- CacheGen: SIGCOMM'24
- HeteGen: MLSys'24
- Hetis: SC'25
- Medha: https://arxiv.org/abs/2409.17264

### Frameworks
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- NVIDIA Dynamo: https://developer.nvidia.com/dynamo
- vllm-ascend: https://github.com/vllm-project/vllm-ascend

### Surveys
- A Survey of Efficient LLM Inference Serving (2025)
- From Attention to Disaggregation: https://arxiv.org/abs/2511.07422
- Theoretically Optimal AF Ratios: https://arxiv.org/abs/2601.21351

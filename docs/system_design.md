# xPU-Simulator System Design Document

## 1. Overview

xPU-Simulator is an analytical performance simulator for LLM inference on heterogeneous accelerators. Given a model architecture, it builds a computation graph, estimates per-operator costs using device-specific roofline models, applies kernel fusion, and reports end-to-end latency.

**Core value proposition**: Predict inference latency across NVIDIA GPUs and Huawei Ascend NPUs without running on actual hardware, enabling rapid architecture exploration and deployment planning.

### Key Metrics

| Metric | Value |
|--------|-------|
| Source lines | ~8,000 (library) + ~3,900 (tests) + ~1,100 (examples) |
| Python files | 45 (library) + 17 (tests) + 3 (examples) |
| Test count | 161 tests, all passing |
| Supported hardware | 4 devices (A100, H100, 910B, 910C) |
| Supported architectures | 8 LLM families via ConfigExtractor |
| Graph extractors | 7 methods |

---

## 2. Architecture

```
                                 User Interface
                    ┌────────────────────────────────────────┐
                    │   CLI (cli.py)   │   Python API        │
                    │   HTML Report    │   Examples           │
                    └────────┬─────────┴──────────┬──────────┘
                             │                    │
                    ┌────────▼────────────────────▼──────────┐
                    │              Frontend Layer              │
                    │                                         │
                    │  ConfigExtractor    DispatchExtractor    │
                    │  GraphBuilder       TorchGraphExtractor  │
                    │  ExportExtractor    ONNXExtractor        │
                    │  ProfilerExtractor                      │
                    └────────────────────┬────────────────────┘
                                        │
                              ComputeGraph (DAG)
                                        │
                    ┌───────────────────▼────────────────────┐
                    │              Core Layer                  │
                    │                                         │
                    │  FusionPass ──► Fused ComputeGraph      │
                    │  PerformanceEvaluator (ASAP scheduler)  │
                    │  CommAwareCostModel (TP/EP)              │
                    └────────┬─────────┬──────────┬──────────┘
                             │         │          │
                    ┌────────▼──┐ ┌───▼────┐ ┌──▼──────────┐
                    │ GPU       │ │ NPU    │ │ Serving      │
                    │ Backend   │ │Backend │ │ Simulator    │
                    │ (A100,    │ │(910B,  │ │ (batching,   │
                    │  H100)    │ │ 910C)  │ │  KV cache)   │
                    └───────────┘ └────────┘ └──────────────┘
```

### Design Principles

1. **Separation of graph and hardware**: The computation graph is hardware-agnostic. Cost models are pluggable per device.
2. **Multiple extraction paths**: The same model can be analyzed via config-driven construction (ConfigExtractor), runtime interception (DispatchExtractor), or static tracing (FX/export).
3. **Composable transforms**: Fusion rules, TP sharding, and quantization are graph-to-graph transforms applied before evaluation.
4. **Analytical over empirical**: Default cost estimates use roofline analysis. Empirical calibration is optional via ProfilingDB.

---

## 3. Module Design

### 3.1 Core (`xpu_simulator/core/`)

The core module is hardware-agnostic and defines the fundamental data structures and algorithms.

#### 3.1.1 Operator Model (`operator.py`)

```
OpType (Enum)           TensorSpec                  OpSpec
├── MATMUL              ├── shape: tuple[int,...]    ├── op_type: OpType
├── CONV2D              ├── dtype: Dtype             ├── inputs: list[TensorSpec]
├── LAYER_NORM          └── size_bytes (property)    ├── outputs: list[TensorSpec]
├── SOFTMAX                                          ├── attrs: dict
├── SILU / GELU / RELU  Dtype (Enum)                ├── flops (property)
├── ADD / MUL            ├── FP32 (4B)              ├── memory_bytes (property)
├── ROPE                 ├── FP16/BF16 (2B)         └── arithmetic_intensity
├── EMBEDDING            ├── INT8/FP8 (1B)
├── ALL_REDUCE/GATHER    └── INT4 (0.5B)
├── DEQUANT
└── UNKNOWN
```

**FLOP calculation** is type-dispatched within `OpSpec.flops`:
- MATMUL: `2 * M * K * N` (batched: `2 * B * M * K * N`)
- Elementwise (ADD, MUL, SILU, GELU, RELU): `numel(output)`
- Reduction (LAYER_NORM, SOFTMAX): `5 * numel(input)`
- ROPE: `6 * numel(input)`
- Communication (ALL_REDUCE): `numel(input)` (reduction component)
- DEQUANT: `2 * numel(input)` (scale + zero-point)

**Quantization** is modeled via `QuantConfig(weight_dtype, activation_dtype, group_size)` which changes tensor dtypes in the graph, affecting memory traffic and selecting the correct hardware peak (e.g., INT8 peak on A100 = 624 TOPS vs FP16 = 312 TFLOPS).

#### 3.1.2 Computation Graph (`graph.py`)

```python
class ComputeGraph:
    """DAG of operations, backed by networkx.DiGraph."""
    _graph: nx.DiGraph   # nodes are Node objects, edges are data dependencies
    _next_id: int         # monotonic node ID counter

class Node:
    id: int               # unique within graph
    op: OpSpec            # operation specification
    name: Optional[str]   # human-readable (e.g., "L3.attn_score")
```

The graph is immutable-by-convention after construction. Transforms (fusion, TP sharding) produce new graphs rather than mutating in place.

**Edge semantics**: An edge `(A, B)` means B consumes an output tensor of A. The edge may carry an optional `TensorSpec` for the transferred tensor.

#### 3.1.3 Cost Model Hierarchy (`cost_model.py`, `hardware.py`)

```
CostModel (ABC)
├── RooflineCostModel          # basic max(compute_time, memory_time)
├── GPUCostModel               # Tensor Core vs CUDA Core, kernel launch overhead
├── NPUCostModel               # CA-style tiled pipeline (MTE→CUBE→VECTOR)
├── CalibratedCostModel        # measured latency on hit, analytical fallback
└── CommAwareCostModel          # wraps base model + handles collective ops
```

Each `CostModel.estimate(op: OpSpec) -> OpCost` returns:

```python
@dataclass
class OpCost:
    compute_us: float       # time at peak compute throughput
    memory_us: float        # time at peak memory bandwidth
    latency_us: float       # max(compute_us, memory_us) + overhead
    bound: str              # "compute" | "memory" | "communication"
    flops: int
    bytes_accessed: int
    utilization: float      # actual / peak throughput
```

**GPU cost model** adds:
- Tensor Core detection (MATMUL uses TC peak, others use CUDA core peak)
- Kernel launch overhead (5 us default)
- L2 cache efficiency factor

**NPU cost model** adds:
- Tiled pipeline simulation: MTE (memory transfer) → CUBE (matrix compute) → VECTOR (element-wise)
- Double-buffering (pipeline depth = 2)
- L2 reuse within tiles
- Tile alignment to hardware constraints (e.g., 16-byte alignment)
- Multi-core parallelism across AI cores

#### 3.1.4 Performance Evaluator (`evaluator.py`)

The evaluator takes a `ComputeGraph` and a `CostModel` and produces a `SimResult`:

```
Sequential mode: sum of all op latencies
Overlap mode:    ASAP scheduling with resource contention
```

**ASAP scheduler** with resource tracking:
```python
resource_busy = {"cube": 0.0, "vector": 0.0, "shared": 0.0, "comm": 0.0}
```

For each node in topological order:
1. `ready_time = max(end_time of all predecessors)`
2. `start_time = max(ready_time, resource_busy[resource_type])`
3. `end_time = start_time + op_cost.latency_us`
4. `resource_busy[resource_type] = end_time`

This models:
- Compute-compute serialization (same resource)
- Compute-memory overlap (different resources)
- Communication-compute overlap (comm resource is independent)

**SimResult** provides:
- `total_latency_us`, `total_flops`, `total_bytes`
- `compute_bound_count`, `memory_bound_count`
- `bottleneck_op`, `per_op` (list of `OpResult`)
- `ttft_ms` (prefill latency), `tpot_ms` (decode per-token latency)

#### 3.1.5 Kernel Fusion (`fusion.py`)

Fusion eliminates intermediate memory traffic between ops that can run as a single kernel.

```
FusionRule (ABC)
├── match(graph, node) -> Optional[list[Node]]   # pattern match
└── fuse(nodes) -> OpSpec                          # produce fused op

FusionPass
└── apply(graph) -> (fused_graph, FusionResult)   # match-then-rewrite
```

**Two-phase algorithm**:
1. **Match phase**: For each rule (in priority order), scan nodes in topo order. Claim matched nodes (no overlapping claims).
2. **Rewrite phase**: Build new graph. Fused nodes become single replacement nodes. Unfused nodes copied. Edges rebuilt, skipping internal edges within fused groups.

**Fusion rule sets**:

| Rule Set | Target | Rules |
|----------|--------|-------|
| `GPU_FUSION_RULES` | ConfigExtractor graphs | FlashAttention, SwiGLU, MatMulEpilogue, ElementwiseChain |
| `NPU_FUSION_RULES` | ConfigExtractor + NPU | Above + NPUFormatFusion |
| `DISPATCH_FUSION_RULES` | DispatchExtractor graphs | RMSNorm, RoPE, ResidualAddRMSNorm, DispatchFlashAttention, GroupedMatMulSwiGLU, SwiGLU, MatMulEpilogue, ElementwiseChain |
| `DISPATCH_NPU_FUSION_RULES` | DispatchExtractor + NPU | Above + NPUFormatFusion |

**Dispatch-level fusion rules** (inspired by Huawei msmodeling):

| Rule | Pattern | Result | DeepSeek V3 Matches |
|------|---------|--------|---------------------|
| `RMSNormFusion` | pow→mean→add→rsqrt→mul→mul (6-7 ops) | 1 LAYER_NORM | 245 |
| `ResidualAddRMSNormFusion` | ADD + fused RMSNorm | 1 LAYER_NORM | dependent |
| `RoPEFusion` | cos→sin→neg→mul→add chain | 1 ROPE | variable |
| `DispatchFlashAttentionFusion` | BMM→(scale/mask)→Softmax→BMM | 1 fused MATMUL | 61 |
| `GroupedMatMulSwiGLUFusion` | grouped_mm→SILU→MUL | 1 fused MATMUL | 58 |

#### 3.1.6 Communication Model (`communication.py`, `parallel.py`)

```python
@dataclass
class ParallelConfig:
    tp_size: int = 1    # Tensor Parallelism
    dp_size: int = 1    # Data Parallelism
    ep_size: int = 1    # Expert Parallelism

@dataclass
class InterconnectSpec:
    name: str            # "NVLink" | "HCCS"
    bandwidth_GBs: float # unidirectional bandwidth
    latency_us: float    # per-hop latency
```

Communication cost uses **algorithm selection** (minimum of ring and tree):
- **Ring all-reduce**: `2*(N-1)/N * msg_bytes / bw + 2*(N-1) * latency`
- **Tree (recursive doubling)**: `2*log2(N) * latency + 2 * msg_bytes / bw`

Small messages prefer tree (lower latency), large messages prefer ring (higher bandwidth utilization).

---

### 3.2 Frontend (`xpu_simulator/frontend/`)

The frontend converts models into `ComputeGraph` objects through 7 extraction methods.

#### 3.2.1 ConfigExtractor

**Primary extraction method for LLMs.** Builds graphs analytically from HuggingFace `config.json`.

```
config.json ──► normalize_config() ──► ModelConfig ──► handler.build_layer() ──► ComputeGraph
                                           │
                                    AttentionPattern
                                    QuantConfig
                                    ParallelConfig
```

**Architecture handler registry** (extensible via `register_handler()`):

| model_type | Attention | FFN | Handler |
|------------|-----------|-----|---------|
| llama, mistral, qwen2 | GQA | SwiGLU | StandardTransformerHandler |
| mixtral | GQA | MoE(SwiGLU) | MixtralHandler |
| deepseek_v2 | MLA | Dense + MoE(SwiGLU) | DeepSeekV2Handler |
| gpt2 | MHA (no RoPE) | Dense(GELU) | GPT2Handler |
| falcon | MQA/GQA | Dense(GELU) | FalconHandler |
| gpt_neox | MHA | Dense(GELU) | GPTNeoXHandler |

**GraphBuilder DSL** (used internally by handlers):
```python
b = GraphBuilder(batch, seq, hidden, dtype, parallel=ParallelConfig(tp_size=4))
emb = b.embedding("embed", vocab, hidden)
n1 = b.norm("L0.attn_norm", hidden, emb)
q = b.tp_linear_col("L0.wq", M, K, N, n1)       # column-parallel + ALL_GATHER
o = b.tp_linear_row("L0.wo", M, K_shard, N, attn) # row-parallel + ALL_REDUCE
```

**Sparse attention** changes effective sequence length in attention matmuls:
- Dense: full `S x S` attention matrix
- Top-k (DSA): `S x k` scoring, then `S x k` attention
- Sliding window: `S x W` attention matrix

#### 3.2.2 DispatchExtractor

**Runtime extraction via `TorchDispatchMode`.** Handles any PyTorch model including dynamic control flow, MoE routing, and custom ops.

```
nn.Module (meta device) ──► _RecordingDispatch.__torch_dispatch__() ──► recorded aten ops
                                                                              │
                            _ModuleTracker (forward hooks) ──► module_path ───┘
                                                                              │
                                                    _build_graph() ──► ComputeGraph
```

**Key design decisions**:
- **Meta tensors**: Model runs on `torch.device("meta")` — no GPU needed, only shapes tracked
- **Module tracking**: Forward hooks on every submodule maintain a call stack, enabling layer-aware naming (e.g., `L3.attn_score`, `L3.wo`, `L3.ffn.w1`)
- **60+ aten op mappings**: `_ATEN_OP_MAP` maps `aten.mm.default` → `OpType.MATMUL`, etc.
- **Skip set**: Zero-cost ops (view, reshape, detach, MoE routing bookkeeping) are skipped
- **addmm reordering**: `aten.addmm(bias, input, weight)` → `[input, weight]` for correct shape tracking

**Comparison with ConfigExtractor**:

| Aspect | ConfigExtractor | DispatchExtractor |
|--------|----------------|-------------------|
| Input | config.json dict | nn.Module + example inputs |
| Granularity | Logical ops (1 per linear/attention) | aten ops (6-7 per RMSNorm) |
| Dynamic control flow | No | Yes |
| MoE routing | Analytical (assumed uniform) | Actual routing captured |
| TP support | Native (GraphBuilder) | Post-extraction transform |
| Typical op count (DSv3) | ~1,600 | ~6,500 (→3,400 after fusion) |
| Needs torch model code | No | Yes (or HF model ID) |

#### 3.2.3 Other Extractors

| Extractor | Input | Use Case |
|-----------|-------|----------|
| `TorchGraphExtractor` | `nn.Module` | Simple models via `torch.fx` trace |
| `ExportExtractor` | `nn.Module` | `torch.export` for more control flow |
| `ONNXExtractor` | `.onnx` file | Interop with ONNX ecosystem |
| `ProfilerExtractor` | Chrome trace JSON | Reconstruct graph from `torch.profiler` |

---

### 3.3 Backends (`xpu_simulator/backends/`)

#### 3.3.1 GPU Backend

```python
# Hardware presets
A100_80GB = GPUSpec(
    sm_count=108, clock_ghz=1.41,
    peak_tflops={"fp16": 312, "fp32": 19.5, "int8": 624},
    hbm_bandwidth_GBs=2039, hbm_size_gb=80,
    l2_cache_mb=40, shared_mem_kb=164,
    interconnect=InterconnectSpec("NVLink", 600, 0.5),
)

H100_80GB = GPUSpec(
    sm_count=132, clock_ghz=1.83,
    peak_tflops={"fp16": 989, "fp32": 67, "fp8": 1979, "int8": 1979},
    hbm_bandwidth_GBs=3350, hbm_size_gb=80,
    l2_cache_mb=50, shared_mem_kb=228,
    interconnect=InterconnectSpec("NVLink", 900, 0.5),
)
```

**GPUCostModel** distinguishes:
- Tensor Core ops (MATMUL, CONV2D): use `peak_tflops[dtype]` × compute efficiency (0.70)
- CUDA Core ops (everything else): use `cuda_core_flops` × compute efficiency (0.85)
- Memory bandwidth scaled by memory efficiency factor (0.80)
- Per-op static overhead: 5 µs for Tensor Core ops, 2 µs for CUDA Core ops (kernel dispatch, sync)

#### 3.3.2 NPU Backend

```python
# Hardware presets
ASCEND_910B = AscendSpec(
    num_cores=30,
    cube_peak_tflops={"fp16": 320, "fp8": 640},
    vector_peak_gflops={"fp16": 160, "fp32": 80},
    gm_bandwidth_GBs=1600, gm_size_gb=64,
    l2_size_mb=192, l1_size_kb=1024,
    interconnect=InterconnectSpec("HCCS", 392, 1.0),
)
```

**NPUCostModel** implements a CA-style (Cost Analysis) tiled pipeline model:

```
For MATMUL [M, K] x [K, N]:
  1. Compute tiling: tile_m, tile_k, tile_n (aligned to hardware constraints)
  2. For each tile:
     MTE_in:  load A_tile + B_tile from GM/L2
     CUBE:    tile_m * tile_k * tile_n * 2 FLOPs at per-core peak
     MTE_out: write C_tile to GM
  3. Double-buffered pipeline: overlap MTE_in[i+1] with CUBE[i]
  4. Total = max(sum_MTE, sum_CUBE) across all tiles, divided by num_cores

For elementwise/reduction ops:
  1. Vector tiling: process in chunks fitting UB
  2. MTE_in → VECTOR → MTE_out pipeline
  3. Parallelized across cores
```

**Efficiency factors** (inspired by msmodeling) are applied to raw hardware peaks:
- CUBE compute: 0.70 (FP16/BF16), 0.65 (INT8/FP8), 0.60 (FP32)
- VECTOR compute: 0.80
- MTE bandwidth (memory): 0.60 — applied to both GM→UB and L2→UB transfers
- Static overhead: 5 µs per CUBE op, 2 µs per VECTOR op (kernel dispatch, synchronization)

These bring estimates significantly closer to measured hardware performance.

---

### 3.4 Serving Simulation (`xpu_simulator/serving/`)

End-to-end inference serving with continuous batching, KV cache management, and latency metrics.

```
                    ┌──────────────┐
                    │   Requests   │
                    │  (queue)     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Scheduler   │  ◄── KVCacheAllocator
                    │  (FCFS +     │      (block-based)
                    │  budget)     │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  Per-iteration loop:     │
              │  1. Schedule batch       │
              │  2. Estimate prefill     │◄── ConfigExtractor + CostModel
              │  3. Estimate decode      │◄── ConfigExtractor + CostModel
              │  4. Advance time         │
              │  5. Complete requests    │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │ ServingMetrics│
                    │ avg/p50/p99  │
                    │ TTFT, TPOT   │
                    │ throughput   │
                    └──────────────┘
```

**Request state machine**: `WAITING → PREFILLING → DECODING → DONE`

**KVCacheAllocator**: Block-based allocator with `block_size` tokens per block. A request with `seq_len` tokens needs `ceil(seq_len / block_size)` blocks.

**BatchScheduler**: Continuous batching — each iteration can prefill new requests while decoding existing ones, subject to `max_tokens_budget`.

**Throughput optimizer**: Binary search on `max_batch_size` to find the maximum throughput that satisfies a TPOT SLA constraint.

---

### 3.5 Visualization (`xpu_simulator/utils/`)

#### HTML Report (`html_report.py`)

Single-file interactive HTML with embedded JavaScript (no external dependencies):

```
┌─────────────────────────────────────────────────┐
│  Configuration (grouped: model, attention,       │
│  FFN/MoE, quantization, parallelism)             │
├─────────────────────────────────────────────────┤
│  Feature Badges (GPTQ W4, TP=4, etc.)           │
├─────────────────────────────────────────────────┤
│  Kernel Fusion (bar chart of rules applied)      │
├─────────────────────────────────────────────────┤
│  Device Comparison Table                         │
│  (latency, speedup, ops, FLOPs, memory)          │
├─────────────────────────────────────────────────┤
│  Serving Metrics (if applicable)                 │
│  (throughput, TTFT p50/p99, TPOT p50/p99)        │
├─────────────────────────────────────────────────┤
│  TP Comparison (if applicable)                   │
│  (single vs TP, comm overhead, efficiency)       │
├─────────────────────────────────────────────────┤
│  Model Architecture                              │
│  ├── Pipeline: Embedding → Dense → MoE → Norm    │
│  ├── Dense Block Detail (sub-component bars)     │
│  ├── MoE Block Detail (experts, gate, shared)    │
│  └── MLA/DSA Detail (Q/KV compress, attention)   │
├─────────────────────────────────────────────────┤
│  Per-Layer Latency Chart (stacked bar, per device)│
├─────────────────────────────────────────────────┤
│  Top 15 Most Expensive Operations (table)        │
└─────────────────────────────────────────────────┘
```

Data is embedded as a JSON blob in the HTML. All rendering is client-side JavaScript with inline SVG and Canvas.

---

## 4. Data Flow

### 4.1 Config-Driven Path (Recommended for LLMs)

```
config.json ──► ConfigExtractor.extract()
                    │
                    ├── normalize_config() → ModelConfig
                    ├── handler.build_layer() × num_layers → ComputeGraph
                    │       uses GraphBuilder DSL
                    │       respects QuantConfig, ParallelConfig, Phase
                    │
                    └── ComputeGraph (~1,600 nodes for DSv3)
                            │
                    FusionPass(GPU_FUSION_RULES).apply()
                            │
                    Fused ComputeGraph (~1,300 nodes)
                            │
                    PerformanceEvaluator(GPUCostModel(A100)).run(overlap=True)
                            │
                    SimResult (latency, FLOPs, per-op breakdown)
```

### 4.2 Dispatch-Driven Path (Any PyTorch Model)

```
HuggingFace model ID ──► DispatchExtractor.extract_from_config()
                              │
                              ├── AutoConfig.from_pretrained()
                              ├── AutoModelForCausalLM.from_config(dtype=bf16)
                              │       on torch.device("meta")
                              ├── _ModuleTracker installs forward hooks
                              ├── _RecordingDispatch.__torch_dispatch__()
                              │       records every aten op with shapes + module path
                              ├── _build_graph() → ComputeGraph
                              │       maps aten ops to OpType
                              │       names nodes with L{layer}.{component}
                              │
                              └── ComputeGraph (~6,500 nodes for DSv3)
                                      │
                              FusionPass(DISPATCH_FUSION_RULES).apply()
                                      │
                              Fused ComputeGraph (~3,400 nodes)
                                      │
                              PerformanceEvaluator(NPUCostModel(910B)).run()
                                      │
                              SimResult
```

### 4.3 Serving Simulation Path

```
Model config + Hardware + Requests
        │
ServingSimulator.__init__()
        │
simulator.run(requests)
        │
        ├── for each iteration:
        │   ├── scheduler.schedule() → ScheduledBatch
        │   ├── _estimate_prefill(batch) → prefill_us
        │   │       builds prefill graph via ConfigExtractor
        │   │       evaluates with CostModel
        │   ├── _estimate_decode(batch) → decode_us
        │   │       builds decode graph (seq=1, kv_seq_len=current)
        │   │       evaluates with CostModel
        │   └── advance time, update request states
        │
        └── ServingMetrics (TTFT, TPOT, throughput)
```

---

## 5. Hardware Specifications

### 5.1 NVIDIA GPU

| Spec | A100 80GB | H100 80GB |
|------|-----------|-----------|
| SMs | 108 | 132 |
| Clock | 1.41 GHz | 1.83 GHz |
| FP16 Peak | 312 TFLOPS | 989 TFLOPS |
| FP8/INT8 Peak | 624 TOPS (INT8) | 1,979 TOPS |
| HBM Bandwidth | 2,039 GB/s | 3,350 GB/s |
| HBM Size | 80 GB | 80 GB |
| L2 Cache | 40 MB | 50 MB |
| NVLink BW | 600 GB/s | 900 GB/s |

### 5.2 Huawei Ascend NPU

| Spec | 910B | 910C |
|------|------|------|
| AI Cores | 30 | 32 |
| CUBE FP16 Peak | 320 TFLOPS | 400 TFLOPS |
| CUBE FP8 Peak | 640 TOPS | 800 TOPS |
| VECTOR FP16 Peak | 160 GFLOPS | 200 GFLOPS |
| GM Bandwidth | 1,600 GB/s | 1,800 GB/s |
| GM Size | 64 GB | 128 GB |
| L2 Cache | 192 MB | 256 MB |
| L1 per Core | 1,024 KB | 1,024 KB |
| HCCS BW | 392 GB/s | 600 GB/s |

---

## 6. Key Algorithms

### 6.1 Roofline Model

```
compute_time = flops / peak_flops
memory_time  = bytes_accessed / peak_bandwidth
latency      = max(compute_time, memory_time) + overhead

bound = "compute" if compute_time > memory_time else "memory"
utilization = min(compute_time, memory_time) / max(compute_time, memory_time)
```

The crossover point (ridge point) is at arithmetic intensity = `peak_flops / peak_bandwidth`. Operations above this intensity are compute-bound; below are memory-bound.

### 6.2 NPU Tiled Pipeline

```
For matmul C[M,N] = A[M,K] × B[K,N]:

1. Tiling: choose tile_m, tile_k, tile_n to fit in L1
   - tile_m aligned to 16 (CUBE constraint)
   - tile_k aligned to 32 (CUBE K-dimension)
   - tile_n aligned to 16

2. Per-tile costs:
   MTE_load  = (tile_m*tile_k + tile_k*tile_n) * dtype_bytes / GM_bandwidth
   CUBE_time = 2 * tile_m*tile_k*tile_n / per_core_cube_peak
   MTE_store = tile_m*tile_n * dtype_bytes / GM_bandwidth

3. L2 reuse: if tile fits in L2, subsequent loads use L2 bandwidth (4x faster)

4. Double buffering: pipeline MTE[i+1] with CUBE[i]
   total = max(sum_MTE, sum_CUBE) per core

5. Multi-core: divide tiles across num_cores
   final = total / num_cores
```

### 6.3 ASAP Overlap Scheduling

```python
for node in topological_order:
    resource = get_resource_type(node)  # "cube", "vector", "shared", "comm"
    ready = max(end_time[pred] for pred in predecessors(node))
    start = max(ready, resource_busy[resource])
    end = start + cost(node).latency_us
    resource_busy[resource] = end
```

This allows:
- CUBE and VECTOR ops to overlap (different pipelines on NPU)
- Communication to overlap with compute
- No overlap within the same resource type

### 6.4 Communication Algorithm Selection

```python
def all_reduce_time(msg_bytes, n_ranks, interconnect):
    ring = 2*(n-1)/n * msg / bw + 2*(n-1) * lat
    tree = 2*log2(n) * lat + 2 * msg / bw
    return min(ring, tree)
```

---

## 7. Extensibility

### Adding a New Hardware Target

1. Create `backends/tpu/hardware.py` with a `TPUSpec(HardwareSpec)` class
2. Create `backends/tpu/cost_model.py` with a `TPUCostModel(CostModel)` class
3. Define peak throughput, memory hierarchy, and any pipeline-specific modeling

### Adding a New Model Architecture

**Via ConfigExtractor** (preferred):
```python
class MyArchHandler(ArchitectureHandler):
    def build_layer(self, b: GraphBuilder, layer_idx: int, mc: ModelConfig, prev):
        # Use GraphBuilder DSL to construct the layer graph
        ...

ConfigExtractor.register_handler("my_arch", MyArchHandler())
```

**Via DispatchExtractor** (zero-effort):
```python
graph = DispatchExtractor().extract(my_model, (example_input,))
```

### Adding a New Fusion Rule

```python
class MyFusion(FusionRule):
    @property
    def name(self) -> str:
        return "my_fusion"

    def match(self, graph, node) -> Optional[list[Node]]:
        # Return list of nodes to fuse, or None
        ...

    def fuse(self, nodes) -> OpSpec:
        # Return a single fused OpSpec
        ...

# Add to rule set
MY_RULES = DISPATCH_FUSION_RULES + [MyFusion()]
```

---

## 8. Testing Strategy

| Test Category | Files | Tests | What's Tested |
|--------------|-------|-------|---------------|
| Core operators | test_phase1.py | 6 | Dtype bytes, FLOP calculation, roofline |
| Graph + evaluation | test_phase2.py | 4 | Graph construction, overlap scheduling |
| Extractors | test_extractors.py | 11 | FX, export, profiler extraction |
| Config extraction | test_config_extractor.py | 29 | 8 architectures, MoE, MLA, sparse attention |
| Dispatch extraction | test_dispatch_extractor.py | 11 | MLP, transformer, shapes, dtypes, FLOPs |
| Fusion (config) | test_fusion.py | 9 | FlashAttention, SwiGLU, epilogue, NPU format |
| Fusion (dispatch) | test_dispatch_fusion.py | 14 | RMSNorm, FlashAttn, SwiGLU, idempotency |
| NPU CA model | test_npu_ca_model.py | 8 | Tiling, pipeline, L2 reuse, multi-core |
| Quantization | test_quantization.py | 15 | INT4/INT8/FP8, GPTQ/AWQ parsing |
| Parallelism | test_parallelism.py | 11 | TP/EP, comm costs, graph structure |
| Decode | test_decode.py | 8 | KV cache, decode shapes, TTFT/TPOT |
| Profiling DB | test_profiling_db.py | 11 | Store/lookup, CSV, calibrated model |
| Serving | test_serving.py | 11 | Scheduler, KV allocator, request FSM |
| Efficiency factors | test_efficiency.py | 12 | GPU/NPU efficiency, static overhead, defaults |
| Visualization | test_visualization.py | 7 | Categorization, SVG export, HTML report |

**Test principles**:
- All tests run without GPU (`torch.device("meta")` or CPU)
- Tests verify numerical correctness (FLOP counts, latency bounds) not just absence of errors
- Integration tests verify end-to-end paths (extract → fuse → evaluate)

---

## 9. Performance Characteristics

### DeepSeek V3 671B (batch=1, seq=4096)

| Metric | ConfigExtractor | DispatchExtractor |
|--------|----------------|-------------------|
| Graph nodes | ~1,600 | 6,513 |
| After fusion | ~1,300 | 3,425 |
| Extraction time | <1s | ~5s |
| Total FLOPs | 80.3 TFLOPS | 384.9 TFLOPS* |
| A100 latency | 2.4s | 7.2s |
| 910B latency | 548ms | 105ms |

*DispatchExtractor captures more ops including MoE routing, normalization decomposition, and RoPE components.

### Fusion Impact (DispatchExtractor, DeepSeek V3)

| Fusion Rule | Matches | Ops Eliminated |
|-------------|---------|----------------|
| elementwise_chain | 422 | ~844 |
| matmul_epilogue | 421 | ~842 |
| rms_norm | 245 | ~1,470 |
| dispatch_flash_attention | 61 | ~244 |
| grouped_matmul_swiglu | 58 | ~174 |
| **Total** | **1,207** | **3,088** |

---

## 10. Limitations and Future Work

### Current Limitations

1. **Analytical estimates only**: Roofline models with efficiency factors don't capture warp scheduling or bank conflicts. Use `CalibratedCostModel` for higher accuracy.
2. **No cycle-accurate simulation**: The ASAP scheduler models resource contention at a coarse granularity (resource type), not at the hardware pipeline level.
3. **MoE routing assumed uniform**: ConfigExtractor assumes uniform token distribution across experts. DispatchExtractor captures actual routing but only for the specific input.
4. **TP sharding heuristics**: DispatchExtractor's post-extraction TP transform uses heuristic column/row-parallel assignment based on dimension ratios.
5. **No memory capacity modeling**: The simulator doesn't check if the model + KV cache fit in device memory.

### Potential Enhancements

- **More hardware targets**: TPU, AMD Instinct, Intel Gaudi
- **More attention patterns**: Block sparse, linear attention, Mamba/SSM
- **Intra-op overlap**: Model CUBE/VECTOR overlap within a single op (currently only inter-op)
- **Memory capacity checks**: Warn when model + KV cache exceeds device HBM
- **Multi-node simulation**: Model inter-node communication (InfiniBand, RoCE)
- **Speculative decoding**: Model draft + verify passes in the serving simulator

---

## 11. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.11+ | Model loading, TorchDispatchMode, meta tensors |
| `networkx` | 3.x | Graph data structure |
| `transformers` | 5.x | HuggingFace model loading (optional, for DispatchExtractor) |
| `pytest` | 9.x | Testing |

Optional: `onnx` (for ONNXExtractor), `matplotlib` (for roofline plots).

---

## 12. File Reference

```
xpu_simulator/
├── __init__.py
├── cli.py                          # Command-line interface
├── core/
│   ├── operator.py                 # OpType, Dtype, TensorSpec, OpSpec, QuantConfig
│   ├── graph.py                    # Node, ComputeGraph (networkx wrapper)
│   ├── hardware.py                 # HardwareSpec, MemLevel (abstract)
│   ├── cost_model.py               # CostModel, RooflineCostModel, CalibratedCostModel, CommAwareCostModel
│   ├── evaluator.py                # PerformanceEvaluator, SimResult, OpResult
│   ├── fusion.py                   # FusionRule (10 rules), FusionPass, rule sets
│   ├── parallel.py                 # ParallelConfig, InterconnectSpec
│   ├── communication.py            # all_reduce_time, all_gather_time, etc.
│   ├── kv_cache.py                 # kv_cache_bytes, kv_cache_per_token_bytes
│   └── profiling_db.py             # ProfilingDB (shape_key → measured latency)
├── backends/
│   ├── base.py                     # Backend (abstract)
│   ├── gpu/
│   │   ├── hardware.py             # GPUSpec, A100_80GB, H100_80GB
│   │   └── cost_model.py           # GPUCostModel (Tensor Core + CUDA Core)
│   └── npu/
│       ├── hardware.py             # AscendSpec, ASCEND_910B, ASCEND_910C
│       └── cost_model.py           # NPUCostModel (CA-style tiled pipeline)
├── frontend/
│   ├── base.py                     # GraphExtractor (abstract)
│   ├── config_extractor.py         # ConfigExtractor + architecture handlers
│   ├── config_normalizer.py        # ModelConfig, AttentionPattern, normalize_config
│   ├── dispatch_extractor.py       # DispatchExtractor + _ModuleTracker + _RecordingDispatch
│   ├── graph_builder.py            # GraphBuilder DSL (TP/EP/quant/decode aware)
│   ├── torch_extractor.py          # TorchGraphExtractor (FX trace)
│   ├── export_extractor.py         # ExportExtractor (torch.export)
│   ├── onnx_extractor.py           # ONNXExtractor
│   ├── profiler_extractor.py       # ProfilerExtractor (Chrome trace)
│   ├── op_registry.py              # OpRegistry (name → OpType mapping)
│   └── _torch_utils.py             # Torch utility helpers
├── serving/
│   ├── config.py                   # ServingConfig
│   ├── request.py                  # Request, RequestState
│   ├── scheduler.py                # BatchScheduler, ScheduledBatch
│   ├── kv_cache.py                 # KVCacheAllocator
│   ├── metrics.py                  # ServingMetrics
│   └── simulator.py                # ServingSimulator, find_max_throughput
└── utils/
    ├── categories.py               # categorize_op, CATEGORY_COLORS
    ├── html_report.py              # export_html_report (interactive dashboard)
    ├── visualize.py                # SVG architecture diagrams
    ├── profiling.py                # Profiling utilities
    └── roofline.py                 # Roofline chart generation
reports/                            # Generated HTML reports and Perfetto traces
examples/                           # Example simulation scripts
docs/                               # System design documentation
tests/                              # 171 tests across 17 test files
```

"""GPU profiling for cost model calibration — runs on Google Colab.

Copy-paste into a Colab cell or upload as a script. Uses torch.cuda.Event
for precise kernel timing. No extra dependencies needed.

Usage in Colab:
    !python gpu_profiling_colab.py

Or paste the contents into a notebook cell and run.
Outputs CSV to stdout and saves to gpu_profiling_results.csv.
"""
from __future__ import annotations

import csv
import io
import sys
import time

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

WARMUP_ITERS = 20
MEASURE_ITERS = 100
DTYPE = torch.float16
DEVICE = "cuda"

# Shapes to profile — matching the NPU profiling set + real model dims
MATMUL_SHAPES = [
    # (M, K, N)
    (1, 4096, 4096),          # decode single token
    (1, 4096, 14336),         # decode MLP up (LLaMA-8B)
    (1, 14336, 4096),         # decode MLP down
    (128, 4096, 4096),        # small batch
    (1024, 1024, 1024),       # square
    (1024, 4096, 4096),       # prefill typical
    (1024, 4096, 14336),      # LLaMA-8B MLP up
    (1024, 14336, 4096),      # LLaMA-8B MLP down
    (1024, 4096, 11008),      # LLaMA-7B MLP up
    (1024, 11008, 4096),      # LLaMA-7B MLP down
    (1024, 8192, 8192),       # LLaMA-70B attn
    (1024, 8192, 28672),      # LLaMA-70B MLP up
    (2048, 4096, 4096),       # longer seq
    (4096, 4096, 4096),       # large square
]

VECTOR_SHAPES = [
    # (rows, cols)
    (1, 4096),
    (128, 4096),
    (1024, 4096),
    (1024, 8192),
    (1024, 14336),
    (2048, 4096),
    (4096, 4096),
]


# ------------------------------------------------------------------ #
# Timing utility
# ------------------------------------------------------------------ #

def bench_fn(fn, warmup=WARMUP_ITERS, iters=MEASURE_ITERS) -> float:
    """Benchmark a CUDA function, return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure each iteration individually for median
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms → us

    times.sort()
    median = times[len(times) // 2]
    return median


# ------------------------------------------------------------------ #
# Op profilers
# ------------------------------------------------------------------ #

def profile_matmul(M: int, K: int, N: int) -> dict:
    A = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    B = torch.randn(K, N, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: torch.mm(A, B))
    flops = 2 * M * K * N
    tflops = flops / latency / 1e6  # TFLOPS
    mem_bytes = (M * K + K * N + M * N) * 2  # fp16
    return {
        "op": "MATMUL", "shape": f"{M}x{K}x{N}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(tflops, 2),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_layer_norm(rows: int, cols: int) -> dict:
    x = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    weight = torch.ones(cols, dtype=DTYPE, device=DEVICE)
    bias = torch.zeros(cols, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: F.layer_norm(x, (cols,), weight, bias))
    # LayerNorm FLOPs: mean(N) + var(3N) + normalize(3N) ≈ 7N per row
    flops = 7 * rows * cols
    mem_bytes = (rows * cols + rows * cols) * 2  # read + write
    return {
        "op": "LAYERNORM", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_softmax(rows: int, cols: int) -> dict:
    x = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: F.softmax(x, dim=-1))
    # Softmax FLOPs: max(N) + sub+exp(2N) + sum(N) + div(N) ≈ 5N per row
    flops = 5 * rows * cols
    mem_bytes = (rows * cols + rows * cols) * 2
    return {
        "op": "SOFTMAX", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_silu(rows: int, cols: int) -> dict:
    x = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: F.silu(x))
    # SiLU = x * sigmoid(x): sigmoid(~4 ops) + mul ≈ 5N
    flops = 5 * rows * cols
    mem_bytes = (rows * cols + rows * cols) * 2
    return {
        "op": "SILU", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_gelu(rows: int, cols: int) -> dict:
    x = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: F.gelu(x))
    flops = 8 * rows * cols  # tanh approximation is ~8 ops
    mem_bytes = (rows * cols + rows * cols) * 2
    return {
        "op": "GELU", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_relu(rows: int, cols: int) -> dict:
    x = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: F.relu(x))
    flops = rows * cols  # 1 comparison per element
    mem_bytes = (rows * cols + rows * cols) * 2
    return {
        "op": "RELU", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_add(rows: int, cols: int) -> dict:
    a = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    b = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: torch.add(a, b))
    flops = rows * cols
    mem_bytes = (2 * rows * cols + rows * cols) * 2  # 2 inputs + 1 output
    return {
        "op": "ADD", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_mul(rows: int, cols: int) -> dict:
    a = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    b = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    latency = bench_fn(lambda: torch.mul(a, b))
    flops = rows * cols
    mem_bytes = (2 * rows * cols + rows * cols) * 2
    return {
        "op": "MUL", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


def profile_rope(rows: int, cols: int) -> dict:
    """RoPE: apply rotary embeddings to query/key tensors."""
    half = cols // 2
    x = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
    cos_cached = torch.randn(rows, half, dtype=DTYPE, device=DEVICE)
    sin_cached = torch.randn(rows, half, dtype=DTYPE, device=DEVICE)

    def rope_fn():
        x0, x1 = x[..., :half], x[..., half:]
        y0 = x0 * cos_cached - x1 * sin_cached
        y1 = x0 * sin_cached + x1 * cos_cached
        return torch.cat([y0, y1], dim=-1)

    latency = bench_fn(rope_fn)
    # 4 muls + 1 sub + 1 add + 1 cat ≈ 6 ops per element
    flops = 6 * rows * cols
    mem_bytes = (rows * cols + 2 * rows * half + rows * cols) * 2
    return {
        "op": "ROPE", "shape": f"{rows}x{cols}",
        "latency_us": round(latency, 2),
        "flops": flops, "tflops": round(flops / latency / 1e6, 4),
        "mem_bytes": mem_bytes,
        "arith_intensity": round(flops / mem_bytes, 1),
    }


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def get_gpu_info() -> dict:
    """Get GPU device info."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "memory_gb": round(props.total_mem / 1024**3, 1),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available")
        sys.exit(1)

    gpu_info = get_gpu_info()
    print("=" * 70)
    print("GPU Profiling for xPU-Simulator Calibration")
    print(f"  Device: {gpu_info['name']}")
    print(f"  SMs: {gpu_info['sm_count']}  Memory: {gpu_info['memory_gb']} GB")
    print(f"  Compute: {gpu_info['compute_capability']}  CUDA: {gpu_info['cuda_version']}")
    print(f"  Torch: {gpu_info['torch_version']}")
    print(f"  Warmup: {WARMUP_ITERS}  Measure: {MEASURE_ITERS} (median)")
    print("=" * 70)

    results = []

    # --- MATMUL ---
    print("\n--- MATMUL (torch.mm, fp16, Tensor Cores) ---")
    for M, K, N in MATMUL_SHAPES:
        print(f"  {M}x{K}x{N}...", end=" ", flush=True)
        r = profile_matmul(M, K, N)
        results.append(r)
        print(f"{r['latency_us']:>10.2f} us  {r['tflops']:>8.2f} TFLOPS")

    # --- VECTOR ops ---
    vector_profilers = [
        ("ADD", profile_add),
        ("MUL", profile_mul),
        ("RELU", profile_relu),
        ("SILU", profile_silu),
        ("GELU", profile_gelu),
        ("LAYERNORM", profile_layer_norm),
        ("SOFTMAX", profile_softmax),
        ("ROPE", profile_rope),
    ]

    for op_name, profiler in vector_profilers:
        print(f"\n--- {op_name} ---")
        for rows, cols in VECTOR_SHAPES:
            print(f"  {rows}x{cols}...", end=" ", flush=True)
            r = profiler(rows, cols)
            results.append(r)
            print(f"{r['latency_us']:>10.2f} us")

    # --- Output CSV ---
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "op", "shape", "latency_us", "flops", "tflops", "mem_bytes", "arith_intensity",
    ])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

    csv_str = output.getvalue()

    # Save to file
    fname = f"gpu_profiling_{gpu_info['name'].replace(' ', '_')}.csv"
    with open(fname, "w") as f:
        f.write(f"# GPU: {gpu_info['name']}\n")
        f.write(f"# SMs: {gpu_info['sm_count']}  Memory: {gpu_info['memory_gb']}GB\n")
        f.write(f"# Compute: {gpu_info['compute_capability']}  CUDA: {gpu_info['cuda_version']}\n")
        f.write(f"# Torch: {gpu_info['torch_version']}\n")
        f.write(f"# Warmup: {WARMUP_ITERS}  Iters: {MEASURE_ITERS}\n")
        f.write(csv_str)
    print(f"\n\nResults saved to: {fname}")

    # Also print CSV for easy copy-paste from Colab
    print("\n--- CSV OUTPUT (copy this) ---")
    print(csv_str)

    # --- Summary ---
    print("\n--- SUMMARY ---")
    matmul_results = [r for r in results if r["op"] == "MATMUL"]
    if matmul_results:
        peak_tflops = max(r["tflops"] for r in matmul_results)
        print(f"Peak MATMUL: {peak_tflops:.2f} TFLOPS")

    # Memory bandwidth estimate from memory-bound ops
    membw_ops = [r for r in results if r["op"] in ("ADD", "RELU", "MUL")]
    if membw_ops:
        # BW = bytes / latency
        bws = [r["mem_bytes"] / r["latency_us"] * 1e6 / 1e9 for r in membw_ops
               if r["latency_us"] > 0]
        if bws:
            peak_bw = max(bws)
            avg_bw = sum(bws) / len(bws)
            print(f"Effective mem BW (from ADD/RELU/MUL): peak {peak_bw:.1f} GB/s, avg {avg_bw:.1f} GB/s")

    # Derived efficiency factors
    print("\n--- DERIVED EFFICIENCY FACTORS ---")
    gpu_name = gpu_info["name"]
    if "A100" in gpu_name:
        spec_tflops, spec_bw = 312, 2039
    elif "H100" in gpu_name:
        spec_tflops, spec_bw = 989, 3350
    elif "T4" in gpu_name:
        spec_tflops, spec_bw = 65, 320
    elif "L4" in gpu_name:
        spec_tflops, spec_bw = 121, 300
    elif "V100" in gpu_name:
        spec_tflops, spec_bw = 125, 900
    else:
        spec_tflops, spec_bw = 0, 0

    if spec_tflops > 0 and matmul_results:
        peak = max(r["tflops"] for r in matmul_results)
        eff = peak / spec_tflops
        print(f"  matmul_fp16 efficiency: {eff:.3f}  (measured {peak:.1f} / spec {spec_tflops} TFLOPS)")

    if spec_bw > 0 and membw_ops:
        bws = [r["mem_bytes"] / r["latency_us"] * 1e6 / 1e9 for r in membw_ops
               if r["latency_us"] > 0]
        if bws:
            peak_bw = max(bws)
            eff = peak_bw / spec_bw
            print(f"  memory efficiency:      {eff:.3f}  (measured {peak_bw:.1f} / spec {spec_bw} GB/s)")

    # Static overhead estimate from smallest ops
    small_matmul = [r for r in matmul_results if "1x" in r["shape"]]
    if small_matmul:
        # For tiny matmuls, most time is overhead
        min_lat = min(r["latency_us"] for r in small_matmul)
        print(f"  estimated static overhead (tiny matmul): ~{min_lat:.1f} us")


if __name__ == "__main__":
    main()

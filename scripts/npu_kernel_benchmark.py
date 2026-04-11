"""Simple + fused kernel benchmark on Ascend NPU via torch_npu.

Runs each kernel in an isolated function and measures wall time using
torch.npu.Event (device-side, much more accurate than time.time).
msprof wraps this whole script to also collect per-op cycle data.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import torch
import torch_npu  # noqa: F401  -- required to register NPU ops

DEVICE = f"npu:{int(os.environ.get('NPU_DEVICE', '5'))}"
WARMUP = 5
ITERS = 20
DTYPE = torch.float16

# (name, shape)
VECTOR_SHAPES = [
    ("1x4096",      (1, 4096)),
    ("1x8192",      (1, 8192)),
    ("16x4096",     (16, 4096)),
    ("64x4096",     (64, 4096)),
    ("256x4096",    (256, 4096)),
    ("1024x4096",   (1024, 4096)),
    ("2048x4096",   (2048, 4096)),
    ("4096x4096",   (4096, 4096)),
]
MATMUL_SHAPES = [
    ("256x256x256",       (256, 256, 256)),
    ("512x512x512",       (512, 512, 512)),
    ("1024x1024x1024",    (1024, 1024, 1024)),
    ("1024x4096x4096",    (1024, 4096, 4096)),
    ("2048x4096x4096",    (2048, 4096, 4096)),
    ("4096x4096x4096",    (4096, 4096, 4096)),
    ("1024x4096x14336",   (1024, 4096, 14336)),
]


def make_vec_inputs(shape):
    a = torch.randn(shape, dtype=DTYPE, device=DEVICE)
    b = torch.randn(shape, dtype=DTYPE, device=DEVICE)
    return a, b


def make_mm_inputs(M, K, N):
    a = torch.randn((M, K), dtype=DTYPE, device=DEVICE)
    b = torch.randn((K, N), dtype=DTYPE, device=DEVICE)
    return a, b


# --- Simple kernels ---------------------------------------------------------

def k_add(a, b):        return a + b
def k_mul(a, b):        return a * b
def k_relu(a, _):       return torch.relu(a)
def k_silu(a, _):       return torch.nn.functional.silu(a)
def k_gelu(a, _):       return torch.nn.functional.gelu(a)
def k_layernorm(a, _):  return torch.nn.functional.layer_norm(a, (a.shape[-1],))
def k_softmax(a, _):    return torch.softmax(a, dim=-1)

# --- Fused patterns ---------------------------------------------------------
# Eager torch_npu will not auto-fuse, so these report "un-fused" costs.
# The graph_mode versions below will be JIT-compiled.

def k_add_relu(a, b):           return torch.relu(a + b)
def k_add_gelu(a, b):           return torch.nn.functional.gelu(a + b)
def k_mul_add_relu(a, b):       return torch.relu(a * b + a)
def k_layernorm_gelu(a, _):     return torch.nn.functional.gelu(
    torch.nn.functional.layer_norm(a, (a.shape[-1],))
)

def k_mm(a, b):                 return torch.matmul(a, b)
def k_mm_bias_relu(a, b):
    bias = torch.zeros(b.shape[-1], dtype=DTYPE, device=DEVICE)
    return torch.relu(torch.matmul(a, b) + bias)
def k_mm_gelu(a, b):            return torch.nn.functional.gelu(torch.matmul(a, b))


VECTOR_KERNELS = {
    "add":            k_add,
    "mul":            k_mul,
    "relu":           k_relu,
    "silu":           k_silu,
    "gelu":           k_gelu,
    "layernorm":      k_layernorm,
    "softmax":        k_softmax,
    # Fused vector
    "add_relu":       k_add_relu,
    "add_gelu":       k_add_gelu,
    "mul_add_relu":   k_mul_add_relu,
    "layernorm_gelu": k_layernorm_gelu,
}
MATMUL_KERNELS = {
    "matmul":         k_mm,
    "matmul_bias_relu": k_mm_bias_relu,
    "matmul_gelu":    k_mm_gelu,
}


def time_kernel(fn, a, b):
    """Return median latency in microseconds over ITERS runs."""
    for _ in range(WARMUP):
        fn(a, b)
    torch.npu.synchronize()

    start_evs = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
    stop_evs = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        start_evs[i].record()
        fn(a, b)
        stop_evs[i].record()
    torch.npu.synchronize()

    # Event elapsed_time returns milliseconds
    samples_us = sorted(
        start_evs[i].elapsed_time(stop_evs[i]) * 1000.0 for i in range(ITERS)
    )
    median = samples_us[len(samples_us) // 2]
    return median


def main():
    torch.npu.set_device(DEVICE)
    print(f"Using device: {DEVICE} (torch {torch.__version__})", flush=True)

    out_path = os.environ.get("BENCH_OUT", "/home/Ray/npu_bench_results.csv")
    writer = csv.writer(open(out_path, "w"))
    writer.writerow(["kernel", "shape", "latency_us"])

    # Vector / fused vector
    for shape_name, shape in VECTOR_SHAPES:
        a, b = make_vec_inputs(shape)
        for name, fn in VECTOR_KERNELS.items():
            try:
                us = time_kernel(fn, a, b)
                print(f"  {name:<16} {shape_name:<14} {us:8.2f} us", flush=True)
                writer.writerow([name, shape_name, f"{us:.3f}"])
            except Exception as e:
                msg = str(e).replace("\n", " | ")[:120]
                print(f"  {name:<16} {shape_name:<14} FAIL: {msg}", flush=True)
                writer.writerow([name, shape_name, "FAIL"])
        del a, b

    # Matmul / fused matmul
    for shape_name, (M, K, N) in MATMUL_SHAPES:
        a, b = make_mm_inputs(M, K, N)
        for name, fn in MATMUL_KERNELS.items():
            try:
                us = time_kernel(fn, a, b)
                print(f"  {name:<16} {shape_name:<14} {us:8.2f} us", flush=True)
                writer.writerow([name, shape_name, f"{us:.3f}"])
            except Exception as e:
                msg = str(e).replace("\n", " | ")[:120]
                print(f"  {name:<16} {shape_name:<14} FAIL: {msg}", flush=True)
                writer.writerow([name, shape_name, "FAIL"])
        del a, b

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total wall time: {time.time()-t0:.1f}s")

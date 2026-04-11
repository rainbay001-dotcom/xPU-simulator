"""End-to-end inference stress test for Qwen3-0.6B on Ascend NPU.

Sweeps (prompt_len, new_tokens, batch_size) and reports per-config:
  - TTFT (time to first token)
  - TPOT (time per output token, i.e. decode latency)
  - tokens/sec (decode throughput)
  - peak NPU memory

Timing uses torch.npu.Event to exclude Python dispatch overhead.

Usage on NPU server:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /home/Ray/venv/bin/activate
    ASCEND_RT_VISIBLE_DEVICES=2 python qwen_stress.py
    ASCEND_RT_VISIBLE_DEVICES=2 python qwen_stress.py --msprof   # under msprof
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import torch_npu  # noqa: F401 — registers the NPU backend
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/weight/Qwen/Qwen3-0.6B"
DEVICE = "npu:0"

SWEEP = [
    # (prompt_len, new_tokens, batch_size)
    (32,   64,  1),
    (128,  64,  1),
    (512,  64,  1),
    (1024, 64,  1),
    (128,  128, 1),
    (128,  256, 1),
    (128,  64,  4),
    (128,  64,  8),
    (512,  64,  4),
]

WARMUP_RUNS = 2
TIMED_RUNS = 3


def now_event() -> "torch.npu.Event":
    e = torch.npu.Event(enable_timing=True)
    e.record()
    return e


def run_one(model, tokenizer, prompt_len: int, new_tokens: int, batch: int):
    """Run generate() for one config and return (ttft_ms, tpot_ms, total_ms)."""
    # Build a deterministic prompt of the requested length
    base = "The capital of France is Paris. " * 200
    toks = tokenizer(base, return_tensors="pt").input_ids[0][:prompt_len]
    if toks.shape[0] < prompt_len:
        pad = torch.full((prompt_len - toks.shape[0],), tokenizer.pad_token_id or 0,
                         dtype=toks.dtype)
        toks = torch.cat([toks, pad])
    input_ids = toks.unsqueeze(0).repeat(batch, 1).to(DEVICE)
    attn = torch.ones_like(input_ids)

    # Prefill-only pass for TTFT (generate 1 new token)
    torch.npu.synchronize()
    t0 = now_event()
    _ = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=1, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    t1 = now_event()
    torch.npu.synchronize()
    ttft_ms = t0.elapsed_time(t1)

    # Full generate for TPOT
    torch.npu.synchronize()
    t2 = now_event()
    out = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    t3 = now_event()
    torch.npu.synchronize()
    total_ms = t2.elapsed_time(t3)

    generated = out.shape[1] - input_ids.shape[1]
    # TPOT = (total - prefill) / (generated - 1) — excludes the first token's prefill
    if generated > 1:
        tpot_ms = (total_ms - ttft_ms) / (generated - 1)
    else:
        tpot_ms = float("nan")
    return ttft_ms, tpot_ms, total_ms, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/Ray/qwen_stress_results.csv")
    parser.add_argument("--msprof", action="store_true",
                        help="Skip sweep; run a single small config for msprof capture")
    args = parser.parse_args()

    print(f"[init] loading {MODEL_PATH}")
    torch.npu.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"[init] params = {params/1e6:.1f} M, dtype = {next(model.parameters()).dtype}")

    if args.msprof:
        sweep = [(128, 32, 1)]
        timed_runs = 1
    else:
        sweep = SWEEP
        timed_runs = TIMED_RUNS

    # Global warmup so the first real config doesn't eat compile overhead
    print("[warmup] 2 dry runs")
    for _ in range(WARMUP_RUNS):
        run_one(model, tokenizer, 128, 16, 1)

    rows = []
    print()
    print(f"{'prompt':>7}{'new':>6}{'batch':>7}{'ttft_ms':>10}"
          f"{'tpot_ms':>10}{'tokens/s':>11}{'total_ms':>11}")
    print("-" * 65)
    for plen, ntok, batch in sweep:
        samples = []
        for _ in range(timed_runs):
            samples.append(run_one(model, tokenizer, plen, ntok, batch))
        # median across runs
        samples.sort(key=lambda s: s[2])
        ttft, tpot, total, gen = samples[len(samples) // 2]
        tok_per_s = (batch * gen) / (total / 1000.0) if total > 0 else 0.0
        print(f"{plen:>7}{ntok:>6}{batch:>7}{ttft:>10.2f}"
              f"{tpot:>10.2f}{tok_per_s:>11.1f}{total:>11.2f}")
        rows.append({
            "prompt_len": plen, "new_tokens": ntok, "batch": batch,
            "ttft_ms": round(ttft, 3), "tpot_ms": round(tpot, 3),
            "tokens_per_sec": round(tok_per_s, 2), "total_ms": round(total, 3),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()

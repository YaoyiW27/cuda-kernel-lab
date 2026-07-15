"""Benchmark the custom attention kernel against PyTorch.

Compares against two references:
  - "naive torch": softmax(Q@K^T/sqrt(d)) @ V, materializing the seq x seq scores
    (the same math our kernel does).
  - "SDPA": torch.nn.functional.scaled_dot_product_attention, which dispatches to a
    fused / FlashAttention-style backend — the number to be humbled by.

We sweep sequence length at fixed head dim d. Attention is O(seq^2) in both compute
and (for our materialized version) memory, so time grows quickly with seq.

Writes benchmarks/results/attention.csv. Run from the repo root:

    python kernels/05_attention/benchmark.py
"""

import csv
import math
import os
import statistics

import torch
import torch.nn.functional as F

from test_attention import custom_attention, load_lib, reference

_RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "results", "attention.csv"
)

D = 64
SEQS = [128, 256, 512, 1024, 2048]
WARMUP = 5
ITERS = 100


def time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this benchmark.")

    lib = load_lib()
    print(f"Device: {torch.cuda.get_device_name()} | torch {torch.__version__} "
          f"| CUDA {torch.version.cuda}\n")
    print(f"d={D}")
    print(f"{'seq':>6} {'custom(ms)':>12} {'naive torch(ms)':>16} {'SDPA(ms)':>10} "
          f"{'vs naive':>9} {'vs SDPA':>9}")

    rows = []
    for seq in SEQS:
        Q = torch.randn(seq, D, device="cuda", dtype=torch.float32)
        K = torch.randn(seq, D, device="cuda", dtype=torch.float32)
        V = torch.randn(seq, D, device="cuda", dtype=torch.float32)

        custom_ms = time_ms(lambda: custom_attention(lib, Q, K, V))
        naive_ms = time_ms(lambda: reference(Q, K, V))
        sdpa_ms = time_ms(lambda: F.scaled_dot_product_attention(Q, K, V))

        print(f"{seq:>6} {custom_ms:>12.4f} {naive_ms:>16.4f} {sdpa_ms:>10.4f} "
              f"{naive_ms/custom_ms:>8.2f}x {sdpa_ms/custom_ms:>8.2f}x")
        rows.append({
            "seq": seq, "d": D,
            "custom_ms": round(custom_ms, 5),
            "naive_torch_ms": round(naive_ms, 5),
            "sdpa_ms": round(sdpa_ms, 5),
            "speedup_vs_naive": round(naive_ms / custom_ms, 3),
            "speedup_vs_sdpa": round(sdpa_ms / custom_ms, 3),
        })

    os.makedirs(os.path.dirname(_RESULTS), exist_ok=True)
    with open(_RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {os.path.abspath(_RESULTS)}")


if __name__ == "__main__":
    main()

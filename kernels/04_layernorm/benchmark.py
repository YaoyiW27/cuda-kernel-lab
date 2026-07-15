"""Benchmark the custom LayerNorm kernel against torch.nn.functional.layer_norm.

LayerNorm is memory-bound; the fused single-pass kernel reads x once. We sweep the
hidden size (cols), the axis that's normalized, at a fixed number of rows (tokens).

Writes benchmarks/results/layernorm.csv. Run from the repo root:

    python kernels/04_layernorm/benchmark.py
"""

import csv
import os
import statistics

import torch
import torch.nn.functional as F

from test_layernorm import custom_layernorm, load_lib

_RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "results", "layernorm.csv"
)

ROWS = 8192          # ~tokens in a batch
COLS = [256, 768, 1024, 4096, 12288]  # hidden sizes (BERT/GPT-ish)
EPS = 1e-5
WARMUP = 10
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


def gbps(rows: int, cols: int, ms: float) -> float:
    # read x + write y => 2 * rows * cols * 4 bytes (gamma/beta negligible).
    return (2.0 * rows * cols * 4) / (ms * 1e-3) / 1e9


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this benchmark.")

    lib = load_lib()
    print(f"Device: {torch.cuda.get_device_name()} | torch {torch.__version__} "
          f"| CUDA {torch.version.cuda}\n")
    print(f"rows={ROWS}")
    print(f"{'cols':>8} {'custom(ms)':>12} {'torch(ms)':>12} "
          f"{'custom GB/s':>12} {'torch GB/s':>12} {'speedup':>9}")

    rows_out = []
    for cols in COLS:
        x = torch.randn(ROWS, cols, device="cuda", dtype=torch.float32)
        gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
        beta = torch.randn(cols, device="cuda", dtype=torch.float32)

        custom_ms = time_ms(lambda: custom_layernorm(lib, x, gamma, beta, EPS))
        torch_ms = time_ms(lambda: F.layer_norm(x, (cols,), weight=gamma, bias=beta, eps=EPS))
        c_gbps, t_gbps = gbps(ROWS, cols, custom_ms), gbps(ROWS, cols, torch_ms)
        speedup = torch_ms / custom_ms
        print(f"{cols:>8} {custom_ms:>12.4f} {torch_ms:>12.4f} "
              f"{c_gbps:>12.1f} {t_gbps:>12.1f} {speedup:>8.2f}x")
        rows_out.append({
            "rows": ROWS, "cols": cols,
            "custom_ms": round(custom_ms, 5), "torch_ms": round(torch_ms, 5),
            "custom_gbps": round(c_gbps, 2), "torch_gbps": round(t_gbps, 2),
            "speedup": round(speedup, 3),
        })

    os.makedirs(os.path.dirname(_RESULTS), exist_ok=True)
    with open(_RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {os.path.abspath(_RESULTS)}")


if __name__ == "__main__":
    main()

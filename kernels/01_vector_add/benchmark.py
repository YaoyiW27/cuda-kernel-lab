"""Benchmark the custom vector_add kernel against torch.add.

Methodology:
  - Time with CUDA events (torch.cuda.Event), not wall clock.
  - Warm up, then run 100+ iterations and report the MEDIAN time.
  - Report effective memory throughput in GB/s. Vector add touches 3 arrays
    (read a, read b, write c) => bytes = 3 * n * 4 (float32). It is memory-bound,
    so throughput vs. the GPU's peak bandwidth is the number that matters.

Writes a CSV to benchmarks/results/vector_add.csv. Run from the repo root:

    python kernels/01_vector_add/benchmark.py
"""

import csv
import os
import statistics

import torch

from test_vector_add import custom_vector_add, load_lib

_RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "results", "vector_add.csv"
)

SIZES = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
WARMUP = 10
ITERS = 100


def time_ms(fn) -> float:
    """Median milliseconds over ITERS runs of fn(), timed with CUDA events."""
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
        times.append(start.elapsed_time(end))  # milliseconds
    return statistics.median(times)


def gbps(n: int, ms: float) -> float:
    """Effective throughput in GB/s for an n-element float32 vector add."""
    bytes_moved = 3 * n * 4  # read a, read b, write c
    return bytes_moved / (ms * 1e-3) / 1e9


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this benchmark.")

    lib = load_lib()
    dev = torch.cuda.get_device_name()
    print(f"Device: {dev} | torch {torch.__version__} | CUDA {torch.version.cuda}\n")
    print(f"{'n':>12} {'custom(ms)':>12} {'torch(ms)':>12} "
          f"{'custom GB/s':>12} {'torch GB/s':>12} {'speedup':>9}")

    rows = []
    for n in SIZES:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(a)

        custom_ms = time_ms(lambda: custom_vector_add(lib, a, b))
        torch_ms = time_ms(lambda: torch.add(a, b, out=out))

        c_gbps, t_gbps = gbps(n, custom_ms), gbps(n, torch_ms)
        speedup = torch_ms / custom_ms
        print(f"{n:>12} {custom_ms:>12.4f} {torch_ms:>12.4f} "
              f"{c_gbps:>12.1f} {t_gbps:>12.1f} {speedup:>8.2f}x")
        rows.append({
            "n": n,
            "custom_ms": round(custom_ms, 5),
            "torch_ms": round(torch_ms, 5),
            "custom_gbps": round(c_gbps, 2),
            "torch_gbps": round(t_gbps, 2),
            "speedup": round(speedup, 3),
        })

    os.makedirs(os.path.dirname(_RESULTS), exist_ok=True)
    with open(_RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {os.path.abspath(_RESULTS)}")


if __name__ == "__main__":
    main()

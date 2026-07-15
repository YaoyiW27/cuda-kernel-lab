"""Benchmark naive vs tiled matmul vs torch.matmul (cuBLAS).

Reports execution time (ms) and GFLOPS. A matmul does 2*M*N*K flops (one multiply
+ one add per inner-product term), so GFLOPS = 2*M*N*K / time. cuBLAS (torch.matmul)
is the "how close can a hand-written kernel get" reference.

Expected story: tiled >> naive, both below cuBLAS, gap widening with matrix size —
this is the shared-memory-tiling win the README explains.

Writes benchmarks/results/matmul.csv. Run from the repo root:

    python kernels/02_matmul/benchmark.py
"""

import csv
import os
import statistics

import torch

from test_matmul import custom_matmul, load_libs

_RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "results", "matmul.csv"
)

SIZES = [256, 512, 1024, 2048, 4096]  # square N x N x N
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


def gflops(n: int, ms: float) -> float:
    return (2.0 * n * n * n) / (ms * 1e-3) / 1e9


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this benchmark.")

    naive, tiled = load_libs()
    print(f"Device: {torch.cuda.get_device_name()} | torch {torch.__version__} "
          f"| CUDA {torch.version.cuda}\n")
    print(f"{'N':>6} {'naive(ms)':>11} {'tiled(ms)':>11} {'torch(ms)':>11} "
          f"{'naive GF':>10} {'tiled GF':>10} {'torch GF':>10} {'tiled/naive':>12}")

    rows = []
    for N in SIZES:
        A = torch.randn(N, N, device="cuda", dtype=torch.float32)
        B = torch.randn(N, N, device="cuda", dtype=torch.float32)
        out = torch.empty(N, N, device="cuda", dtype=torch.float32)

        # Naive gets very slow at large N; skip it past 2048 to keep runs short.
        naive_ms = time_ms(lambda: custom_matmul(naive, "matmul_naive", A, B)) if N <= 2048 else float("nan")
        tiled_ms = time_ms(lambda: custom_matmul(tiled, "matmul_tiled", A, B))
        torch_ms = time_ms(lambda: torch.matmul(A, B, out=out))

        n_gf = gflops(N, naive_ms) if naive_ms == naive_ms else float("nan")
        t_gf, c_gf = gflops(N, tiled_ms), gflops(N, torch_ms)
        speedup = (naive_ms / tiled_ms) if naive_ms == naive_ms else float("nan")

        print(f"{N:>6} {naive_ms:>11.4f} {tiled_ms:>11.4f} {torch_ms:>11.4f} "
              f"{n_gf:>10.1f} {t_gf:>10.1f} {c_gf:>10.1f} {speedup:>11.2f}x")
        rows.append({
            "N": N,
            "naive_ms": round(naive_ms, 5) if naive_ms == naive_ms else "",
            "tiled_ms": round(tiled_ms, 5),
            "torch_ms": round(torch_ms, 5),
            "naive_gflops": round(n_gf, 1) if n_gf == n_gf else "",
            "tiled_gflops": round(t_gf, 1),
            "torch_gflops": round(c_gf, 1),
            "tiled_vs_naive_speedup": round(speedup, 3) if speedup == speedup else "",
            "tiled_pct_of_cublas": round(100.0 * t_gf / c_gf, 1),
        })

    os.makedirs(os.path.dirname(_RESULTS), exist_ok=True)
    with open(_RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {os.path.abspath(_RESULTS)}")


if __name__ == "__main__":
    main()

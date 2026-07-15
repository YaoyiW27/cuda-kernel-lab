# CUDA Kernel Mini Lab

Hand-written CUDA kernels for the operations that dominate machine-learning workloads —
vector add, matrix multiply (naive **and** tiled), softmax, layer norm, and scaled
dot-product attention. Every kernel ships with a **correctness test** against PyTorch
and a **benchmark** (CUDA-event timed, median of 100+ runs) comparing the custom
implementation to PyTorch's native op.

The goal is to understand the GPU's low-level compute model in practice: the memory
hierarchy (global vs. shared memory), coalesced access, parallel reductions, and where
kernels are memory-bound vs. compute-bound.

## Kernels

| # | Kernel | Core concept | Status |
|---|--------|--------------|--------|
| 1 | Vector Add | CUDA programming model, grid-stride loops, launch overhead, GB/s | ✅ implemented |
| 2 | **Matrix Multiply** | **naive → tiled shared-memory matmul** (the central optimization studied here) | ✅ implemented |
| 3 | Softmax | parallel reduction + numerical stability (subtract row max) | ✅ implemented |
| 4 | LayerNorm | **fused single-pass** mean/variance via `E[x²] − E[x]²` | ✅ implemented |
| 5 | Attention Score | `softmax(QKᵀ/√d)·V` — composing matmul + softmax | ✅ implemented |

Every kernel exposes an `extern "C"` entry point that operates on **device pointers**
(a PyTorch CUDA tensor's `.data_ptr()`), so benchmarks time the kernel alone with no
host↔device copies in the measured path.

## Project structure

```
cuda-kernel-lab/
├── kernels/
│   ├── 01_vector_add/    vector_add.cu · test_vector_add.py · benchmark.py
│   ├── 02_matmul/        matmul_naive.cu · matmul_tiled.cu · test_matmul.py · benchmark.py
│   ├── 03_softmax/       softmax.cu · test_softmax.py · benchmark.py
│   ├── 04_layernorm/     layernorm.cu · test_layernorm.py · benchmark.py
│   └── 05_attention/     attention_score.cu · test_attention.py · benchmark.py
├── benchmarks/
│   ├── run_all_benchmarks.py    run every test + benchmark in order
│   └── results/                 per-kernel CSV output
├── profiling/                   Nsight (nsys/ncu) write-up
├── run_on_colab.ipynb           one-click build + test + benchmark on a free T4
├── Makefile                     build all kernels -> build/*.so
└── requirements.txt
```

## Running it

**No local NVIDIA GPU?** Open `run_on_colab.ipynb` in [Google Colab](https://colab.research.google.com/),
set the runtime to a T4 GPU, and run all cells — it clones, builds, tests, and
benchmarks end to end.

**On a GPU machine (CUDA Toolkit 12.x + PyTorch):**

```bash
pip install -r requirements.txt   # install torch matching your CUDA — see pytorch.org
make                              # compile every kernel into build/*.so
python benchmarks/run_all_benchmarks.py   # all tests + benchmarks

# or one kernel at a time:
python kernels/02_matmul/test_matmul.py
python kernels/02_matmul/benchmark.py
```

Keep every benchmark on the **same** GPU so the numbers are comparable.

## Methodology

- **Timing:** CUDA events (`torch.cuda.Event`), never wall clock. Warm up, then take
  the **median** of 100+ iterations.
- **Correctness:** compared against PyTorch ground truth with `torch.allclose`.
  Elementwise/reduction kernels use `atol=rtol=1e-5`; matmul/attention use a looser
  `1e-2/1e-3` because float32 accumulation order differs from cuBLAS (expected, not a bug).
- **Reporting:** GPU model, CUDA version, driver, and PyTorch version recorded with results.

## Results

Measured on a single **NVIDIA Tesla T4** (Google Colab) — CUDA 12.8, driver 580.82.07,
PyTorch 2.11.0+cu128. Each number is the **median of 100+ CUDA-event-timed runs** after
warmup. Raw data in [`benchmarks/results/`](benchmarks/results/). T4 reference ceilings:
**~320 GB/s** memory bandwidth, **~8.1 TFLOP/s** FP32.

### Matmul (square N×N, GFLOPS = 2N³/t)

| N | naive (ms) | tiled (ms) | cuBLAS (ms) | tiled GFLOPS | tiled vs naive | % of cuBLAS |
|---|----------:|----------:|-----------:|-------------:|---------------:|------------:|
| 512  | 0.836 | 0.545 | 0.105 | 493 | 1.53× | 19.3% |
| 1024 | 4.521 | 2.835 | 0.496 | 758 | 1.60× | 17.5% |
| 2048 | 41.553 | 27.387 | 4.641 | 627 | 1.52× | 16.9% |
| 4096 | not run¹ | 223.081 | 38.396 | 616 | — | 17.2% |

¹ Naive is capped at N≤2048 in `benchmark.py` and never executed at 4096 — it's
impractically slow there (O(N³) work with zero data reuse). Not an out-of-memory error:
a 4096² fp32 matrix is only ~64 MB, and all three fit easily on the 16 GB T4.

### Other kernels

| Kernel | Size | Custom (ms) | PyTorch (ms) | Speedup | Bottleneck |
|--------|------|------------:|-------------:|--------:|------------|
| vector_add | n=2²⁶ (67M) | 3.390 | 3.359 | 0.99× | memory-bound |
| softmax | 4096×16384 | 5.728 | 3.117 | 0.54× | memory-bound |
| layernorm | 8192×4096 | 1.599 | 1.651 | 1.03× | memory-bound |
| attention | seq=1024, d=64 | 1.540 | 0.238 (SDPA) | 0.15× | memory-bound (O(seq²) HBM) |

## Analysis

Read off the CSVs in `benchmarks/results/`. Three patterns hold across the whole suite.

**1. Fixed dispatch overhead dominates small inputs.** Every kernel posts its worst ratio
vs. PyTorch at its *smallest* size — vector_add 0.57× at n=64K, softmax 0.09× at 128 cols,
layernorm 0.46× at 256 cols. The custom path pays a tens-of-µs kernel-launch + `ctypes`
marshalling cost that torch's C++ dispatch avoids; it only amortizes once the kernel does
enough work. The small-input ratios are overhead-bound, not kernel-bound.

**2. The bandwidth-bound kernels already match PyTorch — and can't beat it.** vector_add
peaks at **251 GB/s ≈ 78% of the 320 GB/s roof** (0.99–1.05× torch across large n);
layernorm reaches **179 GB/s** and edges torch at 4096 cols (1.03×). Once a kernel is
memory-bound and near the bandwidth ceiling, "faster" is not on the menu — *matching* a
tuned library is the ceiling. This is the most useful result in the repo: it's direct
evidence the memory-bound diagnosis is correct.

**3. Every remaining gap is a reuse/fusion gap, not a raw-compute gap.** The three kernels
that trail torch each leave memory traffic on the table, and each has a known fix:

- **matmul** — tiled reaches ~758 GFLOP/s (**9% of FP32 peak, 17% of cuBLAS**). The 16×16
  tile cuts *global loads* ~16× in theory but buys only ~1.5× wall-clock: one output per
  thread → low per-thread arithmetic intensity, uncoalesced `B` loads, and sync overhead.
  Note tiled GFLOP/s *peaks at N=1024 (758) then falls* (627 at 2048, 616 at 4096) as the
  working set outgrows L2. Fix: register/2D blocking + coalescing — see *naive → tiled
  matmul* below.
- **softmax** — peaks at **186 GB/s (58% of roof)** at 4096 cols but *drops to 94 GB/s* at
  16384. It's a 3-pass kernel (max → exp+sum → normalize) that re-reads the row; once a row
  (64 KB at 16384) spills L1, the re-reads hit farther-out memory and effective throughput
  falls. Fix: single-pass "online" softmax + warp-shuffle reductions. (The GB/s figure
  counts only minimal read-x/write-y traffic, so it *understates* the true 3-pass traffic.)
- **layernorm** — best in the suite. The **fused single-pass** (accumulate Σx and Σx²
  together, then `var = E[x²] − E[x]²`) reads x once and holds 0.82–1.03× of torch at hidden
  sizes ≥ 1024. Direct confirmation that the fusion is worth it.
- **attention** — 0.15× of SDPA at seq ≥ 1024, with time growing O(seq²) (0.10 → 5.93 ms as
  seq 128 → 2048). It's three separate kernels (naive QKᵀ, tree-softmax, naive ·V) that all
  materialize the seq×seq scores in HBM. SDPA runs a FlashAttention backend that tiles and
  fuses, never writing seq² to HBM — hence the ~6–7× gap. Custom only "wins" (1.4×) at
  seq=128, where SDPA's fixed overhead dominates. See *A note on FlashAttention* below.

## Caveats / threats to validity

- **Python + `ctypes` dispatch** adds fixed per-call overhead absent from torch's native
  dispatch, penalizing the custom kernels at small sizes. Treat small-input ratios as a
  measure of overhead, not kernel quality.
- **The GB/s model counts intended traffic only** (one read of inputs, one write of
  outputs). Multi-pass kernels (softmax) move more, so their effective GB/s understates real
  DRAM traffic — `ncu` would give the true byte counts.
- **Single T4, fp32, one machine.** Each cell is a median of 100+ runs, but only one such
  median — no cross-run variance is reported. cuBLAS/SDPA may switch algorithms by size, so
  this is a "custom vs. whatever torch dispatches" comparison: the honest real-world
  baseline, not an algorithm-matched one.

## Key optimization: naive → tiled matmul

The naive kernel gives every thread one output element and streams a full row of `A`
and column of `B` straight from **global memory**. Across the grid, each `A`/`B`
element is re-fetched O(N) times. Global memory is high-latency and bandwidth-limited,
so the FP32 ALUs stall waiting on loads — the kernel is **memory-bandwidth bound**.

The tiled kernel stages `TILE×TILE` (16×16) blocks of `A` and `B` into **shared
memory** (on-chip, per-SM, ~100× lower latency and far higher bandwidth than global),
and every thread in the block reuses each staged value `TILE` times before advancing
along `K`. That raises **arithmetic intensity**: global-memory traffic drops by ~`TILE`
(≈16×), moving the kernel off the bandwidth wall toward being **compute-bound**. Two
`__syncthreads()` barriers keep the shared tile consistent (fully loaded before use;
fully read before it's overwritten).

On the T4 above, tiling buys a **~1.5× speedup over naive** and reaches **~17% of
cuBLAS**. The theoretical 16× cut in *global loads* doesn't become a 16× wall-clock win:
this basic 16×16 tiling still issues uncoalesced global loads and computes only one
output per thread, so it stays partly memory- and overhead-bound. Closing the remaining
~6× gap to cuBLAS is exactly what coalescing, register blocking, and warptiling do (see
Future work). Profile it (`profiling/nsys_matmul_report.md`) to watch the DRAM-traffic drop directly.

Possible next steps (documented, not yet implemented): register blocking (each thread
computes a micro-tile of outputs), shared-memory padding to avoid bank conflicts, and
vectorized `float4` loads for coalescing.

## A note on FlashAttention

The attention kernel here is the **textbook** formulation: it materializes the full
`seq × seq` scores matrix in global memory, giving O(seq²) HBM traffic — memory-bound
and a hard wall for long sequences. **FlashAttention's** insight is to *never write
that matrix to HBM*: tile Q/K/V and maintain a running (**online**) softmax in on-chip
memory, fusing the two matmuls and the softmax into one pass. That turns attention
compute-bound and makes long context feasible. This lab stops at the baseline that
makes the FlashAttention win legible; `torch.nn.functional.scaled_dot_product_attention`
(benchmarked as `SDPA`) dispatches to such a fused backend.

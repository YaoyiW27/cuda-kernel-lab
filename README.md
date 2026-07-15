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
| 2 | **Matrix Multiply** | **naive → tiled shared-memory matmul** (the headline optimization) | ✅ implemented |
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

## Benchmark summary

> ⏳ **Placeholder tables — fill in from the Colab run** (`run_on_colab.ipynb` prints
> these and writes `benchmarks/results/*.csv`).

**Environment:** _GPU · CUDA · driver · PyTorch — record after running._

### Matmul (square N×N, GFLOPS = 2N³/t)

| N | naive (ms) | tiled (ms) | torch/cuBLAS (ms) | tiled GFLOPS | tiled vs naive | % of cuBLAS |
|---|-----------|-----------|-------------------|--------------|----------------|-------------|
| 512 | — | — | — | — | — | — |
| 1024 | — | — | — | — | — | — |
| 2048 | — | — | — | — | — | — |
| 4096 | — (skipped) | — | — | — | — | — |

### Other kernels

| Kernel | Size | Custom (ms) | PyTorch (ms) | Speedup | Bottleneck |
|--------|------|-------------|--------------|---------|------------|
| vector_add | n=2²⁶ | — | — | — | memory-bound |
| softmax | 4096×16384 | — | — | — | memory-bound |
| layernorm | 8192×4096 | — | — | — | memory-bound |
| attention | seq=1024, d=64 | — | — (vs SDPA) | — | memory-bound (O(seq²) HBM) |

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
fully read before it's overwritten). This is why tiled pulls far ahead of naive as
matrices grow — quantify it from the benchmark table above, then profile it
(`profiling/nsys_matmul_report.md`) to confirm the DRAM-traffic drop.

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

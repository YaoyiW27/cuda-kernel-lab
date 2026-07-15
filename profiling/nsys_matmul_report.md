# Profiling: Tiled Matmul (Nsight)

_Placeholder — fill in after running on a real GPU. This file is where the profiling
evidence for the README lives._

## How to capture

Nsight Systems (timeline / where time goes across kernels):

```bash
nsys profile -o profiling/matmul_nsys \
    python kernels/02_matmul/benchmark.py
# open profiling/matmul_nsys.nsys-rep in the Nsight Systems GUI
```

Nsight Compute (per-kernel details: occupancy, memory vs. compute, bank conflicts):

```bash
# Profile just the tiled kernel (regex matches the kernel name).
ncu --set full -k "matmul_tiled_kernel" -c 1 -o profiling/matmul_tiled_ncu \
    python kernels/02_matmul/benchmark.py
# open profiling/matmul_tiled_ncu.ncu-rep in the Nsight Compute GUI
```

## What to record here

- [ ] Screenshot of the Nsight Compute "GPU Speed Of Light" section for the tiled
      kernel (compute vs. memory throughput %).
- [ ] Naive vs. tiled: DRAM read bytes (expect a large drop for tiled — that's the
      shared-memory reuse) and the achieved-occupancy / SM-utilization difference.
- [ ] Bank-conflict count on the shared-memory loads (baseline for a future
      padded-tile / conflict-avoidance optimization).
- [ ] One or two sentences: is the tiled kernel memory-bound or compute-bound at
      N=2048/4096, and what's the next bottleneck to attack?
```

// Naive element-wise vector add:  c[i] = a[i] + b[i]
//
// This is the "hello world" of CUDA — the point is to get comfortable with the
// programming model (grid/block/thread hierarchy, kernel launch) and to measure
// how a trivially memory-bound kernel compares to PyTorch's native add.
//
// Build (via the repo Makefile):  make   ->  build/vector_add.so
// The extern "C" wrapper below is what Python loads through ctypes.

#include <cuda_runtime.h>

// One thread per element, plus a grid-stride loop so a single launch config works
// for any n (large or small) without launching an absurd number of blocks.
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

extern "C" {

// Launch the kernel on pointers that ALREADY live on the GPU (e.g. a PyTorch CUDA
// tensor's .data_ptr()). We deliberately do NOT allocate or copy here so the Python
// benchmark can time only the kernel itself with CUDA events — no H2D/D2H transfer
// muddying the measurement.
void vector_add(const float* d_a, const float* d_b, float* d_c, int n) {
    const int block = 256;                     // threads per block (multiple of warp size)
    int grid = (n + block - 1) / block;        // enough blocks to cover n, one thread/element

    // Cap the grid; the grid-stride loop in the kernel handles the remainder, so we
    // never need more than this many blocks even for very large n.
    const int max_grid = 65535;
    if (grid > max_grid) grid = max_grid;
    if (grid < 1)        grid = 1;

    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
}

}  // extern "C"

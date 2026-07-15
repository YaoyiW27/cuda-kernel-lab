// Row-wise softmax with numerical stability.
// Input x is (rows, cols), row-major. For each row we compute:
//     m   = max_j x[j]                 (subtract max for numerical stability)
//     e_j = exp(x[j] - m)
//     y_j = e_j / sum_j e_j
//
// KEY PATTERNS:
//  1. Numerical stability: exp() of a large value overflows to inf. Subtracting the
//     row max keeps every exponent <= 0, so exp() stays in (0, 1]. This is exactly
//     what torch.softmax does internally.
//  2. Parallel reduction: computing the row max and the row sum are both reductions.
//     One block handles one row; threads first reduce their strided slice into a
//     per-thread partial, then a tree reduction in shared memory combines partials
//     in log2(blockDim) steps. __syncthreads() separates each step.
//
// Launch: grid = rows, block = 256 threads (power of two, required by the tree
// reduction below), shared memory = blockDim * sizeof(float).

#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* xrow = x + (size_t)row * cols;
    float* yrow = y + (size_t)row * cols;

    // --- Pass 1: row max (reduction) ---
    float localMax = -INFINITY;
    for (int i = tid; i < cols; i += nthreads)
        localMax = fmaxf(localMax, xrow[i]);
    sdata[tid] = localMax;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float rowMax = sdata[0];
    __syncthreads();

    // --- Pass 2: exp(x - max) into y, and sum (reduction) ---
    float localSum = 0.0f;
    for (int i = tid; i < cols; i += nthreads) {
        float e = expf(xrow[i] - rowMax);
        yrow[i] = e;           // stash numerator; normalize after we know the sum
        localSum += e;
    }
    sdata[tid] = localSum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rowSum = sdata[0];
    __syncthreads();

    // --- Pass 3: normalize ---
    float inv = 1.0f / rowSum;
    for (int i = tid; i < cols; i += nthreads)
        yrow[i] *= inv;
}

extern "C" {

void softmax(const float* d_x, float* d_y, int rows, int cols) {
    int block = 256;  // must be a power of two for the tree reduction
    dim3 grid(rows);
    size_t shmem = block * sizeof(float);
    softmax_kernel<<<grid, block, shmem>>>(d_x, d_y, rows, cols);
}

}  // extern "C"

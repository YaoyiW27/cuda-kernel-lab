// LayerNorm over the last dimension.
// Input x is (rows, cols); gamma/beta are length-cols learnable affine params.
// For each row:
//     mean = (1/C) * sum_j x[j]
//     var  = (1/C) * sum_j (x[j] - mean)^2
//     y[j] = (x[j] - mean) / sqrt(var + eps) * gamma[j] + beta[j]
//
// KEY OPTIMIZATION — fused single pass for mean AND variance:
// The naive approach reads the row twice: once to get the mean, then again to sum
// squared deviations. Instead we accumulate sum(x) and sum(x^2) in ONE pass, then
// use var = E[x^2] - E[x]^2. That halves global-memory reads of x (it's the
// memory-bound part). Both sums are parallel reductions in shared memory, same tree
// pattern as softmax. We reduce two quantities, reusing one shared buffer sequentially.
//
// (Note: E[x^2] - E[x]^2 can lose precision vs a two-pass/Welford scheme for very
// large magnitudes; for normalized-ish activations in float32 it matches torch to
// ~1e-4, which the test allows.)
//
// Launch: grid = rows, block = 256 (power of two), shared = block * sizeof(float).

#include <cuda_runtime.h>
#include <math.h>

__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y,
                                 int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int n = blockDim.x;
    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    // Single pass: accumulate sum and sum-of-squares per thread.
    float s = 0.0f, ss = 0.0f;
    for (int i = tid; i < cols; i += n) {
        float v = xr[i];
        s += v;
        ss += v * v;
    }

    // Reduce sum(x) -> mean.
    sdata[tid] = s;
    __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float mean = sdata[0] / cols;
    __syncthreads();

    // Reduce sum(x^2) -> E[x^2].
    sdata[tid] = ss;
    __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float meanSq = sdata[0] / cols;
    __syncthreads();

    float var = meanSq - mean * mean;
    float invStd = rsqrtf(var + eps);

    for (int i = tid; i < cols; i += n) {
        float norm = (xr[i] - mean) * invStd;
        yr[i] = norm * gamma[i] + beta[i];
    }
}

extern "C" {

void layernorm(const float* d_x, const float* d_gamma, const float* d_beta,
               float* d_y, int rows, int cols, float eps) {
    int block = 256;  // power of two for the tree reduction
    dim3 grid(rows);
    size_t shmem = block * sizeof(float);
    layernorm_kernel<<<grid, block, shmem>>>(d_x, d_gamma, d_beta, d_y,
                                             rows, cols, eps);
}

}  // extern "C"

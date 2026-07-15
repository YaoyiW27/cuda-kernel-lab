// Naive matrix multiply:  C = A @ B, where A is M x K, B is K x N, C is M x N.
// All matrices are row-major.
//
// Each thread computes ONE output element C[row][col] by walking the full K
// dimension, reading A and B straight from global memory. This is the baseline
// we optimize against in matmul_tiled.cu.
//
// Why it's slow: every thread re-reads an entire row of A and column of B from
// global memory. Across the whole grid, each A/B element is fetched from global
// memory O(N)/O(M) times. Global memory has high latency and limited bandwidth,
// so the kernel is memory-bandwidth bound — the FP32 ALUs sit idle waiting on loads.

#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row  (0..M-1)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output col  (0..N-1)

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

extern "C" {

// Operates on device pointers so Python can time only the kernel (no copies).
void matmul_naive(const float* d_A, const float* d_B, float* d_C,
                  int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

}  // extern "C"

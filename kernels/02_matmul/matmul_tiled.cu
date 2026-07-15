// Tiled (shared-memory) matrix multiply:  C = A @ B, A is M x K, B is K x N.
// All matrices are row-major. This is the key optimization exercise in the lab.
//
// THE IDEA — reuse through shared memory:
// The naive kernel re-reads every A/B element from global memory many times.
// Global memory is high-latency and bandwidth-limited; shared memory (on-chip,
// per-SM) is ~100x lower latency and much higher bandwidth. So we load small
// TILE x TILE blocks of A and B into shared memory ONCE, and every thread in the
// block reuses them TILE times before moving to the next tile along K.
//
// Arithmetic intensity goes up: with a TILE x TILE tile, each global-memory load
// is reused ~TILE times, cutting global traffic by ~TILE (16x here). That moves
// the kernel off the memory-bandwidth wall toward being compute-bound — which is
// why tiled matmul is dramatically faster than naive for large matrices.
//
// __syncthreads() barriers make sure the whole tile is loaded before anyone reads
// it, and that everyone is done reading before the tile is overwritten next round.

#include <cuda_runtime.h>

#define TILE 16  // tile / block dimension; block is TILE x TILE threads

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;  // output row this thread writes
    int col = blockIdx.x * TILE + tx;  // output col this thread writes

    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        // Cooperatively stage one TILE x TILE block of A and of B into shared memory.
        // Guard the edges (K, M, N not multiples of TILE) by zero-filling.
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();  // tile fully loaded before use

        // Multiply the two staged tiles; each shared value is reused TILE times.
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // done reading before the next tile overwrites shared mem
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

extern "C" {

void matmul_tiled(const float* d_A, const float* d_B, float* d_C,
                  int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

}  // extern "C"

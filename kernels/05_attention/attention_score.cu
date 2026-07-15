// Scaled dot-product attention (single head):
//     out = softmax(Q @ K^T / sqrt(d)) @ V
// Shapes (row-major): Q (seq, d), K (seq, d), V (seq, d), out (seq, d).
// `scores` is a caller-provided (seq, seq) scratch buffer so the timed path does no
// cudaMalloc.
//
// This kernel COMPOSES the earlier lessons — a matmul, a row softmax, another matmul —
// done as three straightforward launches:
//   1. qk_kernel : scores = (Q @ K^T) * scale        (naive matmul, K read as rows)
//   2. softmax   : scores <- softmax(scores, axis=-1) (stable row softmax)
//   3. av_kernel : out = scores @ V                    (naive matmul)
//
// This is intentionally the *textbook* formulation: it materializes the full
// seq x seq scores matrix in global memory (O(seq^2) HBM traffic). FlashAttention's
// insight is to AVOID that: tile Q/K/V and keep a running ("online") softmax in
// on-chip memory so the seq x seq matrix is never written to HBM — turning attention
// from memory-bound into compute-bound and enabling long sequences. We don't
// implement FlashAttention here, but that's the gap this baseline illustrates.

#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

// scores[i][j] = scale * dot(Q[i, :], K[j, :])
__global__ void qk_kernel(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          float* __restrict__ scores,
                          int seq, int d, float scale) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key index
    if (i < seq && j < seq) {
        float acc = 0.0f;
        for (int k = 0; k < d; ++k)
            acc += Q[i * d + k] * K[j * d + k];       // K^T => index K by row j
        scores[i * seq + j] = acc * scale;
    }
}

// Row-wise stable softmax over `scores` (seq x seq): one block per row.
__global__ void softmax_rows_kernel(float* __restrict__ scores, int seq) {
    int row = blockIdx.x;
    if (row >= seq) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x, n = blockDim.x;
    float* s = scores + (size_t)row * seq;

    float localMax = -INFINITY;
    for (int j = tid; j < seq; j += n) localMax = fmaxf(localMax, s[j]);
    sdata[tid] = localMax; __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float rowMax = sdata[0]; __syncthreads();

    float localSum = 0.0f;
    for (int j = tid; j < seq; j += n) {
        float e = expf(s[j] - rowMax);
        s[j] = e;
        localSum += e;
    }
    sdata[tid] = localSum; __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0]; __syncthreads();
    for (int j = tid; j < seq; j += n) s[j] *= inv;
}

// out[i][k] = dot(scores[i, :], V[:, k])
__global__ void av_kernel(const float* __restrict__ scores,
                          const float* __restrict__ V,
                          float* __restrict__ out,
                          int seq, int d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // value dim
    if (i < seq && k < d) {
        float acc = 0.0f;
        for (int j = 0; j < seq; ++j)
            acc += scores[i * seq + j] * V[j * d + k];
        out[i * d + k] = acc;
    }
}

extern "C" {

// scores must point to a (seq*seq) float device buffer (scratch).
void attention(const float* d_Q, const float* d_K, const float* d_V,
               float* d_scores, float* d_out, int seq, int d) {
    float scale = 1.0f / sqrtf((float)d);

    dim3 block(TILE, TILE);
    dim3 gridQK((seq + TILE - 1) / TILE, (seq + TILE - 1) / TILE);
    qk_kernel<<<gridQK, block>>>(d_Q, d_K, d_scores, seq, d, scale);

    int sm_block = 256;
    softmax_rows_kernel<<<seq, sm_block, sm_block * sizeof(float)>>>(d_scores, seq);

    dim3 gridAV((d + TILE - 1) / TILE, (seq + TILE - 1) / TILE);
    av_kernel<<<gridAV, block>>>(d_scores, d_V, d_out, seq, d);
}

}  // extern "C"

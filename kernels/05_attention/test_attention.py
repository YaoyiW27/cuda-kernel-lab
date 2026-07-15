"""Correctness test for the custom attention kernel vs a PyTorch reference.

Reference: out = softmax(Q @ K^T / sqrt(d)) @ V (and cross-checked against
torch.nn.functional.scaled_dot_product_attention). Tolerance is loose (1e-2/1e-3):
this chains two float32 matmuls and a softmax, so accumulation-order differences add
up. Expected, not a bug.

Run from the repo root after `make`:

    python kernels/05_attention/test_attention.py
"""

import ctypes
import math
import os

import torch
import torch.nn.functional as F

_LIB = os.path.join(os.path.dirname(__file__), "..", "..", "build", "attention_score.so")


def load_lib(path: str = _LIB) -> ctypes.CDLL:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run `make` first.")
    lib = ctypes.CDLL(path)
    # void attention(const float* Q, const float* K, const float* V,
    #                float* scores, float* out, int seq, int d)
    lib.attention.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_int]
    lib.attention.restype = None
    return lib


def custom_attention(lib, Q, K, V):
    """Single-head attention. Q,K,V: (seq, d) CUDA tensors -> out (seq, d)."""
    assert Q.is_cuda and Q.is_contiguous() and Q.dim() == 2
    seq, d = Q.shape
    scores = torch.empty((seq, seq), device="cuda", dtype=torch.float32)  # scratch
    out = torch.empty((seq, d), device="cuda", dtype=torch.float32)
    lib.attention(Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                  scores.data_ptr(), out.data_ptr(), seq, d)
    torch.cuda.synchronize()
    return out


def reference(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    return torch.softmax((Q @ K.transpose(-1, -2)) * scale, dim=-1) @ V


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this test.")

    lib = load_lib()
    shapes = [(128, 64), (256, 64), (512, 128), (300, 96)]  # (seq, d)

    for (seq, d) in shapes:
        Q = torch.randn(seq, d, device="cuda", dtype=torch.float32)
        K = torch.randn(seq, d, device="cuda", dtype=torch.float32)
        V = torch.randn(seq, d, device="cuda", dtype=torch.float32)

        out = custom_attention(lib, Q, K, V)
        ref = reference(Q, K, V)
        sdpa = F.scaled_dot_product_attention(Q, K, V)  # sanity cross-check

        if not torch.allclose(out, ref, atol=1e-2, rtol=1e-3):
            raise SystemExit(f"FAIL seq={seq} d={d}: max err {(out-ref).abs().max():.2e}")
        assert torch.allclose(ref, sdpa, atol=1e-2, rtol=1e-3), "reference disagrees with SDPA"
        print(f"seq={seq:>4} d={d:>4}  OK")

    print("All attention correctness tests passed.")


if __name__ == "__main__":
    main()

"""Correctness tests for the naive and tiled matmul kernels vs torch.matmul.

Note on tolerance: float32 matmul accumulates rounding error over the K dimension,
and the custom kernels sum in a different order than cuBLAS. So we use a looser
tolerance than the elementwise kernels (atol=1e-2, rtol=1e-3) — this is expected,
not a bug. Correctness here means "same result up to float32 accumulation order."

Run from the repo root after `make`:

    python kernels/02_matmul/test_matmul.py
"""

import ctypes
import os

import torch

_BUILD = os.path.join(os.path.dirname(__file__), "..", "..", "build")


def _load(name: str, symbol: str) -> ctypes.CDLL:
    path = os.path.abspath(os.path.join(_BUILD, name))
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run `make` first.")
    lib = ctypes.CDLL(path)
    fn = getattr(lib, symbol)
    # void f(const float* A, const float* B, float* C, int M, int N, int K)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = None
    return lib


def load_libs():
    naive = _load("matmul_naive.so", "matmul_naive")
    tiled = _load("matmul_tiled.so", "matmul_tiled")
    return naive, tiled


def custom_matmul(lib: ctypes.CDLL, symbol: str,
                  A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B using the named kernel. A: (M,K), B: (K,N) -> C: (M,N)."""
    assert A.is_cuda and B.is_cuda and A.is_contiguous() and B.is_contiguous()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    getattr(lib, symbol)(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    return C


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this test.")

    naive, tiled = load_libs()
    # Include a non-square, non-power-of-two case to exercise the edge guards.
    shapes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (300, 200, 150)]

    for (M, N, K) in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = A @ B

        for symbol, lib in [("matmul_naive", naive), ("matmul_tiled", tiled)]:
            C = custom_matmul(lib, symbol, A, B)
            if not torch.allclose(C, ref, atol=1e-2, rtol=1e-3):
                max_err = (C - ref).abs().max().item()
                raise SystemExit(f"FAIL {symbol} M={M} N={N} K={K}: max err {max_err:.2e}")
            print(f"{symbol:>13}  M={M:>4} N={N:>4} K={K:>4}  OK")

    print("All matmul correctness tests passed.")


if __name__ == "__main__":
    main()

"""Correctness test for the custom LayerNorm kernel vs torch.nn.functional.layer_norm.

Tolerance is 1e-4 (not 1e-5): the fused var = E[x^2] - E[x]^2 formula and cuBLAS-style
reductions differ slightly in float32. Expected, not a bug.

Run from the repo root after `make`:

    python kernels/04_layernorm/test_layernorm.py
"""

import ctypes
import os

import torch
import torch.nn.functional as F

_LIB = os.path.join(os.path.dirname(__file__), "..", "..", "build", "layernorm.so")


def load_lib(path: str = _LIB) -> ctypes.CDLL:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run `make` first.")
    lib = ctypes.CDLL(path)
    # void layernorm(const float* x, const float* gamma, const float* beta,
    #                float* y, int rows, int cols, float eps)
    lib.layernorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                              ctypes.c_float]
    lib.layernorm.restype = None
    return lib


def custom_layernorm(lib, x, gamma, beta, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and x.dim() == 2
    rows, cols = x.shape
    y = torch.empty_like(x)
    lib.layernorm(x.data_ptr(), gamma.data_ptr(), beta.data_ptr(), y.data_ptr(),
                  rows, cols, ctypes.c_float(eps))
    torch.cuda.synchronize()
    return y


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this test.")

    lib = load_lib()
    eps = 1e-5
    shapes = [(32, 128), (128, 768), (64, 4096), (256, 12288)]

    for (rows, cols) in shapes:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
        beta = torch.randn(cols, device="cuda", dtype=torch.float32)

        y = custom_layernorm(lib, x, gamma, beta, eps)
        ref = F.layer_norm(x, (cols,), weight=gamma, bias=beta, eps=eps)
        if not torch.allclose(y, ref, atol=1e-4, rtol=1e-4):
            max_err = (y - ref).abs().max().item()
            raise SystemExit(f"FAIL rows={rows} cols={cols}: max err {max_err:.2e}")
        print(f"rows={rows:>4} cols={cols:>6}  OK")

    print("All LayerNorm correctness tests passed.")


if __name__ == "__main__":
    main()

"""Correctness test for the custom softmax kernel vs torch.softmax(dim=-1).

Run from the repo root after `make`:

    python kernels/03_softmax/test_softmax.py
"""

import ctypes
import os

import torch

_LIB = os.path.join(os.path.dirname(__file__), "..", "..", "build", "softmax.so")


def load_lib(path: str = _LIB) -> ctypes.CDLL:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run `make` first.")
    lib = ctypes.CDLL(path)
    # void softmax(const float* x, float* y, int rows, int cols)
    lib.softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int]
    lib.softmax.restype = None
    return lib


def custom_softmax(lib: ctypes.CDLL, x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D (rows, cols) CUDA tensor."""
    assert x.is_cuda and x.is_contiguous() and x.dim() == 2
    rows, cols = x.shape
    y = torch.empty_like(x)
    lib.softmax(x.data_ptr(), y.data_ptr(), rows, cols)
    torch.cuda.synchronize()
    return y


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this test.")

    lib = load_lib()
    shapes = [(32, 128), (128, 1024), (64, 4096), (256, 30000)]

    for (rows, cols) in shapes:
        # Scale up inputs to stress numerical stability (large values -> exp overflow
        # if the max isn't subtracted).
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32) * 10.0
        y = custom_softmax(lib, x)
        ref = torch.softmax(x, dim=-1)
        if not torch.allclose(y, ref, atol=1e-5, rtol=1e-5):
            max_err = (y - ref).abs().max().item()
            raise SystemExit(f"FAIL rows={rows} cols={cols}: max err {max_err:.2e}")
        print(f"rows={rows:>4} cols={cols:>6}  OK")

    print("All softmax correctness tests passed.")


if __name__ == "__main__":
    main()

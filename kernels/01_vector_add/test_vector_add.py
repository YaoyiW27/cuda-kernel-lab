"""Correctness test for the custom vector_add CUDA kernel.

Ground truth is PyTorch's own `a + b`. We compare with torch.allclose at float32
tolerance (atol=rtol=1e-5). Run from the repo root after `make`:

    python kernels/01_vector_add/test_vector_add.py
"""

import ctypes
import os

import torch

_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "build", "vector_add.so"
)


def load_lib(path: str = _LIB_PATH) -> ctypes.CDLL:
    """Load the compiled kernel and declare the vector_add signature for ctypes."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — build the kernels first with `make`."
        )
    lib = ctypes.CDLL(path)
    # void vector_add(const float* a, const float* b, float* c, int n)
    lib.vector_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.vector_add.restype = None
    return lib


def custom_vector_add(lib: ctypes.CDLL, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run the custom kernel on two CUDA tensors and return the result tensor."""
    assert a.is_cuda and b.is_cuda, "inputs must be CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.is_contiguous() and b.is_contiguous()
    c = torch.empty_like(a)
    lib.vector_add(a.data_ptr(), b.data_ptr(), c.data_ptr(), a.numel())
    torch.cuda.synchronize()  # kernel launch is async — wait before reading results
    return c


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run this test.")

    lib = load_lib()
    sizes = [1, 1000, 1 << 20, (1 << 24) + 7]  # include a non-power-of-two size

    for n in sizes:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        c = custom_vector_add(lib, a, b)
        ref = a + b
        if not torch.allclose(c, ref, atol=1e-5, rtol=1e-5):
            max_err = (c - ref).abs().max().item()
            raise SystemExit(f"FAIL at n={n}: max abs error = {max_err:.2e}")
        print(f"n={n:>12}  OK")

    print("All correctness tests passed.")


if __name__ == "__main__":
    main()

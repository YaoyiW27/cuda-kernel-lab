"""Run every kernel's correctness test and benchmark, in order.

Each kernel is self-contained, so we invoke its scripts as subprocesses (this also
keeps each script's local imports working without package plumbing). Per-kernel CSVs
land in benchmarks/results/; this runner just orchestrates and stops on first failure.

Usage (from the repo root, after `make`):

    python benchmarks/run_all_benchmarks.py
"""

import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# (label, test script, benchmark script) in the intended learning order.
KERNELS = [
    ("01 vector_add", "kernels/01_vector_add/test_vector_add.py", "kernels/01_vector_add/benchmark.py"),
    ("02 matmul",     "kernels/02_matmul/test_matmul.py",         "kernels/02_matmul/benchmark.py"),
    ("03 softmax",    "kernels/03_softmax/test_softmax.py",       "kernels/03_softmax/benchmark.py"),
    ("04 layernorm",  "kernels/04_layernorm/test_layernorm.py",   "kernels/04_layernorm/benchmark.py"),
    ("05 attention",  "kernels/05_attention/test_attention.py",   "kernels/05_attention/benchmark.py"),
]


def run(script: str) -> None:
    print(f"\n$ python {script}")
    subprocess.run([sys.executable, os.path.join(_ROOT, script)], check=True, cwd=_ROOT)


def main() -> None:
    for label, test, bench in KERNELS:
        print(f"\n{'=' * 60}\n {label}\n{'=' * 60}")
        run(test)
        run(bench)
    print("\nAll kernels passed correctness and produced benchmark CSVs in "
          "benchmarks/results/.")


if __name__ == "__main__":
    main()

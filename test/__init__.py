"""Shared operator benchmark framework (see test/README.md).

Modules:
    config     — constants, TEST_BACKENDS knob, OPERATORS, BASELINES
    gpu        — idle-GPU auto-selection
    ops        — load_op(): import an operator's op.py from config.OPERATORS
    registry   — register_op / register_baseline, OpImpl / OpResult / SweepResult
    verify     — correctness verification against the operator reference
    benchmark  — performance benchmarking (GPU events / CPU wall clock)
    plot       — comparison and roofline plots
    native     — generic C++/CUDA shared-library build & load (NativeLib)

Notebook usage:
    from test import *
    import test.config as config   # live settings, e.g. config.TEST_BACKENDS = "cpu"
    op = load_op("gemm")
    device = select_gpu()
"""

from . import config
from .config import (CPU_TARGET_MS, DEFAULT_ITERS, DEFAULT_WARMUP, DTYPE,
                     DTYPE_NP, ERROR_ATOL, ERROR_RTOL)
from .gpu import pick_idle_gpu, query_gpu_status, select_gpu
from .ops import list_ops, load_op
from .registry import (OpImpl, OpResult, SweepResult, get_baseline,
                       get_implementation, list_implementations,
                       register_baseline, register_op, resolve_backends)
from .verify import verify_all, verify_op
from .benchmark import benchmark_all, benchmark_op
from .plot import plot_comparison, plot_roofline
from .native import NativeLib, compile_cpp, compile_cu

__all__ = [
    "config", "DTYPE", "DTYPE_NP", "DEFAULT_WARMUP", "DEFAULT_ITERS",
    "CPU_TARGET_MS", "ERROR_RTOL", "ERROR_ATOL",
    "select_gpu", "pick_idle_gpu", "query_gpu_status",
    "list_ops", "load_op",
    "register_op", "register_baseline", "get_implementation",
    "get_baseline", "list_implementations", "resolve_backends",
    "OpImpl", "OpResult", "SweepResult",
    "verify_op", "verify_all", "benchmark_op", "benchmark_all",
    "plot_comparison", "plot_roofline",
    "NativeLib", "compile_cpp", "compile_cu",
]

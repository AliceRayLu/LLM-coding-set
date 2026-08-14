"""GEMM benchmark framework.

Modules:
    config     — constants and the TEST_BACKENDS knob
    gpu        — idle-GPU auto-selection
    registry   — implementation registry (register_gemm)
    verify     — correctness verification against NumPy
    benchmark  — performance benchmarking (GPU events / CPU wall clock)
    plot       — comparison and roofline plots
    native     — C++/CUDA shared-library build, load, and registration

Notebook usage:
    from test import *
    import test.config as config   # live settings, e.g. config.TEST_BACKENDS = "cpu"
    device = select_gpu()
"""

from . import config
from .config import (CPU_TARGET_MS, DEFAULT_ITERS, DEFAULT_WARMUP, DTYPE,
                     DTYPE_NP, ERROR_ATOL, ERROR_RTOL)
from .gpu import pick_idle_gpu, query_gpu_status, select_gpu
from .registry import (GemmImpl, GemmResult, SweepResult, get_implementation,
                       list_implementations, register_gemm, resolve_backends)
from .verify import verify_all, verify_gemm
from .benchmark import benchmark_all, benchmark_gemm
from .plot import plot_gemm_comparison, plot_roofline
from .native import (CPP_IMPLS, CU_IMPLS, compile_cpp, compile_cu, cpp_wrapper,
                     cu_wrapper, find_src, prepare_all)

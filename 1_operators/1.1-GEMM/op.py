"""GEMM operator module for the shared benchmark framework (see test/README.md).

Protocol: NAME / DEFAULT_SIZES / gen_inputs / metric / register are required;
reference, prepare and the label/format hooks are optional. Registered
implementations take fn(*args) -> out with args from gen_inputs — scalar
dimensions are derived from array shapes, so every fn must be size-generic.
"""

import ctypes
from pathlib import Path

import cupy as cp
import numpy as np
from cupy.cuda import cublas

from test import config
from test.native import NativeLib, compile_cpp, compile_cu
from test.registry import (list_implementations, register_baseline,
                           register_op, resolve_backends)

# ─── protocol fields ───

NAME = "gemm"
SIZE_LABEL = "Matrix Size (M = N = K)"
SIZE_COLS = ["M", "N", "K"]
METRIC_LABEL = "GFLOPS"
METRIC_FMT = ".1f"

DEFAULT_SIZES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (256, 512, 128),   # rectangular
    (512, 256, 1024),  # rectangular, large K
]

_FOLDER = Path(__file__).parent  # kernel sources live next to this module


def gen_inputs(size, backend, seed):
    """Return (A, B) with A (M×K), B (K×N) on the requested backend."""
    M, N, K = size
    rng = np.random.RandomState(seed)
    A = rng.randn(M, K).astype(config.DTYPE_NP)
    B = rng.randn(K, N).astype(config.DTYPE_NP)
    if backend == "gpu":
        return cp.asarray(A), cp.asarray(B)
    return np.ascontiguousarray(A), np.ascontiguousarray(B)


def metric(size, time_ms):
    M, N, K = size
    return (2.0 * M * N * K) / (time_ms / 1000.0) / 1e9


def reference(A, B):
    """NumPy FP32 ground truth for correctness verification."""
    return A @ B


def size_key(size):
    return size[0]


def flops(size):
    M, N, K = size
    return 2.0 * M * N * K


def bytes_read(size):
    M, N, K = size
    return (M * K + K * N) * 4


# ─── register(): Python implementations & baselines (called by load_op) ───

_alpha = np.array([1.0], dtype=config.DTYPE_NP)
_beta = np.array([0.0], dtype=config.DTYPE_NP)


def register():
    @register_op(NAME, "NumPy BLAS", backend="cpu")
    def numpy_blas(A, B):
        """NumPy matmul — the optimized multithreaded BLAS. CPU baseline."""
        return A @ B

    @register_op(NAME, "cuBLAS")
    def cublas_sgemm(A, B):
        """cuBLAS SGEMM. Column-major trick:
        C_row(M×N) = A_row(M×K) @ B_row(K×N)
        → C_col(N×M) = B_col(N×K) @ A_col(K×M) in cuBLAS.
        """
        M, K = A.shape
        N = B.shape[1]
        handle = cp.cuda.Device().cublas_handle
        C = cp.zeros((M, N), dtype=config.DTYPE)
        cublas.sgemm(handle,
                     cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
                     N, M, K,                              # m=N, n=M, k=K
                     _alpha.ctypes.data, B.data.ptr, N,     # A_arg = B_col(N×K)
                     A.data.ptr, K,                          # B_arg = A_col(K×M)
                     _beta.ctypes.data, C.data.ptr, N)       # C_col(N×M)
        return C

    @register_op(NAME, "cuBLAS (cupy)")
    def cublas_cupy(A, B):
        """cuBLAS via cp.matmul — convenience wrapper, minor overhead."""
        return cp.matmul(A, B)

    register_baseline(NAME, "cpu", "NumPy BLAS")
    register_baseline(NAME, "gpu", "cuBLAS")


# ─── native implementations (C++/CUDA): build, load, wrappers ───

_cpp = NativeLib(_FOLDER / "gemm_cpu.cpp", compile_cpp)
_cu = NativeLib(_FOLDER / "gemm_gpu.cu", compile_cu)

# (C++ function, display name, max_size, warmup) — max_size gates keep the
# slow single-thread variants from stalling the sweep.
CPP_IMPLS = [
    ("sgemm_naive", "C++ naive (ijk)", 1024, 1),  # stride-N reads, ~2-5 GFLOPS
    ("sgemm_ikj",   "C++ ikj",         4096, 1),  # vectorized inner loop
    ("sgemm_tiled", "C++ tiled",       4096, 1),  # cache-blocked ikj
    ("sgemm_omp",   "C++ OpenMP",      8192, 1),  # row-parallel ikj
]

# (launcher function, display name) — add new kernels here.
CU_IMPLS = [
    ("sgemm_naive_launch", "Naive"),
    ("sgemm_tile_launch",  "tile"),
]

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_INT = ctypes.c_int
_VOID_P = ctypes.c_void_p


def _cpp_wrapper(lib, func_name):
    """Wrap a C++ sgemm(M, N, K, A, B, C); C must be zero-initialized."""
    fn = getattr(lib, func_name)
    fn.argtypes = [_INT, _INT, _INT, _FLOAT_PTR, _FLOAT_PTR, _FLOAT_PTR]
    fn.restype = None

    def gemm(A, B):
        M, K = A.shape
        N = B.shape[1]
        A = np.ascontiguousarray(A, dtype=config.DTYPE_NP)
        B = np.ascontiguousarray(B, dtype=config.DTYPE_NP)
        C = np.zeros((M, N), dtype=config.DTYPE_NP)
        fn(M, N, K,
           A.ctypes.data_as(_FLOAT_PTR),
           B.ctypes.data_as(_FLOAT_PTR),
           C.ctypes.data_as(_FLOAT_PTR))
        return C

    gemm.__name__ = func_name
    gemm.__doc__ = f"C++ {func_name} via ctypes — see gemm_cpu.cpp"
    return gemm


def _cu_wrapper(lib, func_name):
    """Wrap a CUDA launcher(M, N, K, A, B, C, cudaStream_t).

    Launches on the default stream so CUDA-event timing measures the kernel.
    """
    fn = getattr(lib, func_name)
    fn.argtypes = [_INT, _INT, _INT, _VOID_P, _VOID_P, _VOID_P, _VOID_P]
    fn.restype = None

    def gemm(A, B):
        M, K = A.shape
        N = B.shape[1]
        C = cp.zeros((M, N), dtype=config.DTYPE)
        fn(M, N, K, A.data.ptr, B.data.ptr, C.data.ptr, cp.cuda.Stream.null.ptr)
        return C

    gemm.__name__ = func_name
    gemm.__doc__ = f"CUDA {func_name} via ctypes — see gemm_gpu.cu"
    return gemm


def prepare():
    """(Re)compile stale sources, (re)load libraries, (re)register natives.

    Call from the Results cells: after editing gemm_cpu.cpp / gemm_gpu.cu
    you only need to re-run the Results cells.
    """
    active = resolve_backends()
    if "cpu" in active:
        lib = _cpp.load()
        for fname, disp, max_size, warmup in CPP_IMPLS:
            if hasattr(lib, fname):  # sgemm_omp absent if built without OpenMP
                register_op(NAME, disp, backend="cpu",
                            max_size=max_size, warmup=warmup)(
                    _cpp_wrapper(lib, fname))
            else:
                print(f"  (skipping {disp}: {fname} not in library)")
    if "gpu" in active:
        lib = _cu.load()
        for fname, disp in CU_IMPLS:
            if hasattr(lib, fname):
                register_op(NAME, disp)(_cu_wrapper(lib, fname))
            else:
                print(f"  (skipping {disp}: {fname} not in library)")
    print(f"Ready: {len(list_implementations(NAME))} implementations registered.")

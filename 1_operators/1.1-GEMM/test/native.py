"""Build & load native GEMM implementations (C++ / CUDA shared libraries).

Compiled artifacts are versioned by source mtime+size, so an edited source
always compiles to a fresh path — dlopen then maps fresh code (a same-path
dlopen would return the old mapping). Old artifacts are cleaned up on load.
"""

import ctypes
import pathlib
import subprocess
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np

from . import config
from .registry import list_implementations, register_gemm, resolve_backends


def find_src(filename: str) -> pathlib.Path:
    """Locate a source file next to the notebook (this package's parent)."""
    p = pathlib.Path(__file__).resolve().parent.parent / filename
    if not p.exists():
        raise FileNotFoundError(f"{filename} not found in {p.parent}")
    return p


# ─── versioned build ───

def _artifact(src: pathlib.Path) -> pathlib.Path:
    st = src.stat()
    return src.with_name(f"{src.stem}.{st.st_mtime_ns}-{st.st_size}.so")


def _run(cmd: List[str]) -> None:
    print("Compiling:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{r.stderr}")


def compile_cpp(src: pathlib.Path,
                extra_flags: Optional[List[str]] = None) -> pathlib.Path:
    """g++ build of a .cpp source (cached; retries without OpenMP)."""
    out = _artifact(src)
    if out.exists():
        return out
    flags = ["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC"]
    try:
        _run(flags + (extra_flags or []) + ["-o", str(out), str(src)])
    except RuntimeError:
        print("Retrying without OpenMP ...")
        _run([f for f in flags if f != "-fopenmp"]
             + (extra_flags or []) + ["-o", str(out), str(src)])
    return out


def compile_cu(src: pathlib.Path,
               extra_flags: Optional[List[str]] = None) -> pathlib.Path:
    """nvcc build of a .cu source (cached; arch from the current GPU)."""
    out = _artifact(src)
    if out.exists():
        return out
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    flags = ["nvcc", "-O3", f"-arch=sm_{props['major']}{props['minor']}",
             "-Xcompiler", "-fPIC", "-shared"]
    _run(flags + (extra_flags or []) + ["-o", str(out), str(src)])
    return out


class _NativeLib:
    """One native source file: versioned build, load, artifact cleanup."""

    def __init__(self, src: pathlib.Path, compile_fn: Callable):
        self.src = src
        self.compile_fn = compile_fn
        self.lib: Optional[ctypes.CDLL] = None
        self.loaded_path: Optional[str] = None

    def load(self) -> ctypes.CDLL:
        """(Re)compile if the source changed, then (re)load."""
        out = self.compile_fn(self.src)
        if self.loaded_path != str(out):
            self.lib = ctypes.CDLL(str(out))
            self.loaded_path = str(out)
            self._cleanup(out)
            print(f"Loaded {out.name}")
        return self.lib

    def _cleanup(self, current: pathlib.Path) -> None:
        """Remove older artifacts (safe while loaded: unlinked .so files
        keep their memory mapping)."""
        for p in self.src.parent.glob(f"{self.src.stem}.*.so"):
            if p != current:
                p.unlink(missing_ok=True)
        legacy = self.src.with_suffix(".so")
        if legacy.exists() and legacy != current:
            legacy.unlink()


_cpp = _NativeLib(find_src("gemm_cpu.cpp"), compile_cpp)
_cu = _NativeLib(find_src("gemm_gpu.cu"), compile_cu)


# ─── ctypes wrappers ───

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_INT = ctypes.c_int
_VOID_P = ctypes.c_void_p


def cpp_wrapper(lib: ctypes.CDLL, func_name: str) -> Callable:
    """Wrap a C++ sgemm(M, N, K, A, B, C); C must be zero-initialized."""
    fn = getattr(lib, func_name)
    fn.argtypes = [_INT, _INT, _INT, _FLOAT_PTR, _FLOAT_PTR, _FLOAT_PTR]
    fn.restype = None

    def gemm(M: int, N: int, K: int,
             A: np.ndarray, B: np.ndarray) -> np.ndarray:
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


def cu_wrapper(lib: ctypes.CDLL, func_name: str) -> Callable:
    """Wrap a CUDA launcher(M, N, K, A, B, C, cudaStream_t).

    Launches on the default stream so CUDA-event timing measures the kernel.
    """
    fn = getattr(lib, func_name)
    fn.argtypes = [_INT, _INT, _INT, _VOID_P, _VOID_P, _VOID_P, _VOID_P]
    fn.restype = None

    def gemm(M: int, N: int, K: int,
             A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        C = cp.zeros((M, N), dtype=config.DTYPE)
        fn(M, N, K, A.data.ptr, B.data.ptr, C.data.ptr, cp.cuda.Stream.null.ptr)
        return C

    gemm.__name__ = func_name
    gemm.__doc__ = f"CUDA {func_name} via ctypes — see gemm_gpu.cu"
    return gemm


# ─── implementation lists (add new kernels here) ───

CPP_IMPLS: List[Tuple[str, str, Optional[int], Optional[int]]] = [
    ("sgemm_naive", "C++ naive (ijk)", 1024, 1),  # stride-N reads, ~2-5 GFLOPS
    ("sgemm_ikj",   "C++ ikj",         4096, 1),  # vectorized inner loop
    ("sgemm_tiled", "C++ tiled",       4096, 1),  # cache-blocked ikj
    ("sgemm_omp",   "C++ OpenMP",      8192, 1),  # row-parallel ikj
]

CU_IMPLS: List[Tuple[str, str]] = [
    ("sgemm_naive_launch", "Naive"),
    ("sgemm_tile_launch",  "tile"),
]


def prepare_all() -> None:
    """(Re)compile stale sources, (re)load libraries, (re)register natives.

    Call from the Results cells: after editing gemm_cpu.cpp / gemm_gpu.cu
    you only need to re-run the Results cells.
    """
    active = resolve_backends()
    if "cpu" in active:
        lib = _cpp.load()
        for fname, disp, max_size, warmup in CPP_IMPLS:
            if hasattr(lib, fname):  # sgemm_omp absent if built without OpenMP
                register_gemm(disp, backend="cpu",
                              max_size=max_size, warmup=warmup)(
                    cpp_wrapper(lib, fname))
            else:
                print(f"  (skipping {disp}: {fname} not in library)")
    if "gpu" in active:
        lib = _cu.load()
        for fname, disp in CU_IMPLS:
            if hasattr(lib, fname):
                register_gemm(disp)(cu_wrapper(lib, fname))
            else:
                print(f"  (skipping {disp}: {fname} not in library)")
    print(f"Ready: {len(list_implementations())} implementations registered.")

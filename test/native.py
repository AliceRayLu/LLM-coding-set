"""Generic build & load for native C++/CUDA operator implementations.

Compiled artifacts are versioned by source mtime+size, so an edited source
always compiles to a fresh path — dlopen then maps fresh code (a same-path
dlopen would return the old mapping). Old artifacts are cleaned up on load.

Operator-specific ctypes wrappers and implementation lists live in each
operator's op.py (signatures differ per operator).
"""

import ctypes
import pathlib
import subprocess
from typing import Callable, List, Optional

import cupy as cp

from . import config  # noqa: F401  (kept for the live-config import convention)


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


class NativeLib:
    """One native source file: versioned build, load, artifact cleanup."""

    def __init__(self, src: pathlib.Path,
                 compile_fn: Callable[[pathlib.Path, Optional[List[str]]],
                                      pathlib.Path]):
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

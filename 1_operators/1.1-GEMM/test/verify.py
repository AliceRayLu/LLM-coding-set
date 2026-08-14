"""Correctness verification against a NumPy CPU reference."""

from typing import List, Optional, Tuple, Union

import cupy as cp
import numpy as np

from .config import DTYPE_NP, ERROR_ATOL, ERROR_RTOL
from .registry import (GemmImpl, GemmResult, SweepResult, get_implementation,
                       list_implementations, resolve_backends)

DEFAULT_VERIFY_SIZES: List[Tuple[int, int, int]] = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (256, 512, 128),   # rectangular
    (512, 256, 1024),  # rectangular, large K
]


def verify_gemm(impl: GemmImpl, M: int, N: int, K: int,
                rtol: float = ERROR_RTOL,
                atol: float = ERROR_ATOL) -> Tuple[float, bool]:
    """Verify one implementation against the NumPy CPU reference.

    Returns (max_absolute_error, passed).
    """
    rng = np.random.RandomState(42)  # deterministic data
    A_np = rng.randn(M, K).astype(DTYPE_NP)
    B_np = rng.randn(K, N).astype(DTYPE_NP)
    C_ref = A_np @ B_np  # ground truth

    if impl.backend == "gpu":
        C_out = impl.fn(M, N, K, cp.asarray(A_np), cp.asarray(B_np))
        cp.cuda.Stream.null.synchronize()
        C_res = cp.asnumpy(C_out)
    else:  # cpu: numpy in, numpy out
        C_res = impl.fn(M, N, K, A_np, B_np)

    C_res = np.asarray(C_res, dtype=DTYPE_NP)
    max_error = float(np.max(np.abs(C_res - C_ref)))
    passed = bool(np.allclose(C_res, C_ref, rtol=rtol, atol=atol))
    return max_error, passed


def verify_all(sizes: List[Tuple[int, int, int]] = None,
               backends: Optional[Union[str, List[str]]] = None) -> SweepResult:
    """Run the correctness check on registered implementations.

    sizes: list of (M, N, K); defaults to DEFAULT_VERIFY_SIZES.
    backends: "all" / "gpu" / "cpu" / a list; None = config.TEST_BACKENDS.
    """
    sizes = sizes or DEFAULT_VERIFY_SIZES
    active = resolve_backends(backends)
    print(f"Verifying backends: {active}")

    results = []
    for name in list_implementations():
        impl = get_implementation(name)
        if impl.backend not in active:
            continue
        for M, N, K in sizes:
            if impl.max_size and max(M, N, K) > impl.max_size:
                continue
            err, ok = verify_gemm(impl, M, N, K)
            results.append(GemmResult(name=name, backend=impl.backend, M=M, N=N, K=K,
                                      time_ms=0.0, gflops=0.0,
                                      max_error=err, passed=ok))
    return SweepResult(results=results)

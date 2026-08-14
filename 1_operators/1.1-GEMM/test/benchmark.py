"""Performance benchmarking with backend-appropriate timing."""

import time
from typing import List, Optional, Tuple, Union

import cupy as cp
import numpy as np

from .config import CPU_TARGET_MS, DEFAULT_ITERS, DEFAULT_WARMUP, DTYPE_NP
from .registry import (GemmImpl, GemmResult, SweepResult, get_implementation,
                       list_implementations, resolve_backends)
from .verify import verify_gemm


def _benchmark_cpu(impl: GemmImpl, M: int, N: int, K: int,
                   A: np.ndarray, B: np.ndarray,
                   warmup: int, iters: Optional[int],
                   target_ms: float = CPU_TARGET_MS) -> Tuple[float, int]:
    """Wall-clock timing with adaptive iteration count (~target_ms per point)."""
    for _ in range(max(1, warmup)):
        impl.fn(M, N, K, A, B)

    if iters is None:
        t0 = time.perf_counter_ns()
        impl.fn(M, N, K, A, B)
        t_est = (time.perf_counter_ns() - t0) / 1e6
        if t_est < 200.0:  # fast call — average a few samples for a stable estimate
            t0 = time.perf_counter_ns()
            for _ in range(3):
                impl.fn(M, N, K, A, B)
            t_est = (time.perf_counter_ns() - t0) / 3e6
        iters = max(1, min(200, int(target_ms / max(t_est, 1e-3))))

    t0 = time.perf_counter_ns()
    for _ in range(iters):
        impl.fn(M, N, K, A, B)
    avg_ms = (time.perf_counter_ns() - t0) / iters / 1e6
    return avg_ms, iters


def benchmark_gemm(impl: GemmImpl, M: int, N: int, K: int,
                   warmup: Optional[int] = None,
                   iters: Optional[int] = None,
                   skip_verify: bool = False) -> GemmResult:
    """Benchmark one implementation at one size.

    GPU: CUDA-event timing on the default stream.
    CPU: perf_counter wall-clock timing with adaptive iterations.
    """
    warmup = impl.warmup if impl.warmup is not None else (warmup or DEFAULT_WARMUP)

    if skip_verify:
        max_error, passed = 0.0, True
    else:
        max_error, passed = verify_gemm(impl, M, N, K)

    rng = np.random.RandomState(123)
    A_np = rng.randn(M, K).astype(DTYPE_NP)
    B_np = rng.randn(K, N).astype(DTYPE_NP)

    if impl.backend == "gpu":
        A = cp.asarray(A_np)
        B = cp.asarray(B_np)
        iters = impl.iters if impl.iters is not None else (iters or DEFAULT_ITERS)
        for _ in range(warmup):
            impl.fn(M, N, K, A, B)
        cp.cuda.Stream.null.synchronize()

        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(iters):
            impl.fn(M, N, K, A, B)
        end.record()
        end.synchronize()
        avg_ms = cp.cuda.get_elapsed_time(start, end) / iters
    else:
        A = np.ascontiguousarray(A_np)
        B = np.ascontiguousarray(B_np)
        avg_ms, _ = _benchmark_cpu(impl, M, N, K, A, B, warmup, iters)

    # FLOPs: each C[i,j] accumulates K products → 2·M·N·K operations
    gflops = (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e9

    return GemmResult(name=impl.name, backend=impl.backend, M=M, N=N, K=K,
                      time_ms=avg_ms, gflops=gflops,
                      max_error=max_error, passed=passed)


def benchmark_all(sizes: List[Tuple[int, int, int]],
                  implementations: List[str] = None,
                  backends: Optional[Union[str, List[str]]] = None,
                  warmup: Optional[int] = None,
                  iters: Optional[int] = None) -> SweepResult:
    """Benchmark multiple implementations across multiple sizes.

    sizes: list of (M, N, K) tuples.
    implementations: which impls to test (None = all registered).
    backends: "all" / "gpu" / "cpu" / a list; None = config.TEST_BACKENDS.
    warmup/iters: iteration overrides (None = per-impl / default / adaptive).
    """
    if implementations is None:
        implementations = list_implementations()

    active = resolve_backends(backends)
    print(f"Benchmarking backends: {active}")

    results = []
    for name in implementations:
        impl = get_implementation(name)
        if impl.backend not in active:
            continue
        for M, N, K in sizes:
            if impl.max_size and max(M, N, K) > impl.max_size:
                print(f"  Skip {name:<20} ({M}, {N}, {K}) — exceeds max_size={impl.max_size}")
                continue
            print(f"  Benchmarking {name:<20} ({M}, {N}, {K}) ...", end=" ", flush=True)
            r = benchmark_gemm(impl, M, N, K, warmup=warmup, iters=iters)
            print(f"{r.time_ms:8.4f} ms  {r.gflops:8.1f} GFLOPS  "
                  f"{'✓' if r.passed else '✗'}")
            results.append(r)
    return SweepResult(results=results)

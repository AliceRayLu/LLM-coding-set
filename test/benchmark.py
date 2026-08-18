"""Performance benchmarking with backend-appropriate timing."""

import time
from types import ModuleType
from typing import List, Optional, Tuple, Union

import cupy as cp
import numpy as np

from . import config
from .registry import (OpImpl, OpResult, SweepResult, get_implementation,
                       list_implementations, resolve_backends)
from .verify import verify_op


def _benchmark_cpu(impl: OpImpl, args: tuple,
                   warmup: int, iters: Optional[int],
                   target_ms: float = config.CPU_TARGET_MS) -> Tuple[float, int]:
    """Wall-clock timing with adaptive iteration count (~target_ms per point)."""
    for _ in range(max(1, warmup)):
        impl.fn(*args)

    if iters is None:
        t0 = time.perf_counter_ns()
        impl.fn(*args)
        t_est = (time.perf_counter_ns() - t0) / 1e6
        if t_est < 200.0:  # fast call — average a few samples for a stable estimate
            t0 = time.perf_counter_ns()
            for _ in range(3):
                impl.fn(*args)
            t_est = (time.perf_counter_ns() - t0) / 3e6
        iters = max(1, min(200, int(target_ms / max(t_est, 1e-3))))

    t0 = time.perf_counter_ns()
    for _ in range(iters):
        impl.fn(*args)
    avg_ms = (time.perf_counter_ns() - t0) / iters / 1e6
    return avg_ms, iters


def benchmark_op(op: ModuleType, impl: OpImpl, size: Tuple[int, ...],
                 warmup: Optional[int] = None,
                 iters: Optional[int] = None,
                 skip_verify: bool = False) -> OpResult:
    """Benchmark one implementation of an op at one size.

    GPU: CUDA-event timing on the default stream.
    CPU: perf_counter wall-clock timing with adaptive iterations.
    """
    warmup = impl.warmup if impl.warmup is not None else (
        warmup if warmup is not None else config.DEFAULT_WARMUP)

    if skip_verify:
        max_error, passed = 0.0, True
    else:
        max_error, passed = verify_op(op, impl, size)

    args = op.gen_inputs(size, impl.backend, seed=123)

    if impl.backend == "gpu":
        iters = impl.iters if impl.iters is not None else (
            iters if iters is not None else config.DEFAULT_ITERS)
        for _ in range(warmup):
            impl.fn(*args)
        cp.cuda.Stream.null.synchronize()

        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(iters):
            impl.fn(*args)
        end.record()
        end.synchronize()
        avg_ms = cp.cuda.get_elapsed_time(start, end) / iters
    else:
        avg_ms, _ = _benchmark_cpu(impl, args, warmup, iters)

    return OpResult(op=op.NAME, name=impl.name, backend=impl.backend, size=size,
                    time_ms=avg_ms, metric_value=op.metric(size, avg_ms),
                    max_error=max_error, passed=passed)


def benchmark_all(op: ModuleType,
                  sizes: List[Tuple[int, ...]],
                  implementations: List[str] = None,
                  backends: Optional[Union[str, List[str]]] = None,
                  warmup: Optional[int] = None,
                  iters: Optional[int] = None) -> SweepResult:
    """Benchmark multiple implementations of an op across multiple sizes.

    implementations: which impls to test (None = all registered for this op).
    backends: "all" / "gpu" / "cpu" / a list; None = config.TEST_BACKENDS.
    warmup/iters: iteration overrides (None = per-impl / default / adaptive).
    """
    if implementations is None:
        implementations = list_implementations(op.NAME)

    active = resolve_backends(backends)
    print(f"Benchmarking backends: {active}")

    results = []
    for name in implementations:
        impl = get_implementation(op.NAME, name)
        if impl.backend not in active:
            continue
        for size in sizes:
            if impl.max_size and max(size) > impl.max_size:
                print(f"  Skip {name:<20} {size} — exceeds max_size={impl.max_size}")
                continue
            print(f"  Benchmarking {name:<20} {size} ...", end=" ", flush=True)
            r = benchmark_op(op, impl, size, warmup=warmup, iters=iters)
            print(f"{r.time_ms:8.4f} ms  {r.metric_value:8.1f} "
                  f"{op.METRIC_LABEL}  {'✓' if r.passed else '✗'}")
            results.append(r)
    return SweepResult(op=op, results=results)

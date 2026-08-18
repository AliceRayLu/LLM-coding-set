"""Correctness verification against the operator's NumPy reference."""

from types import ModuleType
from typing import List, Optional, Tuple, Union

import cupy as cp
import numpy as np

from . import config
from .registry import (OpImpl, OpResult, SweepResult, get_baseline,
                       get_implementation, list_implementations,
                       resolve_backends)


def verify_op(op: ModuleType, impl: OpImpl, size: Tuple[int, ...],
              rtol: Optional[float] = None,
              atol: Optional[float] = None) -> Tuple[float, bool]:
    """Verify one implementation at one size against the operator reference.

    Returns (max_absolute_error, passed).
    """
    rtol = config.ERROR_RTOL if rtol is None else rtol
    atol = config.ERROR_ATOL if atol is None else atol

    args_np = op.gen_inputs(size, "cpu", seed=42)  # deterministic inputs
    if op.reference is not None:
        expected = op.reference(*args_np)  # operator-specific ground truth
    else:  # fallback: the op's CPU baseline implementation
        base = get_baseline(op.NAME, "cpu")
        expected = get_implementation(op.NAME, base).fn(*args_np)

    if impl.backend == "gpu":
        args = tuple(cp.asarray(a) for a in args_np)
        out = impl.fn(*args)
        cp.cuda.Stream.null.synchronize()
        res = cp.asnumpy(out)
    else:  # cpu: numpy in, numpy out
        res = impl.fn(*args_np)

    res = np.asarray(res, dtype=config.DTYPE_NP)
    max_error = float(np.max(np.abs(res - expected)))
    passed = bool(np.allclose(res, expected, rtol=rtol, atol=atol))
    return max_error, passed


def verify_all(op: ModuleType,
               sizes: List[Tuple[int, ...]] = None,
               backends: Optional[Union[str, List[str]]] = None) -> SweepResult:
    """Run the correctness check on the registered implementations of an op.

    sizes: defaults to op.DEFAULT_SIZES.
    backends: "all" / "gpu" / "cpu" / a list; None = config.TEST_BACKENDS.
    """
    sizes = sizes or op.DEFAULT_SIZES
    active = resolve_backends(backends)
    print(f"Verifying backends: {active}")

    results = []
    for name in list_implementations(op.NAME):
        impl = get_implementation(op.NAME, name)
        if impl.backend not in active:
            continue
        for size in sizes:
            if impl.max_size and max(size) > impl.max_size:
                continue
            err, ok = verify_op(op, impl, size)
            results.append(OpResult(op=op.NAME, name=name, backend=impl.backend,
                                    size=size, time_ms=0.0, metric_value=0.0,
                                    max_error=err, passed=ok))
    return SweepResult(op=op, results=results)

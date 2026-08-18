"""Operator implementation registry and result containers."""

from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Dict, List, Optional, Tuple, Union

from . import config


@dataclass
class OpResult:
    """One correctness/benchmark data point for one operator."""
    op: str
    name: str
    backend: str
    size: Tuple[int, ...]
    time_ms: float
    metric_value: float
    max_error: float
    passed: bool


@dataclass
class OpImpl:
    """A registered operator implementation.

    fn(*args) -> out computes the operator, with args = op.gen_inputs(...):
      - backend="gpu": args/out are cupy float32 arrays
      - backend="cpu": args/out are numpy float32 arrays

    max_size: skip benchmark sizes where any dimension exceeds it
              (protects slow implementations, e.g. pure-Python loops).
    warmup/iters: per-implementation iteration overrides
                  (CPU iters=None → adaptive).
    """
    op: str
    name: str
    backend: str
    fn: Callable
    max_size: Optional[int] = None
    warmup: Optional[int] = None
    iters: Optional[int] = None


@dataclass
class SweepResult:
    """Aggregated results across implementations and sizes of one op."""
    op: ModuleType
    results: List[OpResult] = field(default_factory=list)

    def to_table(self) -> str:
        """Return a formatted ASCII table (size columns from op.SIZE_COLS)."""
        cols = getattr(self.op, "SIZE_COLS", ["Size"])
        metric = getattr(self.op, "METRIC_LABEL", "Metric")
        fmt = getattr(self.op, "METRIC_FMT", ".3f")
        header = (f"{'Implementation':<22} {'Backend':<8} "
                  + " ".join(f"{c:<8}" for c in cols)
                  + f" {'Time(ms)':<12} {metric:<12} {'MaxErr':<12} {'Pass':<6}")
        sep = "-" * len(header)
        rows = []
        for r in self.results:
            size_cells = (r.size if cols != ["Size"]
                          else ["×".join(map(str, r.size))])
            rows.append(
                f"{r.name:<22} {r.backend:<8} "
                + " ".join(f"{v:<8}" for v in size_cells)
                + f" {r.time_ms:<12.4f} {r.metric_value:<12{fmt}} "
                  f"{r.max_error:<12.6f} {'✓' if r.passed else '✗':<6}"
            )
        return "\n".join([sep, header, sep, *rows, sep])


# (op, name) → impl; (op, backend) → default baseline name.
_IMPLS: Dict[Tuple[str, str], OpImpl] = {}
_BASELINES: Dict[Tuple[str, str], str] = {}


def register_op(op: str, name: str, backend: str = "gpu",
                max_size: Optional[int] = None,
                warmup: Optional[int] = None,
                iters: Optional[int] = None):
    """Decorator registering fn(*args) -> out as an operator implementation.

    backend: "gpu" (cupy arrays, CUDA-event timing) or
             "cpu" (numpy arrays, wall-clock timing).
    """
    def decorator(fn: Callable) -> Callable:
        _IMPLS[(op, name)] = OpImpl(op=op, name=name, backend=backend, fn=fn,
                                    max_size=max_size, warmup=warmup,
                                    iters=iters)
        return fn
    return decorator


def register_baseline(op: str, backend: str, name: str) -> None:
    """Mark a registered implementation as the default speedup baseline."""
    if (op, name) not in _IMPLS:
        raise KeyError(f"Cannot set baseline: implementation '{name}' of op "
                       f"'{op}' is not registered yet.")
    _BASELINES[(op, backend)] = name


def list_implementations(op: Optional[str] = None) -> List[str]:
    """Names of registered implementations (of one op, or all ops)."""
    if op is None:
        return sorted({name for _, name in _IMPLS})
    return [name for (o, name) in _IMPLS if o == op]


def get_implementation(op: str, name: str) -> OpImpl:
    """Get a registered implementation of an operator."""
    key = (op, name)
    if key not in _IMPLS:
        raise KeyError(f"Unknown implementation: {op}/{name}. "
                       f"Available: {list_implementations(op)}")
    return _IMPLS[key]


def get_baseline(op: str, backend: str) -> str:
    """Name of the speedup baseline for (op, backend).

    Resolution: config.BASELINES override → op-registered default →
    first registered implementation of that backend.
    """
    override = config.BASELINES.get(op, {}).get(backend)
    if override:
        if (op, override) not in _IMPLS:
            raise KeyError(f"config.BASELINES[{op!r}][{backend!r}] = "
                           f"{override!r} is not a registered implementation.")
        return override
    name = _BASELINES.get((op, backend))
    if name:
        return name
    for (o, n), impl in _IMPLS.items():
        if o == op and impl.backend == backend:
            return n
    raise KeyError(f"No baseline for {op}/{backend}: no implementation "
                   f"registered for that backend.")


def resolve_backends(backends: Optional[Union[str, List[str]]] = None) -> List[str]:
    """Normalize backends to a list of "gpu"/"cpu".

    Accepts None (→ config.TEST_BACKENDS), "all", "gpu", "cpu", or a list.
    """
    if backends is None:
        backends = config.TEST_BACKENDS
    if isinstance(backends, str):
        backends = ["gpu", "cpu"] if backends == "all" else [backends]
    backends = list(backends)
    unknown = [b for b in backends if b not in ("gpu", "cpu")]
    if unknown:
        raise ValueError(f"Unknown backend(s): {unknown}. Use 'gpu', 'cpu', or 'all'.")
    return backends

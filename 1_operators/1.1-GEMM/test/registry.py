"""GEMM implementation registry and result containers."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from . import config


@dataclass
class GemmResult:
    """One correctness/benchmark data point."""
    name: str
    backend: str
    M: int
    N: int
    K: int
    time_ms: float
    gflops: float
    max_error: float
    passed: bool


@dataclass
class GemmImpl:
    """A registered GEMM implementation.

    fn(M, N, K, A, B) -> C computes C = A @ B with A (M×K), B (K×N):
      - backend="gpu": A/B/C are cupy float32 arrays
      - backend="cpu": A/B/C are numpy float32 arrays

    max_size: skip benchmark sizes where any dimension exceeds it
              (protects slow implementations, e.g. pure-Python loops).
    warmup/iters: per-implementation iteration overrides
                  (CPU iters=None → adaptive).
    """
    name: str
    backend: str
    fn: Callable
    max_size: Optional[int] = None
    warmup: Optional[int] = None
    iters: Optional[int] = None


@dataclass
class SweepResult:
    """Aggregated results across implementations and sizes."""
    results: List[GemmResult] = field(default_factory=list)

    def to_table(self) -> str:
        """Return a formatted ASCII table."""
        header = (f"{'Implementation':<22} {'Backend':<8} {'M':<6} {'N':<6} {'K':<6} "
                  f"{'Time(ms)':<12} {'GFLOPS':<12} {'MaxErr':<12} {'Pass':<6}")
        sep = "-" * len(header)
        rows = [
            f"{r.name:<22} {r.backend:<8} {r.M:<6} {r.N:<6} {r.K:<6} "
            f"{r.time_ms:<12.4f} {r.gflops:<12.1f} {r.max_error:<12.6f} "
            f"{'✓' if r.passed else '✗':<6}"
            for r in self.results
        ]
        return "\n".join([sep, header, sep, *rows, sep])


_GEMM_REGISTRY: Dict[str, GemmImpl] = {}


def register_gemm(name: str, backend: str = "gpu",
                  max_size: Optional[int] = None,
                  warmup: Optional[int] = None,
                  iters: Optional[int] = None):
    """Decorator registering fn(M, N, K, A, B) -> C as a GEMM implementation.

    backend: "gpu" (cupy arrays, CUDA-event timing) or
             "cpu" (numpy arrays, wall-clock timing).
    """
    def decorator(fn: Callable) -> Callable:
        _GEMM_REGISTRY[name] = GemmImpl(name=name, backend=backend, fn=fn,
                                        max_size=max_size, warmup=warmup,
                                        iters=iters)
        return fn
    return decorator


def list_implementations() -> List[str]:
    """Names of all registered GEMM implementations."""
    return list(_GEMM_REGISTRY)


def get_implementation(name: str) -> GemmImpl:
    """Get a registered GEMM implementation by name."""
    if name not in _GEMM_REGISTRY:
        raise KeyError(f"Unknown implementation: {name}. "
                       f"Available: {list_implementations()}")
    return _GEMM_REGISTRY[name]


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

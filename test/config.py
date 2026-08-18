"""Global configuration for the operator benchmark framework."""

import cupy as cp
import numpy as np

# Data type for all operator implementations (FP32).
DTYPE = cp.float32
DTYPE_NP = np.float32

# Benchmark defaults.
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
CPU_TARGET_MS = 1500.0  # adaptive CPU benchmark target: ~1.5 s per (impl, size)

# FP32 correctness tolerances (rtol/atol for np.allclose).
# FP32 has ~7 decimal digits; after many fused multiply-adds the accumulated
# error can reach ~1e-3, so 1e-2 is a safe gate.
ERROR_RTOL = 1e-2
ERROR_ATOL = 1e-2

# Which backends to test: "all" (default), "gpu", or "cpu".
# Read via `config.TEST_BACKENDS` (not a star-imported copy) so changes
# take effect without restarting the kernel:
#     import test.config as config
#     config.TEST_BACKENDS = "cpu"
TEST_BACKENDS = "all"

# Idle-GPU selection criteria (see test.gpu).
IDLE_UTIL_MAX = 5.0         # %   max GPU compute utilization
IDLE_MEM_MAX_GB = 2.0       # GB  max GPU memory used
IDLE_SAMPLES = 2            #     consecutive samples required
IDLE_SAMPLE_INTERVAL = 0.3  # s   between samples

# Operator sources: op name → folder (relative to the repo root) that
# contains the operator's op.py. load_op() imports and registers it.
OPERATORS = {
    "gemm": "1_operators/1.1-GEMM",
}

# Speedup-baseline overrides, read live. Each op.py registers a default
# baseline via register_baseline(); entries here take precedence:
#     BASELINES = {"gemm": {"gpu": "cuBLAS (cupy)"}}
BASELINES = {}

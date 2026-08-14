"""Global configuration for the GEMM benchmark framework."""

import cupy as cp
import numpy as np

# Data type for all GEMM implementations (FP32).
DTYPE = cp.float32
DTYPE_NP = np.float32

# Benchmark defaults.
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
CPU_TARGET_MS = 1500.0  # adaptive CPU benchmark target: ~1.5 s per (impl, size)

# FP32 correctness tolerances (rtol/atol for np.allclose).
# FP32 has ~7 decimal digits; after K multiply-adds the accumulated error
# can reach ~1e-3 for large K, so 1e-2 is a safe gate.
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

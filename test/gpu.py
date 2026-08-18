"""Idle-GPU detection and selection (via nvidia-smi)."""

import os
import subprocess
import time
from typing import List, Optional

import cupy as cp

from . import config

_SMI_QUERY = "index,name,utilization.gpu,memory.used,memory.total"


def query_gpu_status() -> List[dict]:
    """Query nvidia-smi: one dict per GPU (index, name, util %, mem GB)."""
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    status = []
    for line in out.stdout.strip().splitlines():
        idx, name, util, mem_used, mem_total = line.split(",")
        status.append({
            "index": int(idx),
            "name": name.strip(),
            "util": float(util),
            "mem_used_gb": float(mem_used) / 1024.0,
            "mem_total_gb": float(mem_total) / 1024.0,
        })
    return status


def _is_idle(s: dict) -> bool:
    return s["util"] < config.IDLE_UTIL_MAX and s["mem_used_gb"] < config.IDLE_MEM_MAX_GB


def pick_idle_gpu(num_gpus: int) -> Optional[int]:
    """Return a cupy device index whose GPU is idle in every sample, else None.

    Handles CUDA_VISIBLE_DEVICES remapping (nvidia-smi reports physical
    indices; cupy reports indices into the visible set).
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical_to_cupy = {
        phys: i for i, phys in enumerate(int(x) for x in visible.split(",") if x.strip())
        if i < num_gpus
    }

    idle_counts = {i: 0 for i in range(num_gpus)}
    for _ in range(config.IDLE_SAMPLES):
        for s in query_gpu_status():
            phys = s["index"]
            cupy_idx = physical_to_cupy.get(phys) if physical_to_cupy else (
                phys if phys < num_gpus else None)
            if cupy_idx is not None and _is_idle(s):
                idle_counts[cupy_idx] += 1
        time.sleep(config.IDLE_SAMPLE_INTERVAL)

    for idx, count in idle_counts.items():
        if count == config.IDLE_SAMPLES:
            return idx
    return None


def _format_status(status: List[dict]) -> str:
    header = (f"{'Index':<6} {'GPU':<24} {'Util%':<8} "
              f"{'MemUsed(GB)':<14} {'MemTotal(GB)':<14} {'Idle?'}")
    rows = [
        f"{s['index']:<6} {s['name']:<24} {s['util']:<8.0f} "
        f"{s['mem_used_gb']:<14.1f} {s['mem_total_gb']:<14.1f} "
        f"{'✓' if _is_idle(s) else '✗'}"
        for s in status
    ]
    return "\n".join([header, *rows])


def select_gpu() -> Optional[cp.cuda.Device]:
    """Select an idle GPU and make it current for this process.

    CPU-only mode (config.TEST_BACKENDS == "cpu") returns None without
    touching CUDA. Raises SystemExit(1) with a status report when no
    GPU is idle.
    """
    if config.TEST_BACKENDS == "cpu":
        print("CPU-only mode: skipping GPU selection and GPU implementations.")
        return None

    num_gpus = cp.cuda.runtime.getDeviceCount()
    try:
        idx = pick_idle_gpu(num_gpus)
    except FileNotFoundError:
        print("[warn] nvidia-smi not found — cannot check idleness, falling back to GPU 0.")
        idx = 0
    except (subprocess.SubprocessError, ValueError, IndexError) as e:
        print(f"[warn] could not query GPU status ({e}) — falling back to GPU 0.")
        idx = 0

    if idx is None:
        print("=" * 70)
        print("No idle GPU found — exiting.")
        print(f"Idle criteria: utilization < {config.IDLE_UTIL_MAX}% and "
              f"memory used < {config.IDLE_MEM_MAX_GB} GB "
              f"({config.IDLE_SAMPLES} consecutive samples).")
        print("Current GPU status (nvidia-smi):")
        try:
            print(_format_status(query_gpu_status()))
        except Exception as e:
            print(f"  (status query failed: {e})")
        print("=" * 70)
        print("Tip: set config.TEST_BACKENDS = 'cpu' to run CPU-only tests.")
        raise SystemExit(1)

    dev = cp.cuda.Device(idx)
    dev.use()  # all subsequent cupy operations use this GPU
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"Selected idle GPU {dev.id} (of {num_gpus}): {props['name'].decode()}")
    print(f"Compute Capability: {props['major']}.{props['minor']}  |  "
          f"SMs: {props['multiProcessorCount']}")
    print(f"Memory: {props['totalGlobalMem'] / 1024**3:.1f} GB")
    return dev

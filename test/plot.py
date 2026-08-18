"""Visualization: operator comparison and roofline plots."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .registry import SweepResult, get_baseline

_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0",
           "#00BCD4", "#E91E63", "#3F51B5", "#795548", "#607D8B"]
_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "*"]


def plot_comparison(sweep: SweepResult,
                    title: Optional[str] = None,
                    figsize: Optional[tuple] = None,
                    log_y: bool = False,
                    baselines: Optional[Dict[str, str]] = None) -> plt.Figure:
    """Plot metric vs size — one row per backend (GPU / CPU).

    Left panel: metric (op.METRIC_LABEL). Right panel: speedup vs the
    backend's baseline (op-registered default, config.BASELINES override,
    or the `baselines` argument).
    """
    op = sweep.op

    backends: List[str] = []
    for r in sweep.results:
        if r.backend not in backends:
            backends.append(r.backend)

    fig, axes = plt.subplots(len(backends), 2,
                             figsize=figsize or (14, 6 * len(backends)),
                             squeeze=False)

    for row, backend in enumerate(backends):
        ax_metric, ax_sp = axes[row]

        names: List[str] = []
        for r in sweep.results:
            if r.backend == backend and r.name not in names:
                names.append(r.name)
        grouped = {n: [r for r in sweep.results if r.name == n and r.backend == backend]
                   for n in names}

        baseline = (baselines or {}).get(backend) or get_baseline(op.NAME, backend)
        if baseline not in grouped:
            baseline = names[0]
        baseline_metric = {op.size_key(r.size): r.metric_value
                           for r in grouped[baseline]}

        # --- Left: metric vs size ---
        for idx, name in enumerate(names):
            color, marker = _COLORS[idx % 10], _MARKERS[idx % 10]
            xs = [op.size_key(r.size) for r in grouped[name]]
            ys = [r.metric_value for r in grouped[name]]
            ax_metric.plot(xs, ys, color=color, marker=marker, linewidth=2,
                           markersize=8, label=name, zorder=3)
            for x, y in zip(xs, ys):
                ax_metric.annotate(f"{y:{op.METRIC_FMT}}", (x, y),
                                   textcoords="offset points", xytext=(0, 10),
                                   fontsize=8, ha="center", color=color)

        ax_metric.set_xlabel(op.SIZE_LABEL, fontsize=11)
        ax_metric.set_ylabel(op.METRIC_LABEL, fontsize=11)
        ax_metric.set_title(f"{backend.upper()} — "
                            f"{title or op.NAME.upper() + ' Performance'}", fontsize=13)
        ax_metric.legend(fontsize=9, loc="upper left")
        ax_metric.grid(True, alpha=0.3, linestyle="--")
        if log_y:
            ax_metric.set_yscale("log", base=2)
        ax_metric.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

        # --- Right: speedup vs baseline ---
        all_speedups: List[float] = []
        for idx, name in enumerate(names):
            if name == baseline:
                continue
            color, marker = _COLORS[idx % 10], _MARKERS[idx % 10]
            xs, speedups = [], []
            for r in grouped[name]:
                b = baseline_metric.get(op.size_key(r.size))
                if b:
                    xs.append(op.size_key(r.size))
                    speedups.append(r.metric_value / b)
            all_speedups.extend(speedups)
            ax_sp.plot(xs, speedups, color=color, marker=marker, linewidth=2,
                       markersize=8, label=name, zorder=3)
            for x, sp in zip(xs, speedups):
                lbl = f"{sp:.3f}x" if sp < 0.1 else f"{sp:.2f}x"
                ax_sp.annotate(lbl, (x, sp), textcoords="offset points",
                               xytext=(0, 10), fontsize=8, ha="center", color=color)

        ax_sp.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                      alpha=0.7, label=f"Baseline ({baseline})")
        ax_sp.set_xlabel(op.SIZE_LABEL, fontsize=11)
        ax_sp.set_ylabel(f"Speedup vs {baseline}", fontsize=11)
        ax_sp.set_title(f"Speedup vs {baseline} ({backend.upper()})", fontsize=13)
        ax_sp.legend(fontsize=9)
        ax_sp.grid(True, alpha=0.3, linestyle="--")
        if all_speedups and min(all_speedups) < 0.05:
            ax_sp.set_yscale("log", base=2)

    plt.tight_layout()
    return fig


def plot_roofline(sweep: SweepResult,
                  peak_fp32_tflops: float = 19.5,
                  memory_bw_gbs: float = 2039.0) -> plt.Figure:
    """Plot a simple roofline model with measured data points (GPU only).

    Requires the op to provide flops(size) / bytes_read(size) hooks and a
    GFLOPS metric (e.g. gemm); raises ValueError otherwise.
    """
    op = sweep.op
    if not (hasattr(op, "flops") and hasattr(op, "bytes_read")):
        raise ValueError(f"op '{op.NAME}' does not provide flops()/bytes_read() "
                         f"— the roofline plot is unavailable for it.")
    if op.METRIC_LABEL != "GFLOPS":
        raise ValueError(f"op '{op.NAME}' metric is '{op.METRIC_LABEL}', "
                         f"not GFLOPS — the roofline plot is unavailable for it.")

    gpu_results = [r for r in sweep.results if r.backend == "gpu"]
    if not gpu_results:
        raise ValueError("No GPU results in sweep — the roofline is GPU-specific.")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline curves
    oi = np.logspace(-1, 4, 100)  # FLOP/Byte
    roofline = np.minimum(np.full_like(oi, peak_fp32_tflops * 1000),
                          memory_bw_gbs * oi)
    ax.loglog(oi, roofline, "k-", linewidth=2, label="Roofline", zorder=2)
    ax.fill_between(oi, roofline, alpha=0.05, color="black")

    ridge = peak_fp32_tflops * 1000 / memory_bw_gbs
    ax.axvline(ridge, color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"Ridge: {ridge:.1f} FLOP/Byte",
                xy=(ridge, peak_fp32_tflops * 1000),
                xytext=(ridge * 1.3, peak_fp32_tflops * 800),
                fontsize=9, arrowprops=dict(arrowstyle="->", alpha=0.5))

    # Measured points
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in gpu_results:
        grouped[r.name].append(r)

    for idx, (name, results) in enumerate(grouped.items()):
        color, marker = _COLORS[idx % 10], _MARKERS[idx % 10]
        for r in results:
            intensity = op.flops(r.size) / op.bytes_read(r.size)
            ax.scatter([intensity], [r.metric_value], color=color, marker=marker,
                       s=100, zorder=4, edgecolors="white", linewidth=0.5)
            ax.annotate("×".join(map(str, r.size)), (intensity, r.metric_value),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=7, color=color)
        ax.scatter([], [], color=color, marker=marker, s=80, label=name)

    ax.set_xlabel("Operational Intensity (FLOP / Byte)", fontsize=12)
    ax.set_ylabel("GFLOPS", fontsize=12)
    ax.set_title(f"Roofline Model (Peak: {peak_fp32_tflops} TFLOPS, "
                 f"BW: {memory_bw_gbs} GB/s)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig

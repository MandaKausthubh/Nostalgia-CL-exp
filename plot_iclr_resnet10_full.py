"""Plot CIFAR-10 / MNIST / CIFAR-100 validation accuracy for cnn-cl-resnet10-full.

Pulls run histories from local wandb (localhost:8080), renders ICLR-quality
PNG + PDF per (task, view) and saves the raw series to CSV.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate Nostal
    python plot_iclr_resnet10_full.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# ICLR-style defaults
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.dpi": 150,
})

TASKS = ["cifar10", "mnist", "cifar100"]
TASK_DISPLAY = {
    "cifar10": "CIFAR-10",
    "mnist": "MNIST",
    "cifar100": "CIFAR-100",
}

# Method -> colour / line style (consistent across all panels)
METHOD_STYLE = {
    "nostalgia":       {"color": "#1f77b4", "ls": "-",  "lw": 2.0},
    "naive_adam":      {"color": "#d62728", "ls": "--", "lw": 2.0},
    "ewc":             {"color": "#2ca02c", "ls": "-.", "lw": 2.0},
    "gpm":             {"color": "#ff7f0e", "ls": ":",  "lw": 2.4},
    "agem":            {"color": "#9467bd", "ls": "--", "lw": 2.0},
    "ewc_nostalgia":   {"color": "#8c564b", "ls": "-",  "lw": 2.0},
    "sdft":            {"color": "#17becf", "ls": "-.", "lw": 2.0},
}
METHOD_ORDER = ["nostalgia", "naive_adam", "ewc", "gpm", "agem", "ewc_nostalgia"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_runs(project: str = "cnn-cl-resnet10-full") -> pd.DataFrame:
    """Fetch per-run history, concatenate into a single long DataFrame."""
    api = wandb.Api()
    frames = []
    for r in api.runs(project):
        # Run name pattern: <method>-full (e.g. "nostalgia-full", "ewc_nostalgia-full")
        method = r.name.replace("-full", "")
        hist = r.history(
            keys=[
                "cifar10/validation/acc",
                "mnist/validation/acc",
                "cifar100/validation/acc",
                "trainer/global_step",
                "_step",
            ]
        )
        if hist.empty:
            print(f"[warn] empty history for {r.name} ({r.id})")
            continue
        hist = hist.rename(columns={"_step": "log_step"})
        hist["method"] = method
        hist["run_id"] = r.id
        # Aggregate duplicates on trainer/global_step (multiple tasks log per step)
        hist = hist.dropna(subset=["trainer/global_step"])
        # One row per (run, task, global_step) -> melt to long
        frames.append(hist)

    raw = pd.concat(frames, ignore_index=True)

    long = raw.melt(
        id_vars=["method", "run_id", "trainer/global_step", "log_step"],
        value_vars=[f"{t}/validation/acc" for t in TASKS],
        var_name="task",
        value_name="val_acc",
    )
    long["task"] = long["task"].str.replace("/validation/acc", "", regex=False)
    long = long.dropna(subset=["val_acc"])
    return long


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across runs, grouped by (method, task, trainer/global_step)."""
    g = df.groupby(["method", "task", "trainer/global_step"])["val_acc"]
    agg = g.agg(["mean", "std", "count"]).reset_index()
    return agg.sort_values(["method", "task", "trainer/global_step"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _method_runs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return per-method data with all runs preserved (for shading)."""
    return {m: df[df["method"] == m].copy() for m in METHOD_ORDER}


def plot_per_task(df: pd.DataFrame, outdir: Path) -> None:
    """One figure per task: all methods on the same axes (ICLR-style)."""
    # df is long-form: columns = method, task, trainer/global_step, val_acc
    runs_by_method = _method_runs(df)
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for method in METHOD_ORDER:
            m_df = runs_by_method.get(method)
            if m_df is None or m_df.empty:
                continue
            t_df = m_df[m_df["task"] == task]
            if t_df.empty:
                continue
            style = METHOD_STYLE[method]
            # Group by global step for mean/std shading across runs
            g = t_df.groupby("trainer/global_step")["val_acc"]
            xs = sorted(g.groups.keys())
            means = np.array([g.get_group(x).mean() for x in xs])
            stds = np.array([g.get_group(x).std() for x in xs])
            label = method.replace("_", "\\_") if method == "ewc_nostalgia" else method
            ax.plot(xs, means, label=label, **style)
            if len(xs) > 1 and (stds > 0).any():
                ax.fill_between(xs, means - stds, means + stds,
                                color=style["color"], alpha=0.12, linewidth=0)
        ax.set_xlabel("Global training step")
        ax.set_ylabel("Validation accuracy")
        ax.set_title(f"{TASK_DISPLAY[task]} — validation accuracy")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        # Legend below axes, single row
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=len(labels), fontsize=9)
        fig.subplots_adjust(bottom=0.22)
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"val_acc_{task}.{ext}")
        plt.close(fig)
        print(f"[ok] wrote val_acc_{task}.{{png,pdf}}")


def plot_grid(df: pd.DataFrame, outdir: Path) -> None:
    """3-panel grid (CIFAR-10 / MNIST / CIFAR-100), single legend."""
    runs_by_method = _method_runs(df)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        for method in METHOD_ORDER:
            m_df = runs_by_method.get(method)
            if m_df is None or m_df.empty:
                continue
            t_df = m_df[m_df["task"] == task]
            if t_df.empty:
                continue
            style = METHOD_STYLE[method]
            g = t_df.groupby("trainer/global_step")["val_acc"]
            xs = sorted(g.groups.keys())
            means = np.array([g.get_group(x).mean() for x in xs])
            stds = np.array([g.get_group(x).std() for x in xs])
            label = method.replace("_", "\\_") if method == "ewc_nostalgia" else method
            ax.plot(xs, means, label=label, **style)
            if len(xs) > 1 and (stds > 0).any():
                ax.fill_between(xs, means - stds, means + stds,
                                color=style["color"], alpha=0.12, linewidth=0)
        ax.set_xlabel("Global step")
        ax.set_title(TASK_DISPLAY[task])
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Validation accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(labels), fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"val_acc_grid.{ext}")
    plt.close(fig)
    print(f"[ok] wrote val_acc_grid.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="cnn-cl-resnet10-full")
    p.add_argument("--outdir", default="iclr_figures")
    p.add_argument("--csv", default="val_acc_long.csv")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] fetching runs from project '{args.project}' ...")
    long = load_runs(args.project)
    long.to_csv(outdir / args.csv, index=False)
    print(f"[ok] saved raw long-form CSV -> {outdir / args.csv} ({len(long)} rows)")

    agg = aggregate(long)
    agg.to_csv(outdir / "val_acc_agg.csv", index=False)
    print(f"[ok] saved aggregated CSV -> {outdir / 'val_acc_agg.csv'}")

    print(f"[info] runs per method: {long.groupby('method')['run_id'].nunique().to_dict()}")
    print(f"[info] global-step range per task:")
    for t in TASKS:
        sub = long[long["task"] == t]
        if not sub.empty:
            print(f"  {t}: {sub['trainer/global_step'].min()} -> {sub['trainer/global_step'].max()}")

    plot_per_task(long, outdir)
    plot_grid(long, outdir)
    print(f"[done] figures in {outdir}/")


if __name__ == "__main__":
    main()

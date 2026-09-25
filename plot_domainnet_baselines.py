"""DomainNet baselines comparison figure (local MPS sweep).

Reads the stdout logs produced by `run_domainnet_mps.sh` (offline W&B keeps
history in a binary .wandb blob, so the log text is the parseable source).

One log per method, each containing the full sequential run over the 6
DomainNet domains. Reuses `parse_log`/`summarize` from `plot_k_ablation.py`,
which computes, per task t:
    acc_after_own(t) = last validation acc for t inside t's own Phase-2 block
    acc_final(t)     = last validation acc for t anywhere in the run
    forgetting(t)    = acc_after_own(t) - acc_final(t)      (>= 0 if forgotten)
averaged over all but the last task (the last task cannot be forgotten).

Log names: s<seed>_domainnet_vit_<method>.log

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate Nostal
    python plot_domainnet_baselines.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_k_ablation import parse_log, summarize

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

# Dispatch priority order from run_domainnet_mps.sh; method -> display label.
METHOD_ORDER = ["nostalgia", "ewc_nostalgia", "gpm", "naive_adam", "ewc", "agem", "sdft"]
LABELS = {
    "nostalgia": "Nostalgia",
    "ewc_nostalgia": "EWC+Nostalgia",
    "gpm": "GPM",
    "naive_adam": "naive",
    "ewc": "EWC",
    "agem": "A-GEM",
    "sdft": "SDFT",
}
OURS = {"nostalgia", "ewc_nostalgia"}
NAIVE = "naive_adam"
LOG_RE = re.compile(r"s(\d+)_domainnet_vit_(\w+)\.log$")

# Short domain labels: domainnet_clipart -> clipart
def short(task: str) -> str:
    return task.replace("domainnet_", "")


def discover(log_dir: Path) -> list[tuple]:
    """[(method, seed, path)]."""
    runs = []
    for path in sorted(log_dir.glob("*.log")):
        m = LOG_RE.search(path.name)
        if not m:
            continue
        seed, method = int(m.group(1)), m.group(2)
        runs.append((method, seed, path))
    return runs


def sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def plot_bars(rows: pd.DataFrame, outdir: Path) -> None:
    """Two-panel bar chart: avg forgetting (lower better) | avg final acc."""
    rows = rows.sort_values("avg_forgetting", ascending=True)
    methods = list(rows["method"])
    labels = [LABELS.get(m, m) for m in methods]
    colors = ["#1f77b4" if m in OURS else "#7f7f7f" for m in methods]
    y = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.42 * len(methods) + 1.9))

    ax = axes[0]
    ax.barh(y, rows["avg_forgetting"], color=colors,
            xerr=rows["avg_forgetting_std"].fillna(0.0),
            error_kw=dict(ecolor="#333333", lw=1.0, capsize=3))
    naive = rows[rows["method"] == NAIVE]
    if not naive.empty:
        ax.axvline(float(naive["avg_forgetting"].iloc[0]), color="#d62728",
                   ls="--", lw=1.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average forgetting  (lower = better)")
    ax.set_title("Forgetting by method")

    ax = axes[1]
    ax.barh(y, rows["avg_final_acc"], color=colors,
            xerr=rows["avg_final_acc_std"].fillna(0.0),
            error_kw=dict(ecolor="#333333", lw=1.0, capsize=3))
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Average final accuracy  (higher = better)")
    ax.set_title("Retention by method")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"domainnet_baselines.{ext}")
    plt.close(fig)
    print(f"[ok] wrote domainnet_baselines.{{png,pdf}} -> {outdir}")


def plot_heatmap(per_task: pd.DataFrame, outdir: Path, metric: str) -> None:
    """method x domain matrix of `metric` (forgetting or acc_final)."""
    methods = sorted(per_task["method"].unique(), key=sort_key)
    tasks = sorted(per_task["task"].unique())
    mat = np.full((len(methods), len(tasks)), np.nan)
    for i, m in enumerate(methods):
        for j, t in enumerate(tasks):
            vals = per_task[(per_task["method"] == m) & (per_task["task"] == t)][metric]
            if not vals.empty:
                mat[i, j] = float(vals.mean())

    fig, ax = plt.subplots(figsize=(1.05 * len(tasks) + 2.4, 0.42 * len(methods) + 1.6))
    cmap = "Reds" if metric == "forgetting" else "Greens"
    im = ax.imshow(mat, cmap=cmap, aspect="auto",
                   vmin=0.0, vmax=np.nanmax(mat) if np.isfinite(mat).any() else 1.0)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels([short(t) for t in tasks], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([LABELS.get(m, m) for m in methods])
    ax.set_title(f"Per-domain {metric}")
    for i in range(len(methods)):
        for j in range(len(tasks)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#222222")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"domainnet_{metric}_by_domain.{ext}")
    plt.close(fig)
    print(f"[ok] wrote domainnet_{metric}_by_domain.{{png,pdf}} -> {outdir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs/domainnet_mps")
    p.add_argument("--outdir", default="iclr_figures/domainnet_baselines")
    p.add_argument("--allow_partial", action="store_true",
                   help="also summarize runs that did not finish all 6 domains")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = discover(log_dir)
    if not runs:
        raise SystemExit(f"no s<seed>_domainnet_vit_<method>.log found in {log_dir}")

    records = []
    per_task_records = []
    for method, seed, path in runs:
        try:
            parsed = parse_log(path)
        except ValueError as e:
            print(f"[skip] {path.name}: {e}")
            continue
        if not parsed["completed"] and not args.allow_partial:
            print(f"[skip] {path.name}: run not finished yet "
                  f"({len(parsed['per_task'])}/{len(parsed['order'])} tasks seen)")
            continue
        if not parsed["per_task"]:
            print(f"[skip] {path.name}: no completed task yet")
            continue
        s = summarize(parsed)
        records.append({"method": method, "seed": seed, "log": path.name, **s})
        for task, vals in parsed["per_task"].items():
            per_task_records.append({"method": method, "seed": seed, "task": task, **vals})
        print(f"[{path.name}] avg_forgetting={s['avg_forgetting']:.4f} "
              f"avg_final_acc={s['avg_final_acc']:.4f} "
              f"last_task_final_acc={s['last_task_final_acc']:.4f} "
              f"(tasks={s['forgettable_tasks']})")

    if not records:
        raise SystemExit("no completed DomainNet runs yet")

    all_records = pd.DataFrame(records)
    all_records.to_csv(outdir / "domainnet_baselines_runs.csv", index=False)

    per_task = pd.DataFrame(per_task_records)
    per_task.to_csv(outdir / "domainnet_baselines_per_task.csv", index=False)

    rows = (all_records
            .groupby("method")
            .agg(avg_forgetting=("avg_forgetting", "mean"),
                 avg_forgetting_std=("avg_forgetting", "std"),
                 avg_final_acc=("avg_final_acc", "mean"),
                 avg_final_acc_std=("avg_final_acc", "std"),
                 last_task_final_acc=("last_task_final_acc", "mean"),
                 seeds=("avg_forgetting", "size"))
            .reset_index())
    rows["_ord"] = rows["method"].map(sort_key)
    rows = rows.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    rows.to_csv(outdir / "domainnet_baselines_summary.csv", index=False)

    print(f"[ok] wrote CSVs -> {outdir}")
    print(rows.to_string(index=False))

    plot_bars(rows, outdir)
    plot_heatmap(per_task, outdir, "forgetting")
    plot_heatmap(per_task, outdir, "acc_final")


if __name__ == "__main__":
    main()

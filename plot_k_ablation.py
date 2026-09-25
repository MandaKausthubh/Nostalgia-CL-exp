"""Forgetting-vs-k ablation figure for the Nostalgia rank sweep (local MPS).

Reads the stdout logs produced by `run_k_ablation_mps.sh` (offline W&B keeps
history in a binary .wandb blob, so the log text is the parseable source).

Metric, per task t in the sequential order [cifar10, mnist, cifar100]:
    acc_after_own(t) = last validation acc for t inside t's own Phase-2 block
    acc_final(t)     = last validation acc for t anywhere in the run
    forgetting(t)    = acc_after_own(t) - acc_final(t)      (>= 0 if forgotten)

Averaged over all but the last task (the last task cannot be forgotten).

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate Nostal
    python plot_k_ablation.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

VAL_RE = re.compile(r"(\w+)/validation/acc=([\d.]+)")
PHASE2_RE = re.compile(r"\[Phase 2\] Finetuning for '(\w+)'")
NAIVE_COLOR = "#d62728"
CURVE_COLOR = "#1f77b4"


def parse_log(path: Path) -> dict:
    """Extract per-task acc-after-own / acc-final from one training log."""
    txt = path.read_text(errors="replace")

    val = [(m.start(), m.group(1), float(m.group(2))) for m in VAL_RE.finditer(txt)]
    phase2 = [(m.start(), m.group(1)) for m in PHASE2_RE.finditer(txt)]
    order = [name for _, name in phase2]
    if not order:
        raise ValueError(f"{path}: no Phase-2 markers found")

    # Phase-2 block boundaries: from a task's marker up to the next task's marker.
    starts = [p for p, _ in phase2] + [len(txt)]

    after_own = {}
    for i, task in enumerate(order):
        lo, hi = starts[i], starts[i + 1]
        in_block = [a for pos, t, a in val if t == task and lo <= pos < hi]
        if in_block:
            after_own[task] = in_block[-1]

    final = {}
    for _, t, a in val:
        final[t] = a  # finditer is in file order -> last write wins

    per_task = {}
    for task in order:
        if task in after_own and task in final:
            per_task[task] = {
                "acc_after_own": after_own[task],
                "acc_final": final[task],
                "forgetting": after_own[task] - final[task],
            }
    completed = "sequential training pipeline completed!" in txt
    return {"order": order, "per_task": per_task, "final": final, "completed": completed}


def summarize(parsed: dict) -> dict:
    order = parsed["order"]
    per_task = parsed["per_task"]
    forgettable = order[:-1]  # last task cannot be forgotten
    f = [per_task[t]["forgetting"] for t in forgettable if t in per_task]
    finals = [per_task[t]["acc_final"] for t in order if t in per_task]
    return {
        "avg_forgetting": float(np.mean(f)) if f else float("nan"),
        "avg_final_acc": float(np.mean(finals)) if finals else float("nan"),
        "last_task_final_acc": per_task[order[-1]]["acc_final"] if order[-1] in per_task else float("nan"),
        "forgettable_tasks": len(f),
    }


def discover(log_dir: Path) -> list[tuple]:
    """[(k_key, seed, path)]. Filenames: [s<seed>_]<prefix>_k<k>.log / _naive.log."""
    runs = []
    for path in sorted(log_dir.glob("*.log")):
        if path.name == "driver.log":
            continue
        ms = re.search(r"s(\d+)_", path.name)
        seed = int(ms.group(1)) if ms else 0
        m = re.search(r"_k(\d+)\.log$", path.name)
        if m:
            runs.append((int(m.group(1)), seed, path))
        elif path.name.endswith("_naive.log"):
            runs.append(("naive", seed, path))
    return runs


def plot(rows: pd.DataFrame, naive: dict | None, outdir: Path) -> None:
    ks = sorted(rows["k"].unique())
    has_std = rows["avg_forgetting_std"].notna().any()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    ax = axes[0]
    ax.plot(ks, rows["avg_forgetting"], marker="o", color=CURVE_COLOR, lw=2.0,
            label="Nostalgia")
    if has_std:
        lo = rows["avg_forgetting"] - rows["avg_forgetting_std"].fillna(0.0)
        hi = rows["avg_forgetting"] + rows["avg_forgetting_std"].fillna(0.0)
        ax.fill_between(ks, lo, hi, color=CURVE_COLOR, alpha=0.15, linewidth=0)
    if naive is not None:
        ax.axhline(naive["avg_forgetting"], color=NAIVE_COLOR, ls="--", lw=2.0,
                   label="naive (no projection)")
        ax.annotate("naive", xy=(ks[-1], naive["avg_forgetting"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="right", color=NAIVE_COLOR, fontsize=9)
    ax.set_xlabel("Hessian rank $k$")
    ax.set_ylabel("Average forgetting")
    ax.set_title("Forgetting vs rank")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ks, rows["avg_final_acc"], marker="s", color=CURVE_COLOR, lw=2.0,
            label="avg final acc")
    ax.plot(ks, rows["last_task_final_acc"], marker="^", color="#2ca02c", lw=2.0,
            ls="-.", label="final-task acc (plasticity)")
    if naive is not None:
        ax.axhline(naive["avg_final_acc"], color=NAIVE_COLOR, ls="--", lw=1.4)
        ax.axhline(naive["last_task_final_acc"], color="#2ca02c", ls=":", lw=1.4)
    ax.set_xlabel("Hessian rank $k$")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Retention vs plasticity")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"forgetting_vs_k.{ext}")
    plt.close(fig)
    print(f"[ok] wrote forgetting_vs_k.{{png,pdf}} -> {outdir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs/k_ablation_strengthened")
    p.add_argument("--outdir", default="iclr_figures/k_ablation")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = discover(log_dir)
    if not runs:
        raise SystemExit(f"no *_k<k>.log found in {log_dir}")

    nost_records = []
    naive_records = []
    per_task_records = []
    for key, seed, path in runs:
        parsed = parse_log(path)
        if not parsed["completed"]:
            print(f"[skip] {path.name}: run not finished yet")
            continue
        if not parsed["per_task"]:
            print(f"[skip] {path.name}: no completed task yet")
            continue
        s = summarize(parsed)
        rec = {"k": key, "seed": seed, "log": path.name, **s}
        (naive_records if key == "naive" else nost_records).append(rec)
        for task, vals in parsed["per_task"].items():
            per_task_records.append({"k": key, "seed": seed, "task": task, **vals})
        print(f"[{path.name}] avg_forgetting={s['avg_forgetting']:.4f} "
              f"avg_final_acc={s['avg_final_acc']:.4f} "
              f"last_task_final_acc={s['last_task_final_acc']:.4f} "
              f"(tasks={s['forgettable_tasks']})")

    if not nost_records:
        raise SystemExit("no completed Nostalgia runs yet")

    all_records = pd.DataFrame(nost_records + naive_records)
    all_records.to_csv(outdir / "k_ablation_runs.csv", index=False)

    per_task = pd.DataFrame(per_task_records)
    per_task.to_csv(outdir / "k_ablation_per_task.csv", index=False)

    nost = all_records[all_records["k"] != "naive"]
    rows = (nost.groupby("k")
                .agg(avg_forgetting=("avg_forgetting", "mean"),
                     avg_forgetting_std=("avg_forgetting", "std"),
                     avg_final_acc=("avg_final_acc", "mean"),
                     last_task_final_acc=("last_task_final_acc", "mean"),
                     seeds=("avg_forgetting", "size"))
                .reset_index()
                .sort_values("k"))
    rows.to_csv(outdir / "k_ablation_summary.csv", index=False)

    naive_summary = None
    naive = all_records[all_records["k"] == "naive"]
    if not naive.empty:
        naive_summary = {
            "avg_forgetting": float(naive["avg_forgetting"].mean()),
            "avg_final_acc": float(naive["avg_final_acc"].mean()),
            "last_task_final_acc": float(naive["last_task_final_acc"].mean()),
        }
    print(f"[ok] wrote CSVs -> {outdir}")
    print(rows.to_string(index=False))
    if naive_summary:
        print(f"naive: avg_forgetting={naive_summary['avg_forgetting']:.4f} "
              f"avg_final_acc={naive_summary['avg_final_acc']:.4f}")

    plot(rows, naive_summary, outdir)


if __name__ == "__main__":
    main()

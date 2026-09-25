# Nostalgia — Local Experiment Guide (macOS / MPS)

Reproduces the continual-learning experiments run on this machine. All runs are
single-device Apple MPS. Everything goes through `train.py`; the wrapper scripts
below only assemble its CLI.

---

## 1. Environment

Conda env **`Nostal`** (Python 3.13, torch 2.12, lightning 2.6.5, peft 0.19.1,
transformers 5.9.0).

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Nostal
```

**MPS rule:** pass `--precision 32-true`. The bf16 default is unreliable on MPS.

Two platform constraints to be aware of:

- The macOS system bash is **3.2** (`no declare -A`), so the production
  `run_domainnet_sweep.sh` cannot run locally. The `*_mps.sh` runners are
  bash-3.2-safe equivalents with reduced local-compute budgets.
- **No checkpointing exists.** An interrupted run restarts from scratch.

---

## 2. DomainNet sweep (7 methods × 6 domains, single seed)

Local runner: `run_domainnet_mps.sh`.

```bash
mkdir -p logs/domainnet_mps
SEED=0 /bin/bash run_domainnet_mps.sh > logs/domainnet_mps/driver.log 2>&1
```

Runs sequentially, one method at a time: `nostalgia ewc_nostalgia gpm
naive_adam ewc agem sdft`. Per-method log:
`logs/domainnet_mps/s0_domainnet_vit_<method>.log`.

Fixed configuration:

| knob | value |
|---|---|
| backbone | `vit` (HF `google/vit-base-patch16-224`), image size 224 |
| LoRA | r=8, alpha=16, dropout=0.05 |
| optimizer | adamw, lr 3e-4, head_lr 5e-4, wd 1e-4, grad_clip 1.0 |
| budget | ph1=3, ph2=5, warmup=600, total_steps=11000, bs=16 |
| class caps | `--max_train_per_class 100 --max_val_per_class 10` |
| validation | `--val_every_n_epochs 5` |
| Nostalgia | k=24, accumulation_rounds=2, hessian bs=16, num_samples=2000 |
| wandb | offline, project `domainnet-cl-mps` |

Env overrides (all optional): `SEED`, `PH1`, `PH2`, `TOTAL_STEPS`, `BS`, `LR`,
`MAX_TRAIN_PER_CLASS`, `MAX_VAL_PER_CLASS`, `VAL_EPOCHS`, `LORA_R`, `K`,
`HESS_ROUNDS`, `HESS_SAMPLES`, `METHODS`, `DRY_RUN=1`.

Data layout — `DATA_ROOT_DN` (default `$HOME/data/domainnet`) must contain:

```
domainnet/
  clipart/ infograph/ painting/ quickdraw/ real/ sketch/   # <class>/<file>.jpg
  {clipart,...,sketch}_train.txt
  {clipart,...,sketch}_test.txt
```

A `DRY_RUN=1` pass prints every assembled `train.py` command without running.

The Nostalgia Hessian is built in the **LoRA adapter space** (~294,912 params at
r=8), not the full model.

---

## 3. Rank (k) ablation

Setting from `method.tex`: sequential image classification
`cifar10 → mnist → cifar100`, ViT + LoRA, `k ∈ {8, 24, 64}` plus a `naive_adam`
control (no projection).

```bash
/bin/bash run_k_ablation_mps.sh
```

Logs land in `logs/k_ablation_strengthened/` as `s<seed>_vit_lora_k<k>.log` and
`s<seed>_vit_lora_naive.log`. Budget: ph1=1, ph2=6, warmup=200,
total_steps=4000, 100 imgs/class, hessian rounds=2 / samples=2000.

Override the axes with `KS="8 24 64"`, `SEEDS="0 1 2"`, `WITH_NAIVE=0`.

> **Budget trap.** The per-task LR schedule is a warmup+decay over
> `--total_steps`, reset at every task/phase transition. If `total_steps` is
> smaller than the largest task's Phase-2 step count, LR decays to 0 mid-task
> and that task silently never trains. Size `total_steps` to the largest task
> (`imgs / bs × epochs`), not to the smallest.

---

## 4. Figures

Both scripts parse the plain-text stdout logs (offline W&B keeps run history
only in a binary `.wandb` blob, so the log text is the parseable source).

DomainNet baselines (7-method comparison + per-domain heatmaps):

```bash
python plot_domainnet_baselines.py \
    --log_dir logs/domainnet_mps \
    --outdir iclr_figures/domainnet_baselines
```

Outputs: `domainnet_baselines.{png,pdf}` (forgetting / retention bars),
`domainnet_{forgetting,acc_final}_by_domain.{png,pdf}` (method × domain
heatmaps), and three CSVs (`_runs`, `_per_task`, `_summary`).

Runs that have not printed `sequential training pipeline completed!` are
skipped; pass `--allow_partial` to include them.

k-ablation (forgetting vs rank):

```bash
python plot_k_ablation.py --log_dir logs/k_ablation_strengthened \
    --outdir iclr_figures/k_ablation
```

---

## 5. Metric definitions

Per task `t` in the sequential order:

```
acc_after_own(t) = last validation acc for t inside t's own Phase-2 block
acc_final(t)     = last validation acc for t anywhere in the run
forgetting(t)    = acc_after_own(t) - acc_final(t)     # >= 0 if forgotten
```

Averages over all but the last task (the last task cannot be forgotten).

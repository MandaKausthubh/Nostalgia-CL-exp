#!/usr/bin/env bash
# Kaggle TPU v3-8 runner for the DomainNet CL sweep.
#
# Kaggle differences from the RunPod path this script handles:
#   * /kaggle/input is READ-ONLY -> Phase-1 cache + checkpoints + wandb must go
#     under /kaggle/working (PHASE1_CACHE_DIR / CHECKPOINT_DIR here).
#   * /kaggle/working has a ~20 GB output cap, so checkpoints are the thing to
#     watch; the resume bundle already prunes per-task copies.
#   * torch_xla is NOT preinstalled -> INSTALL_DEPS=1 pip-installs torch_xla and
#     pytorch-adapt (DomainNet's list-file loader needs the latter).
#   * Sessions are preempted at ~12h -> RESUME=auto by default.
#   * One process drives all 8 chips (Lightning XLA strategy). Runs are serial.
#
# Usage (from a Kaggle notebook cell):
#   !bash /kaggle/working/Nostalgia-CL-exp/run_kaggle_tpu.sh
#   MODE=smoke !bash .../run_kaggle_tpu.sh          # 2 domains, 2 methods, tiny
#
# Override any axis with env: SEEDS, METHODS, BACKBONES, PH2, K, LORA_R, ...

set -euo pipefail

# ----- Paths -------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/kaggle/working/Nostalgia-CL-exp}"
DATA_ROOT_DN="${DATA_ROOT_DN:-/kaggle/input/datasets/kausthubhmanda/domainnet-fulldataset/domainnet}"
WANDB_DIR="${WANDB_DIR:-/kaggle/working/wandb_logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/kaggle/working/checkpoints}"
PHASE1_CACHE_DIR="${PHASE1_CACHE_DIR:-/kaggle/working/phase1_cache}"

# ----- Accelerator -------------------------------------------------------
export ACCEL="${ACCEL:-tpu}"
export DEVICES="${DEVICES:-8}"          # v3-8 -> 8 chips, single process
export STRATEGY="${STRATEGY:-xla}"
export PRECISION="${PRECISION:-bf16-true}"   # NOT bf16-mixed on XLA
export NUM_WORKERS="${NUM_WORKERS:-0}"       # ignored off-CUDA anyway

# ----- Run policy --------------------------------------------------------
export RESUME="${RESUME:-auto}"        # preemption-safe; never prompts
export PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
# Kaggle TPU VMs usually have internet, but keep wandb local-first. Set
# WANDB_MODE=online (and WANDB_API_KEY) to stream to the dashboard.
export WANDB_MODE="${WANDB_MODE:-offline}"

export DATA_ROOT_DN WANDB_DIR CHECKPOINT_DIR PHASE1_CACHE_DIR

MODE="${MODE:-full}"   # full | smoke

echo "====================================================================="
echo "Kaggle TPU runner"
echo "  REPO_DIR         = $REPO_DIR"
echo "  DATA_ROOT_DN     = $DATA_ROOT_DN"
echo "  WANDB_DIR        = $WANDB_DIR"
echo "  CHECKPOINT_DIR   = $CHECKPOINT_DIR"
echo "  PHASE1_CACHE_DIR = $PHASE1_CACHE_DIR"
echo "  MODE             = $MODE"
echo "  ACCEL/DEVICES    = $ACCEL / $DEVICES   (strategy=$STRATEGY, precision=$PRECISION)"
echo "  RESUME           = $RESUME"
echo "  WANDB_MODE       = $WANDB_MODE"
echo "====================================================================="

# ----- 1. Deps -----------------------------------------------------------
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    echo "=== [1/4] Deps ==="
    python -m pip install -q --upgrade pip

    # torch_xla must match the installed torch; let pip resolve it, then report.
    if ! python -c "import torch_xla" >/dev/null 2>&1; then
        echo "  installing torch_xla ..."
        python -m pip install -q torch_xla
    fi
    # DomainNet loader imports pytorch_adapt.datasets.
    if ! python -c "import pytorch_adapt" >/dev/null 2>&1; then
        echo "  installing pytorch-adapt ..."
        python -m pip install -q pytorch-adapt
    fi

    python - <<'PY'
import torch
print(f"  torch = {torch.__version__}")
try:
    import torch_xla
    import torch_xla.runtime as xr
    print(f"  torch_xla = {torch_xla.__version__}")
    print(f"  xla device count = {xr.global_device_count()}")
except Exception as exc:  # noqa: BLE001
    print(f"  [FATAL] torch_xla unavailable: {exc}")
    raise SystemExit(1)
PY
else
    echo "=== [1/4] Deps skipped (INSTALL_DEPS=0) ==="
fi

# ----- 2. Writable dirs --------------------------------------------------
echo "=== [2/4] Writable dirs ==="
for d in "$WANDB_DIR" "$CHECKPOINT_DIR" "$PHASE1_CACHE_DIR"; do
    mkdir -p "$d"
    if ! touch "$d/.write_test" 2>/dev/null; then
        echo "[FATAL] $d is not writable"
        exit 1
    fi
    rm -f "$d/.write_test"
    echo "  [ok] $d"
done

# ----- 3. Sanity: repo + dataset layout ----------------------------------
echo "=== [3/4] Sanity ==="
[ -d "$REPO_DIR" ] || { echo "[FATAL] repo not found at $REPO_DIR"; exit 1; }
[ -d "$DATA_ROOT_DN" ] || { echo "[FATAL] dataset not found at $DATA_ROOT_DN"; exit 1; }

# The loader (pytorch_adapt) expects <root>/domainnet/{domain}_{train,test}.txt.
# It strips a trailing 'domainnet' basename, so either the parent or the
# domainnet dir itself is acceptable here.
_probe="$DATA_ROOT_DN"
if [ ! -d "$_probe/clipart" ] && [ -d "$_probe/domainnet/clipart" ]; then
    _probe="$_probe/domainnet"
fi
for d in clipart infograph painting quickdraw real sketch; do
    if [ ! -d "$_probe/$d" ] || [ ! -f "$_probe/${d}_train.txt" ]; then
        echo "[FATAL] missing $_probe/$d (or ${d}_train.txt)"
        exit 1
    fi
done
echo "  [ok] 6 domain folders + split lists present under $_probe"

# Read-only input is expected; only warn if we resolved to something else.
case "$DATA_ROOT_DN" in
    /kaggle/input/*) echo "  [ok] dataset under read-only /kaggle/input (expected)" ;;
    *) echo "  [warn] DATA_ROOT_DN is not under /kaggle/input: $DATA_ROOT_DN" ;;
esac

DF_OUT="$(df -h /kaggle/working 2>/dev/null | tail -1 || true)"
[ -n "$DF_OUT" ] && echo "  /kaggle/working: $DF_OUT  (~20 GB output cap)"

# ----- 4. Launch sweep ---------------------------------------------------
echo "=== [4/4] Sweep ==="
cd "$REPO_DIR"

if [ "$MODE" = "smoke" ]; then
    echo "  smoke: 2 domains, nostalgia + naive_adam, resnet18, tiny budget"
    TASKS="domainnet_clipart domainnet_real" \
    SEEDS="0" \
    BACKBONES="resnet18" \
    METHODS="nostalgia naive_adam" \
    PH1=1 PH2=1 WARMUP=10 TOTAL_STEPS=50 VAL_EPOCHS=1 \
    bash run_domainnet_sweep.sh
else
    bash run_domainnet_sweep.sh
fi

echo "Kaggle TPU run finished."

#!/usr/bin/env bash
# RunPod pod bootstrapper. Idempotent. Run after first SSH.
#
# Usage:
#   bash setup_pod.sh
#
# Env vars (optional, else use defaults):
#   REPO_URL          git URL to clone
#   REPO_DIR          target dir (default: /workspace/Nostalgia_Project)
#   VOLUME_MOUNT      persistent volume mount point (default: /workspace/volume)
#   DATA_ROOT_DN      DomainNet dataset path (default: $VOLUME_MOUNT/domainnet)
#   WANDB_DIR         wandb log dir (default: $VOLUME_MOUNT/wandb)
#   KAGGLE_USER       Kaggle username (for Option A dataset download)
#   KAGGLE_KEY        Kaggle API key
#   PIP_INDEX         extra pip index URL (default: empty)
#
# Requires: curl, git, python3, pip. CUDA toolkit assumed from RunPod template.

set -euo pipefail

REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-/workspace/Nostalgia_Project}"
VOLUME_MOUNT="${VOLUME_MOUNT:-/workspace/volume}"
DATA_ROOT_DN="${DATA_ROOT_DN:-$VOLUME_MOUNT/domainnet}"
WANDB_DIR="${WANDB_DIR:-$VOLUME_MOUNT/wandb}"
KAGGLE_USER="${KAGGLE_USER:-}"
KAGGLE_KEY="${KAGGLE_KEY:-}"

echo "=== RunPod setup starting ==="
echo "REPO_DIR       = $REPO_DIR"
echo "VOLUME_MOUNT   = $VOLUME_MOUNT"
echo "DATA_ROOT_DN   = $DATA_ROOT_DN"
echo "WANDB_DIR      = $WANDB_DIR"

# ----------------------------------------------------------------------------
# 1. Verify mount
# ----------------------------------------------------------------------------
if ! mountpoint -q "$VOLUME_MOUNT" 2>/dev/null; then
    echo "[warn] $VOLUME_MOUNT not a mountpoint. Creating dir on root disk."
    echo "[warn] Pod will lose data on stop. Attach persistent volume in RunPod UI."
    mkdir -p "$VOLUME_MOUNT"
fi

mkdir -p "$DATA_ROOT_DN" "$WANDB_DIR"

# ----------------------------------------------------------------------------
# 2. Python deps
# ----------------------------------------------------------------------------
echo "=== Installing Python deps ==="
python3 -m pip install --quiet --upgrade pip

# Core
python3 -m pip install --quiet \
    torch torchvision \
    lightning \
    wandb \
    peft \
    transformers \
    datasets \
    kaggle \
    timm

# Verify torch + CUDA
python3 - <<'PY'
import torch
print(f"torch = {torch.__version__}")
print(f"cuda available = {torch.cuda.is_available()}")
print(f"cuda devices = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  device {i}: {torch.cuda.get_device_name(i)}")
PY

# ----------------------------------------------------------------------------
# 3. Clone repo
# ----------------------------------------------------------------------------
if [ -z "$REPO_URL" ]; then
    echo "[skip] REPO_URL not set. Skipping clone. Assume repo already at $REPO_DIR."
elif [ -d "$REPO_DIR/.git" ]; then
    echo "=== Repo already cloned at $REPO_DIR ==="
    (cd "$REPO_DIR" && git pull --ff-only || echo "[warn] git pull failed; keeping existing")
else
    echo "=== Cloning $REPO_URL ==="
    git clone "$REPO_URL" "$REPO_DIR"
fi

# ----------------------------------------------------------------------------
# 4. Install repo-level deps (if any)
# ----------------------------------------------------------------------------
if [ -f "$REPO_DIR/requirements.txt" ]; then
    python3 -m pip install --quiet -r "$REPO_DIR/requirements.txt"
fi

# ----------------------------------------------------------------------------
# 5. DomainNet download (Option A: Kaggle)
# ----------------------------------------------------------------------------
if [ -d "$DATA_ROOT_DN/clipart/train" ] && [ -d "$DATA_ROOT_DN/real/train" ]; then
    echo "=== DomainNet already present at $DATA_ROOT_DN ==="
else
    if [ -n "$KAGGLE_USER" ] && [ -n "$KAGGLE_KEY" ]; then
        echo "=== Downloading DomainNet via Kaggle ==="
        export KAGGLE_USERNAME="$KAGGLE_USER"
        export KAGGLE_KEY="$KAGGLE_KEY"
        mkdir -p "$DATA_ROOT_DN"
        cd "$DATA_ROOT_DN"
        kaggle datasets download -d kausthubhmanda/domainnet-fulldataset \
            -p "$DATA_ROOT_DN" --unzip
        cd -
    else
        echo "[skip] KAGGLE_USER / KAGGLE_KEY not set."
        echo "       Download DomainNet manually into $DATA_ROOT_DN."
        echo "       Expected layout: <DATA_ROOT_DN>/<domain>/{train,test}"
        echo "       Domains: clipart, infograph, painting, quickdraw, real, sketch"
    fi
fi

# ----------------------------------------------------------------------------
# 6. Dataset layout sanity check
# ----------------------------------------------------------------------------
echo "=== Dataset layout check ==="
expected=(clipart infograph painting quickdraw real sketch)
for d in "${expected[@]}"; do
    if [ -d "$DATA_ROOT_DN/$d/train" ]; then
        n=$(find "$DATA_ROOT_DN/$d/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  [ok] $d/train ($n classes)"
    else
        echo "  [missing] $d/train"
    fi
done

# ----------------------------------------------------------------------------
# 7. WandB login (if key in env)
# ----------------------------------------------------------------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "=== Logging into wandb ==="
    wandb login --relogin "$WANDB_API_KEY" >/dev/null
else
    echo "[skip] WANDB_API_KEY not set. wandb will run offline."
fi

# ----------------------------------------------------------------------------
# 8. Persist env vars in bashrc
# ----------------------------------------------------------------------------
BASHRC="$HOME/.bashrc"
touch "$BASHRC"
grep -qxF "export DATA_ROOT_DN=$DATA_ROOT_DN" "$BASHRC" \
    || echo "export DATA_ROOT_DN=$DATA_ROOT_DN" >> "$BASHRC"
grep -qxF "export WANDB_DIR=$WANDB_DIR" "$BASHRC" \
    || echo "export WANDB_DIR=$WANDB_DIR" >> "$BASHRC"
echo "=== Wrote DATA_ROOT_DN, WANDB_DIR to $BASHRC ==="

# ----------------------------------------------------------------------------
# 9. tmux for long runs
# ----------------------------------------------------------------------------
if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1 \
        || yum install -y tmux >/dev/null 2>&1 \
        || echo "[warn] could not install tmux; install manually if needed"
fi

echo ""
echo "=== Setup done ==="
echo ""
echo "Next:"
echo "  1. export KAGGLE_USER=... KAGGLE_KEY=... WANDB_API_KEY=...   (if not yet)"
echo "  2. bash setup_pod.sh                                          (re-run to fill)"
echo "  3. cd $REPO_DIR/CL_exp"
echo "  4. Smoke test:"
echo "       DATA_ROOT_DN=$DATA_ROOT_DN BACKBONE=resnet18 \\"
echo "       PH1=2 PH2=2 TOTAL_STEPS=200 WARMUP=50 \\"
echo "       BS=64 ACCUM=2 METHODS='nostalgia naive_adam' \\"
echo "       bash run_image_benchmarks.sh"
echo "  5. Full sweep:"
echo "       tmux new -s cl"
echo "       DATA_ROOT_DN=$DATA_ROOT_DN BACKBONE=resnet18 bash run_image_benchmarks.sh"
echo "       Ctrl-b d to detach"

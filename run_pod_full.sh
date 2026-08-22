#!/usr/bin/env bash
# RunPod single-GPU full DomainNet sweep.
# Paste into Jupyter cell (one-shot, top-to-bottom) or `bash run_pod_full.sh`.
#
# Stages:
#   1. Sanity (paths, GPU, dataset layout).
#   2. Smoke (tiny budget, 2 tasks, 1 method) — fast fail.
#   3. Full sweep (6 tasks × N methods).
#
# Override via env: TASKS=... METHODS=... BACKBONE=... bash run_pod_full.sh

set -euo pipefail

# ---------- Config ----------
export DATA_ROOT_DN="${DATA_ROOT_DN:-/workspace/data/domainnet}"
export WANDB_DIR="${WANDB_DIR:-/workspace/data/wandb}"
REPO_DIR="${REPO_DIR:-/workspace/Nostalgia-CL-exp}"

# RunPod pod: 1 GPU by default. Override if multi-GPU pod.
export ACCEL="${ACCEL:-gpu}"
export DEVICES="${DEVICES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

BACKBONE="${BACKBONE:-resnet18}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

# Methods (override via METHODS=...).
METHODS="${METHODS:-nostalgia naive_adam ewc gpm agem sdft}"

# Tasks (DomainNet, 6 domains).
TASKS="${TASKS:-domainnet_clipart domainnet_infograph domainnet_painting domainnet_quickdraw domainnet_real domainnet_sketch}"

# Smoke vs full.
MODE="${MODE:-full}"   # smoke | full

if [ "$MODE" = "smoke" ]; then
    PH1=1; PH2=1; WARMUP=10; TOTAL_STEPS=50; LOG_EVERY=10
    BS=32; ACCUM=1
    TASKS="domainnet_clipart domainnet_real"
    METHODS="nostalgia naive_adam"
else
    PH1=3; PH2=5; WARMUP=100; TOTAL_STEPS=1000; LOG_EVERY=50
    if [ "$BACKBONE" = "vit" ] || [ "$BACKBONE" = "siglip" ]; then
        BS=64; ACCUM=2
    else
        BS=128; ACCUM=2
    fi
fi

LR="1e-3"
HEAD_LR="5e-4"
WEIGHT_DECAY="1e-4"
GRAD_CLIP="1.0"

mkdir -p "$WANDB_DIR"

# ---------- 1. Sanity ----------
echo "=== [1/3] Sanity ==="
echo "REPO_DIR      = $REPO_DIR"
echo "DATA_ROOT_DN  = $DATA_ROOT_DN"
echo "WANDB_DIR     = $WANDB_DIR"
echo "BACKBONE      = $BACKBONE"
echo "METHODS       = $METHODS"
echo "TASKS         = $TASKS"
echo "MODE          = $MODE"
echo "ACCEL/DEVICES = $ACCEL / $DEVICES"

[ -d "$REPO_DIR" ] || { echo "[FATAL] repo not found at $REPO_DIR"; exit 1; }
[ -d "$DATA_ROOT_DN" ] || { echo "[FATAL] dataset not found at $DATA_ROOT_DN"; exit 1; }
for d in clipart infograph painting quickdraw real sketch; do
    if [ ! -d "$DATA_ROOT_DN/$d/train" ]; then
        echo "[FATAL] missing $DATA_ROOT_DN/$d/train"
        exit 1
    fi
done
echo "[ok] all 6 domain folders present"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[warn] nvidia-smi missing; GPU may be unavailable"
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

cd "$REPO_DIR"
[ -f train.py ] || { echo "[FATAL] train.py not in $REPO_DIR"; exit 1; }

# ---------- 2. Smoke ----------
if [ "$MODE" = "smoke" ]; then
    echo ""
    echo "=== [2/3] Smoke run (already in smoke config) ==="
else
    echo ""
    echo "=== [2/3] Smoke run (auto, tiny budget) ==="
    SMOKE_TASKS="domainnet_clipart domainnet_real"
    SMOKE_METHOD="nostalgia"
    python train.py \
        --backbone "$BACKBONE" \
        --image_size "$IMAGE_SIZE" \
        --tasks $SMOKE_TASKS \
        --method "$SMOKE_METHOD" \
        --data_root "$DATA_ROOT_DN" \
        --data_root_domainnet "$DATA_ROOT_DN" \
        --max_length 32 \
        --num_workers "$NUM_WORKERS" \
        --pin_memory \
        --epochs_phase1 1 \
        --epochs_phase2 1 \
        --warmup_steps 5 \
        --total_steps 30 \
        --batch_size 16 \
        --accumulate_grad_batches 1 \
        --accelerator "$ACCEL" \
        --devices "$DEVICES" \
        --log_every_n_steps 5 \
        --val_check_interval 1.0 \
        --wandb_project "domainnet-cl-smoke" \
        --wandb_name "smoke_${BACKBONE}_${SMOKE_METHOD}" \
        --base_optimizer adamw --lr 1e-3 --head_lr 5e-4 \
        --weight_decay 1e-4 --grad_clip_val 1.0 \
        --k 16 --nostalgia_accumulation_rounds 1 \
        --nostalgia_max_hessian_batch 8 --nostalgia_num_samples 100
    echo "[ok] smoke passed"
fi

# ---------- 3. Full sweep ----------
echo ""
echo "=== [3/3] Full sweep ==="

for method in $METHODS; do
    exp_name="domainnet_${BACKBONE}_${method}"

    extra_args="--base_optimizer adamw --lr $LR --head_lr $HEAD_LR --weight_decay $WEIGHT_DECAY --grad_clip_val $GRAD_CLIP"

    case "$method" in
        nostalgia|gpm|ewc_nostalgia)
            extra_args="${extra_args} --k 32 --nostalgia_accumulation_rounds 3 --nostalgia_max_hessian_batch 16 --nostalgia_num_samples 1000"
            ;;
    esac
    case "$method" in
        ewc|ewc_nostalgia)
            extra_args="${extra_args} --ewc_lambda 400.0"
            ;;
    esac
    case "$method" in
        agem)
            extra_args="${extra_args} --agem_mem_size 1000"
            ;;
    esac
    case "$method" in
        sdft)
            extra_args="${extra_args} --sdft_lambda_distillation 1.0 --sdft_temperature 2.0"
            ;;
    esac

    echo ""
    echo "------------------------------------------------------------------"
    echo "Running: backbone=$BACKBONE method=$method tasks=$TASKS"
    echo "------------------------------------------------------------------"

    python train.py \
        --backbone "$BACKBONE" \
        --image_size "$IMAGE_SIZE" \
        --tasks $TASKS \
        --method "$method" \
        --data_root "$DATA_ROOT_DN" \
        --data_root_domainnet "$DATA_ROOT_DN" \
        --max_length 32 \
        --num_workers "$NUM_WORKERS" \
        --pin_memory \
        --epochs_phase1 "$PH1" \
        --epochs_phase2 "$PH2" \
        --warmup_steps "$WARMUP" \
        --total_steps "$TOTAL_STEPS" \
        --batch_size "$BS" \
        --accumulate_grad_batches "$ACCUM" \
        --accelerator "$ACCEL" \
        --devices "$DEVICES" \
        --log_every_n_steps "$LOG_EVERY" \
        --val_check_interval 1.0 \
        --wandb_project "domainnet-cl" \
        --wandb_name "$exp_name" \
        $extra_args

    echo "[done] $exp_name"
done

echo ""
echo "=== All runs complete ==="

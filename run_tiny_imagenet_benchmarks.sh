#!/usr/bin/env bash
set -euo pipefail

# Split Tiny ImageNet — full fine-tuning benchmark across all CL methods.
# Usage:
#   bash run_tiny_imagenet_benchmarks.sh
# Override via env vars, e.g.:
#   BACKBONE=resnet18 BS=128 bash run_tiny_imagenet_benchmarks.sh

ACCEL="${ACCEL:-mps}"
DEVICES="${DEVICES:-1}"

# Full training budget (not a capped smoke test).
PH1="20"            # head-alignment epochs per task
PH2="40"            # full-finetuning epochs per task
WARMUP="500"        # linear-warmup steps (per task / phase)
TOTAL_STEPS="8000"  # scheduler horizon tuned to ~40 epochs of full Tiny ImageNet
BS="128"            # standard ImageNet-style batch for ResNet-18 @ 64x64
LR="1e-3"           # cosine/lamb-style base for ResNet-18 full FT
HEAD_LR="5e-4"      # smaller head LR during alignment
WEIGHT_DECAY="1e-4" # ImageNet-style WD
GRAD_CLIP="1.0"
LOG_EVERY="100"
VAL_EVERY="1.0"     # validate once per epoch (full val pass)

DATA_ROOT="${DATA_ROOT:-$HOME/data}"
DATA_ROOT_TINY="${DATA_ROOT_TINY:-$HOME/data}"
BACKBONE="${BACKBONE:-resnet18}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"

TASKS=(
    "tinyimg_t0" "tinyimg_t1" "tinyimg_t2" "tinyimg_t3"
)

METHODS=(
    # "naive_adam"
    # "nostalgia"
    # "ewc"
    # "agem"
    # "gpm"
    # "ewc_nostalgia"
    "sdft"
)

for method in "${METHODS[@]}"; do
    exp_name="tinyimagenet_split_${BACKBONE}_${method}_fullft"

    extra_args="--base_optimizer adamw --lr $LR --head_lr $HEAD_LR --weight_decay $WEIGHT_DECAY --grad_clip_val $GRAD_CLIP"
    if [ "$method" = "nostalgia" ] || [ "$method" = "gpm" ] || [ "$method" = "ewc_nostalgia" ]; then
        # MPS-safe projection estimation: cap samples and rank to avoid OOM.
        extra_args="${extra_args} --k 64 --nostalgia_accumulation_rounds 5 --nostalgia_max_hessian_batch 32 --nostalgia_num_samples 2000"
    fi
    if [ "$method" = "ewc" ] || [ "$method" = "ewc_nostalgia" ]; then
        extra_args="${extra_args} --ewc_lambda 400.0"
    fi
    if [ "$method" = "agem" ]; then
        extra_args="${extra_args} --agem_mem_size 2000"
    fi
    if [ "$method" = "sdft" ]; then
        extra_args="${extra_args} --sdft_lambda_distillation 0.1 --sdft_temperature 2.0"
    fi

    echo "====================================================================="
    echo "Running: backbone=$BACKBONE  method=$method  tasks=${TASKS[*]}"
    echo "====================================================================="

    python train.py \
        --backbone "$BACKBONE" \
        --image_size "$IMAGE_SIZE" \
        --tasks "${TASKS[@]}" \
        --method "$method" \
        --data_root "$DATA_ROOT" \
        --data_root_tinyimagenet "$DATA_ROOT_TINY" \
        --max_length 32 \
        --epochs_phase1 "$PH1" \
        --epochs_phase2 "$PH2" \
        --warmup_steps "$WARMUP" \
        --total_steps "$TOTAL_STEPS" \
        --batch_size "$BS" \
        --accelerator "$ACCEL" \
        --devices "$DEVICES" \
        --log_every_n_steps "$LOG_EVERY" \
        --val_check_interval "$VAL_EVERY" \
        --wandb_project "tiny-imagenet-cl" \
        --wandb_name "$exp_name" \
        $extra_args

    echo ""
    echo "Finished: $exp_name"
    echo ""
done

echo "All Tiny ImageNet benchmark runs completed."

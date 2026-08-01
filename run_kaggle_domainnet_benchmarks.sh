#!/usr/bin/env bash
set -euo pipefail

# DomainNet full fine-tuning benchmark on Kaggle.
# Usage inside a Kaggle notebook / script:
#   bash run_kaggle_domainnet_benchmarks.sh
# Override via env vars, e.g.:
#   METHODS=("nostalgia") BS=64 bash run_kaggle_domainnet_benchmarks.sh

ACCEL="${ACCEL:-gpu}"
DEVICES="${DEVICES:-1}"

# Full training budget.
PH1="20"            # head-alignment epochs per domain
PH2="40"            # full-finetuning epochs per domain
WARMUP="500"        # linear-warmup steps
TOTAL_STEPS="10000" # scheduler horizon per full-finetuning domain
BS="64"             # Kaggle T4/P100 safe batch for ResNet-18 @ 224x224
LR="1e-3"           # ResNet-18 full FT
HEAD_LR="5e-4"
WEIGHT_DECAY="1e-4"
GRAD_CLIP="1.0"
LOG_EVERY="100"
VAL_EVERY="1.0"

# Kaggle input paths (default; override with env vars if needed).
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/kausthubhmanda/domainnet-fulldataset}"
DATA_ROOT_DN="${DATA_ROOT_DN:-$DATA_ROOT}"

BACKBONE="${BACKBONE:-resnet18}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

TASKS=(
    "domainnet_clipart" "domainnet_infograph" "domainnet_painting"
    "domainnet_quickdraw" "domainnet_real" "domainnet_sketch"
)

METHODS=(
    "naive_adam"
    "nostalgia"
    "ewc"
    "agem"
    "gpm"
    "ewc_nostalgia"
    "sdft"
)

for method in "${METHODS[@]}"; do
    exp_name="domainnet_split_${BACKBONE}_${method}_fullft"

    extra_args="--base_optimizer adamw --lr $LR --head_lr $HEAD_LR --weight_decay $WEIGHT_DECAY --grad_clip_val $GRAD_CLIP"
    if [ "$method" = "nostalgia" ] || [ "$method" = "gpm" ] || [ "$method" = "ewc_nostalgia" ]; then
        # Kaggle-safe projection estimation.
        extra_args="${extra_args} --k 64 --nostalgia_accumulation_rounds 5 --nostalgia_max_hessian_batch 32 --nostalgia_num_samples 2000"
    fi
    if [ "$method" = "ewc" ] || [ "$method" = "ewc_nostalgia" ]; then
        extra_args="${extra_args} --ewc_lambda 400.0"
    fi
    if [ "$method" = "agem" ]; then
        extra_args="${extra_args} --agem_mem_size 2000"
    fi
    if [ "$method" = "sdft" ]; then
        extra_args="${extra_args} --sdft_lambda_distillation 1.0 --sdft_temperature 2.0"
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
        --data_root_domainnet "$DATA_ROOT_DN" \
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
        --wandb_project "domainnet-cl" \
        --wandb_name "$exp_name" \
        $extra_args

    echo ""
    echo "Finished: $exp_name"
    echo ""
done

echo "All DomainNet benchmark runs completed."

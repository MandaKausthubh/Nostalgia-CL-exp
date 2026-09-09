#!/usr/bin/env bash
set -euo pipefail

# Full-scale DomainNet benchmark for the unified train.py pipeline.
# Run on GPU; override paths via env vars:
#   DATA_ROOT_DN=/path/to/domainnet bash run_image_benchmarks.sh

ACCEL="${ACCEL:-gpu}"
DEVICES="${DEVICES:-4}"

# Reduced training budget for ICLR deadline (4x A100).
PH1="5"             # head-alignment epochs per domain
PH2="15"            # full-finetuning epochs per domain
WARMUP="400"        # linear-warmup steps
TOTAL_STEPS="3000"  # scheduler horizon per full-finetuning domain
LR="1e-3"           # ResNet-18 / ViT-B/16 full FT
HEAD_LR="5e-4"
WEIGHT_DECAY="1e-4"
GRAD_CLIP="1.0"
LOG_EVERY="100"
VAL_EVERY="1.0"     # validate once per epoch

# Default DomainNet root. Point at folder containing the `domainnet` subfolder.
DATA_ROOT_DN="${DATA_ROOT_DN:-/kaggle/input/datasets/kausthubhmanda/domainnet-fulldataset}"

# Per-task checkpointing. Set PUSH_TO_HUB=1 to upload each task checkpoint.
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_HUB_NAMESPACE="${HF_HUB_NAMESPACE:-${HF_USERNAME:-${HF_ORG:-}}}"
HF_HUB_PRIVATE="${HF_HUB_PRIVATE:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

BACKBONE="${BACKBONE:-vit}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Per-GPU batch size: 4x A100 (80GB) — saturate all GPUs.
if [ -z "${BS:-}" ]; then
    if [ "$BACKBONE" = "vit" ] || [ "$BACKBONE" = "siglip" ]; then
        BS="96"
    else
        BS="384"
    fi
fi

# Gradient accumulation: effective batch 4x larger (1024 ViT, 1536 ResNet).
if [ -z "${ACCUM:-}" ]; then
    if [ "$BACKBONE" = "vit" ] || [ "$BACKBONE" = "siglip" ]; then
        ACCUM="2"
    else
        ACCUM="1"
    fi
fi

TASKS=(
    "domainnet_clipart" "domainnet_infograph" "domainnet_painting"
    "domainnet_quickdraw" "domainnet_real" "domainnet_sketch"
)

ALL_METHODS=(
    "naive_adam"
    "nostalgia"
    "ewc"
    "agem"
    "gpm"
    "ewc_nostalgia"
    "sdft"
)

# Override subset via env: METHODS="nostalgia naive_adam" bash run_image_benchmarks.sh
if [ -n "${METHODS:-}" ]; then
    METHODS=()
    for m in $METHODS; do
        METHODS+=("$m")
    done
else
    METHODS=("${ALL_METHODS[@]}")
fi

for method in "${METHODS[@]}"; do
    exp_name="domainnet_split_${BACKBONE}_${method}_fullft"

    extra_args="--base_optimizer adamw --lr $LR --head_lr $HEAD_LR --weight_decay $WEIGHT_DECAY --grad_clip_val $GRAD_CLIP"
    if [ "$method" = "nostalgia" ] || [ "$method" = "gpm" ] || [ "$method" = "ewc_nostalgia" ]; then
        # GPU-safe projection estimation.
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

    checkpoint_args=(--checkpoint_dir "$CHECKPOINT_DIR")
    if [ "$PUSH_TO_HUB" = "1" ] || [ "$PUSH_TO_HUB" = "true" ]; then
        checkpoint_args+=(--push_to_hub)
        if [ -n "$HF_HUB_NAMESPACE" ]; then
            checkpoint_args+=(--hf_hub_namespace "$HF_HUB_NAMESPACE")
        fi
        if [ "$HF_HUB_PRIVATE" = "1" ] || [ "$HF_HUB_PRIVATE" = "true" ]; then
            checkpoint_args+=(--hf_hub_private)
        fi
    fi

    echo "====================================================================="
    echo "Running: backbone=$BACKBONE  method=$method  tasks=${TASKS[*]}"
    echo "====================================================================="

    python train.py \
        --backbone "$BACKBONE" \
        --image_size "$IMAGE_SIZE" \
        --tasks "${TASKS[@]}" \
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
        --strategy ddp_find_unused_parameters_true \
        --log_every_n_steps "$LOG_EVERY" \
        --val_check_interval "$VAL_EVERY" \
        --wandb_project "domainnet-cl" \
        --wandb_name "$exp_name" \
        $extra_args \
        "${checkpoint_args[@]}"

    echo ""
    echo "Finished: $exp_name"
    echo ""
done

echo "All DomainNet benchmark runs completed."

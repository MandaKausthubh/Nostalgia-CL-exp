#!/usr/bin/env bash
set -euo pipefail

# Image benchmark loop for the unified train.py pipeline.

ACCEL="gpu"
DEVICES="1"
PH1="1"
PH2="1"
WARMUP="5"
TOTAL_STEPS="30"
MAX_TRAIN="100"
MAX_VAL="50"
BS="8"
LOG_EVERY="10"
DATA_ROOT="./data"
SUFFIX="grid"

BACKBONES=(
    "resnet10:32"
    "resnet18:32"
    "vit:224"
    "siglip:224"
)

BENCHMARKS=(
    "cifar100_split:cifar100_t0 cifar100_t1"
    "tinyimagenet_split:tinyimg_t0 tinyimg_t1"
    "imagenet100_split:imagenet100_t0 imagenet100_t1"
    "domainnet:domainnet_real domainnet_sketch"
)

METHODS=(
    "nostalgia"
    "naive_adam"
    "ewc"
    "gpm"
    "agem"
    "sdft"
)

DATA_ROOT_TINY="./data"
DATA_ROOT_DN="./data"

run_exp() {
    local backbone_name="$1"
    local image_size="$2"
    local bench_name="$3"
    local tasks="$4"
    local method="$5"
    local exp_name="${bench_name}_${backbone_name}_${method}_${SUFFIX}"

    echo "====================================================================="
    echo "Running: backbone=$backbone_name  bench=$bench_name  method=$method"
    echo "====================================================================="

    local extra_args=""
    if [ "$method" = "nostalgia" ] || [ "$method" = "gpm" ]; then
        extra_args="--k 4 --nostalgia_accumulation_rounds 1 --nostalgia_max_hessian_batch 4 --nostalgia_num_samples 40"
    fi
    if [ "$method" = "sdft" ]; then
        extra_args="--sdft_lambda_distillation 0.5 --sdft_temperature 2.0"
    fi

    local root_override=""
    if [ "$bench_name" = "tinyimagenet_split" ]; then
        root_override="--data_root_tinyimagenet $DATA_ROOT_TINY"
    fi
    if [ "$bench_name" = "domainnet" ]; then
        root_override="--data_root_domainnet $DATA_ROOT_DN"
    fi

    python train.py \
        --backbone "$backbone_name" \
        --image_size "$image_size" \
        --tasks $tasks \
        --method "$method" \
        --epochs_phase1 "$PH1" \
        --epochs_phase2 "$PH2" \
        --warmup_steps "$WARMUP" \
        --total_steps "$TOTAL_STEPS" \
        --max_train_samples "$MAX_TRAIN" \
        --max_val_samples "$MAX_VAL" \
        --batch_size "$BS" \
        --accelerator "$ACCEL" \
        --devices "$DEVICES" \
        --log_every_n_steps "$LOG_EVERY" \
        --data_root "$DATA_ROOT" \
        $root_override \
        $extra_args \
        --wandb_project "image-cl-benchmarks" \
        --wandb_name "$exp_name"

    echo ""
    echo "Finished: $exp_name"
    echo ""
}

for spec in "${BACKBONES[@]}"; do
    IFS=':' read -r backbone_name image_size <<< "$spec"
    for bench in "${BENCHMARKS[@]}"; do
        IFS=':' read -r bench_name tasks <<< "$bench"
        for method in "${METHODS[@]}"; do
            run_exp "$backbone_name" "$image_size" "$bench_name" "$tasks" "$method"
        done
    done
done

echo "All image benchmark runs completed."

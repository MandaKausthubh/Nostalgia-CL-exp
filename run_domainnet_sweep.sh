#!/usr/bin/env bash
set -euo pipefail

# DomainNet continual-learning benchmark sweep for ICLR.
# All CL methods × all image backbones × N seeds, each over the 6 sequential
# DomainNet domains. Target hardware: 4× A100 80GB on RunPod.
#
# Defaults give a 7 × 3 × 3 = 63-run main table. Override any axis via env:
#   SEEDS="0 1" BACKBONES="resnet18 vit" METHODS="nostalgia naive_adam" \
#       bash run_domainnet_sweep.sh
#   DATA_ROOT_DN=/workspace/data/domainnet bash run_domainnet_sweep.sh
#   BS_SIGLIP=32 PH2=10 bash run_domainnet_sweep.sh   # per-backbone / per-budget knobs
#
# Loop order (per spec): methods → backbones → seeds; tasks fixed per run.

# ----- Hardware / runtime ----------------------------------------------
ACCEL="${ACCEL:-gpu}"
DEVICES="${DEVICES:-4}"
STRATEGY="${STRATEGY:-ddp_find_unused_parameters_true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ----- Training budget (ICLR main) -------------------------------------
PH1="${PH1:-5}"            # head-alignment epochs per domain
PH2="${PH2:-15}"           # full-finetuning epochs per domain
WARMUP="${WARMUP:-400}"    # linear warmup steps per domain
TOTAL_STEPS="${TOTAL_STEPS:-3000}"
LR="${LR:-1e-3}"
HEAD_LR="${HEAD_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-100}"
VAL_EVERY="${VAL_EVERY:-1.0}"     # validate once per epoch
WANDB_PROJECT="${WANDB_PROJECT:-domainnet-cl-iclr}"

# ----- Data -------------------------------------------------------------
DATA_ROOT_DN="${DATA_ROOT_DN:-/workspace/data/domainnet}"

TASKS=(
    "domainnet_clipart"
    "domainnet_infograph"
    "domainnet_painting"
    "domainnet_quickdraw"
    "domainnet_real"
    "domainnet_sketch"
)

# ----- Method / backbone / seed axes ------------------------------------
ALL_METHODS=(
    "naive_adam"
    "nostalgia"
    "ewc"
    "agem"
    "gpm"
    "ewc_nostalgia"
    "sdft"
)

ALL_BACKBONES=(
    "resnet18"
    "vit"
    "siglip"
)

SEEDS="${SEEDS:-0 1 2}"   # ICLR default: 3 seeds per config

# Per-backbone image size.
declare -A IMG_SIZE=(
    ["resnet18"]="${IMG_SIZE_RESNET:-224}"
    ["vit"]="${IMG_SIZE_VIT:-224}"
    ["siglip"]="${IMG_SIZE_SIGLIP:-224}"
)

# Per-GPU batch size on 4× A100 80GB. Effective = BS × ACCUM × DEVICES.
declare -A BS_DEFAULT=(
    ["resnet18"]="${BS_RESNET:-384}"
    ["vit"]="${BS_VIT:-96}"
    ["siglip"]="${BS_SIGLIP:-64}"
)

declare -A ACCUM_DEFAULT=(
    ["resnet18"]="${ACCUM_RESNET:-1}"
    ["vit"]="${ACCUM_VIT:-2}"
    ["siglip"]="${ACCUM_SIGLIP:-2}"
)

# ----- Optional axis overrides ------------------------------------------
if [ -n "${METHODS:-}" ]; then
    _methods=()
    for m in $METHODS; do _methods+=("$m"); done
    METHODS=("${_methods[@]}")
else
    METHODS=("${ALL_METHODS[@]}")
fi

if [ -n "${BACKBONES:-}" ]; then
    _backbones=()
    for b in $BACKBONES; do _backbones+=("$b"); done
    BACKBONES=("${_backbones[@]}")
else
    BACKBONES=("${ALL_BACKBONES[@]}")
fi

# ----- Run plan summary -------------------------------------------------
_n_seeds=$(echo $SEEDS | wc -w | tr -d ' ')
_total_runs=$((${#METHODS[@]} * ${#BACKBONES[@]} * _n_seeds))

echo "====================================================================="
echo "DomainNet CL sweep — ICLR"
echo "  Methods:    ${METHODS[*]}  (${#METHODS[@]})"
echo "  Backbones:  ${BACKBONES[*]}  (${#BACKBONES[@]})"
echo "  Seeds:      $SEEDS  ($_n_seeds)"
echo "  Tasks:      ${TASKS[*]}"
echo "  Devices:    $DEVICES × $ACCEL"
echo "  Total runs: $_total_runs  (each = 6 sequential domains)"
echo "====================================================================="

# ----- Sweep ------------------------------------------------------------
_run_idx=0
for method in "${METHODS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        image_size="${IMG_SIZE[$backbone]}"
        bs="${BS_DEFAULT[$backbone]}"
        accum="${ACCUM_DEFAULT[$backbone]}"

        for seed in $SEEDS; do
            _run_idx=$((_run_idx + 1))
            exp_name="domainnet_${backbone}_${method}_seed${seed}_fullft"

            # Per-method extras.
            extra_args="--base_optimizer adamw --lr $LR --head_lr $HEAD_LR --weight_decay $WEIGHT_DECAY --grad_clip_val $GRAD_CLIP --seed $seed"
            if [ "$method" = "nostalgia" ] || [ "$method" = "gpm" ] || [ "$method" = "ewc_nostalgia" ]; then
                # GPU-safe null-space / GPM subspace estimation.
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

            echo ""
            echo "---------------------------------------------------------------------"
            echo "[$_run_idx/$_total_runs]  backbone=$backbone  method=$method  seed=$seed"
            echo "  exp_name    = $exp_name"
            echo "  image_size  = $image_size"
            echo "  bs/accum    = $bs / $accum  (eff=${bs}×${accum}×${DEVICES}=$((bs * accum * DEVICES)))"
            echo "  tasks       = ${TASKS[*]}"
            echo "---------------------------------------------------------------------"

            python train.py \
                --backbone "$backbone" \
                --image_size "$image_size" \
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
                --batch_size "$bs" \
                --accumulate_grad_batches "$accum" \
                --accelerator "$ACCEL" \
                --devices "$DEVICES" \
                --strategy "$STRATEGY" \
                --log_every_n_steps "$LOG_EVERY" \
                --val_check_interval "$VAL_EVERY" \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_name "$exp_name" \
                $extra_args

            echo "Finished: $exp_name"
        done
    done
done

echo "====================================================================="
echo "All $_total_runs DomainNet runs completed."
echo "====================================================================="

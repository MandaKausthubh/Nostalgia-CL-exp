#!/usr/bin/env bash
set -euo pipefail

# DomainNet continual-learning benchmark sweep for ICLR.
# All CL methods × all image backbones × N seeds, each over the 6 sequential
# DomainNet domains. Target hardware: 4× A100 80GB on RunPod.
#
# LoRA is ON by default (USE_LORA=0 for full-ft fallback): adapter-only
# training keeps the Nostalgia Q matrix in ~0.6M-param space (~60 MB/task)
# instead of 85.8M-param space (5.5 GB/task, cuSOLVER int32 overflow).
#
# Defaults give a 7 × 3 × 3 = 63-run main table. Override any axis via env:
#   SEEDS="0 1" BACKBONES="resnet18 vit" METHODS="nostalgia naive_adam" \
#       bash run_domainnet_sweep.sh
#   DATA_ROOT_DN=$HOME/domainnet bash run_domainnet_sweep.sh
#   BS_SIGLIP=32 PH2=10 bash run_domainnet_sweep.sh   # per-backbone / per-budget knobs
#   USE_LORA=0 K=32 bash run_domainnet_sweep.sh       # full-ft fallback / larger null-space
#
# Loop order: seeds → backbones → methods; tasks fixed per run.

# ----- Hardware / runtime ----------------------------------------------
ACCEL="${ACCEL:-gpu}"
DEVICES="${DEVICES:-1}"
STRATEGY="${STRATEGY:-auto}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16-mixed}"

# ----- Training budget (ICLR main) -------------------------------------
PH1="${PH1:-5}"            # head-alignment epochs per domain
PH2="${PH2:-15}"           # full-finetuning epochs per domain
WARMUP="${WARMUP:-400}"    # linear warmup steps per domain
TOTAL_STEPS="${TOTAL_STEPS:-3000}"
LR="${LR:-1e-3}"
HEAD_LR="${HEAD_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-5}"
VAL_EVERY="${VAL_EVERY:-1.0}"     # validate once per epoch
VAL_EPOCHS="${VAL_EPOCHS:-3}"     # validate every N Phase-2 epochs (--val_every_n_epochs)
# Optional per-task cap on val samples (unset = full val set; changes reported acc).
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
WANDB_PROJECT="${WANDB_PROJECT:-domainnet-cl-iclr}"

# ----- LoRA (default: ON) ------------------------------------------------
USE_LORA="${USE_LORA:-1}"         # 1 = LoRA adapters, 0 = full-ft fallback
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ----- Memory format --------------------------------------------------------
# channels_last speeds up conv backbones (resnet10/18) on Ampere+; ignored for
# vit/siglip. Set CHANNELS_LAST=0 to fall back to contiguous NCHW.
CHANNELS_LAST="${CHANNELS_LAST:-1}"

# ----- Nostalgia Hessian rank --------------------------------------------
# k=24 kept identical to the full-ft setup for method comparability; over the
# ~0.6M-dim adapter space it spans a much larger spectral fraction. Q storage
# drops from 5.5 GB/task to ~60 MB/task. Raise uniformly via K=... if smoke
# shows weak protection.
K="${K:-24}"

# ----- Per-backbone hyperparameters -------------------------------------
# Backbone-specific overrides keyed by backbone name. Nested fallback:
#   per-backbone env (LR_VIT) > global env (LR) > baked-in default.
declare -A LR_MAP=(
    ["resnet18"]="${LR_RESNET18:-${LR_RESNET:-${LR}}}"
    ["vit"]="${LR_VIT:-${LR:-3e-4}}"
    ["siglip"]="${LR_SIGLIP:-${LR:-3e-4}}"
)
declare -A HEAD_LR_MAP=(
    ["resnet18"]="${HEAD_LR_RESNET18:-${HEAD_LR_RESNET:-${HEAD_LR}}}"
    ["vit"]="${HEAD_LR_VIT:-${HEAD_LR}}"
    ["siglip"]="${HEAD_LR_SIGLIP:-${HEAD_LR}}"
)
declare -A WARMUP_MAP=(
    ["resnet18"]="${WARMUP_RESNET18:-${WARMUP_RESNET:-${WARMUP}}}"
    ["vit"]="${WARMUP_VIT:-${WARMUP:-600}}"
    ["siglip"]="${WARMUP_SIGLIP:-${WARMUP:-600}}"
)
declare -A TOTAL_STEPS_MAP=(
    ["resnet18"]="${TOTAL_STEPS_RESNET18:-${TOTAL_STEPS_RESNET:-${TOTAL_STEPS}}}"
    ["vit"]="${TOTAL_STEPS_VIT:-${TOTAL_STEPS}}"
    ["siglip"]="${TOTAL_STEPS_SIGLIP:-${TOTAL_STEPS}}"
)
declare -A WD_MAP=(
    ["resnet18"]="${WD_RESNET18:-${WD_RESNET:-${WEIGHT_DECAY}}}"
    ["vit"]="${WD_VIT:-${WEIGHT_DECAY}}"
    ["siglip"]="${WD_SIGLIP:-${WEIGHT_DECAY:-5e-3}}"
)
declare -A GC_MAP=(
    ["resnet18"]="${GC_RESNET18:-${GC_RESNET:-${GRAD_CLIP}}}"
    ["vit"]="${GC_VIT:-${GRAD_CLIP}}"
    ["siglip"]="${GC_SIGLIP:-${GRAD_CLIP}}"
)
declare -A PH1_MAP=(
    ["resnet18"]="${PH1_RESNET18:-${PH1_RESNET:-${PH1}}}"
    ["vit"]="${PH1_VIT:-${PH1}}"
    ["siglip"]="${PH1_SIGLIP:-${PH1}}"
)
declare -A PH2_MAP=(
    ["resnet18"]="${PH2_RESNET18:-${PH2_RESNET:-${PH2}}}"
    ["vit"]="${PH2_VIT:-${PH2}}"
    ["siglip"]="${PH2_SIGLIP:-${PH2}}"
)
declare -A VAL_EPOCHS_MAP=(
    ["resnet18"]="${VAL_EPOCHS_RESNET18:-${VAL_EPOCHS_RESNET:-${VAL_EPOCHS}}}"
    ["vit"]="${VAL_EPOCHS_VIT:-${VAL_EPOCHS}}"
    ["siglip"]="${VAL_EPOCHS_SIGLIP:-${VAL_EPOCHS}}"
)

# ----- Data -------------------------------------------------------------
DATA_ROOT_DN="${DATA_ROOT_DN:-$HOME/domainnet}"

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

# Per-GPU batch size on single A100 80GB. Effective = BS × ACCUM × DEVICES.
declare -A BS_DEFAULT=(
    ["resnet18"]="${BS_RESNET:-768}"
    ["vit"]="${BS_VIT:-192}"
    ["siglip"]="${BS_SIGLIP:-128}"
)

declare -A ACCUM_DEFAULT=(
    ["resnet18"]="${ACCUM_RESNET:-1}"
    ["vit"]="${ACCUM_VIT:-1}"
    ["siglip"]="${ACCUM_SIGLIP:-1}"
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
if [ "$USE_LORA" = "1" ]; then
    echo "  LoRA:       ON  (r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT)"
else
    echo "  LoRA:       OFF (full finetuning)"
fi
echo "  K:          $K"
echo "  Total runs: $_total_runs  (each = 6 sequential domains)"
echo "====================================================================="

# ----- Sweep ------------------------------------------------------------
# Loop order: seed → backbone → method. Seed 0 completes ALL backbone×method
# combos first, so the paper's main table can be drafted while seeds 1..N run.
_run_idx=0
for seed in $SEEDS; do
    for backbone in "${BACKBONES[@]}"; do
        image_size="${IMG_SIZE[$backbone]}"
        bs="${BS_DEFAULT[$backbone]}"
        accum="${ACCUM_DEFAULT[$backbone]}"
        lr="${LR_MAP[$backbone]}"
        head_lr="${HEAD_LR_MAP[$backbone]}"
        warmup="${WARMUP_MAP[$backbone]}"
        total_steps="${TOTAL_STEPS_MAP[$backbone]}"
        weight_decay="${WD_MAP[$backbone]}"
        grad_clip="${GC_MAP[$backbone]}"
        ph1="${PH1_MAP[$backbone]}"
        ph2="${PH2_MAP[$backbone]}"
        val_epochs="${VAL_EPOCHS_MAP[$backbone]}"

        for method in "${METHODS[@]}"; do
            _run_idx=$((_run_idx + 1))
            if [ "$USE_LORA" = "1" ]; then
                exp_name="domainnet_${backbone}_${method}_seed${seed}_lora"
                lora_args="--use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
            else
                exp_name="domainnet_${backbone}_${method}_seed${seed}_fullft"
                lora_args=""
            fi

            # Per-method extras.
            extra_args="--base_optimizer adamw --lr $lr --head_lr $head_lr --weight_decay $weight_decay --grad_clip_val $grad_clip --seed $seed"
            if [ "$CHANNELS_LAST" = "1" ]; then
                extra_args="${extra_args} --channels_last"
            fi
            if [ -n "$MAX_VAL_SAMPLES" ]; then
                extra_args="${extra_args} --max_val_samples $MAX_VAL_SAMPLES"
            fi
            if [ "$method" = "nostalgia" ] || [ "$method" = "gpm" ] || [ "$method" = "ewc_nostalgia" ]; then
                # Hessian cuts for ICLR sweep speed. With LoRA (default) the
                # eigenspace lives in ~0.3-0.6M-param adapter space:
                # - resnet18 adapters: single Lanczos round, k=$K, bs=32.
                # - vit/siglip adapters: single round, k=$K, bs=16.
                # k=24 matches the full-ft setup (method comparability); Q
                # storage ~60 MB/task. Raise K uniformly if smoke shows weak
                # protection.
                if [ "$backbone" = "resnet18" ]; then
                    hess_bs=32
                else
                    # vit / siglip
                    hess_bs=16
                fi
                extra_args="${extra_args} --k $K --nostalgia_accumulation_rounds 1 --nostalgia_max_hessian_batch ${hess_bs} --nostalgia_num_samples 1000"
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
            echo "  lr/head_lr  = $lr / $head_lr"
            echo "  warmup/tot  = $warmup / $total_steps"
            echo "  ph1/ph2     = $ph1 / $ph2"
            echo "  wd/clip     = $weight_decay / $grad_clip"
            echo "  val_every   = $val_epochs epochs  (val_check_interval=$VAL_EVERY)"
            echo "  lora        = $USE_LORA (r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT)"
            echo "  channels_last = $CHANNELS_LAST"
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
                --epochs_phase1 "$ph1" \
                --epochs_phase2 "$ph2" \
                --warmup_steps "$warmup" \
                --total_steps "$total_steps" \
                --batch_size "$bs" \
                --accumulate_grad_batches "$accum" \
                --accelerator "$ACCEL" \
                --devices "$DEVICES" \
                --strategy "$STRATEGY" \
                --precision "$PRECISION" \
                --log_every_n_steps "$LOG_EVERY" \
                --val_check_interval "$VAL_EVERY" \
                --val_every_n_epochs "$val_epochs" \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_name "$exp_name" \
                $lora_args \
                $extra_args

            echo "Finished: $exp_name"
        done
    done
done

echo "====================================================================="
echo "All $_total_runs DomainNet runs completed."
echo "====================================================================="

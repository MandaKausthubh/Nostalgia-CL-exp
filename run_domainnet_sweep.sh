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
#   GPU_MODE=per_gpu bash run_domainnet_sweep.sh      # 4 concurrent single-GPU runs (faster sweep)
#   LR_VIT=1e-4 WARMUP_VIT=800 bash run_domainnet_sweep.sh   # per-backbone hyperparam override
#
# Loop order (per spec): methods → backbones → seeds; tasks fixed per run.

# ----- Hardware / runtime ----------------------------------------------
ACCEL="${ACCEL:-gpu}"
DEVICES="${DEVICES:-4}"
STRATEGY="${STRATEGY:-ddp_find_unused_parameters_true}"
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
LOG_EVERY="${LOG_EVERY:-100}"
VAL_EVERY="${VAL_EVERY:-1.0}"     # validate once per epoch
VAL_EPOCHS="${VAL_EPOCHS:-3}"     # validate every N Phase-2 epochs (--val_every_n_epochs)
WANDB_PROJECT="${WANDB_PROJECT:-domainnet-cl-iclr}"

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
echo "  Devices:    $DEVICES × $ACCEL   (GPU_MODE=${GPU_MODE:-ddp})"
echo "  Total runs: $_total_runs  (each = 6 sequential domains)"
echo "====================================================================="

# ----- Multi-GPU execution mode ------------------------------------------
# GPU_MODE=ddp     : one run at a time, DDP across all GPUs (lower throughput
#                    due to sync overhead + per-rank Hessian duplication, but
#                    Phase-3 Lanczos shards across ranks → faster per run).
# GPU_MODE=per_gpu : DEVICES concurrent single-GPU runs (one per GPU, no DDP
#                    sync). Grad-accum is scaled ×DEVICES to preserve the
#                    effective batch size. Phase-3 runs single-rank (slower
#                    per run) but ~DEVICES× overall sweep throughput.
GPU_MODE="${GPU_MODE:-ddp}"
PER_GPU_WORKERS="${PER_GPU_WORKERS:-4}"   # dataloader workers per run in per_gpu mode
LOG_DIR="${LOG_DIR:-./sweep_logs}"

# ----- Run one (method, backbone, seed) config --------------------------------
# Args: method backbone seed devices strategy accum_mult gpu_id run_idx
_run_one() {
    local method="$1" backbone="$2" seed="$3"
    local devices="$4" strategy="$5" accum_mult="$6" gpu_id="$7" run_idx="$8"

    local image_size="${IMG_SIZE[$backbone]}"
    local bs="${BS_DEFAULT[$backbone]}"
    local accum=$(( ${ACCUM_DEFAULT[$backbone]} * accum_mult ))
    local lr="${LR_MAP[$backbone]}"
    local head_lr="${HEAD_LR_MAP[$backbone]}"
    local warmup="${WARMUP_MAP[$backbone]}"
    local total_steps="${TOTAL_STEPS_MAP[$backbone]}"
    local weight_decay="${WD_MAP[$backbone]}"
    local grad_clip="${GC_MAP[$backbone]}"
    local ph1="${PH1_MAP[$backbone]}"
    local ph2="${PH2_MAP[$backbone]}"
    local val_epochs="${VAL_EPOCHS_MAP[$backbone]}"
    local workers="$NUM_WORKERS"
    [ -n "$gpu_id" ] && workers="$PER_GPU_WORKERS"

    local exp_name="domainnet_${backbone}_${method}_seed${seed}_fullft"

    # Per-method extras.
    local extra_args="--base_optimizer adamw --lr $lr --head_lr $head_lr --weight_decay $weight_decay --grad_clip_val $grad_clip --seed $seed"
    if [ "$method" = "nostalgia" ] || [ "$method" = "gpm" ] || [ "$method" = "ewc_nostalgia" ]; then
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

    local strategy_args=()
    if [ -n "$strategy" ]; then
        strategy_args=(--strategy "$strategy")
    fi

    echo ""
    echo "---------------------------------------------------------------------"
    echo "[$run_idx/$_total_runs]  backbone=$backbone  method=$method  seed=$seed  gpu=${gpu_id:-all}"
    echo "  exp_name    = $exp_name"
    echo "  image_size  = $image_size"
    echo "  bs/accum    = $bs / $accum  (eff=${bs}×${accum}×${devices}=$((bs * accum * devices)))"
    echo "  lr/head_lr  = $lr / $head_lr"
    echo "  warmup/tot  = $warmup / $total_steps"
    echo "  ph1/ph2     = $ph1 / $ph2"
    echo "  wd/clip     = $weight_decay / $grad_clip"
    echo "  val_every   = $val_epochs epochs  (val_check_interval=$VAL_EVERY)"
    echo "---------------------------------------------------------------------"

    if [ -n "$gpu_id" ]; then
        # Single-GPU run pinned to one device; log to per-run file.
        CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
            --backbone "$backbone" \
            --image_size "$image_size" \
            --tasks "${TASKS[@]}" \
            --method "$method" \
            --data_root "$DATA_ROOT_DN" \
            --data_root_domainnet "$DATA_ROOT_DN" \
            --max_length 32 \
            --num_workers "$workers" \
            --pin_memory \
            --epochs_phase1 "$ph1" \
            --epochs_phase2 "$ph2" \
            --warmup_steps "$warmup" \
            --total_steps "$total_steps" \
            --batch_size "$bs" \
            --accumulate_grad_batches "$accum" \
            --accelerator "$ACCEL" \
            --devices 1 \
            --precision "$PRECISION" \
            --log_every_n_steps "$LOG_EVERY" \
            --val_check_interval "$VAL_EVERY" \
            --val_every_n_epochs "$val_epochs" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_name "$exp_name" \
            $extra_args \
            > "$LOG_DIR/${exp_name}.log" 2>&1
    else
        python train.py \
            --backbone "$backbone" \
            --image_size "$image_size" \
            --tasks "${TASKS[@]}" \
            --method "$method" \
            --data_root "$DATA_ROOT_DN" \
            --data_root_domainnet "$DATA_ROOT_DN" \
            --max_length 32 \
            --num_workers "$workers" \
            --pin_memory \
            --epochs_phase1 "$ph1" \
            --epochs_phase2 "$ph2" \
            --warmup_steps "$warmup" \
            --total_steps "$total_steps" \
            --batch_size "$bs" \
            --accumulate_grad_batches "$accum" \
            --accelerator "$ACCEL" \
            --devices "$devices" \
            "${strategy_args[@]}" \
            --precision "$PRECISION" \
            --log_every_n_steps "$LOG_EVERY" \
            --val_check_interval "$VAL_EVERY" \
            --val_every_n_epochs "$val_epochs" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_name "$exp_name" \
            $extra_args
    fi
    echo "Finished: $exp_name"
}

# ----- Sweep ------------------------------------------------------------
mkdir -p "$LOG_DIR"

# Flatten job list.
JOBS=()
for method in "${METHODS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        for seed in $SEEDS; do
            JOBS+=("${method}|${backbone}|${seed}")
        done
    done
done

if [ "$GPU_MODE" = "per_gpu" ]; then
    echo "GPU_MODE=per_gpu: launching $DEVICES concurrent single-GPU runs per wave."
    echo "Logs: $LOG_DIR/<exp_name>.log"
    _job_idx=0
    _run_idx=0
    while [ "$_job_idx" -lt "${#JOBS[@]}" ]; do
        _pids=()
        _names=()
        for (( g=0; g<DEVICES && _job_idx<${#JOBS[@]}; g++, _job_idx++ )); do
            _run_idx=$((_run_idx + 1))
            IFS='|' read -r _m _b _s <<< "${JOBS[$_job_idx]}"
            _exp="domainnet_${_b}_${_m}_seed${_s}_fullft"
            _run_one "$_m" "$_b" "$_s" 1 "" "$DEVICES" "$g" "$_run_idx" &
            _pids+=($!)
            _names+=("$_exp")
        done
        # Wait for wave; report failures without aborting the sweep.
        for i in "${!_pids[@]}"; do
            if wait "${_pids[$i]}"; then
                echo "[wave done] ${_names[$i]}  OK"
            else
                echo "[wave done] ${_names[$i]}  FAILED (see $LOG_DIR/${_names[$i]}.log)"
            fi
        done
    done
else
    _run_idx=0
    for job in "${JOBS[@]}"; do
        _run_idx=$((_run_idx + 1))
        IFS='|' read -r _m _b _s <<< "$job"
        _run_one "$_m" "$_b" "$_s" "$DEVICES" "$STRATEGY" 1 "" "$_run_idx"
    done
fi

echo "====================================================================="
echo "All $_total_runs DomainNet runs completed."
echo "====================================================================="

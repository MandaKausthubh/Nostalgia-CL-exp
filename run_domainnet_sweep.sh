#!/usr/bin/env bash
set -euo pipefail

# DomainNet continual-learning benchmark sweep for ICLR.
# All CL methods × all image backbones × N seeds, each over the 6 sequential
# DomainNet domains. Target hardware: 4× A100 80GB on RunPod.
#
# LoRA is ON by default (USE_LORA=0 for full-ft fallback): adapter-only
# training keeps the Nostalgia Q matrix in the ~0.3-0.6M-param adapter space
# (~60 MB/task) instead of 85.8M-param space (5.5 GB/task, cuSOLVER overflow).
#
# Runs are dispatched one-per-GPU in parallel; GPUS="0 1 2 3" to pin, default =
# every visible GPU. On TPU (ACCEL=tpu) the 8 v5e chips are driven by a SINGLE
# process (--devices 8 --strategy xla, bf16-true) and runs go strictly serial.
# Defaults give a 7 × 3 × 3 = 63-run main table. Override:
#   SEEDS="0" BACKBONES="resnet18 vit" METHODS="nostalgia ewc_nostalgia" \
#       bash run_domainnet_sweep.sh
#   DATA_ROOT_DN=/home/t-kmanda/domainnet bash run_domainnet_sweep.sh
#   GPUS="0 1" bash run_domainnet_sweep.sh            # limit parallelism
#   BS_SIGLIP=32 PH2=10 bash run_domainnet_sweep.sh   # per-backbone / per-budget knobs
#   USE_LORA=0 K=32 bash run_domainnet_sweep.sh       # full-ft fallback / larger null-space
#   ACCEL=tpu DEVICES=8 bash run_domainnet_sweep.sh   # TPU v5e-8 (one process, all chips)
#
# Loop order: seeds → methods → backbones; tasks fixed per run.

# ----- Hardware / runtime ----------------------------------------------
ACCEL="${ACCEL:-gpu}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ----- Accelerator-specific defaults ------------------------------------
# TPU v5e-8: ONE process drives all 8 chips (SPMD data-parallel). Launching a
# second train.py would contend for the same chips, so TPU mode bypasses the
# per-GPU pool and runs a single job at a time over every chip.
IS_TPU=0
[ "$ACCEL" = "tpu" ] && IS_TPU=1

if [ "$IS_TPU" = "1" ]; then
    DEVICES="${DEVICES:-8}"
    STRATEGY="${STRATEGY:-xla}"
    # XLA wants bf16-true; bf16-mixed (the GPU default) is unreliable there.
    PRECISION="${PRECISION:-bf16-true}"
else
    DEVICES="${DEVICES:-1}"
    STRATEGY="${STRATEGY:-auto}"
    PRECISION="${PRECISION:-bf16-mixed}"
fi

# ----- Parallel dispatch -------------------------------------------------
# GPU: one single-GPU train.py process per GPU, run concurrently. GPUs are
# assigned round-robin; a new run only starts when its slot's previous run has
# finished. TPU: a single slot (N_GPU=1) over all chips.
# Ordering (seed -> method -> backbone) means seed 0's Nostalgia/EWC+Nostalgia
# runs occupy the first wave, so the headline numbers land earliest.
# GPUS="0 1 2 3" to pin; default = every visible GPU.
if [ "$IS_TPU" = "1" ]; then
    GPU_LIST=(0)
elif [ -n "${GPUS:-}" ]; then
    GPU_LIST=()
    for _g in $GPUS; do GPU_LIST+=("$_g"); done
else
    GPU_LIST=()
    if command -v nvidia-smi >/dev/null 2>&1; then
        while read -r _g; do [ -n "$_g" ] && GPU_LIST+=("$_g"); done \
            < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    fi
fi
[ ${#GPU_LIST[@]} -eq 0 ] && GPU_LIST=(0)
N_GPU=${#GPU_LIST[@]}

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

# ----- Crash-resume ------------------------------------------------------
# Each run writes a resumable bundle per task boundary; a crash loses at most
# one domain's Phase-2. RESUME=auto (no prompt — right for unattended sweeps),
# never (always restart), prompt (ask when interactive).
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/checkpoints}"
RESUME="${RESUME:-prompt}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_HUB_NAMESPACE="${HF_HUB_NAMESPACE:-}"
HF_HUB_PRIVATE="${HF_HUB_PRIVATE:-0}"
mkdir -p "$CHECKPOINT_DIR"

HUB_ARGS="--checkpoint_dir $CHECKPOINT_DIR --resume $RESUME"
if [ "$PUSH_TO_HUB" = "1" ]; then
    HUB_ARGS="$HUB_ARGS --push_to_hub"
    [ -n "$HF_HUB_NAMESPACE" ] && HUB_ARGS="$HUB_ARGS --hf_hub_namespace $HF_HUB_NAMESPACE"
    [ "$HF_HUB_PRIVATE" = "1" ] && HUB_ARGS="$HUB_ARGS --hf_hub_private"
fi

# ----- LoRA (always ON — no full finetuning) -----------------------------
# r=8 (alpha=2r=16) halves the adapter Hessian dim vs r=16 (~0.3M vs ~0.6M for
# resnet18), so Lanczos + Q build are ~2x cheaper. Raise LORA_R only if a
# smoke shows the reduced capacity caps Phase-2 accuracy.
USE_LORA="${USE_LORA:-1}"         # keep 1; 0 = full-ft fallback (discouraged)
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ----- Memory format --------------------------------------------------------
# channels_last speeds up conv backbones (resnet10/18) on Ampere+; ignored for
# vit/siglip. Set CHANNELS_LAST=0 to fall back to contiguous NCHW.
# XLA has no channels_last path (the code guards it on torch.cuda), so force off.
CHANNELS_LAST="${CHANNELS_LAST:-1}"
[ "$IS_TPU" = "1" ] && CHANNELS_LAST=0

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
# DomainNet root = dir holding <domain>/ class folders + <domain>_{train,test}.txt.
# Auto-detect over the known pod layouts; explicit DATA_ROOT_DN wins.
if [ -z "${DATA_ROOT_DN:-}" ]; then
    for _cand in "/home/t-kmanda/domainnet" "/home/t-kmanda" \
                 "$HOME/domainnet" "$HOME"; do
        if [ -d "$_cand/clipart" ] || [ -d "$_cand/domainnet/clipart" ]; then
            DATA_ROOT_DN="$_cand"
            break
        fi
    done
fi
DATA_ROOT_DN="${DATA_ROOT_DN:-$HOME/domainnet}"

# Domain order = the sequential CL task order. Override with TASKS="..." for a
# shortened smoke run (e.g. TASKS="domainnet_clipart domainnet_real").
if [ -n "${TASKS:-}" ]; then
    _tasks=()
    for _t in $TASKS; do _tasks+=("$_t"); done
    TASKS=("${_tasks[@]}")
else
    TASKS=(
        "domainnet_clipart"
        "domainnet_infograph"
        "domainnet_painting"
        "domainnet_quickdraw"
        "domainnet_real"
        "domainnet_sketch"
    )
fi

# ----- Method / backbone / seed axes ------------------------------------
# Order = dispatch priority. Paper's headline pair (Nostalgia, EWC+Nostalgia)
# runs first so seed 0 produces the main table before the baselines land.
ALL_METHODS=(
    "nostalgia"
    "ewc_nostalgia"
    "gpm"
    "naive_adam"
    "ewc"
    "agem"
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
if [ "$IS_TPU" = "1" ]; then
    echo "  TPUs:       $DEVICES chips, one process (strategy=$STRATEGY, precision=$PRECISION)"
else
    echo "  GPUs:       ${GPU_LIST[*]}  (${N_GPU} parallel single-GPU runs)"
fi
if [ "$USE_LORA" = "1" ]; then
    echo "  LoRA:       ON  (r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT)"
else
    echo "  LoRA:       OFF (full finetuning)"
fi
echo "  K:          $K"
echo "  Data root:  $DATA_ROOT_DN"
echo "  Resume:     $RESUME  (checkpoint_dir=$CHECKPOINT_DIR)"
echo "  Push hub:   $PUSH_TO_HUB  (namespace=${HF_HUB_NAMESPACE:-<unset>})"
echo "  Total runs: $_total_runs  (each = 6 sequential domains)"
echo "====================================================================="

# ----- Sweep ------------------------------------------------------------
# Loop order: seed → method → backbone. Seed 0 + Nostalgia/EWC+Nostalgia runs
# are dispatched first (first wave on the GPUs); seed 0 finishes before seed 1
# starts, so a complete single-seed table exists as early as possible.
# Runs are launched one-per-GPU in the background and slot-blocked round-robin.
declare -a SLOT_PID=()
for ((_s = 0; _s < N_GPU; _s++)); do SLOT_PID[$_s]=""; done

_run_idx=0
for seed in $SEEDS; do
    for method in "${METHODS[@]}"; do
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
            if [ -n "${PHASE1_CACHE_DIR:-}" ]; then
                extra_args="${extra_args} --phase1_cache_dir $PHASE1_CACHE_DIR"
            fi
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

            # Round-robin slot: block until this slot's previous run finishes.
            # TPU: N_GPU=1 -> strictly serial, one process owns all chips.
            _slot=$(( (_run_idx - 1) % N_GPU ))
            if [ "$IS_TPU" = "1" ]; then
                _slot_label="tpu x$DEVICES"
                _launch=(python)
                _devices_arg="$DEVICES"
            else
                _slot_label="gpu ${GPU_LIST[$_slot]}"
                # Must be a literal env-assignment prefix; a value produced by
                # expansion is parsed as the command name, not an assignment.
                _launch=(env "CUDA_VISIBLE_DEVICES=${GPU_LIST[$_slot]}" python)
                _devices_arg="1"
            fi
            if [ -n "${SLOT_PID[$_slot]}" ]; then
                wait "${SLOT_PID[$_slot]}" \
                    || echo "[warn] previous run on $_slot_label exited non-zero"
            fi

            echo ""
            echo "---------------------------------------------------------------------"
            echo "[$_run_idx/$_total_runs]  $_slot_label  backbone=$backbone  method=$method  seed=$seed"
            echo "  exp_name    = $exp_name"
            echo "  image_size  = $image_size"
            echo "  bs/accum    = $bs / $accum  (eff=${bs}×${accum}=$((bs * accum)) / GPU)"
            echo "  lr/head_lr  = $lr / $head_lr"
            echo "  warmup/tot  = $warmup / $total_steps"
            echo "  ph1/ph2     = $ph1 / $ph2"
            echo "  wd/clip     = $weight_decay / $grad_clip"
            echo "  val_every   = $val_epochs epochs  (val_check_interval=$VAL_EVERY)"
            echo "  lora        = $USE_LORA (r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT)"
            echo "  channels_last = $CHANNELS_LAST"
            echo "  tasks       = ${TASKS[*]}"
            echo "---------------------------------------------------------------------"

            "${_launch[@]}" train.py \
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
                --devices "$_devices_arg" \
                --strategy "$STRATEGY" \
                --precision "$PRECISION" \
                --log_every_n_steps "$LOG_EVERY" \
                --val_check_interval "$VAL_EVERY" \
                --val_every_n_epochs "$val_epochs" \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_name "$exp_name" \
                $HUB_ARGS \
                $lora_args \
                $extra_args &

            SLOT_PID[$_slot]=$!
            echo "Launched: $exp_name  (pid=${SLOT_PID[$_slot]}, $_slot_label)"
        done
    done
done

# Drain remaining in-flight runs.
for ((_s = 0; _s < N_GPU; _s++)); do
    if [ -n "${SLOT_PID[$_s]}" ]; then
        wait "${SLOT_PID[$_s]}" || echo "[warn] run on slot $_s exited non-zero"
    fi
done

echo "====================================================================="
echo "All $_total_runs DomainNet runs completed."
echo "====================================================================="

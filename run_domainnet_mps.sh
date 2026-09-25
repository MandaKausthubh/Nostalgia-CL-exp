#!/usr/bin/env bash
# DomainNet continual-learning sweep on local MPS (bash 3.2-safe).
#
# `run_domainnet_sweep.sh` is the production (4x A100 / TPU) runner, but it uses
# `declare -A` and cannot run under macOS system bash 3.2. This is the local
# equivalent: one method at a time, single MPS device, sequential over the 6
# domains.
#
# Config mirrors run_domainnet_sweep.sh where it matters (ViT backbone, LoRA r8,
# LR 3e-4 / head 5e-4, adamw, grad_clip 1.0, k=24, per-class caps) with two
# local-compute adjustments:
#   - val/class=10 (3.4k samples -> SE ~0.009) and --val_every_n_epochs 5,
#     because Lightning validates ALL started tasks on every val event, so val
#     cost is O(T^2) in domains. At val/class=50 + every-epoch, val alone is
#     ~8.8 h vs ~0.4 h with these settings.
#   - total_steps sized to the largest domain's Phase-2 step count
#     (quickdraw 34,500 imgs / bs16 = 2,156 steps/epoch x 5 epochs = 10,780).
#     Under-sizing decays LR to 0 mid-task and the task silently never trains.
#
# NO CHECKPOINTING exists in this repo: an interrupted method restarts from
# scratch. Methods run sequentially and each takes hours.
#
# Env overrides:
#   METHODS="nostalgia naive_adam ..."   method list (default: all 7)
#   SEED=0
#   PH1=3 PH2=5
#   DATA_ROOT_DN=$HOME/data/domainnet
#   DRY_RUN=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-Nostal}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/domainnet_mps}"
DATA_ROOT_DN="${DATA_ROOT_DN:-$HOME/data/domainnet}"

SEED="${SEED:-0}"
PH1="${PH1:-3}"
PH2="${PH2:-5}"
WARMUP="${WARMUP:-600}"
TOTAL_STEPS="${TOTAL_STEPS:-11000}"
BS="${BS:-16}"
LR="${LR:-3e-4}"
HEAD_LR="${HEAD_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Per-class caps (class-balanced; the DomainNet lever for local compute).
MAX_TRAIN_PER_CLASS="${MAX_TRAIN_PER_CLASS:-100}"
MAX_VAL_PER_CLASS="${MAX_VAL_PER_CLASS:-10}"
VAL_EPOCHS="${VAL_EPOCHS:-5}"

# LoRA r=8 kept identical to run_domainnet_sweep.sh so numbers stay comparable.
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

K="${K:-24}"
# rounds=2 / samples=2000 so k=24 is actually realized (1 round / 1000 samples
# truncated k=24 to k_eff~14 on cifar; Lanczos rank is capped by the HVP budget).
HESS_ROUNDS="${HESS_ROUNDS:-2}"
HESS_BS="${HESS_BS:-16}"
HESS_SAMPLES="${HESS_SAMPLES:-2000}"

WANDB_PROJECT="${WANDB_PROJECT:-domainnet-cl-mps}"
export WANDB_MODE="${WANDB_MODE:-offline}"

TASKS="domainnet_clipart domainnet_infograph domainnet_painting domainnet_quickdraw domainnet_real domainnet_sketch"
ALL_METHODS="nostalgia ewc_nostalgia gpm naive_adam ewc agem sdft"
METHODS="${METHODS:-$ALL_METHODS}"

mkdir -p "$LOG_DIR"

# --- sanity ---------------------------------------------------------------
for d in clipart infograph painting quickdraw real sketch; do
    if [ ! -d "$DATA_ROOT_DN/$d" ] || [ ! -f "$DATA_ROOT_DN/${d}_train.txt" ]; then
        echo "[FATAL] missing $DATA_ROOT_DN/$d (or ${d}_train.txt)"; exit 1
    fi
done
echo "[ok] DomainNet layout verified at $DATA_ROOT_DN"

COMMON_ARGS=(
    --backbone vit --image_size 224
    --tasks $TASKS
    --data_root "$REPO_DIR/data"
    --data_root_domainnet "$DATA_ROOT_DN"
    --max_length 32
    --epochs_phase1 "$PH1" --epochs_phase2 "$PH2"
    --warmup_steps "$WARMUP" --total_steps "$TOTAL_STEPS"
    --batch_size "$BS" --accumulate_grad_batches 1
    --base_optimizer adamw --lr "$LR" --head_lr "$HEAD_LR"
    --weight_decay "$WEIGHT_DECAY" --grad_clip_val "$GRAD_CLIP"
    --max_train_per_class "$MAX_TRAIN_PER_CLASS"
    --max_val_per_class "$MAX_VAL_PER_CLASS"
    --accelerator mps --devices 1 --strategy auto --precision 32-true
    --log_every_n_steps 20 --val_check_interval 1.0
    --val_every_n_epochs "$VAL_EPOCHS"
    --use_lora --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT"
    --wandb_project "$WANDB_PROJECT"
)

echo "====================================================================="
echo "DomainNet CL sweep (MPS, single seed)"
echo "  methods       : $METHODS"
echo "  tasks         : $TASKS"
echo "  seed          : $SEED"
echo "  budget        : ph1=$PH1 ph2=$PH2 warmup=$WARMUP total_steps=$TOTAL_STEPS bs=$BS"
echo "  caps          : train/class=$MAX_TRAIN_PER_CLASS val/class=$MAX_VAL_PER_CLASS val_every=$VAL_EPOCHS"
echo "  LoRA          : r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  k / hessian   : k=$K rounds=$HESS_ROUNDS bs=$HESS_BS samples=$HESS_SAMPLES"
echo "  logs          : $LOG_DIR"
echo "  NOTE: no checkpointing - an interrupted method restarts from scratch."
echo "====================================================================="

CONDA_RUN=(conda run --no-capture-output -n "$CONDA_ENV")
source "$(conda info --base)/etc/profile.d/conda.sh"

for method in $METHODS; do
    name="s${SEED}_domainnet_vit_${method}"
    log="$LOG_DIR/${name}.log"
    extra=""
    case "$method" in
        nostalgia|gpm|ewc_nostalgia)
            extra="--k $K --nostalgia_accumulation_rounds $HESS_ROUNDS --nostalgia_max_hessian_batch $HESS_BS --nostalgia_num_samples $HESS_SAMPLES"
            ;;
    esac
    case "$method" in
        ewc|ewc_nostalgia) extra="$extra --ewc_lambda 400.0" ;;
    esac
    case "$method" in
        agem) extra="$extra --agem_mem_size 2000" ;;
    esac
    case "$method" in
        sdft) extra="$extra --sdft_lambda_distillation 1.0 --sdft_temperature 2.0" ;;
    esac

    echo ""
    echo ">>> $name"
    echo "    log: $log"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "    [dry-run] ${CONDA_RUN[*]} python train.py ${COMMON_ARGS[*]} --method $method --wandb_name $name --seed $SEED $extra"
        continue
    fi
    WANDB_DIR="${WANDB_DIR:-$HOME/wandb_log}" \
        "${CONDA_RUN[@]}" python train.py \
        "${COMMON_ARGS[@]}" \
        --method "$method" \
        --wandb_name "$name" \
        --seed "$SEED" \
        $extra > "$log" 2>&1
    echo "    done: $name"
done

echo ""
echo "=== DomainNet sweep complete ==="
echo "logs in $LOG_DIR"

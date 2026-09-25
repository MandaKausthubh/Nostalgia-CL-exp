#!/usr/bin/env bash
# Rank (k) ablation for Nostalgia on local MPS.
#
# Vision setting from method.tex (ViT backbone, sequential image classification),
# with the class-incremental sequence cifar10 -> mnist -> cifar100. LoRA is ON
# (method.tex: "the projection is applied within a LoRA fine-tuning setup"), so
# the Hessian eigenspace lives in the ~295k-param ViT adapter space, not the
# 85.8M-param full model.
#
# BUDGET NOTE (why these numbers): the per-task LR schedule is a warmup+decay
# over `--total_steps`, and the scheduler is reset at every task/phase transition.
# A first pass used total_steps=500 while cifar100 alone runs ~300 steps/epoch,
# so LR hit 0 mid-task -> tasks never fit (mnist train acc 0.71) -> nothing to
# forget -> the k sweep measured noise. total_steps below is sized to the
# largest task's Phase-2 step count (~10k imgs / bs16 x 6 epochs).
#
# Each k value is a separate train.py run (same everything else). A naive_adam
# run is included as the no-projection control (k -> 0), which anchors the
# forgetting/plasticity trade-off.
#
# Env overrides:
#   KS="8 24 64"          rank values
#   SEEDS="0"             seed list
#   WITH_NAIVE=0          drop the naive_adam control
#   TASKS="..."           task sequence
#   CONDA_ENV=Nostal      conda env holding torch/lightning/peft
#   DRY_RUN=1             print commands only

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-Nostal}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/k_ablation_strengthened}"

KS="${KS:-8 24 64}"
SEEDS="${SEEDS:-0}"
WITH_NAIVE="${WITH_NAIVE:-1}"
TASKS="${TASKS:-cifar10 mnist cifar100}"

# ---- Model / LoRA (method.tex vision config; ViT-B/32 unavailable locally) ----
BACKBONE="${BACKBONE:-vit}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
USE_LORA=1
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ---- Budget (sized so the LR schedule actually completes) ----
PH1="${PH1:-1}"
PH2="${PH2:-6}"
WARMUP="${WARMUP:-200}"
TOTAL_STEPS="${TOTAL_STEPS:-4000}"
BS="${BS:-16}"
ACCUM="${ACCUM:-1}"
LR="${LR:-3e-4}"
HEAD_LR="${HEAD_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Stratified caps. cifar100 gets the same 100/class -> 10k imgs; the schedule
# above is sized to that (the largest task).
MAX_TRAIN_PER_CLASS="${MAX_TRAIN_PER_CLASS:-100}"
MAX_VAL_PER_CLASS="${MAX_VAL_PER_CLASS:-50}"
# Validate every 3rd Phase-2 epoch (task-final epoch is always validated).
VAL_EPOCHS="${VAL_EPOCHS:-3}"

# ---- Hessian budget ----
# lanczos runs k HVP iterations per round, so rank is capped by k AND the HVP
# sample budget. 2 rounds x 2000 samples x bs16 (~125 batches/round) supports
# k=64 without early breakdown (1 round / 1000 samples truncated k=64 to ~k_eff 40).
HESS_ROUNDS="${HESS_ROUNDS:-2}"
HESS_BS="${HESS_BS:-16}"
HESS_SAMPLES="${HESS_SAMPLES:-2000}"

WANDB_PROJECT="${WANDB_PROJECT:-cl-k-ablation}"
# W&B is optional here: default to offline so no API key is needed.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$LOG_DIR"

LORA_ARGS="--use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
COMMON_ARGS=(
    --backbone "$BACKBONE"
    --image_size "$IMAGE_SIZE"
    --tasks $TASKS
    --data_root "$REPO_DIR/data"
    --max_length 32
    --epochs_phase1 "$PH1"
    --epochs_phase2 "$PH2"
    --warmup_steps "$WARMUP"
    --total_steps "$TOTAL_STEPS"
    --batch_size "$BS"
    --accumulate_grad_batches "$ACCUM"
    --base_optimizer adamw
    --lr "$LR"
    --head_lr "$HEAD_LR"
    --weight_decay "$WEIGHT_DECAY"
    --grad_clip_val "$GRAD_CLIP"
    --max_train_per_class "$MAX_TRAIN_PER_CLASS"
    --max_val_per_class "$MAX_VAL_PER_CLASS"
    --accelerator mps
    --devices 1
    --strategy auto
    --precision 32-true
    --log_every_n_steps 5
    --val_check_interval 1.0
    --val_every_n_epochs "$VAL_EPOCHS"
    --wandb_project "$WANDB_PROJECT"
)

echo "====================================================================="
echo "k-ablation (MPS, strengthened)"
echo "  backbone/LoRA : $BACKBONE  image_size=$IMAGE_SIZE  r=$LORA_R alpha=$LORA_ALPHA"
echo "  tasks         : $TASKS"
echo "  k values      : $KS   seeds: $SEEDS   naive control: $WITH_NAIVE"
echo "  budget        : ph1=$PH1 ph2=$PH2 warmup=$WARMUP total_steps=$TOTAL_STEPS bs=$BS"
echo "  caps          : train/class=$MAX_TRAIN_PER_CLASS val/class=$MAX_VAL_PER_CLASS val_every=$VAL_EPOCHS epochs"
echo "  hessian       : rounds=$HESS_ROUNDS bs=$HESS_BS samples=$HESS_SAMPLES"
echo "  logs          : $LOG_DIR"
echo "====================================================================="

run_one() {
    local method="$1"; local k="$2"; local name="$3"; shift 3
    local log="$LOG_DIR/${name}.log"
    echo ""
    echo ">>> $name   (method=$method k=${k:-n/a})"
    echo "    log: $log"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "    [dry-run] ${CONDA_RUN[*]} python train.py ${COMMON_ARGS[*]} $LORA_ARGS --method $method --wandb_name $name $*"
        return 0
    fi
    # WANDB_DIR keeps offline run dirs out of the repo.
    WANDB_DIR="${WANDB_DIR:-$HOME/wandb_log}" \
        "${CONDA_RUN[@]}" python train.py \
        "${COMMON_ARGS[@]}" \
        $LORA_ARGS \
        --method "$method" \
        --wandb_name "$name" \
        "$@" > "$log" 2>&1
    echo "    done: $name  (exit $?)"
}

# conda run with live output; --no-capture-output so the log streams.
CONDA_RUN=(conda run --no-capture-output -n "$CONDA_ENV")
source "$(conda info --base)/etc/profile.d/conda.sh"

for seed in $SEEDS; do
    for k in $KS; do
        run_one nostalgia "$k" "s${seed}_vit_lora_k${k}" \
            --seed "$seed" \
            --k "$k" \
            --nostalgia_accumulation_rounds "$HESS_ROUNDS" \
            --nostalgia_max_hessian_batch "$HESS_BS" \
            --nostalgia_num_samples "$HESS_SAMPLES"
    done
    if [ "$WITH_NAIVE" = "1" ]; then
        run_one naive_adam "" "s${seed}_vit_lora_naive" \
            --seed "$seed" \
            --nostalgia_accumulation_rounds "$HESS_ROUNDS" \
            --nostalgia_max_hessian_batch "$HESS_BS" \
            --nostalgia_num_samples "$HESS_SAMPLES"
    fi
done

echo ""
echo "=== k-ablation complete ==="
echo "logs in $LOG_DIR"

#!/usr/bin/env bash
# Smoke-test LLM continual-learning sweep.
# Tiny data, tiny budget, all 6 methods. MPS-friendly.

set -euo pipefail

TASKS="sst2 agnews"
MODEL="gpt2"
EPOCHS_PHASE1=1
EPOCHS_PHASE2=1
BATCH_SIZE=8
MAX_LENGTH=128
MAX_TRAIN=100
MAX_VAL=50

TOTAL_STEPS=30
WARMUP_STEPS=5

USE_LORA="--use_lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

K=4
NOSTALGIA_NUM_SAMPLES=40
NOSTALGIA_ACCUMULATION_ROUNDS=1
NOSTALGIA_MAX_HESSIAN_BATCH=4
NOSTALGIA_ALPHA=1.0

EWC_LAMBDA=400
EWC_NOSTALGIA_LAMBDA=400
AGEM_MEM_SIZE=100
GPM_THRESHOLD=0.925

LR=5e-4
HEAD_LR=1e-3
WEIGHT_DECAY=0.01
BASE_OPTIMIZER="adamw"

PROJECT="llm-cl-smoke"
LOG_EVERY=10

for M in nostalgia naive_adam ewc gpm agem ewc_nostalgia; do
  echo "=== smoke running $M ==="
  python train.py \
    --method "$M" \
    --model_name "$MODEL" \
    $USE_LORA \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --tasks $TASKS \
    --max_length "$MAX_LENGTH" \
    --epochs_phase1 "$EPOCHS_PHASE1" \
    --epochs_phase2 "$EPOCHS_PHASE2" \
    --max_train_samples "$MAX_TRAIN" \
    --max_val_samples "$MAX_VAL" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --head_lr "$HEAD_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --base_optimizer "$BASE_OPTIMIZER" \
    --accelerator mps \
    --devices 1 \
    --total_steps "$TOTAL_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --k "$K" \
    --nostalgia_num_samples "$NOSTALGIA_NUM_SAMPLES" \
    --nostalgia_accumulation_rounds "$NOSTALGIA_ACCUMULATION_ROUNDS" \
    --nostalgia_max_hessian_batch "$NOSTALGIA_MAX_HESSIAN_BATCH" \
    --nostalgia_alpha "$NOSTALGIA_ALPHA" \
    --ewc_lambda "$EWC_LAMBDA" \
    --ewc_nostalgia_lambda "$EWC_NOSTALGIA_LAMBDA" \
    --agem_mem_size "$AGEM_MEM_SIZE" \
    --gpm_threshold "$GPM_THRESHOLD" \
    --log_every_n_steps "$LOG_EVERY" \
    --val_check_interval 1.0 \
    --wandb_project "$PROJECT" \
    --wandb_name "$M-smoke"
done

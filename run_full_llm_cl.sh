#!/usr/bin/env bash
# Full-length LLM continual-learning sweep over all methods.
# GPT-2 + LoRA on SST-2 / AG-news / Trec / DB-pedia.
# Intended for TPU/GPU; works on MPS with small max_train_samples.

set -euo pipefail

TASKS="sst2 agnews trec dbpedia"
MODEL="gpt2"
EPOCHS_PHASE1=2
EPOCHS_PHASE2=3
BATCH_SIZE=16
MAX_LENGTH=256
MAX_TRAIN=10000
MAX_VAL=1000

# Per-task step budget: ~625 steps/epoch for 10k samples @ bs=16.
# 3 epochs => ~1.9k steps; 4 tasks => ~7.5k steps total.
TOTAL_STEPS=2000
WARMUP_STEPS=200

# LoRA: small adapter rank makes Nostalgia Q memory cheap.
USE_LORA="--use_lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Nostalgia / GPM settings
K=64
NOSTALGIA_NUM_SAMPLES=1000
NOSTALGIA_ACCUMULATION_ROUNDS=5
NOSTALGIA_MAX_HESSIAN_BATCH=8
NOSTALGIA_ALPHA=1.0

# Baselines
EWC_LAMBDA=400
EWC_NOSTALGIA_LAMBDA=400
AGEM_MEM_SIZE=500
GPM_THRESHOLD=0.925

# Optimizer
LR=5e-4
HEAD_LR=1e-3
WEIGHT_DECAY=0.01
BASE_OPTIMIZER="adamw"

PROJECT="llm-cl-gpt2-lora"
LOG_EVERY=50

for M in nostalgia naive_adam ewc gpm agem ewc_nostalgia; do
  echo "=== running $M ==="
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
    --wandb_name "$M-full"
done

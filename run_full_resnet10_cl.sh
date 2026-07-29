#!/usr/bin/env bash
# Full-length ResNet-10 continual-learning sweep over all 5 methods.
# Uses full CIFAR-10 / MNIST / CIFAR-100 train sets and runs real epochs.

set -euo pipefail

TASKS="cifar10 mnist cifar100"
EPOCHS_PHASE1=10
EPOCHS_PHASE2=10
BATCH_SIZE=128
MAX_TRAIN=50000
MAX_VAL=10000

# Per-task step budget: ~391 steps/epoch for CIFAR @ bs=128.
# 10 epochs => ~3.9k steps; MNIST is similar. 5k covers all three tasks.
TOTAL_STEPS=5000
WARMUP_STEPS=500

# Nostalgia / GPM settings for a ~5M param ResNet-10.
# Effective rank should be much larger than the smoke-test k=4.
K=128
NOSTALGIA_NUM_SAMPLES=5000
NOSTALGIA_ACCUMULATION_ROUNDS=5
NOSTALGIA_MAX_HESSIAN_BATCH=32
NOSTALGIA_ALPHA=0.5

# Baselines
EWC_LAMBDA=400
EWC_NOSTALGIA_LAMBDA=400
AGEM_MEM_SIZE=500
GPM_THRESHOLD=0.95

# Optimizer
LR=0.001
HEAD_LR=0.01
WEIGHT_DECAY=0.0001

PROJECT="cnn-cl-resnet10-full"
LOG_EVERY=50

for M in ewc_nostalgia; do
  echo "=== running $M ==="
  python testing/train_cnn.py \
    --method "$M" \
    --tasks $TASKS \
    --epochs_phase1 "$EPOCHS_PHASE1" \
    --epochs_phase2 "$EPOCHS_PHASE2" \
    --max_train_samples "$MAX_TRAIN" \
    --max_val_samples "$MAX_VAL" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --head_lr "$HEAD_LR" \
    --weight_decay "$WEIGHT_DECAY" \
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

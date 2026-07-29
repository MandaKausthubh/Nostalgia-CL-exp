set -euo pipefail
METHODS=(nostalgia naive_adam ewc gpm agem)
for M in "${METHODS[@]}"; do
    echo "=== running $M ==="
    python testing/train_cnn.py \
      --method "$M" \
      --tasks cifar10 cifar100 mnist\
      --epochs_phase1 10 \
      --epochs_phase2 5 \
      --max_train_samples 50000 \
      --max_val_samples 10000 \
      --batch_size 128 \
      --accelerator mps \
      --devices 1 \
      --total_steps 20 \
      --warmup_steps 2 \
      --nostalgia_num_samples 40 \
      --k 4 \
      --nostalgia_accumulation_rounds 1 \
      --nostalgia_max_hessian_batch 4 \
      --log_every_n_steps 5 \
      --wandb_project cnn-cl-resnet10 \
      --wandb_name "$M"
done

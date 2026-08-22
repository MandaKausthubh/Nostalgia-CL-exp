# Continual Learning Experiment

In this project we're experimenting with a custom optimizer.

## Working of Optimizer:
- Each task has it's own classification head
- Works on the principle that if each element lies in the null-space of the average of hessians, we don't experience any reduction in accarcy or increase in loss.
- After computation of gradient (any arbitrary loss function), apply the projection to the null space of hessians.
- Compute the hessians (either only for the task that just ended or all past tasks)

### Algorithm:
```
Tasks: [T_1, T_2, T_3, ...]
Q, Lambda <- None, None

# Downstream head alignment
for task in Tasks:
    only downstream taskhead to be trained

# Full training
for task in Tasks:
    set task head
    if task is T_1:
        # train normally!!
        # This is template, can wary based on the experiment
        gradient <- backprop(model, task)
        model.update(gradient)              # Depending on whether we use Adam, AdamW, SGD
    else:
        # train with Nostalgia
        gradient <- backprop(model, task)
        gradient <- Projection(gradient)    # This part is taken care of inside nostalgia implementation
        model.update(gradient)              # Depending on whether we use Adam, AdamW, SGD
```

## Current Details:
- The optimizer itself is implemented in `utils/nostalgia.py`.
- `train.py` is the single unified CLI entry point for BOTH text and image tasks (task names dispatch the pipeline).
- Text model: `./models_utils/language_model.py` + `./models_utils/nostalgia_optimizer.py`.
- Image model: `./models_utils/image_model.py` (`ImageModelModule`; backbones ResNet-10/18, ViT-B/16, SigLIP-B/16 via `--backbone`).
- Text datasets (SST-2, AG-news, Trec, DB-pedia; also FLAN/SQuAD utilities) in `./datasets_utils/*`.
- Image datasets (CIFAR-10/100, MNIST, Split-CIFAR100, Split-TinyImageNet, ImageNet-100, DomainNet) in `./datasets_utils/image_datasets.py` (`IMAGE_TASK_REGISTRY`).
- `config.py` merges `TEXT_TASK_REGISTRY` + `IMAGE_TASK_REGISTRY` into one `TASK_REGISTRY`.
- Legacy CNN experiment code remains in `testing/` (`cnn_model.py`, `cnn_datasets.py`); superseded by the unified pipeline.

## Tech stack to be used:
- pytorch lightning + pytorch
- Local testing: GPU or MPS (`--accelerator mps --devices 1`)
- Main training targets: TPU v5e x8 (`--accelerator tpu --strategy xla`) and RunPod multi-GPU (4× A100, `--accelerator gpu --strategy ddp_find_unused_parameters_true`)
- `setup_pod.sh` provisions a fresh RunPod pod; `run_pod_full.sh` (single-GPU) and `run_domainnet_sweep.sh` (multi-GPU ICLR sweep: methods × backbones × seeds) are the production runners.

## What I want:
- Modular setup where I can pick:
    - Model (Qwen2.5, Gpt-2, tiny-gpt)
    - Image backbone (resnet10, resnet18, vit, siglip) + image size
    - LoRA (alpha, rank, dropout)
    - quantization
    - learning rate for backbone
    - learning rate for separate downstream head
    - number of gradiant accumulation steps
    - Gradient clipping
    - warmup steps per task
    - total_steps per task
    - epochs for full training

- Modular dataset where I can pick:
    - max length of input sequence
    - batch size for each datasets
    - max samples in datasets (training, validation)

- Benchmarking Experiment:
    - Base optimizer: Adam, AdamW, SGD
    - Whether to use nostalgia projections or not (nostalgia = On/Off)
    - Number of datapoints to be used to compute nostalgia
    - Validate every x steps
    - Wandb logging every y steps

- WandB logging:
    - Head alignment index of task: index that is specific to one task's head alignment.
    - Global step counter: Global index, maintained across tasks throughout the full finetuning process
    - Charts I want to see:
        a. {task}/training/acc          |> During full finetuning only, limited to full finetuning of that task
        b. {task}/training/loss         |> During full finetuning only, limited to full finetuning of that task
        c. {task}/validation/acc        |> During full finetuning only, plotted during full finetuning across all tasks
        c. {task}/validation/loss       |> During full finetuning only, plotted during full finetuning across all tasks
        d. {task}/alignment/loss        |> During downstream task only, but for that task
        e. {task}/alignment/acc         |> During downstream task only, but for that task

    - Charts and their plots:
        a. {task}/training/acc       = training_acc vs global index[only during fine-tuning of specific task]
        b. {task}/training/loss      = training loss vs global index[only during fine-tuning of specific task]
        c. {task}/validation/acc     = validation acc vs global index[full global index]
        c. {task}/validation/loss    = validation loss vs global index[full global index]
        d. {task}/alignment/loss     = alignment loss vs head alignment index of task
        e. {task}/alignment/acc      = alignment acc vs head alignment index of task

## Implementation Plan:

## Methods
7 methods selected via `--method {nostalgia,naive_adam,ewc,gpm,agem,ewc_nostalgia,sdft}`:
- **Nostalgia** — Hessian (Lanczos) null-space gradient projection. `--nostalgia_alpha` controls soft vs hard projection.
- **naive_adam** — raw base optimizer, no protection.
- **GPM** (Gradient Projection Memory) — same null-space projection family, gradient-based subspace. Reuses the Nostalgia projection wrapper; only Q construction differs.
- **EWC** (Elastic Weight Consolidation) — Fisher-diagonal regularization penalty.
- **A-GEM** (Average Gradient Episodic Memory) — replay buffer with gradient angle projection.
- **EWC+Nostalgia** (`ewc_nostalgia`) — hybrid: low-rank Hessian quadratic penalty `lam * Q Λ Qᵀ (θ − θ*)` injected into grads (`baselines/ewc_nostalgia.py`, `--ewc_nostalgia_lambda`).
- **SDFT** (Self-Distillation Fine-Tuning) — frozen teacher snapshot from previous task, KL distillation added during full finetuning (image pipeline).

Implemented in `baselines/` package.
EWC, A-GEM, SDFT and EWC+Nostalgia add their own optimizer wrappers (parallel to NostalgiaOptimizer).
Per-task state (Fisher / replay buffer / subspace / teacher) computed in PhaseSchedulerCallback.on_train_epoch_end.

## Run scripts
- `run_pod_full.sh` — RunPod single-GPU DomainNet run (sanity → smoke → full sweep). Env overrides: `TASKS`, `METHODS`, `BACKBONE`, `MODE=smoke|full`.
- `run_domainnet_sweep.sh` — full ICLR sweep: 7 methods × {resnet18, vit, siglip} × 3 seeds over the 6 DomainNet domains. Env overrides for every axis (`SEEDS`, `BACKBONES`, `METHODS`, `BS_*`, `PH2`, ...).
- `run_cnn_benchmarks.sh`, `run_image_benchmarks.sh`, `run_tiny_imagenet_benchmarks.sh`, `run_kaggle_domainnet_benchmarks.sh` — older per-benchmark runners (some still reference legacy `testing/train_cnn.py`).
- `run_full_llm_cl.sh`, `run_smoke_llm_cl.sh`, `run_full_resnet10_cl.sh` — LLM and ResNet-10 runners.

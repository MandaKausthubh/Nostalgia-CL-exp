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
- The model has been implemented in `./models_utils/language_model.py` and `./models_utils/nostalgia_optimizer.py`.
- Datasets we are experimenting with datasets such as SST-2, AG-news, Trec and DB-pedia. (all classification datasets).
- datasets are implemented in `./datasets_utils/*`.

## Tech stack to be used:
- pytorch lightning + pytorch
- Training device is a TPU (although for local testing keep using GPU or MPS)
- Multi-TPU setup is the main one in which I'm training of TPUV5e x8.

## What I want:
- Modular setup where I can pick:
    - Model (Qwen2.5, Gpt-2, tiny-gpt)
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






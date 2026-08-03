"""Top-level entry point for the modular Nostalgia continual-learning pipeline."""

import argparse
import json
import os
import sys

import lightning.pytorch as pl


def _parse_dataset_overrides(value):
    """Accept either a JSON string or a path to a JSON file."""
    if value is None:
        return None
    if os.path.exists(value):
        with open(value, "r") as f:
            return json.load(f)
    return json.loads(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular Nostalgia continual-learning pipeline",
    )

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_name", type=str, default="gpt2")
    model_group.add_argument("--backbone", type=str, default="resnet10",
                             choices=["resnet10", "resnet18", "vit", "siglip"],
                             help="Image encoder backbone (only used for image tasks)")
    model_group.add_argument("--image_size", type=int, default=None,
                             help="Input image size (default: 32 for ResNets, 224 for ViT/SigLIP)")
    model_group.add_argument("--use_lora", action="store_true")
    model_group.add_argument("--lora_r", type=int, default=8)
    model_group.add_argument("--lora_alpha", type=int, default=16)
    model_group.add_argument("--lora_dropout", type=float, default=0.05)
    model_group.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "4bit", "8bit"],
    )
    model_group.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "bf16", "bf16-mixed", "16", "16-mixed"],
    )
    model_group.add_argument(
        "--pooling",
        type=str,
        default="last",
        choices=["last", "mean", "eos", "first", "before_category"],
    )
    model_group.add_argument("--head_layers", type=int, default=1)

    # Optimizer / training
    opt_group = parser.add_argument_group("Optimizer / training")
    opt_group.add_argument(
        "--base_optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
    )
    opt_group.add_argument("--lr", type=float, default=5e-5)
    opt_group.add_argument("--head_lr", type=float, default=None)
    opt_group.add_argument("--weight_decay", type=float, default=0.01)
    opt_group.add_argument("--sgd_momentum", type=float, default=0.9)
    opt_group.add_argument("--grad_clip_val", type=float, default=1.0)
    opt_group.add_argument("--accumulate_grad_batches", type=int, default=1)
    opt_group.add_argument("--warmup_steps", type=int, default=100)
    opt_group.add_argument("--total_steps", type=int, default=1000)
    opt_group.add_argument("--epochs_phase1", type=int, default=1)
    opt_group.add_argument("--epochs_phase2", type=int, default=1)

    # Dataset
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--tasks",
        nargs="+",
        default=["sst2", "agnews", "trec", "dbpedia"],
        help="List of task names to train on",
    )
    data_group.add_argument("--max_length", type=int, default=512)
    data_group.add_argument("--batch_size", type=int, default=8)
    data_group.add_argument("--max_train_samples", type=int, default=None)
    data_group.add_argument("--max_val_samples", type=int, default=None)
    data_group.add_argument(
        "--dataset_overrides",
        type=str,
        default=None,
        help='JSON string or path to JSON with per-task overrides. Example: \'{"sst2": {"batch_size": 4}}\'',
    )
    data_group.add_argument("--data_root", type=str, default="./data",
                            help="Root folder for datasets")
    data_group.add_argument("--data_root_tinyimagenet", type=str, default=None,
                            help="Folder containing tiny-imagenet-200 (overrides --data_root for tinyimg_* tasks)")
    data_group.add_argument("--data_root_domainnet", type=str, default=None,
                            help="Folder containing the domainnet directory (overrides --data_root for domainnet_* tasks)")
    data_group.add_argument("--num_workers", type=int, default=0,
                            help="DataLoader worker processes")
    data_group.add_argument("--pin_memory", action="store_true",
                            help="Use pin_memory in DataLoaders (speeds up GPU transfer)")

    # Nostalgia
    nostalgia_group = parser.add_argument_group("Nostalgia")
    nostalgia_group.add_argument(
        "--method",
        type=str,
        default="nostalgia",
        choices=["nostalgia", "naive_adam", "ewc", "gpm", "agem", "ewc_nostalgia", "sdft"],
    )
    nostalgia_group.add_argument("--k", type=int, default=20)
    nostalgia_group.add_argument("--nostalgia_accumulation_rounds", type=int, default=5)
    nostalgia_group.add_argument("--nostalgia_max_hessian_batch", type=int, default=8)
    nostalgia_group.add_argument(
        "--nostalgia_num_samples",
        type=int,
        default=None,
        help="Max number of training samples used for Hessian estimation per task",
    )
    nostalgia_group.add_argument(
        "--nostalgia_alpha",
        type=float,
        default=1.0,
        help="Soft-projection factor for Nostalgia (1.0 hard, 0.0 no projection)",
    )
    # Baseline hyperparameters
    nostalgia_group.add_argument("--ewc_lambda", type=float, default=400.0,
                                 help="EWC regularization strength (lam)")
    nostalgia_group.add_argument("--ewc_nostalgia_lambda", type=float, default=400.0,
                                 help="EWC+Nostalgia quadratic penalty strength (lam)")
    nostalgia_group.add_argument("--agem_mem_size", type=int, default=500,
                                 help="A-GEM replay buffer capacity (examples)")
    nostalgia_group.add_argument("--gpm_threshold", type=float, default=0.925,
                                 help="GPM relative singular-value threshold for subspace retention")
    nostalgia_group.add_argument("--sdft_lambda_distillation", type=float, default=1.0,
                                 help="SDFT distillation loss weight")
    nostalgia_group.add_argument("--sdft_temperature", type=float, default=2.0,
                                 help="SDFT distillation temperature")

    # Validation / logging
    log_group = parser.add_argument_group("Validation / logging")
    log_group.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validate every N training steps (int) or fraction of an epoch (float)",
    )
    log_group.add_argument("--log_every_n_steps", type=int, default=50)
    log_group.add_argument("--wandb_project", type=str, default=None)
    log_group.add_argument("--wandb_name", type=str, default=None)
    log_group.add_argument("--wandb_entity", type=str, default=None)
    log_group.add_argument("--run_debug_checks", action="store_true")
    log_group.add_argument("--seed", type=int, default=42)

    # Trainer hardware
    trainer_group = parser.add_argument_group("Trainer")
    trainer_group.add_argument("--accelerator", type=str, default="auto")
    trainer_group.add_argument("--devices", type=str, default="auto")
    trainer_group.add_argument("--strategy", type=str, default="auto")

    args = parser.parse_args()

    # Validate tasks
    from config import TASK_REGISTRY
    unknown = [t for t in args.tasks if t not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid: {list(TASK_REGISTRY)}")

    # Validate dataset_overrides if provided
    if args.dataset_overrides is not None:
        _parse_dataset_overrides(args.dataset_overrides)

    # Infer default image size from backbone for image tasks
    if args.image_size is None:
        args.image_size = 32 if args.backbone in ("resnet10", "resnet18") else 224

    # Default wandb name
    if args.wandb_project and args.wandb_name is None:
        args.wandb_name = "-".join(args.tasks) + f"-{args.model_name}"

    return args


def main():
    args = parse_args()

    # WandB offline when no project is supplied
    if not args.wandb_project:
        os.environ.setdefault("WANDB_MODE", "offline")

    pl.seed_everything(args.seed, workers=True)

    from training.pipeline import run_sequential_pipeline
    run_sequential_pipeline(args)


if __name__ == "__main__":
    main()

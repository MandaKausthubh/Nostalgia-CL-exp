import hashlib
import json
import os
import torch
import wandb
from torch.utils.data import DataLoader, Subset

from config import resolve_device_and_quantization, TASK_REGISTRY, TEXT_TASK_REGISTRY
from datasets_utils.wrappers import TaskClassificationDataset
from models_utils.nostalgia_optimizer import NostalgiaLanguageModelModule
from models_utils.image_model import ImageModelModule
from utils.logging import WandbSummaryWriterWrapper, NostalgiaWandbLogger

import lightning.pytorch as pl
from training.phases import SimpleProgressBar
from training.phase_scheduler import PhaseSchedulerCallback
from training.switching_dataloader import SequentialTaskDataModule


def _parse_dataset_overrides(value):
    """Accept either a JSON string or a path to a JSON file."""
    if value is None:
        return {}
    if os.path.exists(value):
        with open(value, "r") as f:
            return json.load(f)
    return json.loads(value)


def _is_image_task(task_name: str) -> bool:
    return task_name not in TEXT_TASK_REGISTRY


def build_dataset_config(args):
    """Build per-task dataset config, merging global args with overrides."""
    default = {
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "data_root": getattr(args, "data_root", "./data"),
    }
    overrides = _parse_dataset_overrides(getattr(args, "dataset_overrides", None))
    cfg = {}
    for task_name in getattr(args, "tasks", ["sst2", "agnews", "trec", "dbpedia"]):
        task_cfg = {**default, **overrides.get(task_name, {})}
        # Allow image benchmarks to live in separate folders.
        if task_name.startswith("tinyimg_") and getattr(args, "data_root_tinyimagenet", None) is not None:
            task_cfg["data_root"] = args.data_root_tinyimagenet
        if task_name.startswith("domainnet_") and getattr(args, "data_root_domainnet", None) is not None:
            task_cfg["data_root"] = args.data_root_domainnet
        cfg[task_name] = task_cfg
    return cfg


def build_tasks(args, data_modules, default_device):
    """Construct task dicts with per-task loaders and a capped Hessian loader."""
    active_tasks = getattr(args, "tasks", ["sst2", "agnews", "trec", "dbpedia"])
    dataset_config = getattr(args, "dataset_config", build_dataset_config(args))

    tasks = []
    for task_name in active_tasks:
        if task_name not in TASK_REGISTRY:
            continue
        num_classes, _ = TASK_REGISTRY[task_name]
        dm = data_modules[task_name]
        cfg = dataset_config[task_name]

        train_dataset = TaskClassificationDataset(dm.train_ds, num_classes=num_classes)
        hessian_dataset = train_dataset
        hessian_num_samples = getattr(args, "nostalgia_num_samples", None)
        if hessian_num_samples is not None:
            hessian_num_samples = min(hessian_num_samples, len(hessian_dataset))
            hessian_dataset = Subset(hessian_dataset, range(hessian_num_samples))

        tasks.append({
            "name": task_name,
            "train_ds": dm.train_ds,
            "num_classes": num_classes,
            "loader": DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                pin_memory=(default_device.type == "cuda"),
            ),
            "hessian_loader": DataLoader(
                hessian_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                pin_memory=(default_device.type == "cuda"),
            ),
        })
    return tasks


def build_val_dataloaders(tasks, data_modules, default_device, dataset_config):
    """Build validation DataLoaders for all tasks using per-task batch sizes.

    Returns:
        val_dataloaders: list of DataLoader
        val_task_names: list of str (parallel to val_dataloaders)
    """
    val_dataloaders = []
    val_task_names = []

    for task in tasks:
        t_name = task["name"]
        dm = data_modules.get(t_name)
        if dm is None:
            continue

        cfg = dataset_config[t_name]
        val_dataloaders.append(DataLoader(
            TaskClassificationDataset(dm.val_ds, num_classes=task["num_classes"]),
            batch_size=cfg["batch_size"],
            shuffle=False,
            pin_memory=(default_device.type != "mps"),
        ))
        val_task_names.append(t_name)

    return val_dataloaders, val_task_names

def print_global(text, rank=0, string_process_func = None):
    if rank == 0:
        if string_process_func is not None:
            print(string_process_func(text), flush=True)
        else:
            print(text, flush=True)


def _phase1_cache_path(args, tasks, dataset_config):
    """Return cache file path for Phase-1 head-alignment state_dict.

    Key is a sha1 over the Phase-1-relevant config: backbone, pretrained,
    image_size, sorted task names, epochs_phase1, head_lr, and the effective
    batch config (batch_size, max_train_samples). Excludes method and seed —
    Phase-1 output is identical across methods and seeds for a given backbone.
    """
    first_cfg = dataset_config[tasks[0]["name"]]
    key_dict = {
        "backbone": getattr(args, "backbone", None),
        "pretrained": getattr(args, "pretrained", True),
        "image_size": getattr(args, "image_size", None),
        "tasks": sorted(t["name"] for t in tasks),
        "epochs_phase1": args.epochs_phase1,
        "head_lr": args.head_lr,
        "batch_size": first_cfg["batch_size"],
        "max_train_samples": first_cfg["max_train_samples"],
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    key_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
    data_root = first_cfg.get("data_root", "./data")
    cache_dir = os.path.join(data_root, "phase1_cache")
    return os.path.join(cache_dir, f"{key_hash}.pt"), key_dict

def run_sequential_pipeline(args):
    """Full sequential multi-task training pipeline with Nostalgia projection.

    Uses a SINGLE Trainer + single fit() call.  All task/phase switching is
    handled by PhaseSchedulerCallback and SequentialTaskDataModule so that
    XLA's xmp.spawn is only invoked once (fixing the TPU multi-spawn crash).
    """

    # Seed everything for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # TF32 matmul on Ampere+ GPUs for higher throughput (fp32 semantics;
    # used for forward/backward only — Lanczos HVP in Phase 3 runs outside
    # autocast and is unaffected by matmul precision).
    torch.set_float32_matmul_precision("high")

    # Detect device and validate quantization
    default_device, quantization = resolve_device_and_quantization(args)

    # Initialize wandb logger (Lightning handles wandb.init internally)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print_global("Initializing Weights & Biases (wandb) run...", rank=local_rank)

    # Drop unused LM-only defaults when running image tasks to avoid confusing printout.
    _active_tasks = getattr(args, "tasks", [])
    _is_image = _active_tasks and _active_tasks[0] not in TEXT_TASK_REGISTRY
    display_args = vars(args).copy()
    if _is_image:
        for k in ["model_name", "use_lora", "lora_r", "lora_alpha", "lora_dropout",
                  "quantization", "pooling", "head_layers", "max_length"]:
            display_args.pop(k, None)

    print_global(
        display_args, local_rank,
        string_process_func=lambda x: "Arguments for this training are:\n" + str(x)
    )

    wandb_dir = os.environ.get("WANDB_DIR", "/kaggle/tmp/wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    wandb_logger = NostalgiaWandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        dir=wandb_dir,
        save_dir=wandb_dir,
    )

    # 1. Setup Datasets
    print_global("Setting up datasets...", rank=local_rank)
    args.dataset_config = build_dataset_config(args)
    active_tasks = getattr(args, "tasks", ["sst2", "agnews", "trec", "dbpedia"])

    data_modules = {}
    for task_name in active_tasks:
        num_classes, DMClass = TASK_REGISTRY[task_name]
        cfg = args.dataset_config[task_name]
        if _is_image_task(task_name):
            dm = DMClass(
                batch_size=cfg["batch_size"],
                max_train_samples=cfg["max_train_samples"],
                max_val_samples=cfg["max_val_samples"],
                image_size=getattr(args, "image_size", 32),
                data_root=cfg["data_root"],
                num_workers=getattr(args, "num_workers", 0),
                pin_memory=getattr(args, "pin_memory", False),
            )
        else:
            dm = DMClass(
                model_name=args.model_name,
                max_length=cfg["max_length"],
                batch_size=cfg["batch_size"],
                max_train_samples=cfg["max_train_samples"],
                max_val_samples=cfg["max_val_samples"],
            )
        dm.setup()
        data_modules[task_name] = dm

    tasks = build_tasks(args, data_modules, default_device)
    val_dataloaders, val_task_names = build_val_dataloaders(
        tasks, data_modules, default_device, args.dataset_config,
    )
    print_global("Datasets setup completed successfully.", rank=local_rank)

    # 2. Instantiate Model with Task-Specific Heads
    tasks_config = {task["name"]: (task["num_classes"], args.head_layers) for task in tasks}
    print_global(f"Instantiating model with tasks: {tasks_config}...", rank=local_rank)
    print_global(f"Training method: {args.method}", rank=local_rank)

    if active_tasks and _is_image_task(active_tasks[0]):
        # Image pipeline
        tasks_config = {task["name"]: task["num_classes"] for task in tasks}
        model = ImageModelModule(
            lr=args.lr,
            head_lr=args.head_lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
            tasks_config=tasks_config,
            writer=WandbSummaryWriterWrapper(),
            method=args.method,
            base_optimizer_name=getattr(args, "base_optimizer", "adamw"),
            sgd_momentum=getattr(args, "sgd_momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 0.01),
            log_every=getattr(args, "log_every_n_steps", 50),
            ewc_lambda=getattr(args, "ewc_lambda", 400.0),
            ewc_nostalgia_lambda=getattr(args, "ewc_nostalgia_lambda", 400.0),
            agem_mem_size=getattr(args, "agem_mem_size", 500),
            gpm_threshold=getattr(args, "gpm_threshold", 0.925),
            run_debug_checks=getattr(args, "run_debug_checks", False),
            nostalgia_alpha=getattr(args, "nostalgia_alpha", 1.0),
            backbone_name=getattr(args, "backbone", "resnet10"),
            image_size=getattr(args, "image_size", 32),
            pretrained=getattr(args, "pretrained", True),
            sdft_lambda_distillation=getattr(args, "sdft_lambda_distillation", 1.0),
            sdft_temperature=getattr(args, "sdft_temperature", 2.0),
        )
    else:
        # Language model pipeline
        model = NostalgiaLanguageModelModule(
            model_name=args.model_name,
            lr=args.lr,
            head_lr=args.head_lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
            tasks_config=tasks_config,
            writer=WandbSummaryWriterWrapper(),
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            quantization=quantization,
            method=args.method,
            pooling=getattr(args, "pooling", "last"),
            run_debug_checks=getattr(args, "run_debug_checks", False),
            precision=args.precision,
            base_optimizer_name=getattr(args, "base_optimizer", "adamw"),
            sgd_momentum=getattr(args, "sgd_momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 0.01),
            log_every=getattr(args, "log_every_n_steps", 50),
            ewc_lambda=getattr(args, "ewc_lambda", 400.0),
            ewc_nostalgia_lambda=getattr(args, "ewc_nostalgia_lambda", 400.0),
            agem_mem_size=getattr(args, "agem_mem_size", 500),
            gpm_threshold=getattr(args, "gpm_threshold", 0.925),
            nostalgia_alpha=getattr(args, "nostalgia_alpha", 1.0),
        )
    if getattr(model, "writer", None) is not None:
        model.writer.model = model

    # Tell the model which task each val dataloader corresponds to
    model.val_task_names = val_task_names

    # Assign model reference to the custom logger so it can access global_step_counter
    wandb_logger.model = model

    # 2b. Phase-1 head-alignment cache lookup — load cached state_dict if
    # present, then drop all head_align epochs by zeroing args.epochs_phase1
    # BEFORE the PhaseSchedulerCallback builds its schedule. Cache is keyed
    # by backbone/image_size/tasks/batch config only (method + seed excluded).
    args.phase1_cache_path = None  # default: no cache save
    if getattr(args, "epochs_phase1", 0) > 0 and active_tasks and _is_image_task(active_tasks[0]):
        cache_path, cache_key = _phase1_cache_path(args, tasks, args.dataset_config)
        if os.path.exists(cache_path):
            blob = torch.load(cache_path, map_location="cpu")
            model.load_state_dict(blob["state_dict"])
            args.epochs_phase1 = 0  # zero BEFORE callback reads it
            print_global(
                f"[Phase-1 cache] HIT: loaded cached heads from {cache_path} "
                f"(key={json.dumps(cache_key, sort_keys=True)}); "
                f"epochs_phase1 zeroed, head_align epochs dropped",
                rank=local_rank,
            )
        else:
            # Record path so PhaseSchedulerCallback saves at end of final head_align epoch.
            args.phase1_cache_path = cache_path
            print_global(
                f"[Phase-1 cache] MISS: will write to {cache_path} "
                f"after Phase 1 completes",
                rank=local_rank,
            )

    # 3. Build schedule and single Trainer
    scheduler_callback = PhaseSchedulerCallback(tasks, args)
    task_batch_sizes = {
        name: cfg["batch_size"]
        for name, cfg in args.dataset_config.items()
    }
    data_module = SequentialTaskDataModule(
        tasks=tasks,
        val_dataloaders=val_dataloaders,
        val_task_names=val_task_names,
        schedule=scheduler_callback.schedule,
        args=args,
        default_device=default_device,
        task_batch_sizes=task_batch_sizes,
    )

    total_epochs = scheduler_callback.total_epochs

    # Compute val_check_interval clamped to smallest task loader
    val_check_interval = args.val_check_interval
    num_devices = args.devices if isinstance(args.devices, int) else 1
    min_batches = min(len(t["loader"]) for t in tasks)
    effective_batches = max(1, min_batches // num_devices)
    if isinstance(val_check_interval, int) and val_check_interval > effective_batches:
        val_check_interval = effective_batches

    trainer = pl.Trainer(
        max_epochs=total_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        deterministic=True,
        gradient_clip_val=0,              # starts in Phase 1 (disabled); callback sets Phase 2 value
        accumulate_grad_batches=1,        # starts in Phase 1 (no accum); callback sets Phase 2 value
        enable_checkpointing=False,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=val_check_interval,
        callbacks=[SimpleProgressBar(), scheduler_callback],
    )

    # 4. Single fit() — all task/phase switching happens in the callback
    trainer.fit(model, datamodule=data_module)

    print_global("Lifelong learning sequential training pipeline completed!", rank=local_rank)

    # Close wandb run
    if wandb.run is not None:
        wandb.finish()

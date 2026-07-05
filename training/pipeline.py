import os
import wandb
from torch.utils.data import DataLoader

from config import resolve_device_and_quantization
from datasets_utils.sst2_dataset import SST2DataModule
from datasets_utils.agnews_dataset import AGNewsDataModule
from datasets_utils.trec_dataset import TRECDataModule
from datasets_utils.dbpedia_dataset import DBpediaDataModule
from datasets_utils.wrappers import TaskClassificationDataset
from models_utils.nostalgia_optimizer import NostalgiaLanguageModelModule
from utils.logging import WandbSummaryWriterWrapper, NostalgiaWandbLogger

import lightning.pytorch as pl
from training.phases import SimpleProgressBar
from training.phase_scheduler import PhaseSchedulerCallback
from training.switching_dataloader import SequentialTaskDataModule


def build_tasks(args, data_modules, default_device):
    """Construct the list of task dicts, each with name, train_ds, and num_classes."""
    task_defs = [
        {"name": "sst2",     "dm_key": "sst2",     "num_classes": 2},
        {"name": "agnews",   "dm_key": "agnews",   "num_classes": 4},
        {"name": "trec",     "dm_key": "trec",     "num_classes": 6},
        {"name": "dbpedia",  "dm_key": "dbpedia",  "num_classes": 14},
    ]
    active_tasks = getattr(args, "tasks", ["sst2", "agnews", "trec", "dbpedia"])
    task_defs = [td for td in task_defs if td["name"] in active_tasks]
    tasks = []
    for td in task_defs:
        dm = data_modules[td["dm_key"]]
        tasks.append({
            "name": td["name"],
            "train_ds": dm.train_ds,
            "num_classes": td["num_classes"],
            # Keep a pre-built loader for Hessian estimation (Phase 3)
            "loader": DataLoader(
                TaskClassificationDataset(dm.train_ds, num_classes=td["num_classes"]),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=(default_device.type == "cuda"),
            ),
        })
    return tasks


def build_val_dataloaders(tasks, data_modules, default_device, batch_size):
    """Build validation DataLoaders for all tasks.

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

        val_dataloaders.append(DataLoader(
            TaskClassificationDataset(dm.val_ds, num_classes=task["num_classes"]),
            batch_size=batch_size,
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

def run_sequential_pipeline(args):
    """Full sequential multi-task training pipeline with Nostalgia projection.

    Uses a SINGLE Trainer + single fit() call.  All task/phase switching is
    handled by PhaseSchedulerCallback and SequentialTaskDataModule so that
    XLA's xmp.spawn is only invoked once (fixing the TPU multi-spawn crash).
    """

    # Seed everything for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Detect device and validate quantization
    default_device, quantization = resolve_device_and_quantization(args)

    # Initialize wandb logger (Lightning handles wandb.init internally)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print_global("Initializing Weights & Biases (wandb) run...", rank=local_rank)
    print_global(
        args, local_rank,
        string_process_func=lambda x: "Arguments for this training are:\n" + str(x)
    )

    wandb_logger = NostalgiaWandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
    )

    # 1. Setup Datasets
    print_global("Setting up datasets...", rank=local_rank)
    dm_kwargs = dict(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    active_tasks = getattr(args, "tasks", ["sst2", "agnews", "trec", "dbpedia"])
    data_modules = {}
    if "sst2" in active_tasks:
        sst2_dm = SST2DataModule(**dm_kwargs)
        sst2_dm.setup()
        data_modules["sst2"] = sst2_dm
    if "agnews" in active_tasks:
        agnews_dm = AGNewsDataModule(**dm_kwargs)
        agnews_dm.setup()
        data_modules["agnews"] = agnews_dm
    if "trec" in active_tasks:
        trec_dm = TRECDataModule(**dm_kwargs)
        trec_dm.setup()
        data_modules["trec"] = trec_dm
    if "dbpedia" in active_tasks:
        dbpedia_dm = DBpediaDataModule(**dm_kwargs)
        dbpedia_dm.setup()
        data_modules["dbpedia"] = dbpedia_dm

    tasks = build_tasks(args, data_modules, default_device)
    val_dataloaders, val_task_names = build_val_dataloaders(
        tasks, data_modules, default_device, args.batch_size,
    )
    print_global("Datasets setup completed successfully.", rank=local_rank)

    # 2. Instantiate Model with Task-Specific Heads
    tasks_config = {task["name"]: (task["num_classes"], args.head_layers) for task in tasks}
    print_global(f"Instantiating model with tasks: {tasks_config}...", rank=local_rank)
    print_global(f"Training method: {args.method}", rank=local_rank)
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
    )
    if getattr(model, "writer", None) is not None:
        model.writer.model = model

    # Tell the model which task each val dataloader corresponds to
    model.val_task_names = val_task_names

    # Assign model reference to the custom logger so it can access global_step_counter
    wandb_logger.model = model

    # 3. Build schedule and single Trainer
    scheduler_callback = PhaseSchedulerCallback(tasks, args)
    data_module = SequentialTaskDataModule(
        tasks=tasks,
        val_dataloaders=val_dataloaders,
        val_task_names=val_task_names,
        schedule=scheduler_callback.schedule,
        args=args,
        default_device=default_device,
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

import hashlib
import json
import os
import re
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


def _checkpoint_domain(task_name):
    """Use the dataset task as the domain, stripping the DomainNet prefix."""
    return task_name.removeprefix("domainnet_")


class HuggingFaceTaskCheckpointCallback(pl.Callback):
    """Save and optionally upload the model after each completed task."""

    def __init__(self, args, is_image):
        super().__init__()
        self.args = args
        self.is_image = is_image
        self.uploaded_tasks = set()

    def _repo_name(self, task_name):
        model_name = (
            getattr(self.args, "backbone", "image")
            if self.is_image else self.args.model_name
        )
        model_name = model_name.rsplit("/", 1)[-1]
        raw_name = f"{model_name}_{self.args.method}_{_checkpoint_domain(task_name)}"
        return re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-")

    def _save_checkpoint(self, pl_module, task_name):
        checkpoint_dir = os.path.abspath(
            os.path.join(self.args.checkpoint_dir, self._repo_name(task_name))
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_dict = {
            name: value.detach().cpu()
            for name, value in pl_module.state_dict().items()
        }
        torch.save(state_dict, os.path.join(checkpoint_dir, "lightning_state_dict.pt"))
        with open(os.path.join(checkpoint_dir, "checkpoint_config.json"), "w") as config_file:
            json.dump({
                "task": task_name,
                "model_name": getattr(self.args, "model_name", None),
                "backbone": getattr(self.args, "backbone", None),
                "method": self.args.method,
                "domain": _checkpoint_domain(task_name),
            }, config_file, indent=2)

        # Preserve native Transformers/PEFT loading for text checkpoints.
        if not self.is_image and hasattr(pl_module, "model"):
            pl_module.model.save_pretrained(checkpoint_dir)
            tokenizer = getattr(pl_module, "tokenizer_instance", None)
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)
        return checkpoint_dir

    def _upload(self, checkpoint_dir, task_name):
        from huggingface_hub import HfApi

        namespace = getattr(self.args, "hf_hub_namespace", None)
        if not namespace:
            raise ValueError(
                "--push_to_hub requires --hf_hub_namespace or HF_USERNAME/HF_ORG"
            )
        repo_id = f"{namespace}/{self._repo_name(task_name)}"
        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=getattr(self.args, "hf_hub_private", False),
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=checkpoint_dir,
            commit_message=f"Checkpoint after task {task_name}",
        )
        return repo_id

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        scheduler = next(
            (callback for callback in trainer.callbacks
             if isinstance(callback, PhaseSchedulerCallback)),
            None,
        )
        if scheduler is None or epoch >= len(scheduler.schedule):
            return

        task_name, phase, _ = scheduler.schedule[epoch]
        next_epoch = epoch + 1
        is_task_end = (
            phase == "nostalgia"
            and (next_epoch >= len(scheduler.schedule)
                 or scheduler.schedule[next_epoch][0] != task_name)
        )
        if not is_task_end or task_name in self.uploaded_tasks:
            return

        if trainer.is_global_zero:
            checkpoint_dir = self._save_checkpoint(pl_module, task_name)
            print(f"[Checkpoint] Saved task '{task_name}' to {checkpoint_dir}", flush=True)
            if getattr(self.args, "push_to_hub", False):
                repo_id = self._upload(checkpoint_dir, task_name)
                print(f"[Checkpoint] Pushed task '{task_name}' to {repo_id}", flush=True)
            # Crash-resume bundle — runs AFTER PhaseSchedulerCallback.on_train_epoch_end
            # (callbacks list order), so Q_memory / fisher / replay buffer are fresh.
            self._save_resume_bundle(pl_module, scheduler, task_name)
            self.uploaded_tasks.add(task_name)

        trainer.strategy.barrier("task_checkpoint")

    def _save_resume_bundle(self, pl_module, scheduler, task_name):
        """Persist a complete resumable bundle (local + optional HF Hub mirror)."""
        from training.resume import (
            build_bundle,
            experiment_key,
            resume_dir,
            save_bundle,
            snapshot_task_copy,
            upload_bundle,
            write_progress,
        )

        args = self.args
        tasks = scheduler.tasks
        dataset_config = getattr(args, "dataset_config", build_dataset_config(args))
        key_hash, key = experiment_key(args, tasks, dataset_config)

        bundle = build_bundle(
            model=pl_module,
            scheduler_cb=scheduler,
            args=args,
            key_hash=key_hash,
            key=key,
            tasks=tasks,
            wandb_run_id=(wandb.run.id if wandb.run is not None else None),
            val_metrics=getattr(pl_module, "_val_accs_per_task", {}) or {},
        )

        directory = resume_dir(args)
        latest_path = os.path.join(directory, "latest.pt")
        save_bundle(latest_path, bundle)
        write_progress(directory, bundle)
        snapshot_task_copy(directory, latest_path, bundle["resume_after_idx"])
        print(
            f"[Resume] Bundle saved after task '{task_name}' "
            f"({bundle['resume_after_idx']}/{len(tasks)} tasks) -> {directory}",
            flush=True,
        )

        if getattr(args, "push_to_hub", False):
            path = os.path.join(directory, "progress.json")
            repo_id = upload_bundle(args, [path, latest_path])
            print(f"[Resume] Pushed resume bundle to {repo_id}", flush=True)


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

        # Resolve I/O parallelism once: pin_memory + num_workers only matter on CUDA.
        # For CPU (MPS) and TPU we keep num_workers=0 because worker processes can't
        # access the accelerator memory mapping and would actually slow things down.
        on_cuda = (default_device.type == "cuda")
        nw = getattr(args, "num_workers", 0) if on_cuda else 0
        pin = on_cuda
        prefetch = 4 if nw > 0 else None  # prefetch_factor requires num_workers > 0
        persistent = bool(nw > 0)

        tasks.append({
            "name": task_name,
            "train_ds": dm.train_ds,
            "num_classes": num_classes,
            "loader": DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                num_workers=nw,
                pin_memory=pin,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
            ),
            "hessian_loader": DataLoader(
                hessian_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                num_workers=nw,
                pin_memory=pin,
                # Not persistent: iterated only at task end (~N-1 times total).
                # Keeping its pool alive would hoard idle workers next to the
                # long-lived train/val pools.
                persistent_workers=False,
                prefetch_factor=prefetch,
            ),
        })
    return tasks


def build_val_dataloaders(tasks, data_modules, default_device, dataset_config, args):
    """Build validation DataLoaders for all tasks using per-task batch sizes.

    Returns:
        val_dataloaders: list of DataLoader
        val_task_names: list of str (parallel to val_dataloaders)
    """
    val_dataloaders = []
    val_task_names = []

    # Validation is a read-only, infrequent pass that Lightning runs over ALL
    # tasks at once (every val loader is open simultaneously), so per-loader
    # worker pools multiply. Reusing the training I/O config here (workers=8,
    # prefetch_factor=4, persistent, at 224px train batch sizes) pins tens of GB
    # of host RAM the instant Phase-2 validation first runs -> cgroup OOM kills
    # sshd (kex_exchange_identification reset). Val is off the critical path:
    # use a small, non-persistent pool and minimal prefetch.
    on_cuda = (default_device.type == "cuda")
    nw = getattr(args, "num_workers", 0) if on_cuda else 0
    val_nw = min(nw, 2)
    pin = on_cuda
    prefetch = 1 if val_nw > 0 else None

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
            num_workers=val_nw,
            pin_memory=pin,
            persistent_workers=False,
            prefetch_factor=prefetch,
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
        "use_lora": getattr(args, "use_lora", False),
        "lora_r": getattr(args, "lora_r", None),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", None),
        "tasks": sorted(t["name"] for t in tasks),
        "epochs_phase1": args.epochs_phase1,
        "head_lr": args.head_lr,
        "batch_size": first_cfg["batch_size"],
        "max_train_samples": first_cfg["max_train_samples"],
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    key_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
    # Cache must live somewhere writable. data_root is read-only on Kaggle
    # (/kaggle/input/...) and can be read-only on other mounts, so prefer an
    # explicit --phase1_cache_dir, then the checkpoint dir, then data_root.
    cache_root = getattr(args, "phase1_cache_dir", None) or getattr(args, "checkpoint_dir", None)
    if cache_root:
        cache_dir = os.path.join(os.path.abspath(cache_root), "phase1_cache")
    else:
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
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # fixed image size → autotune conv algos

    # Detect device and validate quantization
    default_device, quantization = resolve_device_and_quantization(args)

    # wandb logger is constructed later (after resume detection) so it can join
    # the original run id. local_rank is needed for all the rank-0 prints below.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Drop unused LM-only defaults when running image tasks to avoid confusing printout.
    _active_tasks = getattr(args, "tasks", [])
    _is_image = _active_tasks and _active_tasks[0] not in TEXT_TASK_REGISTRY
    display_args = vars(args).copy()
    if _is_image:
        # LoRA keys stay: they now configure the image pipeline too.
        for k in ["model_name", "quantization", "pooling", "head_layers", "max_length"]:
            display_args.pop(k, None)

    print_global(
        display_args, local_rank,
        string_process_func=lambda x: "Arguments for this training are:\n" + str(x)
    )

    wandb_dir = os.path.expanduser(
        os.environ.get("WANDB_DIR", "~/wandb_log")
    )
    os.makedirs(wandb_dir, exist_ok=True)

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
        tasks, data_modules, default_device, args.dataset_config, args,
    )
    print_global("Datasets setup completed successfully.", rank=local_rank)

    # 1b. Crash-resume detection. Find a bundle matching this exact config and
    # ask (or auto-resume). Rank 0 prompts and publishes the decision; other DDP
    # ranks read it from disk (no process group exists this early).
    from training.resume import (
        experiment_key, find_resume, prompt_resume, read_decision, resume_dir,
        restore_bundle, summarize, wait_for_decision, write_decision,
    )

    key_hash, exp_key = experiment_key(args, tasks, args.dataset_config)
    decision_dir = resume_dir(args)
    resume_bundle = None
    resume_requested = False
    resume_wandb_run_id = None

    if local_rank == 0:
        # Clear any stale decision from a previous launch so non-rank-0 ranks
        # cannot read it before we publish the new one.
        try:
            os.remove(os.path.join(decision_dir, "decision.json"))
        except OSError:
            pass
        resume_bundle = find_resume(args, tasks, args.dataset_config)
        if resume_bundle is not None:
            summary = summarize(resume_bundle, args)
            resume_requested = prompt_resume(summary, getattr(args, "resume", "prompt"))
        write_decision(decision_dir, resume_requested)
    else:
        resume_requested = wait_for_decision(decision_dir)
        if resume_requested:
            # Rank 0 already downloaded the bundle; read it from the local dir.
            from training.resume import _load_bundle

            resume_bundle = _load_bundle(os.path.join(decision_dir, "latest.pt"))

    if resume_requested:
        resume_wandb_run_id = resume_bundle.get("wandb_run_id")
        # Set these BEFORE PhaseSchedulerCallback is built: epochs_phase1=0 drops
        # all head_align epochs (heads are inside the bundle), resume_after_idx
        # skips completed tasks' Phase-2 blocks.
        args.epochs_phase1 = 0
        args.resume_after_idx = int(resume_bundle.get("resume_after_idx", 0))
        print_global(
            f"[Resume] Accepted: completed "
            f"{len(resume_bundle.get('completed_tasks', []))}/"
            f"{len(resume_bundle.get('tasks', []))} tasks — "
            f"resuming after task {args.resume_after_idx}",
            rank=local_rank,
        )
        if resume_wandb_run_id:
            print_global(f"[Resume] WandB run id: {resume_wandb_run_id}", rank=local_rank)
    else:
        print_global(
            f"[Resume] Starting fresh (key={key_hash[:12]}).", rank=local_rank,
        )

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
            use_lora=getattr(args, "use_lora", False),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            channels_last=getattr(args, "channels_last", False),
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

    # Build the wandb logger now that we know whether we are joining an existing
    # run. WandbLogger already sets resume="allow"; passing id makes it continue
    # the original run instead of creating a new one.
    print_global("Initializing Weights & Biases (wandb) run...", rank=local_rank)
    wandb_logger = NostalgiaWandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        dir=wandb_dir,
        save_dir=wandb_dir,
        id=resume_wandb_run_id,
    )

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

    # Restore the full CL state (weights, Q/Lambda, fisher/theta_star, replay
    # buffer, teacher, counters, RNG) onto the freshly built model + callback.
    # Must happen BEFORE pl.Trainer is constructed: configure_optimizers reads
    # Q_memory at the first transition, which is the first epoch on resume.
    if resume_requested:
        restore_bundle(resume_bundle, model, scheduler_callback, args)
        print_global(
            f"[Resume] Restored bundle: Q={'yes' if scheduler_callback.Q_memory is not None else 'no'}, "
            f"fisher={'yes' if scheduler_callback.fisher_memory is not None else 'no'}, "
            f"replay={'yes' if scheduler_callback.replay_buffer is not None else 'no'}, "
            f"teacher={'yes' if scheduler_callback.teacher_memory is not None else 'no'}",
            rank=local_rank,
        )

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

    checkpoint_callback = HuggingFaceTaskCheckpointCallback(
        args=args,
        is_image=_is_image,
    )
    trainer = pl.Trainer(
        max_epochs=total_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        deterministic=False,               # deterministic=True disables cudnn.benchmark autotuning
        benchmark=True,                    # fixed image size → autotune conv algos for throughput
        gradient_clip_val=0,              # starts in Phase 1 (disabled); callback sets Phase 2 value
        accumulate_grad_batches=1,        # starts in Phase 1 (no accum); callback sets Phase 2 value
        enable_checkpointing=False,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=val_check_interval,
        callbacks=[SimpleProgressBar(), scheduler_callback, checkpoint_callback],
    )

    # 4. Single fit() — all task/phase switching happens in the callback
    trainer.fit(model, datamodule=data_module)

    print_global("Lifelong learning sequential training pipeline completed!", rank=local_rank)

    # Close wandb run
    if wandb.run is not None:
        wandb.finish()

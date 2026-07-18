"""PhaseSchedulerCallback — manages task/phase transitions for single-Trainer sequential learning.

Schedule layout (global Phase 1 → sequential Phase 2 + Phase 3):

  Global Phase 1:  head_align epochs for task_1, then task_2, …, then task_N
                   (backbone frozen, pretrained representations used for all)
  Sequential Phase 2+3:
      for each task i:
          Phase 2  — full finetuning epochs (backbone + head_i)
          Phase 3  — Hessian estimation (inline, at last Phase 2 epoch)

This ensures every head is aligned to the *same* pretrained backbone,
avoiding the problem where per-task Phase 1 overwrites a head after the
backbone has already drifted.
"""

import gc
import torch
import lightning.pytorch as pl

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    class DummyXM:
        def mark_step(self):
            pass
    xm = DummyXM()


from utils.hessians import compute_single_domain_eigenspace
from utils.accumulate import accumulate_hessian_eigenspace_stable
from utils.TPU import broadcast_Q_Lambda


class PhaseSchedulerCallback(pl.Callback):
    """Epoch-level scheduler that drives multi-task / multi-phase training."""

    def __init__(self, tasks, args):
        super().__init__()
        self.tasks = tasks
        self.args = args
        self.method = getattr(args, "method", "nostalgia")

        # Build epoch → (task_name, phase, task_idx) schedule
        # ── Global Phase 1: align ALL heads first ──
        self.schedule = []
        for task_idx, task in enumerate(tasks, start=1):
            for _ in range(args.epochs_phase1):
                self.schedule.append((task["name"], "head_align", task_idx))

        # ── Sequential Phase 2: finetune each task in order ──
        for task_idx, task in enumerate(tasks, start=1):
            for _ in range(args.epochs_phase2):
                self.schedule.append((task["name"], "nostalgia", task_idx))

        self.total_epochs = len(self.schedule)

        # Track the previous epoch's (task, phase) to detect transitions
        self._prev_task = None
        self._prev_phase = None

        # Accumulated Hessian eigenspace
        self.Q_memory = None
        self.Lambda_memory = None

    # ------------------------------------------------------------------
    # Batch start: sanity-check that the correct task's data is served
    # ------------------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Verify the dataloader is serving the expected task's data.

        Logs label range + distribution for the first batch of every epoch.
        If you see label_max=1 during an agnews epoch (4-class), the
        SequentialTaskDataModule has an off-by-one bug and is still serving
        SST-2 data — that is the root cause of the rising loss.
        """
        if not getattr(self.args, "run_debug_checks", False):
            return
        if batch_idx != 0 or not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if epoch >= len(self.schedule):
            return
        task_name, phase, task_idx = self.schedule[epoch]

        # Extract labels — supports (input_ids, attn_mask, labels) tuples
        # and dict-style batches.
        labels = None
        if isinstance(batch, (list, tuple)):
            labels = batch[-1]
        elif isinstance(batch, dict):
            labels = batch.get("labels", batch.get("label", batch.get("target", batch.get("targets", None))))

        if labels is not None and torch.is_tensor(labels):
            unique_vals, counts = labels.unique(return_counts=True)
            print(
                f"\n  [DATA CHECK epoch={epoch}] "
                f"task={task_name!r}, phase={phase}, "
                f"label_range=[{labels.min().item()}, {labels.max().item()}], "
                f"dist={dict(zip(unique_vals.tolist(), counts.tolist()))}",
                flush=True,
            )
        else:
            print(
                f"\n  [DATA CHECK epoch={epoch}] task={task_name!r}, phase={phase} "
                f"(could not extract labels from batch type={type(batch).__name__})",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Epoch start: switch task / phase / parameters / optimizer
    # ------------------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch >= len(self.schedule):
            return

        task_name, phase, task_idx = self.schedule[epoch]
        task_changed = (task_name != self._prev_task)
        phase_changed = (phase != self._prev_phase)

        if not task_changed and not phase_changed:
            return  # nothing to do — same task & phase as last epoch

        # --- Print banner on task or phase change ---
        if task_changed and trainer.is_global_zero:
            print(f"\n{'='*60}")
            print(f"TASK {task_idx}: {task_name.upper()}")
            print(f"{'='*60}")

        if trainer.is_global_zero:
            print(f"\n[Phase {'1' if phase == 'head_align' else '2'}] "
                  f"{'Aligning head' if phase == 'head_align' else 'Finetuning'} "
                  f"for '{task_name}'  (epoch {epoch})", flush=True)

        # --- Switch active task & phase on the model ---
        pl_module.active_task = task_name
        pl_module.training_phase = phase
        pl_module.logging_disabled = False

        # --- Update active task name on datamodule for dynamic dataloader ---
        if getattr(trainer, "datamodule", None) is not None:
            trainer.datamodule.active_task_name = task_name

        # --- Freeze / unfreeze parameters ---
        if phase == "head_align":
            # Freeze backbone
            for p in pl_module.backbone.parameters():
                p.requires_grad = False
            # Reset alignment step counter for this task
            pl_module.alignment_step_counter = 0
            # Disable validation during head alignment
            if not hasattr(self, "_saved_limit_val"):
                self._saved_limit_val = trainer.limit_val_batches
            trainer.limit_val_batches = 0
            # Phase 1: no gradient clipping, no accumulation
            trainer.gradient_clip_val = 0
            trainer.accumulate_grad_batches = 1
        else:
            # Unfreeze trainable backbone params
            for name, p in pl_module.backbone.named_parameters():
                if name in pl_module.trainable_backbone_param_names:
                    p.requires_grad = True
            # Re-enable validation
            if hasattr(self, "_saved_limit_val"):
                trainer.limit_val_batches = self._saved_limit_val
            # Phase 2: enable gradient clipping and accumulation
            trainer.gradient_clip_val = getattr(self.args, "grad_clip_val", 1.0)
            trainer.accumulate_grad_batches = getattr(self.args, "accumulate_grad_batches", 1)

            # Attach Q_memory for NostalgiaOptimizer (skip for naive_adam)
            if self.method == "nostalgia":
                if self.Q_memory is not None and trainer.is_global_zero:
                    print(f"  Attaching nostalgia projection Q of shape "
                          f"{list(self.Q_memory.shape)}", flush=True)
                pl_module.Q_memory = self.Q_memory
                pl_module.Lambda_memory = self.Lambda_memory

        if trainer.is_global_zero:
            print(f"  grad_clip_val={trainer.gradient_clip_val}, "
                  f"accumulate_grad_batches={trainer.accumulate_grad_batches}",
                  flush=True)

        # Freeze/unfreeze heads — only active head trainable
        for head_name, head in pl_module.task_head_list.items():
            for p in head.parameters():
                p.requires_grad = (head_name == task_name)

        # Item 6: Verify the optimizer actually updates the correct head
        if trainer.is_global_zero:
            print("\n  [OPTIMIZER HEAD TRAINABILITY CHECK]")
            for name, head in pl_module.task_head_list.items():
                print(f"    {name:<10}: {any(p.requires_grad for p in head.parameters())}")
            print()

        # Log trainable parameter count to catch silent freeze bugs
        if trainer.is_global_zero:
            n_trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {n_trainable:,}", flush=True)

        # --- Re-configure optimizer on any transition ---
        # Explicitly break references in old optimizer to free device memory
        if hasattr(trainer, "optimizers") and trainer.optimizers:
            for opt in list(trainer.optimizers):
                if hasattr(opt, "base_optimizer"):
                    opt.base_optimizer = None
                if hasattr(opt, "projection_params"):
                    opt.projection_params = None
                if hasattr(opt, "nostalgia_Q"):
                    opt.nostalgia_Q = None
                if hasattr(opt, "scaling"):
                    opt.scaling = None
            trainer.optimizers.clear()
        if hasattr(trainer, "lr_scheduler_configs"):
            trainer.lr_scheduler_configs.clear()

        gc.collect()
        xm.mark_step()

        trainer.strategy.setup_optimizers(trainer)

        # ── FIX: Reset LR scheduler after optimizer rebuild ──────────────
        # When configure_optimizers() is called with total_steps=N (per-task),
        # some Lightning versions fast-forward the new scheduler's last_epoch
        # to match trainer.global_step. If global_step >= total_steps by the
        # time we reach task 2+, the linear-decay scheduler gives LR = 0 →
        # gradients are computed but never applied → loss and accuracy are flat
        # regardless of what the data contains.
        #
        # Resetting last_epoch = -1 and re-stepping forces the scheduler to
        # restart from epoch 0 for every new task/phase, giving the full
        # warmup+decay budget to each task independently.
        for sched_config in getattr(trainer, "lr_scheduler_configs", []):
            sched = sched_config.scheduler
            if hasattr(sched, "last_epoch") and sched.last_epoch > 0:
                if trainer.is_global_zero:
                    print(
                        f"  [LR scheduler reset] last_epoch was {sched.last_epoch} "
                        f"(global_step={trainer.global_step}). Resetting to 0.",
                        flush=True,
                    )
                sched.last_epoch = -1
                sched._step_count = 1
                sched.step()  # primes scheduler to epoch 0

        # Log effective LR and optimizer parameter counts for diagnostics
        if trainer.is_global_zero:
            for i, opt in enumerate(trainer.optimizers):
                for j, pg in enumerate(opt.param_groups):
                    n = sum(p.numel() for p in pg["params"] if p.requires_grad)
                    print(
                        f"  [Optimizer {i} group {j}] lr={pg['lr']:.2e}, "
                        f"trainable_params={n:,}",
                        flush=True,
                    )

        gc.collect()
        xm.mark_step()

        self._prev_task = task_name
        self._prev_phase = phase

    # ------------------------------------------------------------------
    # Epoch end: run Hessian estimation at task boundaries
    # ------------------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch >= len(self.schedule):
            return

        task_name, phase, task_idx = self.schedule[epoch]

        # Only run Phase 3 at the end of a task's Phase 2 block
        if phase != "nostalgia":
            return

        # Check if next epoch is a different task (or we're at the end)
        next_epoch = epoch + 1
        is_last = (next_epoch >= len(self.schedule))
        next_is_different_task = (
            not is_last and self.schedule[next_epoch][0] != task_name
        )

        if not is_last and not next_is_different_task:
            return  # still more nostalgia epochs for this task

        # Mark task as completed
        pl_module.completed_tasks.add(task_name)

        # Skip Hessian estimation for naive_adam baseline
        if self.method != "nostalgia":
            if trainer.is_global_zero:
                print(f"  [Phase 3] Skipped (method={self.method})")
            return

        if trainer.is_global_zero:
            print(f"\n[Phase 3] Estimating Hessian eigenspace for '{task_name}'...",
                  flush=True)

        # --- Hessian estimation ---
        self._run_hessian_estimation(pl_module, task_name, task_idx)

    # ------------------------------------------------------------------
    # Hessian estimation (Phase 3) — runs on all ranks
    # ------------------------------------------------------------------
    def _run_hessian_estimation(self, pl_module, task_name, task_idx):
        device = pl_module.device

        # Prefer the dedicated Hessian loader if available, else fall back.
        loader = None
        for t in self.tasks:
            if t["name"] == task_name:
                loader = t.get("hessian_loader", t["loader"])
                break

        if loader is None:
            print(f"  WARNING: no loader found for task '{task_name}', skipping Phase 3")
            return

        accumulation_rounds = getattr(self.args, "nostalgia_accumulation_rounds", 5)
        max_hessian_batch = getattr(self.args, "nostalgia_max_hessian_batch", 8)

        old_active_task = pl_module.active_task
        old_logging_disabled = getattr(pl_module, "logging_disabled", False)
        pl_module.active_task = task_name
        pl_module.logging_disabled = True

        try:
            Q_new, Lambda_new = compute_single_domain_eigenspace(
                model=pl_module,
                k=self.args.k,
                device=device,
                train_loader=loader,
                accumulation_rounds=accumulation_rounds,
                max_hessian_batch=max_hessian_batch,
            )

            # CRITICAL: restore model to train mode after Hessian estimation
            pl_module.train()

            print(f"  Computed task Q shape: {list(Q_new.shape)}, "
                  f"Lambda shape: {list(Lambda_new.shape)}")

            _flush_memory(device)

            self.Q_memory, self.Lambda_memory = accumulate_hessian_eigenspace_stable(
                Q_old=self.Q_memory,
                Lambda_old=self.Lambda_memory,
                Q_new=Q_new,
                Lambda_new=Lambda_new,
                t=task_idx+1,
                k=self.args.k,
            )
            print(f"  Accumulated Q memory shape: {list(self.Q_memory.shape)}, "
                  f"Lambda memory shape: {list(self.Lambda_memory.shape)}")

            del Q_new, Lambda_new
            _flush_memory(device)

            # Broadcast across TPU replicas
            self.Q_memory, self.Lambda_memory = broadcast_Q_Lambda(
                self.Q_memory, self.Lambda_memory
            )

        except Exception as e:
            # Don't let Hessian estimation crash the entire training run.
            # If it fails, we simply skip projection for this task.
            import traceback
            print(f"  ERROR in Hessian estimation for '{task_name}': {e}")
            traceback.print_exc()
            print(f"  Continuing without updating projection space.")
            _flush_memory(device)
        finally:
            pl_module.active_task = old_active_task
            pl_module.logging_disabled = old_logging_disabled
            pl_module.train()


def _flush_memory(device):
    """Release accelerator memory caches after heavy computation."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "xla" or device.type == "privateuseone":
        xm.mark_step()
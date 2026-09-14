"""SequentialTaskDataModule — switches DataLoader per epoch based on the phase schedule.

Lightning calls `train_dataloader()` at the start of each epoch, so we use the
Trainer's `current_epoch` to look up which task should provide data.
"""

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from datasets_utils.wrappers import TaskClassificationDataset


class DynamicTaskDataLoader(DataLoader):
    def __init__(self, datamodule, **kwargs):
        self.datamodule = datamodule
        # Initialize super class with the first task's dataset to satisfy structural checks
        if not getattr(datamodule, "task_loaders", None):
            raise RuntimeError(
                "SequentialTaskDataModule.task_loaders is empty — "
                "check args.tasks and TASK_REGISTRY membership."
            )
        first_loader = list(datamodule.task_loaders.values())[0]
        super().__init__(first_loader.dataset)

    @property
    def active_loader(self):
        # Determine the active task from the datamodule (updated by callback)
        task_name = getattr(self.datamodule, "active_task_name", None)
        if task_name is None:
            # Fallback based on current_epoch if not set yet
            trainer = getattr(self.datamodule, "trainer", None)
            schedule = getattr(self.datamodule, "schedule", [])
            # Defensive: if schedule is empty (misconfiguration upstream),
            # fall back to the first registered task loader rather than
            # crashing with IndexError. Caller will surface the real bug.
            if not schedule:
                task_loaders = getattr(self.datamodule, "task_loaders", {})
                if task_loaders:
                    task_name = next(iter(task_loaders))
                else:
                    raise RuntimeError(
                        "SequentialTaskDataModule.schedule and task_loaders are "
                        "both empty — check args.tasks / args.epochs_phase1+2."
                    )
            elif trainer is None:
                task_name = schedule[0][0]
            else:
                epoch = trainer.current_epoch
                if epoch < len(schedule):
                    task_name = schedule[epoch][0]
                else:
                    task_name = schedule[-1][0]
        return self.datamodule.task_loaders[task_name]

    def __iter__(self):
        return iter(self.active_loader)

    def __len__(self):
        return len(self.active_loader)

    def __getattribute__(self, name):
        delegated = {
            "dataset", "batch_size", "num_workers", "pin_memory", "drop_last",
            "timeout", "sampler", "batch_sampler", "collate_fn",
            "worker_init_fn", "prefetch_factor", "persistent_workers"
        }
        if name in delegated:
            active_loader = super().__getattribute__("active_loader")
            return getattr(active_loader, name)
        return super().__getattribute__(name)


class SequentialTaskDataModule(pl.LightningDataModule):
    """Returns the correct task DataLoader based on the epoch schedule."""

    def __init__(
        self,
        tasks,
        val_dataloaders,
        val_task_names,
        schedule,
        args,
        default_device,
        task_batch_sizes=None,
    ):
        super().__init__()
        self.schedule = schedule
        self.val_dataloaders_list = val_dataloaders
        self.val_task_names = val_task_names
        self.active_task_name = None
        # Surface misconfiguration early instead of crashing later in train_dataloader.
        if not schedule:
            print(
                f"[SequentialTaskDataModule] WARNING: schedule is empty "
                f"(len(schedule)=0, len(tasks)={len(tasks)}). "
                f"args.epochs_phase1+epochs_phase2 must produce a non-empty "
                f"epoch schedule.",
                flush=True,
            )

        if task_batch_sizes is None:
            task_batch_sizes = {}

        # Pre-build a DataLoader per task (keyed by task name)
        self.task_loaders = {}
        for task in tasks:
            task_name = task["name"]
            batch_size = task_batch_sizes.get(task_name, getattr(args, "batch_size", 8))
            self.task_loaders[task_name] = DataLoader(
                TaskClassificationDataset(task["train_ds"], num_classes=task["num_classes"]),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(default_device.type == "cuda"),
            )

    def train_dataloader(self):
        return DynamicTaskDataLoader(self)

    def val_dataloader(self):
        if self.val_dataloaders_list:
            return self.val_dataloaders_list
        return None

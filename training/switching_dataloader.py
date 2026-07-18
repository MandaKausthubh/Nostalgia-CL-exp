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
        first_loader = list(datamodule.task_loaders.values())[0]
        super().__init__(first_loader.dataset)

    @property
    def active_loader(self):
        # Determine the active task from the datamodule (updated by callback)
        task_name = getattr(self.datamodule, "active_task_name", None)
        if task_name is None:
            # Fallback based on current_epoch if not set yet
            trainer = getattr(self.datamodule, "trainer", None)
            if trainer is None:
                task_name = self.datamodule.schedule[0][0]
            else:
                epoch = trainer.current_epoch
                if epoch < len(self.datamodule.schedule):
                    task_name = self.datamodule.schedule[epoch][0]
                else:
                    task_name = self.datamodule.schedule[-1][0]
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

"""Image classification datasets for the main continual-learning pipeline.

All datasets emit the dict-batch contract that `TaskClassificationDataset`
(datasets_utils/wrappers.py) expects:

    {"input_ids": image_tensor (C, H, W),
     "attention_mask": torch.ones(1, dtype=torch.long),  # dummy
     "raw_label": int}

The dummy length-1 attention mask keeps A-GEM's `ReplayBuffer.sample_batch`
padding logic happy.
"""

import functools
import os
from typing import Optional, List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import lightning.pytorch as pl
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class ImageClassificationDataset(Dataset):
    """Wrap a torchvision-style dataset and emit dict batches."""

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return {
            "input_ids": img,
            "raw_label": int(label),
            "attention_mask": torch.ones(1, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ClassSubsetDataset(Dataset):
    """Filter a classification dataset to a subset of classes and remap labels."""

    def __init__(self, base_dataset, class_ids: List[int]):
        self.base_dataset = base_dataset
        self.class_ids = sorted(set(class_ids))
        self._id_map = {old_id: new_id for new_id, old_id in enumerate(self.class_ids)}
        self._class_set = set(self.class_ids)

        self.indices = []
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            old_id = int(label)
            if old_id in self._class_set:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        return img, self._id_map[int(label)]


def _maybe_subset(ds, max_samples):
    """Stratified subsample — keeps class balance."""
    if max_samples is None or max_samples <= 0:
        return ds

    n = min(max_samples, len(ds))
    labels = [int(ds[i][1]) for i in range(len(ds))]
    by_class = {}
    for idx, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(idx)
    num_classes = len(by_class)
    per_class = max(1, n // num_classes)

    selected = []
    for lab, idxs in by_class.items():
        take = min(per_class, len(idxs))
        selected.extend(idxs[:take])
    if len(selected) < n:
        remaining = [i for i in range(len(ds)) if i not in set(selected)]
        selected.extend(remaining[: (n - len(selected))])
    return Subset(ds, selected[:n])


def _build_image_transform(in_channels: int, image_size: int = 32):
    """Build a transform that produces a (C, image_size, image_size) tensor."""
    tfs = []
    if in_channels == 1:
        tfs.append(transforms.Grayscale(num_output_channels=3))
    tfs.append(transforms.Resize((image_size, image_size)))
    tfs.append(transforms.ToTensor())
    return transforms.Compose(tfs)


# ---------------------------------------------------------------------------
# Base DataModule
# ---------------------------------------------------------------------------

class BaseImageDataModule(pl.LightningDataModule):
    """Shared image DataModule skeleton."""

    IN_CHANNELS = 3
    NUM_CLASSES = 10

    def __init__(
        self,
        batch_size: int = 32,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        image_size: int = 32,
        num_workers: int = 0,
        data_root: str = "./data",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = None
        self.val_ds = None
        self.transform = _build_image_transform(self.IN_CHANNELS, image_size)

    def _load_train(self):
        raise NotImplementedError

    def _load_val(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        train_raw = self._load_train()
        val_raw = self._load_val()
        train_raw = _maybe_subset(train_raw, self.hparams.max_train_samples)
        val_raw = _maybe_subset(val_raw, self.hparams.max_val_samples)
        self.train_ds = ImageClassificationDataset(train_raw, transform=self.transform)
        self.val_ds = ImageClassificationDataset(val_raw, transform=self.transform)


# ---------------------------------------------------------------------------
# Existing torchvision datasets
# ---------------------------------------------------------------------------

class CIFAR10DataModule(BaseImageDataModule):
    IN_CHANNELS = 3
    NUM_CLASSES = 10

    def _load_train(self):
        return torchvision.datasets.CIFAR10(
            root=self.hparams.data_root, train=True, download=True, transform=None
        )

    def _load_val(self):
        return torchvision.datasets.CIFAR10(
            root=self.hparams.data_root, train=False, download=True, transform=None
        )


class CIFAR100DataModule(BaseImageDataModule):
    IN_CHANNELS = 3
    NUM_CLASSES = 100

    def _load_train(self):
        return torchvision.datasets.CIFAR100(
            root=self.hparams.data_root, train=True, download=True, transform=None
        )

    def _load_val(self):
        return torchvision.datasets.CIFAR100(
            root=self.hparams.data_root, train=False, download=True, transform=None
        )


class MNISTDataModule(BaseImageDataModule):
    IN_CHANNELS = 1
    NUM_CLASSES = 10

    def _load_train(self):
        return torchvision.datasets.MNIST(
            root=self.hparams.data_root, train=True, download=True, transform=None
        )

    def _load_val(self):
        return torchvision.datasets.MNIST(
            root=self.hparams.data_root, train=False, download=True, transform=None
        )


# ---------------------------------------------------------------------------
# Split / benchmark datasets
# ---------------------------------------------------------------------------

def _load_cifar100_parsed_or_pickled(data_root: str, train: bool):
    """Load CIFAR-100 from parsed image-folder layout if present, else pickled torchvision.

    Parsed Kaggle layout expected:
        <data_root>/train/<class>/
        <data_root>/test/<class>/
    or  <data_root>/cifar100/train/<class>/
        <data_root>/cifar100/test/<class>/
    """
    split_name = "train" if train else "test"
    candidates = [
        os.path.join(data_root, split_name),
        os.path.join(data_root, "cifar100", "images", split_name),
        os.path.join(data_root, "cifar100", split_name),
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            base = ImageFolder(folder)
            if len(base.classes) > 0:
                return base

    # Fall back to original pickled CIFAR-100 (no download; avoids read-only FS on Kaggle).
    try:
        return torchvision.datasets.CIFAR100(
            root=data_root, train=train, download=False, transform=None
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"CIFAR-100 not found at {data_root}. "
            "Provide parsed image folders (train/ + test/) or original pickled CIFAR-100."
        ) from e


class SplitCIFAR100DataModule(BaseImageDataModule):
    """10 tasks x 10 classes from CIFAR-100."""

    IN_CHANNELS = 3
    NUM_CLASSES = 10

    def __init__(self, task_id: int = 0, **kwargs):
        self.task_id = task_id
        super().__init__(**kwargs)

    def _class_ids(self):
        start = self.task_id * self.NUM_CLASSES
        return list(range(start, start + self.NUM_CLASSES))

    def _load_train(self):
        base = _load_cifar100_parsed_or_pickled(self.hparams.data_root, train=True)
        return _ClassSubsetDataset(base, self._class_ids())

    def _load_val(self):
        base = _load_cifar100_parsed_or_pickled(self.hparams.data_root, train=False)
        return _ClassSubsetDataset(base, self._class_ids())


class _TinyImageNetValDataset(Dataset):
    """Tiny ImageNet validation set, parsed from val_annotations.txt."""

    def __init__(self, root: str, class_to_idx: Dict[str, int], class_ids: List[int]):
        self.root = root
        self.class_to_idx = class_to_idx
        self.class_ids = set(class_ids)
        self._id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(class_ids))}

        annotations_file = os.path.join(root, "val_annotations.txt")
        self.samples = []
        if os.path.exists(annotations_file):
            with open(annotations_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    old_id = class_to_idx.get(wnid)
                    if old_id is None or old_id not in self.class_ids:
                        continue
                    path = os.path.join(root, "images", fname)
                    if os.path.exists(path):
                        self.samples.append((path, old_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, old_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, self._id_map[old_id]


class SplitTinyImageNetDataModule(BaseImageDataModule):
    """4 tasks x 50 classes from Tiny ImageNet."""

    IN_CHANNELS = 3
    NUM_CLASSES = 50

    def __init__(self, task_id: int = 0, **kwargs):
        self.task_id = task_id
        super().__init__(**kwargs)

    def _class_ids(self):
        start = self.task_id * self.NUM_CLASSES
        return list(range(start, start + self.NUM_CLASSES))

    def _load_train(self):
        root = os.path.join(self.hparams.data_root, "tiny-imagenet-200", "train")
        base = ImageFolder(root)
        return _ClassSubsetDataset(base, self._class_ids())

    def _load_val(self):
        root = os.path.join(self.hparams.data_root, "tiny-imagenet-200", "val")
        train_root = os.path.join(self.hparams.data_root, "tiny-imagenet-200", "train")
        train_base = ImageFolder(train_root)
        return _TinyImageNetValDataset(root, train_base.class_to_idx, self._class_ids())


class ImageNet100DataModule(BaseImageDataModule):
    """10 tasks x 10 classes from an ImageNet-100 subset."""

    IN_CHANNELS = 3
    NUM_CLASSES = 10

    def __init__(self, task_id: int = 0, **kwargs):
        self.task_id = task_id
        super().__init__(**kwargs)

    def _imagenet100_classes(self):
        root = os.path.join(self.hparams.data_root, "imagenet100")
        classes_file = os.path.join(root, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, "r") as f:
                return [line.strip() for line in f if line.strip()]
        train_root = os.path.join(root, "train")
        if os.path.isdir(train_root):
            return sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
        return []

    def _class_ids(self):
        start = self.task_id * self.NUM_CLASSES
        return list(range(start, start + self.NUM_CLASSES))

    def _load_train(self):
        root = os.path.join(self.hparams.data_root, "imagenet100", "train")
        base = ImageFolder(root)
        return _ClassSubsetDataset(base, self._class_ids())

    def _load_val(self):
        root = os.path.join(self.hparams.data_root, "imagenet100", "val")
        base = ImageFolder(root)
        return _ClassSubsetDataset(base, self._class_ids())


class _DomainNetFolder(Dataset):
    """ImageFolder-like dataset using a shared class_to_idx map."""

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

    def __init__(self, root: str, class_to_idx: Dict[str, int], samples: Optional[List[Tuple[str, int]]] = None):
        self.root = root
        self.class_to_idx = class_to_idx
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            if os.path.isdir(root):
                for class_name in sorted(os.listdir(root)):
                    class_dir = os.path.join(root, class_name)
                    if not os.path.isdir(class_dir):
                        continue
                    old_id = class_to_idx.get(class_name)
                    if old_id is None:
                        continue
                    for fname in sorted(os.listdir(class_dir)):
                        if fname.lower().endswith(self.IMG_EXTS):
                            self.samples.append((os.path.join(class_dir, fname), old_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, target


def _split_samples_per_class(samples: List[Tuple[str, int]], val_ratio: float = 0.1):
    """Deterministic per-class train/val split."""
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for path, target in samples:
        by_class.setdefault(target, []).append((path, target))

    train_samples, val_samples = [], []
    for target, class_samples in by_class.items():
        class_samples = sorted(class_samples)
        n_val = max(1, int(len(class_samples) * val_ratio)) if len(class_samples) >= 10 else 1
        step = max(1, len(class_samples) // n_val)
        val_indices = set(range(0, len(class_samples), step)) if step > 0 else set()
        val_indices = set(sorted(val_indices)[:n_val])
        for i, item in enumerate(class_samples):
            if i in val_indices:
                val_samples.append(item)
            else:
                train_samples.append(item)
    return train_samples, val_samples


def _build_domainnet_class_map(data_root: str) -> Dict[str, int]:
    """Shared 345-class name -> index map across all six DomainNet domains."""
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    class_names = set()
    root = os.path.join(data_root, "domainnet")
    for domain in domains:
        domain_root = os.path.join(root, domain)
        if not os.path.isdir(domain_root):
            continue
        for class_name in os.listdir(domain_root):
            if os.path.isdir(os.path.join(domain_root, class_name)):
                class_names.add(class_name)
    return {name: idx for idx, name in enumerate(sorted(class_names))}


class DomainNetDataModule(BaseImageDataModule):
    """One task per DomainNet domain; 345 shared classes per task."""

    IN_CHANNELS = 3
    NUM_CLASSES = 345

    def __init__(self, domain_name: str = "real", **kwargs):
        self.domain_name = domain_name
        super().__init__(**kwargs)
        self.class_to_idx = _build_domainnet_class_map(self.hparams.data_root)

    def _load_train(self):
        root = os.path.join(self.hparams.data_root, "domainnet", self.domain_name)
        val_root = os.path.join(root, "val")
        if os.path.isdir(val_root):
            return _DomainNetFolder(root, self.class_to_idx)

        full = _DomainNetFolder(root, self.class_to_idx)
        train_samples, val_samples = _split_samples_per_class(full.samples, val_ratio=0.1)
        self._train_samples = train_samples
        self._val_samples = val_samples
        return _DomainNetFolder(root, self.class_to_idx, samples=train_samples)

    def _load_val(self):
        root = os.path.join(self.hparams.data_root, "domainnet", self.domain_name)
        val_root = os.path.join(root, "val")
        if os.path.isdir(val_root):
            return _DomainNetFolder(val_root, self.class_to_idx)

        if hasattr(self, "_val_samples"):
            return _DomainNetFolder(root, self.class_to_idx, samples=self._val_samples)

        full = _DomainNetFolder(root, self.class_to_idx)
        _, val_samples = _split_samples_per_class(full.samples, val_ratio=0.1)
        return _DomainNetFolder(root, self.class_to_idx, samples=val_samples)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IMAGE_TASK_REGISTRY = {
    "cifar10": (10, CIFAR10DataModule),
    "cifar100": (100, CIFAR100DataModule),
    "mnist": (10, MNISTDataModule),
}

for _tid in range(10):
    IMAGE_TASK_REGISTRY[f"cifar100_t{_tid}"] = (
        10,
        functools.partial(SplitCIFAR100DataModule, task_id=_tid),
    )

for _tid in range(4):
    IMAGE_TASK_REGISTRY[f"tinyimg_t{_tid}"] = (
        50,
        functools.partial(SplitTinyImageNetDataModule, task_id=_tid),
    )

for _tid in range(10):
    IMAGE_TASK_REGISTRY[f"imagenet100_t{_tid}"] = (
        10,
        functools.partial(ImageNet100DataModule, task_id=_tid),
    )

for _domain in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
    IMAGE_TASK_REGISTRY[f"domainnet_{_domain}"] = (
        345,
        functools.partial(DomainNetDataModule, domain_name=_domain),
    )

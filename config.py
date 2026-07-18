"""Central configuration helpers for the Nostalgia continual-learning pipeline."""

import json
import os
from typing import Any, Dict, Optional, Tuple, Type

import torch

from datasets_utils.sst2_dataset import SST2DataModule
from datasets_utils.agnews_dataset import AGNewsDataModule
from datasets_utils.trec_dataset import TRECDataModule
from datasets_utils.dbpedia_dataset import DBpediaDataModule
from datasets_utils.base_class import BaseTextDataModule


def _has_xla_runtime() -> bool:
    """Return True if torch_xla is available and an XLA device is present."""
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device_hw() is not None
    except Exception:
        return False


#: task_name -> (num_classes, DataModuleClass)
TASK_REGISTRY: Dict[str, Tuple[int, Type[BaseTextDataModule]]] = {
    "sst2": (2, SST2DataModule),
    "agnews": (4, AGNewsDataModule),
    "trec": (6, TRECDataModule),
    "dbpedia": (14, DBpediaDataModule),
}


def resolve_device_and_quantization(args: Any) -> Tuple[torch.device, Optional[str]]:
    """Detect accelerator and validate quantization compatibility.

    Returns:
        default_device: torch.device to use for host-side decisions (pin_memory, etc.)
        quantization:   None, "4bit", or "8bit"
    """
    accelerator = getattr(args, "accelerator", None)

    if accelerator == "tpu" or _has_xla_runtime():
        import torch_xla.core.xla_model as xm
        default_device = xm.xla_device()
    elif accelerator == "cuda":
        default_device = torch.device("cuda")
    elif accelerator == "mps":
        default_device = torch.device("mps")
    elif accelerator == "cpu":
        default_device = torch.device("cpu")
    else:
        # Auto-detect in priority order: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            default_device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            default_device = torch.device("mps")
        else:
            default_device = torch.device("cpu")

    quantization = getattr(args, "quantization", None)
    if quantization not in (None, "4bit", "8bit"):
        raise ValueError(
            f"quantization must be None, '4bit', or '8bit'; got {quantization!r}"
        )

    if quantization in ("4bit", "8bit") and default_device.type in ("xla",):
        raise ValueError(
            "BitsAndBytes quantization is not compatible with TPU / XLA."
        )

    return default_device, quantization


def default_trainer_kwargs(args: Any) -> Dict[str, Any]:
    """Return validated Trainer kwargs from the parsed CLI args."""
    accelerator = getattr(args, "accelerator", "auto")
    devices = getattr(args, "devices", "auto")
    strategy = getattr(args, "strategy", "auto")
    precision = getattr(args, "precision", "32-true")

    if accelerator == "tpu" and strategy in (None, "auto"):
        strategy = "xla"

    return {
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy,
        "precision": precision,
    }

from __future__ import annotations
 
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
 
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import lightning.pytorch as pl




class BaseTextDataset(Dataset, ABC):
    """
    Abstract dataset for causal-LM fine-tuning.
 
    Subclasses override `format_example()`, which must return either:
      • str                 → labels == input_ids (full-sequence loss)
      • (full_str, prompt)  → prompt tokens masked to -100 (response-only loss)
 
    The second form is preferred for instruction tuning because it prevents
    the model from "wasting" gradient steps memorising the prompt.
    """
 
    def __init__(
        self,
        hf_split,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
 
        if max_samples is not None:
            hf_split = hf_split.select(range(min(max_samples, len(hf_split))))
 
        formatted = [self.format_example(ex) for ex in hf_split]
 
        # Unpack into parallel lists regardless of return type
        if formatted and isinstance(formatted[0], tuple):
            texts, prompts   = zip(*formatted)
            self.texts   = list(texts)
            self.prompts = list(prompts)
        else:
            self.texts   = list(formatted)
            self.prompts = [None] * len(self.texts)

        # Store raw labels if present in the HF dataset split
        self.raw_labels = None
        if hf_split is not None:
            for col in ["label", "label_class", "target", "targets"]:
                if col in hf_split.column_names:
                    self.raw_labels = hf_split[col]
                    break
 
    @abstractmethod
    def format_example(self, example: dict) -> Union[str, Tuple[str, str]]:
        """
        Return a single training string, OR a (full_text, prompt) tuple.
        When a tuple is returned the loss on prompt tokens is masked out.
        """
        ...
 
    def __len__(self) -> int:
        return len(self.texts)
 
    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
 
        # Mask prompt tokens so loss is computed on the response only.
        # Note: BPE tokenizers can split boundary tokens differently when
        # tokenizing a substring vs. the full string, so prompt_len is an
        # approximation — accurate to within a token or two in practice.
        prompt = self.prompts[idx]
        if prompt is not None:
            prompt_len = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"].shape[-1]
            labels[:prompt_len] = -100   # CrossEntropyLoss ignores -100
 
        res = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        if self.raw_labels is not None:
            res["raw_label"] = self.raw_labels[idx]
        return res
 
 
class BaseTextDataModule(pl.LightningDataModule, ABC):
    """
    Abstract DataModule. Subclasses implement `setup()` to assign
    `self.train_ds` and `self.val_ds` using the Dataset subclasses below.
    """
 
    def __init__(
        self,
        model_name:         str            = "gpt2",
        max_length:         int            = 512,
        batch_size:         int            = 8,
        num_workers:        int            = 4,
        max_train_samples: Optional[int]  = None,
        max_val_samples:   Optional[int]  = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT-family models have no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
 
    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """
        Load and preprocess data, assigning `self.train_ds` and `self.val_ds`.
        """
        self.train_ds = None
        self.val_ds   = None
 
    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Must call setup() before requesting dataloaders"
        return DataLoader(
            self.train_ds,
            batch_size  = self.hparams.batch_size,
            shuffle     = True,
            num_workers = self.hparams.num_workers,
            pin_memory  = False,
        )
 
    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "Must call setup() before requesting dataloaders"
        return DataLoader(
            self.val_ds,
            batch_size  = self.hparams.batch_size,
            shuffle     = False,
            num_workers = self.hparams.num_workers,
            pin_memory  = False,
        )
 
 


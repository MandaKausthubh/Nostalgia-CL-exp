from typing import Optional, Tuple
from datasets_utils.base_class import BaseTextDataset, BaseTextDataModule
from datasets import load_dataset


_FLAN_PROMPT = "### System:\n{system}\n\n### User:\n{question}\n\n### Assistant:\n"
_FLAN_FULL   = _FLAN_PROMPT + "{response}"
 
 
class FLANDataset(BaseTextDataset):
    """
    Open-Orca/FLAN instruction-following dataset.
    HuggingFace columns used: system_prompt | question | response
 
    Returns (full_text, prompt) so the loss is computed only on the
    assistant response — the standard approach for instruction tuning.
    """
 
    def format_example(self, ex: dict) -> Tuple[str, str]:
        system   = (ex.get("system_prompt") or "").strip()
        question = (ex.get("question")      or "").strip()
        response = (ex.get("response")      or "").strip()
 
        prompt = _FLAN_PROMPT.format(system=system, question=question)
        full   = _FLAN_FULL.format(system=system, question=question, response=response)
        return full, prompt
 
 
class FLANDataModule(BaseTextDataModule):
    """
    DataModule for Open-Orca/FLAN.
 
    FLAN ships with a single 'train' split, so we carve 5 % off as a
    held-out validation set (seeded for reproducibility).
 
    Tip: FLAN is large (~1 M rows). Use max_train_samples during development.
    """
 
    def setup(self, stage: Optional[str] = None):
        raw  = load_dataset("Open-Orca/FLAN", split="train")
        spl  = raw.train_test_split(test_size=0.05, seed=42)
 
        self.train_ds = FLANDataset(
            spl["train"], self.tokenizer,
            self.hparams.max_length, self.hparams.max_train_samples,
        )
        self.val_ds = FLANDataset(
            spl["test"], self.tokenizer,
            self.hparams.max_length, self.hparams.max_val_samples,
        )
 
 

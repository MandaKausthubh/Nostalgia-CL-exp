from typing import Optional
from datasets import load_dataset
from datasets_utils.base_class import BaseTextDataset, BaseTextDataModule


_TREC_TEMPLATE = (
    "Classify the following question into one of the categories: "
    "Description, Entity, Abbreviation, Human, Numeric, Location.\n\n"
    "Question: {text}\n\n"
    "### Category:\n{label}"
)

_LABEL_NAMES = ["Description", "Entity", "Abbreviation", "Human", "Numeric", "Location"]


class TRECDataset(BaseTextDataset):
    """
    SetFit/TREC-QC question classification dataset.
    HuggingFace columns used: text | label_coarse (int 0-5)

    Labels: 0=Description, 1=Entity, 2=Abbreviation, 3=Human, 4=Numeric, 5=Location
    """

    def __init__(self, hf_split, tokenizer, max_length: int = 512, max_samples: Optional[int] = None):
        super().__init__(hf_split, tokenizer, max_length, max_samples)
        # Override raw_labels to be label_coarse (the 6-class target) of the truncated split
        if hf_split is not None:
            if max_samples is not None:
                hf_split = hf_split.select(range(min(max_samples, len(hf_split))))
            if "label_coarse" in hf_split.column_names:
                self.raw_labels = hf_split["label_coarse"]

    def format_example(self, ex: dict) -> str:
        text = ex["text"].strip()
        label = _LABEL_NAMES[ex["label_coarse"]]
        return _TREC_TEMPLATE.format(text=text, label=label)


class TRECDataModule(BaseTextDataModule):
    """
    DataModule for SetFit/TREC-QC.
    Has train and test splits (no validation split — we use test as val).
    """

    def setup(self, stage: Optional[str] = None):
        self.train_ds = TRECDataset(
            load_dataset("SetFit/TREC-QC", split="train"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_train_samples,
        )
        self.val_ds = TRECDataset(
            load_dataset("SetFit/TREC-QC", split="test"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_val_samples,
        )

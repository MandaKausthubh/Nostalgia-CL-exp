from typing import Optional
from datasets import load_dataset
from datasets_utils.base_class import BaseTextDataset, BaseTextDataModule


_AGNEWS_TEMPLATE = (
    "Classify the following news article into one of: "
    "World, Sports, Business, Sci/Tech.\n\n"
    "{text}\n\n"
    "### Category:\n{label}"
)

_LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


class AGNewsDataset(BaseTextDataset):
    """
    fancyzhx/ag_news topic classification dataset.
    HuggingFace columns used: text | label (int 0-3)

    Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
    """

    def format_example(self, ex: dict) -> str:
        text = ex["text"].strip()
        label = _LABEL_NAMES[ex["label"]]
        return _AGNEWS_TEMPLATE.format(text=text, label=label)


class AGNewsDataModule(BaseTextDataModule):
    """
    DataModule for fancyzhx/ag_news.
    Has official train / test splits (no validation split — we use test as val).
    """

    def setup(self, stage: Optional[str] = None):
        self.train_ds = AGNewsDataset(
            load_dataset("fancyzhx/ag_news", split="train"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_train_samples,
        )
        self.val_ds = AGNewsDataset(
            load_dataset("fancyzhx/ag_news", split="test"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_val_samples,
        )

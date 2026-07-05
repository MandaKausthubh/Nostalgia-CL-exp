from typing import Optional
from datasets import load_dataset
from datasets_utils.base_class import BaseTextDataset, BaseTextDataModule


_SST2_TEMPLATE = (
    "Classify the sentiment of the following sentence as positive or negative.\n\n"
    "{sentence}\n\n"
    "### Sentiment:\n{label}"
)

_LABEL_NAMES = ["negative", "positive"]


class SST2Dataset(BaseTextDataset):
    """
    SST-2 (GLUE) sentiment classification dataset.
    HuggingFace columns used: sentence | label (int 0-1)

    Labels: 0=negative, 1=positive
    """

    def format_example(self, ex: dict) -> str:
        sentence = ex["sentence"].strip()
        label = _LABEL_NAMES[ex["label"]]
        return _SST2_TEMPLATE.format(sentence=sentence, label=label)


class SST2DataModule(BaseTextDataModule):
    """
    DataModule for GLUE sst2.
    Has train, validation, and test splits. Since test split has no labels,
    we use validation split for validation and test evaluation.
    """

    def setup(self, stage: Optional[str] = None):
        self.train_ds = SST2Dataset(
            load_dataset("glue", "sst2", split="train"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_train_samples,
        )
        self.val_ds = SST2Dataset(
            load_dataset("glue", "sst2", split="validation"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_val_samples,
        )

from typing import Optional
from datasets import load_dataset
from datasets_utils.base_class import BaseTextDataset, BaseTextDataModule


_CNN_TEMPLATE = (
    "Summarize the following news article:\n\n"
    "{article}\n\n"
    "### Summary:\n{summary}"
)
 
 
class CNNDailyMailDataset(BaseTextDataset):
    """
    pszemraj/cnn_dailymail-cleaned summarization dataset.
    HuggingFace columns used: text (article body) | summary
 
    Falls back to the original CNN/DailyMail column names
    ('article' / 'highlights') in case of schema variation.
 
    Tip: news articles are long. Use max_length >= 1024 to avoid heavy
    truncation — at 512 tokens most articles will be cut mid-sentence.
    """
 
    def format_example(self, ex: dict) -> str:
        article = (ex.get("text")    or ex.get("article")    or "").strip()
        summary = (ex.get("summary") or ex.get("highlights") or "").strip()
        return _CNN_TEMPLATE.format(article=article, summary=summary)
 
 
class CNNDailyMailDataModule(BaseTextDataModule):
    """
    DataModule for pszemraj/cnn_dailymail-cleaned.
    Has official train / validation / test splits.
    """
 
    def setup(self, stage: Optional[str] = None):
        self.train_ds = CNNDailyMailDataset(
            load_dataset("pszemraj/cnn_dailymail-cleaned", split="train"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_train_samples,
        )
        self.val_ds = CNNDailyMailDataset(
            load_dataset("pszemraj/cnn_dailymail-cleaned", split="validation"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_val_samples,
        )
 

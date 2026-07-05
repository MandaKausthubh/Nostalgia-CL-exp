from typing import Optional
from datasets import load_dataset
from datasets_utils.base_class import BaseTextDataset, BaseTextDataModule


_DBPEDIA_TEMPLATE = (
    "Classify the following Wikipedia article into one of the categories: "
    "Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, "
    "Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work.\n\n"
    "Title: {title}\n"
    "Content: {content}\n\n"
    "### Category:\n{label}"
)

_LABEL_NAMES = [
    "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
    "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
    "Plant", "Album", "Film", "Written Work"
]


class DBpediaDataset(BaseTextDataset):
    """
    fancyzhx/dbpedia_14 ontology classification dataset.
    HuggingFace columns used: title | content | label (int 0-13)

    Labels: 0=Company, 1=Educational Institution, 2=Artist, 3=Athlete, 4=Office Holder,
            5=Mean Of Transportation, 6=Building, 7=Natural Place, 8=Village, 9=Animal,
            10=Plant, 11=Album, 12=Film, 13=Written Work
    """

    def format_example(self, ex: dict) -> str:
        title = ex["title"].strip()
        content = ex["content"].strip()
        label = _LABEL_NAMES[ex["label"]]
        return _DBPEDIA_TEMPLATE.format(title=title, content=content, label=label)


class DBpediaDataModule(BaseTextDataModule):
    """
    DataModule for fancyzhx/dbpedia_14.
    Has official train and test splits (no validation split — we use test as val).
    """

    def setup(self, stage: Optional[str] = None):
        self.train_ds = DBpediaDataset(
            load_dataset("fancyzhx/dbpedia_14", split="train"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_train_samples,
        )
        self.val_ds = DBpediaDataset(
            load_dataset("fancyzhx/dbpedia_14", split="test"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_val_samples,
        )

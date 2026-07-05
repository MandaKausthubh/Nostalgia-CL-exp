import torch
from torch.utils.data import Dataset


class TaskClassificationDataset(Dataset):
    """
    Wraps a base dataset to:
    1. Assign integer target labels for classification.
    2. Remove the "labels" key to prevent conflicting target lookups in _shared_step.
    """
    def __init__(self, base_dataset, num_classes):
        self.base_dataset = base_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if "labels" in item:
            del item["labels"]

        if "raw_label" in item:
            item["target"] = torch.tensor(item["raw_label"] % self.num_classes, dtype=torch.long)
            del item["raw_label"]
        else:
            # For summarization/QA tasks without natural labels, construct a deterministic
            # and learnable semantic target based on the variable input text content.
            # We strip the static template prefix and use the first alphanumeric character.
            text = getattr(self.base_dataset, "texts", [None])[idx]
            if text is None:
                val = idx
            else:
                # Strip known static prefixes to get the variable content
                for prefix in [
                    "Summarize the following news article:\n\n",
                    "Context:\n",
                    "Classify the following news article into one of: World, Sports, Business, Sci/Tech.\n\n",
                    "Classify the sentiment of the following sentence as positive or negative.\n\n",
                    "Classify the following question into one of the categories: Description, Entity, Abbreviation, Human, Numeric, Location.\n\n",
                    "Classify the following Wikipedia article into one of the categories: Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work.\n\n"
                ]:
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                        break
                
                text_clean = text.strip()
                char = 'a'
                for c in text_clean:
                    if c.isalnum():
                        char = c.lower()
                        break
                val = ord(char)
                
            item["target"] = torch.tensor(val % self.num_classes, dtype=torch.long)

        return item

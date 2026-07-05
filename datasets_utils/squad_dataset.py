from typing import Optional
from datasets import load_dataset
from datasets_utils.base_class import BaseTextDataset, BaseTextDataModule


_SQUAD_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer: {answer}"
)
 
# SQuAD v2 introduced ~50 % unanswerable questions (answers["text"] == []).
# We replace them with an explicit natural-language refusal string so the
# model learns to say "I don't know" rather than hallucinating an answer.
_UNANSWERABLE = "This question cannot be answered from the given context."
 
 
class SQuADv2Dataset(BaseTextDataset):
    """
    rajpurkar/squad_v2 extractive QA dataset.
    HuggingFace columns used: context | question | answers (dict)
 
    answers["text"] is a list of valid spans; an empty list means the
    question is unanswerable (new in v2 vs v1).
    """
 
    def format_example(self, ex: dict) -> str:
        context  = ex["context"].strip()
        question = ex["question"].strip()
        spans    = ex["answers"]["text"]
        # Use the first annotator's answer, or the refusal string
        answer   = spans[0].strip() if spans else _UNANSWERABLE
        return _SQUAD_TEMPLATE.format(context=context, question=question, answer=answer)
 
 
class SQuADv2DataModule(BaseTextDataModule):
    """
    DataModule for rajpurkar/squad_v2.
    Has official train / validation splits (no public test labels).
    """
 
    def setup(self, stage: Optional[str] = None):
        self.train_ds = SQuADv2Dataset(
            load_dataset("rajpurkar/squad_v2", split="train"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_train_samples,
        )
        self.val_ds = SQuADv2Dataset(
            load_dataset("rajpurkar/squad_v2", split="validation"),
            self.tokenizer,
            self.hparams.max_length, self.hparams.max_val_samples,
        )
 
 

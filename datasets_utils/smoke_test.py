from datasets_utils.flan_dataset import FLANDataModule
from datasets_utils.squad_dataset import SQuADv2DataModule
from datasets_utils.cnn_datast import CNNDailyMailDataModule
from datasets_utils.base_class import BaseTextDataModule


# Smoke test to verify datasets load and tokenize without errors.
if __name__ == "__main__":
 
    def _inspect(name: str, dm: BaseTextDataModule) -> None:
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        shapes = {k: tuple(v.shape) for k, v in batch.items()}
        n_masked = (batch["labels"][0] == -100).sum().item()
        print(f"\n{'─'*55}")
        print(f"  {name}")
        print(f"  train rows : {len(dm.train_ds):,}  |  val rows : {len(dm.val_ds):,}")
        print(f"  batch      : {shapes}")
        print(f"  labels[-100] in example[0]: {n_masked} tokens masked")
 
    # A) FLAN ──────────────────────────────────────────────────────────────────
    # _inspect("Open-Orca/FLAN", FLANDataModule(
    #     model_name="gpt2",
    #     max_length=512,
    #     batch_size=4,
    #     max_train_samples=1_000,
    #     max_val_samples=100,
    # ))
 
    # B) CNN DailyMail ─────────────────────────────────────────────────────────
    _inspect("pszemraj/cnn_dailymail-cleaned", CNNDailyMailDataModule(
        model_name="gpt2",
        max_length=1024,         # articles are long — 512 truncates heavily
        batch_size=4,
        max_train_samples=1_000,
        max_val_samples=100,
    ))
 
    # C) SQuAD v2 ──────────────────────────────────────────────────────────────
    _inspect("rajpurkar/squad_v2", SQuADv2DataModule(
        model_name="gpt2",
        max_length=512,
        batch_size=4,
        max_train_samples=1_000,
        max_val_samples=100,
    ))
 





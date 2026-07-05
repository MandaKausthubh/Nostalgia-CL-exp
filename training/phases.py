"""Shared training utilities — progress bar and memory helpers."""

import sys
from tqdm.auto import tqdm
from lightning.pytorch.callbacks import TQDMProgressBar


class SimpleProgressBar(TQDMProgressBar):
    """Minimal tqdm.auto progress bar — works in both terminals and notebooks."""

    def _make_tqdm(self, desc, leave=True):
        disable = self.is_disabled or (self.trainer is not None and not self.trainer.is_global_zero)
        return tqdm(desc=desc, disable=disable, leave=leave,
                    file=sys.stdout, smoothing=0, bar_format=self.BAR_FORMAT)

    def init_train_tqdm(self):
        return self._make_tqdm(self.train_description)

    def init_validation_tqdm(self):
        return tqdm(disable=True)

    def init_test_tqdm(self):
        return self._make_tqdm("Testing")

    def init_predict_tqdm(self):
        return self._make_tqdm(self.predict_description)

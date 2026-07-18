import wandb
from lightning.pytorch.loggers import WandbLogger as PLWandbLogger


class NostalgiaWandbLogger(PLWandbLogger):
    """WandbLogger that maps trainer-local steps to persistent phase counters.

    Phase 1 (head_align) is logged normally.
    Phase 2 (nostalgia) uses ``global_step_counter`` as the wandb step.
    Phase 3 is suppressed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def log_metrics(self, metrics, step=None):
        if self.model is not None:
            if getattr(self.model, "logging_disabled", False):
                return
            phase = getattr(self.model, "training_phase", None)
            if phase == "nostalgia":
                step = getattr(self.model, "global_step_counter", step)
            elif phase == "head_align":
                step = getattr(self.model, "alignment_step_counter", step)
        super().log_metrics(metrics, step=step)


class WandbSummaryWriterWrapper:
    """Translates ``add_scalars`` calls into ``wandb.log()`` so the
    NostalgiaOptimizer can log projection metrics without depending on
    TensorBoard.

    Only logs on the rank that owns the wandb run (rank 0 on TPU).
    """

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        model = getattr(self, "model", None)
        if model is not None and getattr(model, "logging_disabled", False):
            return
        if wandb.run is None:
            return  # non-rank-0 processes don't have a wandb run
        log_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        if global_step is not None:
            log_dict["nostalgia/global_step"] = int(global_step)
        wandb.log(log_dict)

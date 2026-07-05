import lightning.pytorch as pl


class NostalgiaCallback(pl.Callback):
    def __init__(
        self,
        hessian_dataloader,
        epochs_per_cycle: int = 5,
        k: int = 20,
        warmup_cycles: int = 1,   # skip Q update on the very first cycle
    ):
        self.hessian_dataloader = hessian_dataloader
        self.epochs_per_cycle = epochs_per_cycle
        self.k = k
        self.warmup_cycles = warmup_cycles
        self.cycle_count = 0
        self._epoch_in_cycle = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self._epoch_in_cycle += 1

        if self._epoch_in_cycle < self.epochs_per_cycle:
            return

        # Cycle complete
        self._epoch_in_cycle = 0
        self.cycle_count += 1

        if self.cycle_count <= self.warmup_cycles:
            print(f"[Nostalgia] Cycle {self.cycle_count} complete — still in warmup, skipping Q update.")
            return

        print(f"[Nostalgia] Cycle {self.cycle_count} complete — updating Q.")
        self._update_Q(trainer, pl_module)

    def _update_Q(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        was_training = pl_module.training
        pl_module.eval()

        try:
            Q, scaling = compute_hessian_eigenvectors(
                pl_module,
                self.hessian_dataloader,
                k=self.k,
            )
            optimizer.zero_grad()
            optimizer.set_Q(Q.detach(), scaling.detach())
        finally:
            pl_module.train(was_training)

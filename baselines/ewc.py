"""EWC (Elastic Weight Consolidation).

Adds a quadratic penalty around the previous task's solution:
    L = L_task + (lam / 2) * sum_i  fisher_i * (theta_i - theta_star_i)^2

Gradient contribution:  g += lam * fisher * (theta - theta_star)

The penalty is applied to backbone parameters only (heads are task-specific
and do not need consolidation).

Implemented as an optimizer wrapper parallel to `NostalgiaOptimizer`.
"""

from typing import List, Optional, Any, Dict
import torch
from torch.optim.optimizer import Optimizer


class EWCOptimizer(Optimizer):
    """Wraps a base optimizer and injects the EWC penalty into gradients.

    Args:
        params:            list of backbone parameters that EWC regularizes.
        base_optimizer:   underlying optimizer (Adam/AdamW/SGD).
        device:           torch device.
        dtype:            compute dtype for the penalty.
        fisher:           dict {id(p): tensor same shape as p} — Fisher diagonal.
                          None disables the penalty (first task).
        theta_star:       dict {id(p): tensor same shape as p} — previous solution.
        lam:              EWC strength (scalar).
        writer:           optional wandb writer for logging penalty magnitude.
        log_every:         log cadence.
        starting_step:    initial global step (for log x-axis continuity).
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        base_optimizer: Optimizer,
        device: torch.device,
        dtype: torch.dtype,
        fisher: Optional[Dict[int, torch.Tensor]] = None,
        theta_star: Optional[Dict[int, torch.Tensor]] = None,
        lam: float = 400.0,
        writer: Optional[Any] = None,
        log_every: int = 50,
        starting_step: int = 0,
    ):
        super().__init__(params, {})
        self.base_optimizer = base_optimizer
        self.projection_params = list(params)
        self.device = device
        self.dtype = dtype
        self.fisher = fisher
        self.theta_star = theta_star
        self.lam = float(lam)
        self.writer = writer
        self.log_every = log_every
        self.step_count = starting_step
        self._last_penalty: Optional[float] = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_state(self, fisher: Dict[int, torch.Tensor], theta_star: Dict[int, torch.Tensor]):
        """Attach Fisher and previous-solution tensors (called by the callback)."""
        self.fisher = {
            pid: f.to(self.device, self.dtype) for pid, f in fisher.items()
        }
        self.theta_star = {
            pid: t.to(self.device, self.dtype) for pid, t in theta_star.items()
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_penalty(self):
        """Add lam * fisher * (theta - theta_star) to each param's .grad."""
        if self.fisher is None or self.theta_star is None:
            return None

        penalty_norm_sq = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for p in self.projection_params:
            if p.grad is None:
                continue
            f = self.fisher.get(id(p))
            ts = self.theta_star.get(id(p))
            if f is None or ts is None:
                continue
            # (theta - theta_star), element-wise
            delta = p.data - ts
            # penalty gradient = lam * fisher * delta
            penalty = self.lam * (f * delta)
            p.grad.add_(penalty)
            penalty_norm_sq = penalty_norm_sq + (penalty * penalty).sum()

        return penalty_norm_sq.item() if penalty_norm_sq.numel() > 0 else 0.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):  # type: ignore
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        penalty_val = self._apply_penalty()
        self._last_penalty = penalty_val

        loss_out = self.base_optimizer.step()
        if loss is None:
            loss = loss_out

        # Log penalty magnitude
        if (
            self.writer is not None
            and penalty_val is not None
            and self.step_count % self.log_every == 0
        ):
            try:
                self.writer.add_scalars(
                    "EWC", {"penalty": float(penalty_val)}, self.step_count
                )
            except Exception:
                pass

        self.step_count += 1
        return loss

    # ------------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        sd = self.base_optimizer.state_dict()
        sd["ewc_step_count"] = self.step_count
        return sd

    def load_state_dict(self, state_dict):
        if "ewc_step_count" in state_dict:
            self.step_count = state_dict["ewc_step_count"]
        self.base_optimizer.load_state_dict(state_dict)

    # Mirror NostalgiaOptimizer property forwarding so Lightning can drive it.
    @property
    def param_groups(self):
        if not hasattr(self, "base_optimizer"):
            return []
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        if not hasattr(self, "base_optimizer"):
            self.__dict__["param_groups"] = value
        else:
            self.base_optimizer.param_groups = value

    @property
    def defaults(self):
        if not hasattr(self, "base_optimizer"):
            return {}
        return self.base_optimizer.defaults

    @defaults.setter
    def defaults(self, value):
        if not hasattr(self, "base_optimizer"):
            self.__dict__["defaults"] = value
        else:
            self.base_optimizer.defaults = value

    @property
    def state(self):
        if not hasattr(self, "base_optimizer"):
            return getattr(self, "_dummy_state", {})
        return self.base_optimizer.state

    @state.setter
    def state(self, value):
        if not hasattr(self, "base_optimizer"):
            self._dummy_state = value
        else:
            self.base_optimizer.state = value


# ---------------------------------------------------------------------------
# Fisher diagonal estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_targets(batch):
    if isinstance(batch, dict):
        return batch.get("target", batch.get("labels", batch.get("label")))
    if isinstance(batch, (list, tuple)):
        return batch[-1]
    return None


def compute_fisher_diagonal(model, loader, device, max_batches: Optional[int] = None) -> Dict[int, torch.Tensor]:
    """Empirical Fisher diagonal: F_i = E[(dL/dtheta_i)^2].

    Uses squared per-sample gradients averaged over the task-end loader.
    Only backbone (trainable) parameters are recorded.

    Args:
        model:   LightningModule with `.backbone` and `get_backbone_params_dict()`.
        loader:  DataLoader for the task that just finished.
        device:  accelerator device.
        max_batches: optional cap on number of batches used.

    Returns:
        dict {id(p): tensor same shape as p} holding the Fisher diagonal.
    """
    model.eval()
    backbone_params = {name: p for name, p in model.get_backbone_params_dict().items()}
    param_ids = {id(p) for p in backbone_params.values()}

    fisher: Dict[int, torch.Tensor] = {
        id(p): torch.zeros_like(p, device=device, dtype=torch.float32)
        for p in backbone_params.values()
    }

    n_samples = 0
    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        # Build inputs/targets on device
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            targets = batch.get("target", batch.get("labels", batch.get("label"))).to(device)
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                input_ids, attention_mask, targets = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device) if attention_mask is not None else None
                targets = targets.to(device)
            else:
                input_ids, targets = batch
                input_ids = input_ids.to(device)
                attention_mask = None
                targets = targets.to(device)
        else:
            continue

        bs = input_ids.shape[0]
        # Per-sample squared gradients: loop over the batch to get true empirical Fisher.
        # (Cheap enough for a task-end estimation pass.)
        for i in range(bs):
            model.zero_grad()
            logits = model(
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1] if attention_mask is not None else None,
                task_name=model.active_task,
            )
            loss = model.criterion(logits, targets[i:i+1])
            loss.backward()

            for p in backbone_params.values():
                if p.grad is None:
                    continue
                fisher[id(p)].add_(p.grad.detach().float() ** 2)

            n_samples += 1

    if n_samples > 0:
        for pid in fisher:
            fisher[pid].div_(n_samples)

    # Snapshot theta_star = current backbone params (detached clones)
    model.zero_grad()
    model.train()
    return fisher


def snapshot_theta_star(model, device) -> Dict[int, torch.Tensor]:
    """Snapshot current backbone parameters as the EWC anchor."""
    return {
        id(p): p.detach().to(device, dtype=torch.float32).clone()
        for p in model.get_backbone_params_dict().values()
    }
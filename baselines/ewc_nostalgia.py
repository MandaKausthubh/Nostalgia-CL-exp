"""EWC + Nostalgia hybrid.

Uses the accumulated Hessian eigenspace (Q, Lambda) as a full low-rank
quadratic penalty instead of the diagonal Fisher used by EWC:

    L_reg = (lam / 2) * (theta - theta_star)^T (Q diag(Lambda) Q^T) (theta - theta_star)

Gradient contribution on the flattened backbone gradient:

    g_reg = lam * Q diag(Lambda) Q^T (theta - theta_star)

This is EWC's quadratic anchor loss, but the curvature is the global
Hessian eigenspace accumulated across past tasks rather than a diagonal
Fisher approximation. The optimizer otherwise behaves as AdamW/SGD on the
task loss plus this regularizer.
"""

from typing import List, Optional, Any, Dict
import torch
from torch.optim.optimizer import Optimizer


class EWCNostalgiaOptimizer(Optimizer):
    """Wraps a base optimizer and injects a low-rank Hessian quadratic penalty.

    Args:
        params:          backbone parameters to regularize.
        base_optimizer:  underlying optimizer.
        device:          torch device.
        dtype:           compute dtype for the penalty.
        Q:               (N, k) orthonormal accumulated eigenspace, or None.
        Lambda:          (k,) accumulated eigenvalues, or None.
        theta_star:      dict {id(p): tensor} previous backbone solution.
        lam:             penalty strength.
        writer:          optional wandb writer.
        log_every:       log cadence.
        starting_step:   initial global step.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        base_optimizer: Optimizer,
        device: torch.device,
        dtype: torch.dtype,
        Q: Optional[torch.Tensor] = None,
        Lambda: Optional[torch.Tensor] = None,
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
        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.nostalgia_Lambda: Optional[torch.Tensor] = None
        self.theta_star = theta_star
        self.lam = float(lam)
        self.writer = writer
        self.log_every = log_every
        self.step_count = starting_step
        self._last_penalty: Optional[float] = None

        self.param_numels = [p.numel() for p in self.projection_params]
        self.num_params = sum(self.param_numels)

        if Q is not None and Lambda is not None:
            self.set_Q(Q, Lambda)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_Q(self, Q: torch.Tensor, Lambda: torch.Tensor):
        if Q.shape[0] != self.num_params:
            raise ValueError(
                f"Q has {Q.shape[0]} rows, expected {self.num_params} "
                f"(sum of projection parameter sizes)."
            )
        self.nostalgia_Q = Q.to(self.device, self.dtype)
        self.nostalgia_Lambda = Lambda.to(self.device, self.dtype)

    @torch.no_grad()
    def set_theta_star(self, theta_star: Dict[int, torch.Tensor]):
        self.theta_star = {
            pid: t.to(self.device, self.dtype) for pid, t in theta_star.items()
        }

    # ------------------------------------------------------------------
    def _flatten_params(self) -> torch.Tensor:
        flat = []
        for p in self.projection_params:
            flat.append(p.data.view(-1).to(self.device, self.dtype))
        return torch.cat(flat)

    def _flatten_theta_star(self) -> torch.Tensor:
        flat = []
        for p in self.projection_params:
            ts = self.theta_star.get(id(p)) if self.theta_star is not None else None
            if ts is None:
                flat.append(torch.zeros(p.numel(), device=self.device, dtype=self.dtype))
            else:
                flat.append(ts.view(-1).to(self.device, self.dtype))
        return torch.cat(flat)

    def _unflatten_to_grads(self, flat_grad: torch.Tensor):
        pointer = 0
        for p, n in zip(self.projection_params, self.param_numels):
            slice_ = flat_grad[pointer:pointer + n].view_as(p).to(p.dtype)
            if p.grad is None:
                p.grad = slice_.clone()
            else:
                p.grad.add_(slice_)
            pointer += n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_penalty(self) -> Optional[float]:
        """Add lam * Q Lambda Q^T (theta - theta_star) to .grad."""
        if self.nostalgia_Q is None or self.nostalgia_Lambda is None or self.theta_star is None:
            return None

        flat_theta = self._flatten_params()
        flat_theta_star = self._flatten_theta_star()
        delta = flat_theta - flat_theta_star  # (N,)

        # Q^T delta  -> (k,)
        coeffs = self.nostalgia_Q.T @ delta
        # scale by eigenvalues
        scaled = coeffs * self.nostalgia_Lambda
        # lift back to parameter space
        penalty = self.lam * (self.nostalgia_Q @ scaled)

        if torch.isnan(penalty).any() or torch.isinf(penalty).any():
            print("[EWCNostalgiaOptimizer] WARNING: NaN/Inf in penalty, skipping")
            return None

        self._unflatten_to_grads(penalty)
        return (penalty * penalty).sum().item()

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

        if (
            self.writer is not None
            and penalty_val is not None
            and self.step_count % self.log_every == 0
        ):
            try:
                self.writer.add_scalars(
                    "EWC_Nostalgia", {"penalty": float(penalty_val)}, self.step_count
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
        sd["ewc_nostalgia_step_count"] = self.step_count
        return sd

    def load_state_dict(self, state_dict):
        if "ewc_nostalgia_step_count" in state_dict:
            self.step_count = state_dict["ewc_nostalgia_step_count"]
        self.base_optimizer.load_state_dict(state_dict)

    # Property forwarding (same pattern as NostalgiaOptimizer / EWCOptimizer).
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

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Any
from utils.hessians import *


class NostalgiaOptimizer(Optimizer):
    """
    Wraps a base optimizer and applies a Nostalgia-style gradient projection:
        g' = g - Q (Q^T g)

    Projection is applied ONLY to the specified projection_params.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        base_optimizer: Optimizer,
        device: torch.device,
        dtype: torch.dtype,
        writter: Optional[Any] = None,
        starting_step: int = 0,
        log_every: int = 50,
    ):
        super().__init__(params, {})  # Dummy call to satisfy Optimizer base class
        self.base_optimizer = base_optimizer


        self.projection_params = list(params)
        self.device = device
        self.dtype = dtype

        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.scaling: Optional[torch.Tensor] = None
        self.writter = writter

        self.log_every = log_every
        self.ema_beta = 0.98
        self.proj_ratio_ema: Optional[float] = None
        self.step_count = starting_step

        # Fixed parameter layout (ordering matters!)
        self.param_numels = [p.numel() for p in self.projection_params]
        self.num_params = sum(self.param_numels)
        self.k_max: Optional[int] = None

    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_Q(self, Q: Optional[torch.Tensor], scaling: Optional[torch.Tensor] = None):
        """
        Q: [num_params, k] matrix of eigenvectors, or None to disable projection.
        scaling: optional [k] or [k, k] eigenvalue-based scaling
        """
        if Q is None:
            self.nostalgia_Q = None
            self.scaling = None
            return

        print(
            "[set_Q before copy]",
            torch.isfinite(Q).all().item(),
            Q.abs().max().item()
        )

        if Q.shape[0] != self.num_params:
            raise ValueError(
                f"Q has {Q.shape[0]} rows, expected {self.num_params} "
                f"(sum of projection parameter sizes)."
            )
        self.nostalgia_Q = Q.to(self.device, self.dtype)
        self.scaling = scaling.to(self.device, self.dtype) if scaling is not None else None

        print(
            "[set_Q after copy]",
            torch.isfinite(self.nostalgia_Q).all().item(),
            self.nostalgia_Q.abs().max().item()
        )

    # ------------------------------------------------------------------
    def _flatten_grads(self) -> torch.Tensor:
        """
        Flatten gradients of ALL projection params in fixed order.
        Missing gradients are treated as zeros.
        """
        flat_grads = []
        for p in self.projection_params:
            if p.grad is None:
                flat_grads.append(torch.zeros(
                    p.numel(), device=self.device, dtype=self.dtype
                ))
            else:
                flat_grads.append(p.grad.view(-1))
        return torch.cat(flat_grads)

    # ------------------------------------------------------------------
    def _unflatten_to_grads(self, flat_grad: torch.Tensor):
        """
        Write projected flat gradient back into parameter .grad fields.
        """
        pointer = 0
        for p, n in zip(self.projection_params, self.param_numels):
            grad_slice = flat_grad[pointer:pointer + n].view_as(p)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)
            pointer += n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):  # type: ignore
        # 1. Save parameter values before standard optimizer step
        params_before = [p.detach().clone() for p in self.projection_params]

        # 2. Execute base optimizer step (AdamW, etc.)
        loss = self.base_optimizer.step(closure)

        # 3. Apply projection to the actual parameter update if Q is injected
        try:
            if self.nostalgia_Q is not None:
                updates = []
                for p, p_before in zip(self.projection_params, params_before):
                    updates.append((p - p_before).view(-1))
                flat_updates = torch.cat(updates)

                if torch.isnan(flat_updates).any() or torch.isinf(flat_updates).any():
                    print("[NostalgiaOptimizer] WARNING: NaN/Inf in parameter updates, skipping projection")
                    return loss

                coeffs = self.nostalgia_Q.T @ flat_updates

                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    print("[NostalgiaOptimizer] WARNING: NaN/Inf in coefficients, skipping projection")
                    return loss

                # Optional eigenvalue-aware scaling
                if self.scaling is not None:
                    c_scaling = torch.median(self.scaling) + 1e-12
                    if self.scaling.ndim == 1:
                        coeffs = coeffs * (self.scaling / (c_scaling + self.scaling))
                    else:
                        coeffs = (self.scaling / (c_scaling + self.scaling)) @ coeffs

                projection = self.nostalgia_Q @ coeffs

                if torch.isnan(projection).any() or torch.isinf(projection).any():
                    print("[NostalgiaOptimizer] WARNING: NaN/Inf in projection, skipping")
                    return loss

                flat_updates_projected = flat_updates - projection

                if torch.isnan(flat_updates_projected).any() or torch.isinf(flat_updates_projected).any():
                    print("[NostalgiaOptimizer] WARNING: NaN/Inf in projected updates, skipping")
                    return loss

                # Write the projected parameters back
                pointer = 0
                for p, p_before, n in zip(self.projection_params, params_before, self.param_numels):
                    update_slice = flat_updates_projected[pointer:pointer + n].view_as(p)
                    p.copy_(p_before + update_slice)
                    pointer += n

                # Log projection ratio to wandb
                if self.writter is not None and self.step_count % self.log_every == 0:
                    ratio = (torch.norm(flat_updates_projected) / (torch.norm(flat_updates) + 1e-12)).item()
                    if self.proj_ratio_ema is None:
                        self.proj_ratio_ema = ratio
                    else:
                        self.proj_ratio_ema = self.ema_beta * self.proj_ratio_ema + (1 - self.ema_beta) * ratio

                    self.writter.add_scalars('Nostalgia', {
                        'Projection_Ratio': ratio,
                        'Projection_Ratio_EMA': self.proj_ratio_ema,
                    }, self.step_count)

        except Exception as e:
            print(f"[NostalgiaOptimizer] ERROR during projection step: {e}")

        finally:
            self.step_count += 1
        return loss

    # ------------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        sd = self.base_optimizer.state_dict()
        sd['nostalgia_Q'] = self.nostalgia_Q
        sd['scaling'] = self.scaling
        sd['step_count'] = self.step_count
        return sd

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

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




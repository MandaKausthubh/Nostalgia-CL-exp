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
        alpha: float = 1.0,
    ):
        super().__init__(params, {})  # Dummy call to satisfy Optimizer base class
        self.base_optimizer = base_optimizer


        self.projection_params = list(params)
        self.device = device
        self.dtype = dtype

        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.scaling: Optional[torch.Tensor] = None
        self.writter = writter

        self.alpha = alpha
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
    def _project_gradients(self) -> Optional[torch.Tensor]:
        """
        Project gradients onto the null-space of the remembered eigenspace:
            g' = g - alpha * Q (Q^T g)
        alpha=1.0 is hard null-space projection; alpha<1.0 softens plasticity loss.
        Writes the projected gradients back into parameter .grad fields and
        returns the flattened original gradient (for logging).
        """
        if self.nostalgia_Q is None:
            return None

        flat_grads = self._flatten_grads()

        if torch.isnan(flat_grads).any() or torch.isinf(flat_grads).any():
            print("[NostalgiaOptimizer] WARNING: NaN/Inf in gradients, skipping projection")
            return flat_grads

        coeffs = self.nostalgia_Q.T @ flat_grads

        if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
            print("[NostalgiaOptimizer] WARNING: NaN/Inf in Q^T g coefficients, skipping projection")
            return flat_grads

        # Optional eigenvalue-aware scaling
        if self.scaling is not None:
            c_scaling = torch.median(self.scaling) + 1e-12
            if self.scaling.ndim == 1:
                coeffs = coeffs * (self.scaling / (c_scaling + self.scaling))
            else:
                coeffs = (self.scaling / (c_scaling + self.scaling)) @ coeffs

        projection = self.alpha * (self.nostalgia_Q @ coeffs)

        if torch.isnan(projection).any() or torch.isinf(projection).any():
            print("[NostalgiaOptimizer] WARNING: NaN/Inf in projection, skipping")
            return flat_grads

        flat_grads_projected = flat_grads - projection

        if torch.isnan(flat_grads_projected).any() or torch.isinf(flat_grads_projected).any():
            print("[NostalgiaOptimizer] WARNING: NaN/Inf in projected gradients, skipping")
            return flat_grads

        self._unflatten_to_grads(flat_grads_projected)
        return flat_grads

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):  # type: ignore
        # Lightning passes a closure that computes the forward/backward pass.
        # Run it first so the accumulated gradients in p.grad are up-to-date.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Project raw gradients before the base optimizer consumes them.
        flat_grads_before = None
        try:
            flat_grads_before = self._project_gradients()
        except Exception as e:
            print(f"[NostalgiaOptimizer] ERROR during gradient projection: {e}")

        # Execute base optimizer step on the (possibly projected) gradients.
        # Call without closure because we already executed it above.
        loss_out = self.base_optimizer.step()
        if loss is None:
            loss = loss_out

        # Log projection ratio to wandb.
        if flat_grads_before is not None and self.nostalgia_Q is not None:
            if self.writter is not None and self.step_count % self.log_every == 0:
                flat_grads_after = self._flatten_grads()  # projected gradient currently in .grad
                ratio = (torch.norm(flat_grads_after) / (torch.norm(flat_grads_before) + 1e-12)).item()
                if self.proj_ratio_ema is None:
                    self.proj_ratio_ema = ratio
                else:
                    self.proj_ratio_ema = self.ema_beta * self.proj_ratio_ema + (1 - self.ema_beta) * ratio

                self.writter.add_scalars('Nostalgia', {
                    'Projection_Ratio': ratio,
                    'Projection_Ratio_EMA': self.proj_ratio_ema,
                }, self.step_count)

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




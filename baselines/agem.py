"""A-GEM (Average Gradient Episodic Memory).

Maintains a replay buffer of examples from past tasks. At each step:
    1. compute task gradient g on the current batch
    2. compute reference gradient g_ref on a random replay batch
    3. if <g, g_ref> < 0  ->  g -= (<g, g_ref> / <g_ref, g_ref>) * g_ref
       else               ->  g unchanged
    4. base optimizer steps on (possibly projected) g

Projection operates on flattened backbone gradients only.
"""

from typing import List, Optional, Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import random


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Ring buffer of (input_ids, attention_mask, label, task) tuples.

    Stores CPU tensors so device memory is unaffected between steps.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._store: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, str]] = []
        self._pos = 0  # next write index

    def __len__(self):
        return len(self._store)

    def add(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
            label: torch.Tensor, task: str):
        item = (
            input_ids.detach().cpu(),
            None if attention_mask is None else attention_mask.detach().cpu(),
            label.detach().cpu(),
            str(task),
        )
        if len(self._store) < self.capacity:
            self._store.append(item)
        else:
            self._store[self._pos] = item
            self._pos = (self._pos + 1) % self.capacity

    def sample_batch(self, batch_size: int, device: torch.device, max_length: Optional[int] = None
                     ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, str]]:
        """Random sample a replay batch; returns None if buffer empty."""
        if len(self._store) == 0:
            return None
        bs = min(batch_size, len(self._store))
        indices = random.sample(range(len(self._store)), bs)
        input_ids_list = []
        attn_list = []
        label_list = []
        task_names = set()
        for idx in indices:
            ii, am, lb, tk = self._store[idx]
            input_ids_list.append(ii)
            attn_list.append(am)
            label_list.append(lb)
            task_names.add(tk)

        input_ids = torch.stack(input_ids_list, dim=0).to(device)
        # label_list contains 0-dim scalars (labels[i]); stack -> 1-d
        labels = torch.stack(label_list, dim=0).to(device)
        # attention masks may have varying shapes; pad to max length across the batch.
        if attn_list[0] is not None:
            max_len = max(am.shape[-1] for am in attn_list)
            padded = torch.zeros((bs, max_len), dtype=attn_list[0].dtype)
            for i, am in enumerate(attn_list):
                L = am.shape[-1]
                padded[i, :L] = am
            attention_mask = padded.to(device)
            # Also pad input_ids if needed (shouldn't be — both come from same tokenizer)
            if input_ids.shape[-1] < max_len:
                pad_len = max_len - input_ids.shape[-1]
                pad = torch.zeros((bs, pad_len), dtype=input_ids.dtype, device=device)
                input_ids = torch.cat([input_ids, pad], dim=-1)
        else:
            attention_mask = None

        # Pick a task name (A-GEM averages across all past tasks; any single task works for the ref gradient)
        task_name = next(iter(task_names))
        return input_ids, attention_mask, labels, task_name


def fill_replay_buffer(model, loader, mem_size: int, task_name: str) -> ReplayBuffer:
    """Fill a replay buffer with up to `mem_size` examples from `loader`.

    One example = one row of a batch (input_ids[i], attention_mask[i], label[i]).
    """
    buf = ReplayBuffer(mem_size)
    for batch in loader:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch.get("target", batch.get("labels", batch.get("label")))
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
            else:
                input_ids, labels = batch
                attention_mask = None
        else:
            continue

        for i in range(input_ids.shape[0]):
            if len(buf) >= mem_size:
                return buf
            am_i = attention_mask[i] if attention_mask is not None else None
            buf.add(input_ids[i], am_i, labels[i], task_name)

    return buf


# ---------------------------------------------------------------------------
# A-GEM optimizer
# ---------------------------------------------------------------------------

class AGEMOptimizer(Optimizer):
    """Wraps a base optimizer and applies A-GEM gradient projection.

    Args:
        params:          backbone parameters to project.
        base_optimizer:  underlying optimizer.
        device:          torch device.
        dtype:           compute dtype.
        replay_buffer:   ReplayBuffer or None (first task -> no projection).
        replay_bs:       batch size sampled from the replay buffer per step.
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
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_bs: int = 8,
        writer: Optional[Any] = None,
        log_every: int = 50,
        starting_step: int = 0,
    ):
        super().__init__(params, {})
        self.base_optimizer = base_optimizer
        self.projection_params = list(params)
        self.device = device
        self.dtype = dtype
        self.replay_buffer = replay_buffer
        self.replay_bs = replay_bs
        self.writer = writer
        self.log_every = log_every
        self.step_count = starting_step
        self.proj_count = 0
        self.param_numels = [p.numel() for p in self.projection_params]
        self.num_params = sum(self.param_numels)
        self.model_ref: Optional[Any] = None  # set by set_model_ref so we can forward replay batches

    # ------------------------------------------------------------------
    def set_model_ref(self, model):
        """Attach a reference to the LightningModule so we can run forward/backward on replay batches."""
        self.model_ref = model

    def set_replay_buffer(self, buf: ReplayBuffer):
        self.replay_buffer = buf

    # ------------------------------------------------------------------
    def _flatten_grads(self) -> torch.Tensor:
        flat = []
        for p in self.projection_params:
            if p.grad is None:
                flat.append(torch.zeros(p.numel(), device=self.device, dtype=self.dtype))
            else:
                flat.append(p.grad.view(-1).to(self.dtype))
        return torch.cat(flat)

    def _unflatten_to_grads(self, flat_grad: torch.Tensor):
        pointer = 0
        for p, n in zip(self.projection_params, self.param_numels):
            slice_ = flat_grad[pointer:pointer + n].view_as(p)
            if p.grad is None:
                p.grad = slice_.clone().to(p.dtype)
            else:
                p.grad.copy_(slice_.to(p.dtype))
            pointer += n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_replay_gradient(self) -> Optional[torch.Tensor]:
        """Forward + backward on a random replay batch; return flattened backbone grad or None."""
        if self.replay_buffer is None or len(self.replay_buffer) == 0 or self.model_ref is None:
            return None

        sample = self.replay_buffer.sample_batch(self.replay_bs, self.device)
        if sample is None:
            return None
        input_ids, attention_mask, labels, task_name = sample

        # Zero grads of backbone params only (heads are separate, not projected)
        for p in self.projection_params:
            if p.grad is not None:
                p.grad.zero_()

        model = self.model_ref
        saved_task = model.active_task
        saved_phase = model.training_phase
        saved_logging = getattr(model, "logging_disabled", False)
        model.active_task = task_name
        model.logging_disabled = True
        try:
            with torch.enable_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, task_name=task_name)
                loss = model.criterion(logits, labels)
                loss.backward()
        except Exception as e:
            print(f"[A-GEM] replay forward failed: {e}")
            return None
        finally:
            model.active_task = saved_task
            model.training_phase = saved_phase
            model.logging_disabled = saved_logging

        return self._flatten_grads()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _project(self) -> Optional[float]:
        """A-GEM projection. Returns 1.0 if projection was applied, else 0.0, or None if skipped."""
        if self.replay_buffer is None or len(self.replay_buffer) == 0 or self.model_ref is None:
            return None

        g = self._flatten_grads()
        g_ref = self._compute_replay_gradient()
        if g_ref is None:
            return None

        # Restore g back into .grad (the replay forward just zeroed and wrote g_ref)
        # Note: _compute_replay_gradient zeroed grads then backward wrote g_ref into them.
        # We need g (the task gradient) back in .grad before projection math, but we
        # computed g_ref into the .grad fields. So re-read g from a saved copy.
        # Simpler: we already captured g before calling replay; restore it now.
        # Actually _flatten_grads was called BEFORE _compute_replay_gradient, so g is stale
        # only if .grad was modified. It WAS modified (replay backward wrote into .grad).
        # So: write g back into .grad, then compute dot products on flattened tensors.
        self._unflatten_to_grads(g)

        # Now .grad holds g again. Recompute flat g from current .grad to be safe
        g = self._flatten_grads()

        dot = torch.dot(g, g_ref)
        denom = torch.dot(g_ref, g_ref) + 1e-12
        ratio = (dot / denom).item()
        if dot.item() < 0:
            g_proj = g - ratio * g_ref
            self._unflatten_to_grads(g_proj)
            return 1.0
        else:
            # g already in .grad; nothing to do
            return 0.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):  # type: ignore
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        try:
            proj_flag = self._project()
        except Exception as e:
            print(f"[A-GEM] projection error: {e}")
            proj_flag = None

        if proj_flag == 1.0:
            self.proj_count += 1

        loss_out = self.base_optimizer.step()
        if loss is None:
            loss = loss_out

        if (
            self.writer is not None
            and proj_flag is not None
            and self.step_count % self.log_every == 0
        ):
            try:
                self.writer.add_scalars(
                    "AGEM",
                    {"projection_ratio": float(self.proj_count / max(1, self.step_count + 1))},
                    self.step_count,
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
        sd["agem_step_count"] = self.step_count
        sd["agem_proj_count"] = self.proj_count
        return sd

    def load_state_dict(self, state_dict):
        if "agem_step_count" in state_dict:
            self.step_count = state_dict["agem_step_count"]
        if "agem_proj_count" in state_dict:
            self.proj_count = state_dict["agem_proj_count"]
        self.base_optimizer.load_state_dict(state_dict)

    # Property forwarding (same pattern as NostalgiaOptimizer).
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
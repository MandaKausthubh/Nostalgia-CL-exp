# from torch.func import jvp, functional_call
# from torch.nn.utils import parameters_to_vector

# try:
#     import torch_xla.core.xla_model as xm
#     import torch_xla
#     import torch_xla.runtime as xr
#     HAS_XLA = True
# except ImportError:
#     HAS_XLA = False
#     class DummyXR:
#         def world_size(self):
#             return 1
#         def global_ordinal(self):
#             return 0
#     xr = DummyXR()
    
#     class DummyXM:
#         def is_master_ordinal(self):
#             return True
#         def mark_step(self):
#             pass
#         def all_reduce(self, reduce_type, tensor):
#             return tensor
#         def all_gather(self, tensor, dim=0):
#             return tensor
#     xm = DummyXM()

# import torch
# import gc

# # Flag to track if we're in TPU distributed mode
# _TPU_DISTRIBUTED = None

# def _is_tpu_distributed():
#     """Check if we're running in TPU distributed mode (xmp.spawn)."""
#     global _TPU_DISTRIBUTED
#     if _TPU_DISTRIBUTED is None:
#         try:
#             _TPU_DISTRIBUTED = xr.world_size() > 1 if HAS_XLA else False
#         except Exception:
#             _TPU_DISTRIBUTED = False
#     return _TPU_DISTRIBUTED



# def flatten_params(params):
#     return parameters_to_vector(params.values())


# def unflatten(vec, params_template):
#     new_params = {}
#     pointer = 0
#     for name, p in params_template.items():
#         num = p.numel()
#         new_params[name] = vec[pointer:pointer+num].view_as(p)
#         pointer += num
#     return new_params


# def hvp_flat(vec, params, model, inputs, targets, loss_fn):
#     """
#     Computes Hessian-vector product Hv using double backward.

#     Every call creates a FULLY INDEPENDENT computation graph:
#     fresh params → forward → grad → dot → grad → detach.
#     Nothing is retained across calls.
#     """
#     # Ensure vec is completely detached
#     vec = vec.detach().clone()

#     # Fresh params with grad enabled — no connection to prior graphs
#     fresh_params = {
#         name: p.detach().clone().requires_grad_(True)
#         for name, p in params.items()
#     }

#     # Forward pass
#     if isinstance(inputs, dict):
#         inputs_detached = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#     else:
#         inputs_detached = inputs.detach()
#     inputs_proc = model.preprocess_inputs(inputs_detached)
#     representations = functional_call(model.backbone, fresh_params, (inputs_proc,))
#     outputs = model.task_head_list[model.active_task](representations)
#     loss = loss_fn(outputs, targets.detach())

#     # First gradient (creates graph for second derivative)
#     grads = torch.autograd.grad(
#         loss,
#         tuple(fresh_params.values()),
#         create_graph=True,
#         allow_unused=True
#     )

#     # Replace None grads with zeros
#     grads = tuple(
#         g if g is not None else torch.zeros_like(p)
#         for g, p in zip(grads, fresh_params.values())
#     )

#     grad_vec = torch.nn.utils.parameters_to_vector(grads)

#     # Dot product with the query vector
#     grad_dot_vec = (grad_vec * vec).sum()

#     # Second gradient (HVP) — release graph completely
#     hv = torch.autograd.grad(
#         grad_dot_vec,
#         tuple(fresh_params.values()),
#         retain_graph=False,
#         allow_unused=True
#     )

#     hv = tuple(
#         g if g is not None else torch.zeros_like(p)
#         for g, p in zip(hv, fresh_params.values())
#     )

#     # Detach and clone immediately to sever ALL graph references
#     hv_vec = torch.nn.utils.parameters_to_vector(hv).detach().clone()

#     # Distributed reduction
#     if _is_tpu_distributed():
#         hv_vec = xm.all_reduce(xm.REDUCE_SUM, hv_vec)
#         world_size = xr.world_size()
#         hv_vec = hv_vec / world_size

#     return hv_vec


# def lanczos(hvp_fn, dim, k, device):
#     """
#     Lanczos algorithm with XLA-safe memory management.

#     Key insight: XLA builds a lazy computation graph. If we keep all
#     intermediate tensors on XLA across iterations, the graph grows
#     until it OOMs the device. The fix is to:
#       1. xm.mark_step() to compile+execute the graph after each HVP
#       2. Move Q, alpha, beta to CPU between iterations so XLA can't
#          "see" them as part of the current graph
#       3. Only bring the specific columns needed back to XLA for each step
#     """
#     dtype = torch.float32
#     rank = xr.global_ordinal() if _is_tpu_distributed() else 0

#     # Store Lanczos vectors on CPU to prevent XLA graph accumulation
#     Q_cpu = torch.zeros(dim, k, dtype=dtype)       # CPU
#     alpha_cpu = torch.zeros(k, dtype=dtype)         # CPU
#     beta_cpu = torch.zeros(k - 1, dtype=dtype)      # CPU

#     # Initial random vector — generate on device, normalize, move to CPU
#     q = torch.randn(dim, device=device, dtype=dtype)
#     q = q / q.norm()
#     xm.mark_step()
#     Q_cpu[:, 0] = q.detach().cpu()
#     del q

#     actual_k = k  # Track actual rank (may be less if early exit)

#     for j in range(k):
#         # if rank == 0:
#         #     print(f"[Lanczos iteration]: {j}/{k}")

#         # ── Step 1: bring current q_j to device ──────────────────────
#         q_j = Q_cpu[:, j].to(device)

#         # ── Step 2: compute HVP (fully self-contained graph) ─────────
#         v = hvp_fn(q_j)

#         # Force XLA to compile and execute THIS iteration's graph
#         xm.mark_step()

#         # ── Step 3: Lanczos recurrence (on device) ───────────────────
#         # Detach v so subsequent ops don't extend the HVP graph
#         v = v.detach()

#         alpha_j = torch.dot(q_j, v)
#         xm.mark_step()
#         alpha_cpu[j] = alpha_j.item()

#         v = v - alpha_j * q_j

#         if j > 0:
#             q_prev = Q_cpu[:, j - 1].to(device)
#             v = v - beta_cpu[j - 1].item() * q_prev
#             del q_prev

#         # Full reorthogonalization — bring columns one at a time
#         for i in range(j + 1):
#             qi = Q_cpu[:, i].to(device)
#             coeff = torch.dot(qi, v)
#             v = v - coeff * qi
#             del qi

#         if j < k - 1:
#             xm.mark_step()
#             beta_j = v.norm()
#             beta_val = beta_j.item()
#             beta_cpu[j] = beta_val

#             if beta_val < 1e-10:
#                 if rank == 0:
#                     print(f"Early exit at step {j}")
#                 actual_k = j + 1
#                 break

#             q_next = (v / beta_j).detach()
#             xm.mark_step()
#             Q_cpu[:, j + 1] = q_next.cpu()
#             del q_next

#         # Clean up device tensors from this iteration
#         del v, q_j
#         xm.mark_step()
#         gc.collect()

#     if rank == 0:
#         print(f"Rank of Lanczos: {actual_k}")

#     # Truncate to actual rank
#     Q_cpu = Q_cpu[:, :actual_k]
#     alpha_cpu = alpha_cpu[:actual_k]
#     beta_cpu = beta_cpu[:max(1, actual_k - 1)]

#     # Build tridiagonal matrix (on CPU — it's tiny: k×k)
#     T = torch.diag(alpha_cpu)
#     for i in range(len(beta_cpu)):
#         if i + 1 < actual_k:
#             T[i, i + 1] = beta_cpu[i]
#             T[i + 1, i] = beta_cpu[i]

#     return T, Q_cpu   # Both on CPU


# def compute_Q_for_task(model, k, device, train_loader):
#     model.eval()

#     batch = next(iter(train_loader))
#     if len(batch) == 3:
#         input_ids, attention_mask, targets = batch
#         inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
#     else:
#         inputs, targets = batch

#     if isinstance(inputs, dict):
#         inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#     else:
#         inputs = inputs.to(device)
#     targets = targets.to(device)

#     # Snapshot backbone params once — hvp_flat clones them each call
#     params = {
#         name: p.detach()
#         for name, p in model.get_backbone_params_dict().items()
#     }
#     param_dim = sum(p.numel() for p in params.values())

#     if xr.global_ordinal() == 0:
#         print(f"[compute_Q_for_task] param_dim={param_dim:,}, k={k}")

#     def hvp_operator(v):
#         return hvp_flat(
#             v,
#             params,
#             model,
#             inputs,
#             targets,
#             model.criterion
#         )

#     # T and Q_cpu are returned on CPU
#     T, Q_cpu = lanczos(
#         hvp_operator,
#         dim=param_dim,
#         k=k,
#         device=device
#     )

#     # Clean up the HVP closure references
#     del inputs, targets, params
#     xm.mark_step()
#     gc.collect()

#     T_cpu = T.detach().cpu().float()
#     T_cpu = 0.5 * (T_cpu + T_cpu.T)

#     # Eigendecomposition on CPU (T is small: k×k)
#     eps = 1e-6
#     T_cpu += eps * torch.eye(T_cpu.shape[0])
#     eigvals, eigvecs = torch.linalg.eigh(T_cpu)

#     eigvals = eigvals.to(T.device)
#     eigvecs = eigvecs.to(T.device)

#     # Reorder to descending eigenvalue order
#     idx = torch.argsort(eigvals, descending=True)
#     eigvals = eigvals[idx]
#     eigvecs = eigvecs[:, idx]

#     # Lift Ritz vectors back to full parameter space (CPU matmul — safe)
#     Q_full = Q_cpu @ eigvecs      # (param_dim, k)

#     # Move results to XLA device
#     Q_full = Q_full.to(device=device, dtype=torch.float32)
#     eigvals = eigvals.to(device=device, dtype=torch.float32)

#     qtq = Q_full.T @ Q_full
#     eye = torch.eye(qtq.shape[0], device=qtq.device)
#     err = (qtq - eye).abs().max()
#     # xm.master_print(f"[compute_Q_for_task] Q_full orthogonality error: {err.item():.7e}")

#     return Q_full, eigvals


# def update_Q_Lambda_for_all_past_domains(
#     self,
#     past_domains,
#     rank,
# ):
#     """
#     Stable across-domain Hessian memory accumulation.

#     Computes running average:
#         H_bar_t = ((t-1)/t) H_bar_{t-1} + (1/t) H_t

#     using weighted PSD factor merge.
#     """

#     Q_memory = None
#     Lambda_memory = None

#     k = self.config.hessian_eigenspace_dim

#     for i, domain in enumerate(past_domains):

#         xm.master_print(
#             f"[Rank {rank}] Processing domain {domain}"
#         )

#         # -------------------------------------
#         # compute task/domain Hessian
#         # -------------------------------------
#         Q_new, Lambda_new = self.update_Q_Lambda_for_single_domain(
#             domain,
#             rank,
#         )

#         # -------------------------------------
#         # first domain
#         # -------------------------------------
#         if Q_memory is None:
#             Q_memory = Q_new
#             Lambda_memory = Lambda_new

#         else:
#             t = i + 1

#             alpha = (t - 1) / t
#             beta = 1.0 / t

#             # ---------------------------------
#             # weighted PSD factors
#             # ---------------------------------
#             sqrt_old = torch.sqrt(
#                 Lambda_memory.clamp_min(0)
#             )
#             sqrt_new = torch.sqrt(
#                 Lambda_new.clamp_min(0)
#             )

#             F_old = (
#                 math.sqrt(alpha)
#                 * Q_memory
#                 * sqrt_old.unsqueeze(0)
#             )

#             F_new = (
#                 math.sqrt(beta)
#                 * Q_new
#                 * sqrt_new.unsqueeze(0)
#             )

#             # ---------------------------------
#             # factor merge
#             # ---------------------------------
#             F_global = torch.cat(
#                 [F_old, F_new],
#                 dim=1,
#             )

#             # ---------------------------------
#             # recover low-rank eigenspace
#             # ---------------------------------
#             Q_memory, Lambda_memory = recover_eigenspace_from_factor(
#                 F_global=F_global,
#                 k=k,
#             )

#         xm.mark_step()

#         if rank == 0:
#             err = check_orthogonality(Q_memory)

#             xm.master_print(
#                 f"[MASTER] After domain {domain}\n"
#                 f"Q shape: {Q_memory.shape}\n"
#                 f"Lambda shape: {Lambda_memory.shape}\n"
#                 f"Orthogonality error: {err}"
#             )

#     return Q_memory, Lambda_memory



# def recover_eigenspace_from_factor(
#     F_global: torch.Tensor,
#     k: int,
#     eps: float = 1e-8,
# ):
#     """
#     Recover low-rank eigenspace from PSD factor matrix.

#     Given:
#         F_global ∈ R^{n × m}

#     Computes eigenspace of:
#         H ≈ F F^T

#     using Gram trick:
#         G = F^T F

#     Returns:
#         Q ∈ R^{n × k_eff}
#         Lambda ∈ R^{k_eff}
#     """

#     # -------------------------------------
#     # small Gram matrix
#     # -------------------------------------
#     G = F_global.T @ F_global

#     # force symmetry
#     G = 0.5 * (G + G.T)

#     # -------------------------------------
#     # eigendecompose Gram
#     # -------------------------------------
#     eigvals, V = torch.linalg.eigh(G)

#     idx = torch.argsort(eigvals, descending=True)

#     eigvals = eigvals[idx]
#     V = V[:, idx]

#     # -------------------------------------
#     # remove numerical junk
#     # -------------------------------------
#     valid = eigvals > eps

#     eigvals = eigvals[valid]
#     V = V[:, valid]

#     if eigvals.numel() == 0:
#         raise RuntimeError(
#             "No valid eigenvalues found in factor recovery."
#         )

#     k_eff = min(k, eigvals.shape[0])

#     eigvals = eigvals[:k_eff]
#     V = V[:, :k_eff]

#     # -------------------------------------
#     # recover left singular vectors
#     # Q = F V Σ^{-1}
#     # -------------------------------------
#     singular_vals = torch.sqrt(
#         eigvals.clamp_min(eps)
#     )

#     Q = F_global @ V
#     Q = Q / singular_vals.unsqueeze(0)

#     # -------------------------------------
#     # mandatory orthonormal cleanup
#     # -------------------------------------
#     Q, _ = torch.linalg.qr(Q, mode="reduced")

#     Lambda = eigvals

#     # CRITICAL: Ensure Q and Lambda are fully materialized before returning
#     Q = Q.detach().contiguous()
#     Lambda = Lambda.detach().contiguous()
#     xm.mark_step()

#     return Q, Lambda


from torch.func import functional_call
from torch.nn.utils import parameters_to_vector

import math
import gc
import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    import torch_xla.runtime as xr
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

    class DummyXR:
        def world_size(self):
            return 1
        def global_ordinal(self):
            return 0

    xr = DummyXR()

    class DummyXM:
        REDUCE_SUM = "sum"

        def is_master_ordinal(self):
            return True
        def mark_step(self):
            pass
        def all_reduce(self, reduce_type, tensor):
            return tensor
        def all_gather(self, tensor, dim=0):
            return tensor
        def master_print(self, *args, **kwargs):
            print(*args, **kwargs)

    xm = DummyXM()


# ---------------------------------------------------------------------------
# Detect execution environment once at import time
# ---------------------------------------------------------------------------

_TPU_DISTRIBUTED = None

def _is_tpu_distributed():
    global _TPU_DISTRIBUTED
    if _TPU_DISTRIBUTED is None:
        try:
            _TPU_DISTRIBUTED = HAS_XLA and xr.world_size() > 1
        except Exception:
            _TPU_DISTRIBUTED = False
    return _TPU_DISTRIBUTED


def _is_gpu():
    return torch.cuda.is_available() and not HAS_XLA


def _is_mps():
    return (
        not HAS_XLA
        and not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _offload_to_cpu():
    """True for backends that have no lazy graph — GPU and MPS."""
    return _is_gpu() or _is_mps()


def _free_memory(device=None):
    """
    Free accelerator memory after a computation-heavy step.
    - TPU : xm.mark_step() flushes the lazy graph — the real mechanism.
    - GPU : torch.cuda.empty_cache() releases the caching allocator.
    - MPS : torch.mps.empty_cache() does the same for the Metal allocator.
    mark_step() is always called; it is a no-op on non-XLA builds.
    """
    xm.mark_step()
    gc.collect()
    dev_type = device.type if isinstance(device, torch.device) else None
    if _is_gpu() or dev_type == "cuda":
        torch.cuda.empty_cache()
    elif _is_mps() or dev_type == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------

def flatten_params(params):
    return parameters_to_vector(params.values())


def unflatten(vec, params_template):
    new_params = {}
    pointer = 0
    for name, p in params_template.items():
        num = p.numel()
        new_params[name] = vec[pointer:pointer + num].view_as(p)
        pointer += num
    return new_params


# ---------------------------------------------------------------------------
# HVP
# ---------------------------------------------------------------------------

def hvp_flat(vec, params, model, inputs, targets, loss_fn):
    """
    Hessian-vector product via double backward.

    Fixes vs. original:
      - p.detach() (no .clone()) on fresh_params — saves one full-param copy.
        The create_graph=True grad call does NOT alias back to `params` because
        requires_grad_(True) on a detached leaf is safe.
      - Explicit `del` + _free_memory() before returning so the double-backward
        graph is fully released whether we're on GPU or TPU.
      - Distributed reduction unchanged.
    """
    vec = vec.detach()

    # Detach-only (no clone) to save one param-sized allocation.
    # requires_grad_(True) on a detach leaf is correct — autograd treats it
    # as a fresh leaf; no aliasing to the original param graph.
    fresh_params = {
        name: p.detach().requires_grad_(True)
        for name, p in params.items()
    }

    # ── Forward ──────────────────────────────────────────────────────────
    if isinstance(inputs, dict):
        inputs_detached = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
    else:
        inputs_detached = inputs.detach()

    inputs_proc = model.preprocess_inputs(inputs_detached)
    representations = functional_call(model.backbone, fresh_params, (inputs_proc,))
    outputs = model.task_head_list[model.active_task](representations)
    loss = loss_fn(outputs, targets.detach())

    # ── First backward (keep graph) ───────────────────────────────────────
    grads = torch.autograd.grad(
        loss,
        tuple(fresh_params.values()),
        create_graph=True,
        allow_unused=True,
    )
    grads = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, fresh_params.values())
    )
    grad_vec = parameters_to_vector(grads)

    # ── Dot with query, second backward (release graph) ───────────────────
    grad_dot = (grad_vec * vec).sum()

    hv = torch.autograd.grad(
        grad_dot,
        tuple(fresh_params.values()),
        retain_graph=False,
        allow_unused=True,
    )
    hv = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(hv, fresh_params.values())
    )
    hv_vec = parameters_to_vector(hv).detach()

    # ── Explicit cleanup — critical on GPU where gc is not lazy-graph-aware ──
    del fresh_params, grads, grad_vec, grad_dot, hv, loss, outputs
    del representations, inputs_proc
    _free_memory()

    # ── Distributed reduction (TPU only) ──────────────────────────────────
    if _is_tpu_distributed():
        hv_vec = xm.all_reduce(xm.REDUCE_SUM, hv_vec)
        hv_vec = hv_vec / xr.world_size()

    return hv_vec


# ---------------------------------------------------------------------------
# Lanczos
# ---------------------------------------------------------------------------

def lanczos(hvp_fn, dim, k, device):
    """
    Lanczos with full reorthogonalisation.

    Two storage strategies depending on backend:

    TPU  — vectors stay ON device between iterations.
           xm.mark_step() flushes the lazy graph each step, which is what
           actually prevents graph accumulation. No CPU round-trips needed.
           Returns Q on device.

    GPU / MPS — no lazy graph, so vectors are offloaded to CPU between
           iterations. At most one vector lives on the accelerator at a time.
           pin_memory=True on CUDA for fast async H2D copies (not supported
           on MPS). Returns Q on CPU.

    In both cases T (k×k tridiagonal) is returned on CPU.
    """
    dtype = torch.float32
    rank  = xr.global_ordinal() if _is_tpu_distributed() else 0
    offload = _offload_to_cpu()          # False on TPU, True on GPU/MPS
    pin     = _is_gpu()                  # pin_memory only valid for CUDA

    alpha_buf = torch.zeros(k,      dtype=dtype)   # always CPU, tiny
    beta_buf  = torch.zeros(k - 1,  dtype=dtype)   # always CPU, tiny

    if offload:
        # Lanczos vectors stored in CPU RAM, one column per iteration.
        # For a 2.5B-param model each fp32 vector is 10 GB — keep exactly k
        # of them, never all on device simultaneously.
        Q_store = torch.zeros(dim, k, dtype=dtype,
                              pin_memory=pin)        # CPU, page-locked on CUDA
    else:
        # TPU: keep everything on device; mark_step handles graph flushing.
        Q_store = torch.zeros(dim, k, dtype=dtype, device=device)

    # ── Initial random unit vector ────────────────────────────────────────
    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / q.norm()
    xm.mark_step()
    if offload:
        Q_store[:, 0].copy_(q.cpu())
        del q
    else:
        Q_store[:, 0] = q
        del q
    _free_memory(device)

    actual_k = k

    for j in range(k):

        # ── Load q_j onto device ──────────────────────────────────────────
        if offload:
            q_j = Q_store[:, j].to(device, non_blocking=pin)
        else:
            q_j = Q_store[:, j]          # already on device

        # ── HVP ───────────────────────────────────────────────────────────
        v = hvp_fn(q_j)
        xm.mark_step()
        v = v.detach()

        # ── α_j = q_j · v ────────────────────────────────────────────────
        alpha_j = torch.dot(q_j, v)
        xm.mark_step()
        alpha_buf[j] = alpha_j.item()

        # ── v ← v − α_j q_j − β_{j-1} q_{j-1} ──────────────────────────
        v = v - alpha_j * q_j

        if j > 0:
            if offload:
                q_prev = Q_store[:, j - 1].to(device, non_blocking=pin)
            else:
                q_prev = Q_store[:, j - 1]
            v = v - beta_buf[j - 1].item() * q_prev
            del q_prev

        # ── Full reorthogonalisation (one vector at a time) ───────────────
        for i in range(j + 1):
            if offload:
                qi = Q_store[:, i].to(device, non_blocking=pin)
            else:
                qi = Q_store[:, i]
            coeff = torch.dot(qi, v)
            v = v - coeff * qi
            if offload:
                del qi
            # Prevent CUDA command-queue overflow on long sweeps
            if _is_gpu() and i % 64 == 63:
                torch.cuda.synchronize()

        # ── β_j = ‖v‖, normalise, store ──────────────────────────────────
        if j < k - 1:
            xm.mark_step()
            beta_j   = v.norm()
            beta_val = beta_j.item()
            beta_buf[j] = beta_val

            if beta_val < 1e-10:
                if rank == 0:
                    print(f"[Lanczos] Early exit at step {j}")
                actual_k = j + 1
                del v, q_j
                _free_memory(device)
                break

            q_next = (v / beta_j).detach()
            xm.mark_step()
            if offload:
                Q_store[:, j + 1].copy_(q_next.cpu())
                del q_next
            else:
                Q_store[:, j + 1] = q_next
                del q_next

        del v, q_j
        _free_memory(device)

    if rank == 0:
        print(f"[Lanczos] Effective rank: {actual_k}")

    # Truncate to actual rank
    Q_store   = Q_store[:, :actual_k].contiguous()
    alpha_buf = alpha_buf[:actual_k]
    beta_buf  = beta_buf[:max(1, actual_k - 1)]

    # Build tridiagonal T (CPU, tiny: actual_k × actual_k)
    T = torch.diag(alpha_buf)
    for i in range(min(len(beta_buf), actual_k - 1)):
        T[i, i + 1] = beta_buf[i]
        T[i + 1, i] = beta_buf[i]

    # Q_store is on CPU for GPU/MPS, on device for TPU
    return T, Q_store


# ---------------------------------------------------------------------------
# _single_lanczos_pass  (internal helper — one Lanczos call on one batch)
# ---------------------------------------------------------------------------

def _single_lanczos_pass(model, k, device, inputs, targets):
    """
    Run a single Lanczos pass on a pre-prepared batch.

    Args:
        model:   the LightningModule (must already be in eval mode)
        k:       Lanczos rank
        device:  accelerator device
        inputs:  dict or tensor, already on *device*
        targets: tensor, already on *device*

    Returns:
        Q_full:  (param_dim, k)  orthonormal Ritz vectors  [on device]
        eigvals: (k,)            corresponding eigenvalues [on device]
    """
    # ── Snapshot params (detached, no grad) ───────────────────────────────
    params = {
        name: p.detach()
        for name, p in model.get_backbone_params_dict().items()
    }
    param_dim = sum(p.numel() for p in params.values())

    # ── HVP operator ──────────────────────────────────────────────────────
    def hvp_op(v):
        return hvp_flat(v, params, model, inputs, targets, model.criterion)

    # ── Lanczos → T (k×k, CPU) + Q_store (param_dim×k, CPU or device) ───
    T, Q_store = lanczos(hvp_op, dim=param_dim, k=k, device=device)

    # Free params — no longer needed
    del params
    _free_memory(device)

    # ── Eigendecompose T on CPU ────────────────────────────────────────────
    T_cpu = T.detach().cpu().float()
    T_cpu = 0.5 * (T_cpu + T_cpu.T)
    T_cpu = T_cpu + 1e-6 * torch.eye(T_cpu.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(T_cpu)   # ascending

    # Descending order
    idx      = torch.argsort(eigvals, descending=True)
    eigvals  = eigvals[idx]
    eigvecs  = eigvecs[:, idx]

    # ── Lift Ritz vectors: Q_full = Q_store @ eigvecs ────────────────────
    if not _offload_to_cpu():
        # TPU: lift and QR on device
        eigvecs_dev = eigvecs.to(device)
        Q_full      = Q_store @ eigvecs_dev
        del Q_store, eigvecs, eigvecs_dev
        Q_full, _   = torch.linalg.qr(Q_full, mode="reduced")
        eigvals     = eigvals.to(device=device, dtype=torch.float32)

        Q_full  = Q_full.detach().contiguous()
        eigvals = eigvals.detach().contiguous()
        xm.mark_step()
    else:
        # GPU / MPS: lift and QR on CPU, then move
        Q_full_cpu  = Q_store @ eigvecs
        del Q_store, eigvecs
        gc.collect()

        Q_full_cpu, _ = torch.linalg.qr(Q_full_cpu, mode="reduced")

        Q_full  = Q_full_cpu.to(device=device, dtype=torch.float32)
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        del Q_full_cpu
        _free_memory(device)

    return Q_full, eigvals


# ---------------------------------------------------------------------------
# compute_single_domain_eigenspace  (Algorithm 3 from the paper)
# ---------------------------------------------------------------------------

def compute_single_domain_eigenspace(
    model, k, device, train_loader,
    accumulation_rounds=5,
    max_hessian_batch=8,
):
    """
    Compute the top-k Hessian eigenspace for a single domain/task,
    averaged over multiple Lanczos rounds (Algorithm 3, Section 4.4).

    Instead of relying on a single noisy Lanczos call, this function
    runs Ea independent Lanczos passes — each on a different mini-batch
    — and aggregates the resulting PSD factors:

        F_local = (1/√Ea) · [F_1 | F_2 | ... | F_Ea]

    where  F_e = Q_e · diag(√(max(Λ_e, 0))).

    The eigenspace is then recovered from F_local via the Gram trick
    (Section 4.5), with optional distributed synchronisation.

    Args:
        model:                the LightningModule
        k:                    Lanczos rank per round
        device:               accelerator device
        train_loader:         DataLoader for this task/domain
        accumulation_rounds:  Ea — number of Lanczos passes (default 5)
        max_hessian_batch:    cap per-batch sample count for OOM safety
                              during double-backward HVP (default 8)

    Returns:
        Q:      (param_dim, k_eff)  orthonormal basis   [on device]
        Lambda: (k_eff,)            eigenvalues         [on device]
    """
    model.eval()
    Ea = accumulation_rounds
    rank = xr.global_ordinal() if _is_tpu_distributed() else 0

    if rank == 0:
        params_snapshot = model.get_backbone_params_dict()
        param_dim = sum(p.numel() for p in params_snapshot.values())
        f_gb = param_dim * k * Ea * 4 / 1e9
        print(
            f"[SingleDomainEigenspace] param_dim={param_dim:,}, k={k}, "
            f"Ea={Ea}, F_local RAM ≈ {f_gb:.2f} GB"
        )
        del params_snapshot

    # ── Collect PSD factors across Ea rounds ──────────────────────────────
    factors = []  # list of (param_dim, k) tensors
    loader_iter = iter(train_loader)

    for e in range(Ea):
        # Fetch next batch, cycling if loader is exhausted
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        # Prepare batch → (inputs, targets) on device
        if isinstance(batch, dict):
            # dict-style batch (from TaskClassificationDataset)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            targets = batch["target"]
        elif len(batch) == 3:
            input_ids, attention_mask, targets = batch
        else:
            input_ids, targets = batch
            attention_mask = None

        # Cap batch size for OOM safety during double-backward
        actual_bs = input_ids.shape[0]
        hess_bs = min(actual_bs, max_hessian_batch)

        if attention_mask is not None:
            inputs = {
                "input_ids":      input_ids[:hess_bs].to(device),
                "attention_mask": attention_mask[:hess_bs].to(device),
            }
        else:
            inputs = input_ids[:hess_bs].to(device)
        targets_dev = targets[:hess_bs].to(device)

        if rank == 0:
            print(
                f"  [Round {e+1}/{Ea}] batch_size={hess_bs}/{actual_bs}"
            )

        # ── Single Lanczos pass ───────────────────────────────────────────
        Q_e, Lambda_e = _single_lanczos_pass(
            model, k, device, inputs, targets_dev,
        )

        # ── Form PSD factor:  F_e = Q_e · diag(√(max(Λ_e, 0))) ──────────
        sqrt_lam = Lambda_e.clamp_min(0).sqrt()      # (k,)
        F_e = Q_e * sqrt_lam.unsqueeze(0)             # (n, k)

        # Store on CPU to avoid accumulating Ea full-sized tensors on device
        factors.append(F_e.detach().cpu())

        del Q_e, Lambda_e, F_e, sqrt_lam
        del inputs, targets_dev
        _free_memory(device)

    # ── Concatenate and scale: F_local = (1/√Ea) · [F_1 | ... | F_Ea] ────
    F_local_cpu = torch.cat(factors, dim=1) / math.sqrt(Ea)   # (n, k·Ea)
    del factors
    gc.collect()

    if rank == 0:
        print(f"  F_local shape: {list(F_local_cpu.shape)}")

    # ── Gram matrix: G_local = F_local^T · F_local  (small: kEa × kEa) ───
    G_local = F_local_cpu.T @ F_local_cpu   # CPU, (k·Ea, k·Ea)
    G_local = G_local.to(device=device, dtype=torch.float32)
    xm.mark_step()

    # ── Distributed sync: G = all_reduce_mean(G_local) ────────────────────
    if _is_tpu_distributed():
        G = xm.all_reduce(xm.REDUCE_SUM, G_local)
        G = G / xr.world_size()
        xm.mark_step()
    else:
        G = G_local
    del G_local

    # Enforce symmetry
    G = 0.5 * (G + G.T)

    # ── Eigendecompose G → top-k positive eigenpairs ──────────────────────
    # G is small (k·Ea × k·Ea), e.g. 100×100 for k=20,Ea=5. Safe on CPU.
    G_cpu = G.detach().cpu().float()
    del G

    eigvals_g, V = torch.linalg.eigh(G_cpu)   # ascending
    idx = torch.argsort(eigvals_g, descending=True)
    eigvals_g = eigvals_g[idx]
    V         = V[:, idx]

    # Keep only positive eigenvalues
    eps = 1e-8
    valid = eigvals_g > eps
    eigvals_g = eigvals_g[valid]
    V         = V[:, valid]

    if eigvals_g.numel() == 0:
        raise RuntimeError(
            "[SingleDomainEigenspace] No valid positive eigenvalues in Gram matrix."
        )

    k_eff = min(k, eigvals_g.shape[0])
    sigma = eigvals_g[:k_eff]        # top-k eigenvalues (= Λ_τ)
    V     = V[:, :k_eff]             # corresponding eigenvectors

    # ── Recover basis: Q = F_local · V · diag(σ^{-1/2}) ──────────────────
    # Done on CPU since F_local_cpu is (n, k·Ea).
    inv_sqrt_sigma = 1.0 / torch.sqrt(sigma.clamp_min(eps))

    Q_cpu = F_local_cpu @ V          # (n, k_eff)  — CPU matmul
    Q_cpu = Q_cpu * inv_sqrt_sigma.unsqueeze(0)
    del F_local_cpu, V, inv_sqrt_sigma
    gc.collect()

    # ── QR for numerical stability ────────────────────────────────────────
    Q_cpu, _ = torch.linalg.qr(Q_cpu, mode="reduced")

    # ── Move to device ────────────────────────────────────────────────────
    Q      = Q_cpu.to(device=device, dtype=torch.float32)
    Lambda = sigma.to(device=device, dtype=torch.float32)
    del Q_cpu, sigma
    _free_memory(device)

    # Materialize for clean graph on XLA
    Q      = Q.detach().contiguous()
    Lambda = Lambda.detach().contiguous()
    xm.mark_step()

    # ── Sanity check ──────────────────────────────────────────────────────
    with torch.no_grad():
        Q_double = Q.cpu().double()
        qtq = Q_double.T @ Q_double
        err = (qtq - torch.eye(qtq.shape[0], device=torch.device("cpu"), dtype=torch.double)).abs().max()
        if rank == 0:
            print(
                f"  [SingleDomainEigenspace] k_eff={k_eff}, "
                f"orthogonality error: {err.item():.2e}"
            )
        del Q_double, qtq
    _free_memory(device)

    # Restore model to train mode
    model.train()

    return Q, Lambda


def compute_Q_for_task(model, k, device, train_loader):
    """
    Backward-compatibility wrapper for compute_single_domain_eigenspace.
    Runs with accumulation_rounds=1.
    """
    return compute_single_domain_eigenspace(
        model=model,
        k=k,
        device=device,
        train_loader=train_loader,
        accumulation_rounds=1,
    )


# ---------------------------------------------------------------------------
# Multi-domain accumulation — structure unchanged, cleanups added
# ---------------------------------------------------------------------------

def recover_eigenspace_from_factor(F_global, k, eps=1e-8):
    """
    Given F ∈ R^{n×m} representing H ≈ F F^T, recover the top-k eigenspace
    via the Gram trick G = F^T F  (m×m, cheap when m = 2k).
    """
    G = F_global.T @ F_global
    G = 0.5 * (G + G.T)

    eigvals, V = torch.linalg.eigh(G)

    idx     = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    V       = V[:, idx]

    valid   = eigvals > eps
    eigvals = eigvals[valid]
    V       = V[:, valid]

    if eigvals.numel() == 0:
        raise RuntimeError("No valid eigenvalues found in factor recovery.")

    k_eff   = min(k, eigvals.shape[0])
    eigvals = eigvals[:k_eff]
    V       = V[:, :k_eff]

    singular_vals = torch.sqrt(eigvals.clamp_min(eps))
    Q = F_global @ V / singular_vals.unsqueeze(0)
    Q, _ = torch.linalg.qr(Q, mode="reduced")

    Q      = Q.detach().contiguous()
    eigvals = eigvals.detach().contiguous()
    xm.mark_step()

    return Q, eigvals


def update_Q_Lambda_for_all_past_domains(self, past_domains, rank):
    """
    Running-average Hessian accumulation:
        H_bar_t = ((t-1)/t) H_bar_{t-1} + (1/t) H_t
    """
    Q_memory      = None
    Lambda_memory = None
    k = self.config.hessian_eigenspace_dim

    for i, domain in enumerate(past_domains):
        xm.master_print(f"[Rank {rank}] Processing domain {domain}")

        Q_new, Lambda_new = self.update_Q_Lambda_for_single_domain(domain, rank)

        if Q_memory is None:
            Q_memory      = Q_new
            Lambda_memory = Lambda_new
        else:
            t     = i + 1
            alpha = (t - 1) / t
            beta  = 1.0 / t

            sqrt_old = Lambda_memory.clamp_min(0).sqrt()
            sqrt_new = Lambda_new.clamp_min(0).sqrt()

            F_old    = math.sqrt(alpha) * Q_memory * sqrt_old.unsqueeze(0)
            F_new    = math.sqrt(beta)  * Q_new    * sqrt_new.unsqueeze(0)
            F_global = torch.cat([F_old, F_new], dim=1)
            del F_old, F_new

            Q_memory, Lambda_memory = recover_eigenspace_from_factor(F_global, k)
            del F_global
            _free_memory()

        xm.mark_step()

        if rank == 0:
            err = (Q_memory.T @ Q_memory
                   - torch.eye(Q_memory.shape[1], device=Q_memory.device)).abs().max()
            xm.master_print(
                f"[MASTER] After domain {domain}\n"
                f"  Q shape       : {Q_memory.shape}\n"
                f"  Lambda shape  : {Lambda_memory.shape}\n"
                f"  Orth error    : {err.item():.2e}"
            )

    return Q_memory, Lambda_memory

import torch
import gc
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# XLA-safe imports — same pattern as hessians.py
# ---------------------------------------------------------------------------
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

    class DummyXM:
        def mark_step(self):
            pass
    xm = DummyXM()


def _is_xla(tensor: torch.Tensor) -> bool:
    return tensor.device.type == 'xla'


def _mark_step():
    """Flush the XLA lazy graph. No-op on non-XLA backends."""
    xm.mark_step()


def _needs_cpu_offload(tensor: torch.Tensor) -> bool:
    """Check if tensor needs to be moved to CPU for numerical operations."""
    # TPU ('xla') and MPS may have issues with certain linalg operations
    return tensor.device.type in ('xla', 'mps')


def _safe_qr(X: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable QR with automatic CPU offload when required.

    On XLA: materializes X via mark_step() before the CPU transfer, then
    mark_step() again after the result is back on device so the transfer
    node is compiled immediately and doesn't bloat the next graph.
    """
    original_device = X.device
    original_dtype = X.dtype

    if _needs_cpu_offload(X):
        # Force XLA to compile/execute everything that produced X,
        # so the .to("cpu") transfer is a clean synchronous copy
        # rather than pulling in a giant uncompiled graph.
        _mark_step()

        X_cpu = X.detach().to("cpu", dtype=torch.float32)
        Q_cpu, _ = torch.linalg.qr(X_cpu, mode="reduced")
        result = Q_cpu.to(device=original_device, dtype=original_dtype)

        # Compile the CPU→device transfer immediately so it doesn't
        # get merged into subsequent operations' graph.
        _mark_step()
        return result

    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q


def _safe_eigh(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safe symmetric eigendecomposition with automatic CPU offload.
    Same mark_step pattern as _safe_qr.
    """
    original_device = S.device
    original_dtype = S.dtype

    if _needs_cpu_offload(S):
        _mark_step()

        S_cpu = S.detach().to("cpu", dtype=torch.float32)
        eigvals_cpu, eigvecs_cpu = torch.linalg.eigh(S_cpu)

        eigvals = eigvals_cpu.to(device=original_device, dtype=original_dtype)
        eigvecs = eigvecs_cpu.to(device=original_device, dtype=original_dtype)

        _mark_step()
        return eigvals, eigvecs

    return torch.linalg.eigh(S)


def _diag_from_vector(v: torch.Tensor) -> torch.Tensor:
    """
    Converts eigenvalue vector to diagonal matrix.
    """
    if v.ndim == 2:
        return v
    return torch.diag(v)


def accumulate_hessian_eigenspace_stable(
    Q_old: Optional[torch.Tensor],
    Lambda_old: Optional[torch.Tensor],
    Q_new: torch.Tensor,
    Lambda_new: torch.Tensor,
    t: int,
    k: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stable ACCUMULATE step with XLA-safe graph management.

    Maintains:
        H_bar_t = ((t-1)/t) * H_bar_{t-1} + (1/t) * H_t

    using a numerically stable low-rank eigenspace update.

    On TPU (XLA), the lazy computation graph must be flushed at key
    boundaries to prevent HBM exhaustion.  Each phase below is
    separated by a mark_step() so XLA compiles and executes the
    preceding operations before starting the next batch.

    Args:
        Q_old: (N, k_old) previous basis
        Lambda_old: (k_old,) previous eigenvalues
        Q_new: (N, k_new) new task basis
        Lambda_new: (k_new,) new eigenvalues
        t: task index (1-based)
        k: rank cap
        eps: numerical threshold

    Returns:
        Q_t: (N, k_eff) orthonormal basis
        Lambda_t: (k_eff,) eigenvalues
    """

    # -----------------------------
    # sanitize eigenvalues
    # -----------------------------
    if Lambda_new.ndim == 2:
        Lambda_new = torch.diag(Lambda_new)

    Q_new = _safe_qr(Q_new)       # mark_step inside on XLA

    # -----------------------------
    # first task — just truncate
    # -----------------------------
    if Q_old is None or Lambda_old is None:
        k_eff = min(k, Q_new.shape[1], Lambda_new.shape[0])

        Q_out = Q_new[:, :k_eff].detach().contiguous()
        L_out = Lambda_new[:k_eff].detach().contiguous()
        _mark_step()

        return Q_out, L_out

    if Lambda_old.ndim == 2:
        Lambda_old = torch.diag(Lambda_old)

    Q_old = _safe_qr(Q_old)       # mark_step inside on XLA

    alpha = (t - 1) / t
    beta = 1.0 / t

    # ============================================================
    # PHASE 1: Build the merged basis B = orth([Q_old, Q_res])
    # ============================================================
    overlap = Q_old.T @ Q_new
    Q_res = Q_new - Q_old @ overlap

    # Flush so that the matmuls above are compiled before we
    # branch on res_norm (which forces a scalar .item() sync).
    _mark_step()

    res_norm = torch.norm(Q_res).item()

    if res_norm < eps:
        B = Q_old
    else:
        Q_res = _safe_qr(Q_res)          # mark_step inside
        B = torch.cat([Q_old, Q_res], dim=1)
        B = _safe_qr(B)                  # mark_step inside
        del Q_res

    # Free intermediates before the next phase
    del overlap
    gc.collect()

    # ============================================================
    # PHASE 2: Build small-space averaged Hessian S
    # ============================================================
    A_old = B.T @ Q_old
    A_new = B.T @ Q_new

    Lambda_old_diag = _diag_from_vector(Lambda_old)
    Lambda_new_diag = _diag_from_vector(Lambda_new)

    S_old = A_old @ Lambda_old_diag @ A_old.T
    S_new = A_new @ Lambda_new_diag @ A_new.T

    S = alpha * S_old + beta * S_new

    # enforce symmetry
    S = 0.5 * (S + S.T)

    # Free projection matrices — only B, S, and eigvecs needed next
    del A_old, A_new, S_old, S_new, Lambda_old_diag, Lambda_new_diag
    _mark_step()

    # ============================================================
    # PHASE 3: Eigendecompose S, select top-k
    # ============================================================
    eigvals, eigvecs = _safe_eigh(S)     # mark_step inside on XLA
    del S

    idx = torch.argsort(eigvals, descending=True)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Flush before the boolean mask (forces .item() / .any() sync)
    _mark_step()

    valid = eigvals > eps

    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]

    if eigvals.numel() == 0:
        print("[ACCUMULATE warning] no valid eigvals, fallback to Q_old")
        Q_fallback = Q_old[:, :1].detach().contiguous()
        L_fallback = Lambda_old[:1].detach().contiguous()
        _mark_step()
        return Q_fallback, L_fallback

    k_eff = min(k, eigvals.shape[0])

    eigvals = eigvals[:k_eff]
    eigvecs = eigvecs[:, :k_eff]

    # ============================================================
    # PHASE 4: Lift back to full parameter space
    # ============================================================
    Q_t = B @ eigvecs
    del B, eigvecs
    Q_t = _safe_qr(Q_t)

    # Materialize final result so subsequent code gets a clean tensor
    Q_t = Q_t.detach().contiguous()
    eigvals = eigvals.detach().contiguous()
    _mark_step()

    # -----------------------------
    # sanity check
    # -----------------------------
    qtq = Q_t.T @ Q_t
    err = (
        qtq
        - torch.eye(
            qtq.shape[0],
            device=qtq.device,
            dtype=qtq.dtype,
        )
    ).abs().max()
    _mark_step()

    return Q_t, eigvals

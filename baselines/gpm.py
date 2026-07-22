"""GPM (Gradient Projection Memory).

Builds a subspace basis from the gradients of the task loss w.r.t. backbone
parameters. The basis Q has the same `[num_params, k]` shape contract as the
Nostalgia Hessian eigenspace, so `NostalgiaOptimizer.set_Q` accepts it
unchanged and the same `accumulate_hessian_eigenspace_stable` routine merges
it with the running memory.

Reference: Saha et al., "Gradient Projection Memory for Continual Learning",
ICLR 2021.
"""

from typing import Optional, Tuple
import torch
import gc

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

    class DummyXM:
        def mark_step(self):
            pass
    xm = DummyXM()


def compute_gpm_subspace(
    model,
    loader,
    device,
    threshold: float = 0.925,
    k: int = 20,
    max_batches: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect backbone-loss gradients over `max_batches` batches and return
    an orthonormal basis Q (num_params, k_eff) + pseudo-eigenvalues Lambda (k_eff,).

    Args:
        model:     LightningModule with `.backbone` and `get_backbone_params_dict()`.
        loader:    DataLoader for the task that just ended.
        device:    accelerator device.
        threshold: relative singular-value threshold; directions with
                   s_i > threshold * s_max are retained.
        k:         rank cap.
        max_batches: number of batches whose gradients form the columns of G.

    Returns:
        Q:      (num_params, k_eff) orthonormal basis [on device, float32]
        Lambda: (k_eff,) pseudo-eigenvalues (= retained singular values ** 2)
    """
    model.eval()
    backbone_params = list(model.get_backbone_params_dict().values())
    param_numels = [p.numel() for p in backbone_params]
    num_params = sum(param_numels)

    grad_columns = []  # each (num_params,) on CPU
    n_collected = 0
    loader_iter = iter(loader)

    for b_idx in range(max_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            try:
                batch = next(loader_iter)
            except StopIteration:
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

        # Zero backbone grads
        for p in backbone_params:
            if p.grad is not None:
                p.grad.zero_()

        try:
            with torch.enable_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                               task_name=model.active_task)
                loss = model.criterion(logits, targets)
                loss.backward()
        except Exception as e:
            print(f"[GPM] forward/backward failed on batch {b_idx}: {e}")
            continue

        # Flatten backbone grads into a single column
        flat = []
        for p, n in zip(backbone_params, param_numels):
            if p.grad is None:
                flat.append(torch.zeros(n, device=device, dtype=torch.float32))
            else:
                flat.append(p.grad.detach().view(-1).float())
        col = torch.cat(flat)
        grad_columns.append(col.cpu())
        n_collected += 1

        model.zero_grad()
        del input_ids, attention_mask, targets, loss, logits, flat, col

    if n_collected == 0:
        raise RuntimeError("[GPM] no gradients collected; loader empty?")

    # Stack into G: (num_params, n_collected)
    G = torch.stack(grad_columns, dim=1)  # CPU
    del grad_columns
    gc.collect()

    # SVD on CPU (numerically stable; num_params may be large but n_collected is small)
    # Use the Gram trick: GG^T is huge; G^T G is (n_collected x n_collected).
    # Right singular vectors V (n_collected x n_collected), singular values s.
    # Left singular vectors U = G V / s.
    GtG = G.T @ G                      # (n_collected, n_collected)
    GtG = 0.5 * (GtG + GtG.T)          # enforce symmetry
    eigvals, V = torch.linalg.eigh(GtG)  # ascending

    # Descending order
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    V = V[:, idx]

    # Singular values = sqrt(eigenvalues of G^T G)
    s = torch.sqrt(eigvals.clamp_min(0.0))

    # Threshold by relative magnitude
    s_max = s[0].item() if s.numel() > 0 else 0.0
    if s_max <= 0:
        # Degenerate — return a single random direction
        q = torch.randn(num_params, dtype=torch.float32)
        q = q / q.norm()
        Q = q.unsqueeze(1).to(device=device, dtype=torch.float32)
        Lambda = torch.ones(1, dtype=torch.float32, device=device)
        return Q, Lambda

    keep_mask = s > (threshold * s_max)
    s_kept = s[keep_mask]
    V_kept = V[:, keep_mask]

    if s_kept.numel() == 0:
        # Threshold too aggressive — keep at least one direction
        s_kept = s[:1]
        V_kept = V[:, :1]

    # Cap at k
    k_eff = min(k, s_kept.numel())
    s_kept = s_kept[:k_eff]
    V_kept = V_kept[:, :k_eff]

    # Recover left singular vectors: U = G V / s
    inv_s = 1.0 / s_kept.clamp_min(1e-12)
    U = (G @ V_kept) * inv_s.unsqueeze(0)   # (num_params, k_eff), CPU

    # QR for orthonormality
    Q_cpu, _ = torch.linalg.qr(U, mode="reduced")

    # Pseudo-eigenvalues = singular values ** 2 (matches Hessian eigenvalue scale roughly)
    Lambda_cpu = (s_kept ** 2).contiguous()

    Q = Q_cpu.to(device=device, dtype=torch.float32).detach().contiguous()
    Lambda = Lambda_cpu.to(device=device, dtype=torch.float32).detach().contiguous()

    del G, GtG, eigvals, V, s, V_kept, s_kept, inv_s, U, Q_cpu
    gc.collect()

    print(f"[GPM] collected {n_collected} gradient columns, kept rank {k_eff} "
          f"(threshold={threshold}, s_max={s_max:.4e})")

    model.train()
    return Q, Lambda
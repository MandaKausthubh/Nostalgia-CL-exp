"""SDFT (Self-Distillation Fine-Tuning) helper.

Freezes a copy of the model trained on the previous task and distills it
while training the next task. The distillation term is a scaled KL divergence
between the current (student) logits and the frozen teacher logits.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_sdft_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Return the scaled KL divergence used by SDFT.

    Args:
        student_logits: (B, K) logits from the current model.
        teacher_logits: (B, K) logits from the frozen teacher.
        temperature:    soft-target temperature.

    Returns:
        Scalar tensor = KL(student/T || teacher/T) * T^2.
    """
    T = float(temperature)
    kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    )
    return kl * (T * T)


def snapshot_teacher(model, device) -> torch.nn.Module:
    """Build a frozen eval copy of `model` from its state dict.

    Avoids `deepcopy` so trainer / writer references are not carried over.
    The returned model is on `device`, in eval mode, and has requires_grad=False.
    """
    hparams = dict(model.hparams)
    # The teacher does not need a writer reference.
    hparams.pop("writer", None)

    # Re-instantiate the same class; assumes the class can be reconstructed from
    # its hyperparameters (true for CNNModelModule).
    teacher = type(model)(**hparams)
    teacher.load_state_dict(model.state_dict())
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher

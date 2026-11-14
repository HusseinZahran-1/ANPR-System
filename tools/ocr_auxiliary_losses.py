"""Loss utilities for OCR development experiments; not runtime correction rules."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def freeze_batch_norm_stats(model: torch.nn.Module) -> None:
    """Freeze only BatchNorm running statistics for an auxiliary forward.

    Affine parameters still receive gradients. Recurrent layers retain their
    training mode, required by cuDNN for backward. Call ``model.train()`` before
    the next ordinary batch to restore its normal BatchNorm behavior.
    """
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.eval()


def masked_teacher_kl(student: torch.Tensor, teacher: torch.Tensor,
                      lengths: torch.Tensor, selected: torch.Tensor, *,
                      temperature: float = 1.0) -> torch.Tensor:
    """Average teacher KL over selected rows' valid CTC timesteps only.

    Inputs are matching (batch, time, classes) logits. The caller decides which
    reviewed training examples receive retention supervision. Teacher logits
    are always detached. Padding and unselected rows contribute no gradient.
    An empty selection returns a differentiable zero on the student's device.
    """
    if student.ndim != 3 or teacher.shape != student.shape:
        raise ValueError('student and teacher must have matching (batch,time,classes) shapes')
    batch, time, _ = student.shape
    if lengths.shape != (batch,) or selected.shape != (batch,) or selected.dtype != torch.bool:
        raise ValueError('lengths and boolean selection must have one item per batch row')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if lengths.dtype not in (torch.int32, torch.int64) or bool(((lengths < 0) | (lengths > time)).any()):
        raise ValueError('integer lengths must be between zero and the time dimension')
    mask = selected.to(student.device)[:, None] & (
        torch.arange(time, device=student.device)[None, :] < lengths.to(student.device)[:, None])
    if not bool(mask.any()):
        return student[mask].sum() * 0.0
    # Select before computing probabilities: arbitrary padded values must not
    # contaminate loss or gradients, even if those values contain NaN/Inf.
    student_valid = student[mask].float() / temperature
    teacher_valid = teacher.detach().to(student.device)[mask].float() / temperature
    return F.kl_div(F.log_softmax(student_valid, dim=-1),
                    F.softmax(teacher_valid, dim=-1), reduction='batchmean') * temperature**2

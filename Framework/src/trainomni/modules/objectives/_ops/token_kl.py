"""FP32 dense teacher-to-student token KL divergence."""

from __future__ import annotations

from trainomni.core.errors import ObjectiveError


def dense_token_kl(student_logits, teacher_logits, *, temperature: float):
    import torch
    from torch.nn import functional

    if student_logits.shape != teacher_logits.shape:
        raise ObjectiveError(
            "student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if not teacher_logits.is_floating_point():
        raise ObjectiveError("teacher logits must be floating point")
    student_log_probs = functional.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    teacher_probs = functional.softmax(teacher_logits.float() / temperature, dim=-1)
    teacher_log_probs = functional.log_softmax(
        teacher_logits.float() / temperature, dim=-1
    )
    return torch.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        dim=-1,
        dtype=torch.float32,
    )

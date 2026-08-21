"""Small native scheduler set with serializable PyTorch state."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR

from trainomni.specs.run import SchedulerSpec


def build_scheduler(spec: SchedulerSpec, optimizer, *, total_steps: int):
    def scale(step: int) -> float:
        if spec.warmup_steps and step < spec.warmup_steps:
            return max(step / spec.warmup_steps, 0.0)
        if spec.name == "constant":
            return 1.0
        decay_steps = max(total_steps - spec.warmup_steps, 1)
        progress = min(max((step - spec.warmup_steps) / decay_steps, 0.0), 1.0)
        if spec.name == "linear":
            decay = 1.0 - progress
        else:
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return spec.min_lr_ratio + (1.0 - spec.min_lr_ratio) * decay

    return LambdaLR(optimizer, lr_lambda=scale)

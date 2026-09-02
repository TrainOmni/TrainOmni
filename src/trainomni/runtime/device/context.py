"""Device placement and autocast policy owned by the runtime."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch

from trainomni.contracts.batch import OmniBatch
from trainomni.core.errors import SpecError


class DeviceContext:
    def __init__(self, device: str, precision: str) -> None:
        try:
            self.device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise SpecError(f"invalid run device {device!r}: {exc}") from exc
        self.precision = precision
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise SpecError(f"run requests {device!r}, but CUDA is unavailable")
        if precision not in {"fp32", "bf16_mixed", "fp16_mixed", "bf16_true"}:
            raise SpecError(f"unsupported precision: {precision!r}")
        if precision == "fp16_mixed" and self.device.type == "cpu":
            raise SpecError("fp16 autocast is not supported by the CPU runtime")
        if (
            precision in {"bf16_mixed", "bf16_true"}
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise SpecError("BF16 was requested, but the CUDA device does not support it")

    def prepare_model(self, model: Any) -> Any:
        if self.precision == "bf16_true":
            return model.to(device=self.device, dtype=torch.bfloat16)
        return model.to(self.device)

    def move(self, value: Any) -> Any:
        """Move model-forward inputs and apply true-precision input casting."""

        if isinstance(value, torch.Tensor):
            value = value.to(self.device, non_blocking=True)
            if self.precision == "bf16_true" and value.is_floating_point():
                value = value.to(torch.bfloat16)
            return value
        if isinstance(value, Mapping):
            return {key: self.move(inner) for key, inner in value.items()}
        if isinstance(value, tuple):
            return tuple(self.move(inner) for inner in value)
        if isinstance(value, list):
            return [self.move(inner) for inner in value]
        return value

    def move_exact(self, value: Any) -> Any:
        """Move identity-bearing labels/supervision without changing dtype."""

        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: self.move_exact(inner) for key, inner in value.items()}
        if isinstance(value, tuple):
            return tuple(self.move_exact(inner) for inner in value)
        if isinstance(value, list):
            return [self.move_exact(inner) for inner in value]
        return value

    def move_batch(self, batch: OmniBatch) -> OmniBatch:
        return replace(
            batch,
            model_inputs=self.move(batch.model_inputs),
            labels=self.move_exact(batch.labels),
            supervision=self.move_exact(batch.supervision),
        )

    def autocast(self):
        if self.precision in {"fp32", "bf16_true"}:
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16_mixed" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

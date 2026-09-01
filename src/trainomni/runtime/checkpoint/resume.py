"""RNG capture and restoration used by exact resume."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch

from trainomni.core.errors import CheckpointError


def _cpu_rng_tensor(value: Any, *, owner: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CheckpointError(f"checkpoint {owner} RNG state is not a tensor")
    if value.dtype != torch.uint8:
        raise CheckpointError(f"checkpoint {owner} RNG state is not a ByteTensor")
    return value.detach().cpu()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy
    except ImportError:
        pass
    else:
        state["numpy"] = numpy.random.get_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" not in state or "torch_cpu" not in state:
        raise CheckpointError("checkpoint is missing required RNG state")
    random.setstate(state["python"])
    torch.set_rng_state(_cpu_rng_tensor(state["torch_cpu"], owner="CPU"))
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise CheckpointError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise CheckpointError("checkpoint CUDA RNG state is not a sequence")
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointError(
                "checkpoint CUDA RNG state count differs from visible CUDA devices"
            )
        torch.cuda.set_rng_state_all(
            [
                _cpu_rng_tensor(value, owner=f"CUDA device {index}")
                for index, value in enumerate(cuda_states)
            ]
        )
    if "numpy" in state:
        try:
            import numpy
        except ImportError as exc:
            raise CheckpointError(
                "checkpoint contains NumPy RNG state but NumPy is unavailable"
            ) from exc
        numpy.random.set_state(state["numpy"])

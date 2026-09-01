"""FSDP2 full-state bridge for TrainOmni's portable checkpoint format."""

from __future__ import annotations

from typing import Any


class FSDP2StateAdapter:
    name = "torch_fsdp2_full_state"
    capture_is_collective = True
    # get_state_dict owns the internal collective sequence. TrainOmni
    # coordinates all pure rank-local runtime capture before entering it and
    # coordinates exceptions once it returns; failures inside a stuck backend
    # collective are bounded by the process-group timeout rather than by a
    # second TrainOmni collective.
    failure_boundary = "torch-distributed-process-group-timeout"

    @staticmethod
    def capture(model: Any, optimizer: Any):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
        )

        return get_state_dict(
            model,
            optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    @staticmethod
    def load_model(model: Any, model_state: dict[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            model,
            model_state,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )

    @staticmethod
    def load_training(
        model: Any,
        optimizer: Any,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
    ) -> None:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )

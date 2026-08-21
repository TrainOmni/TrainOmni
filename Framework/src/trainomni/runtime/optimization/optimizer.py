"""Optimizer construction from run configuration and parameter-policy output."""

from __future__ import annotations

from typing import Any

import torch

from trainomni.core.errors import SpecError
from trainomni.modules.parameters.protocol import ParameterSelection
from trainomni.specs.run import OptimizerSpec


def optimizer_metadata(optimizer: Any, spec: OptimizerSpec) -> dict[str, Any]:
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        state_dtypes = set()
        state_tensor_count = 0
        for parameter in group["params"]:
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor):
                    state_tensor_count += 1
                    state_dtypes.add(str(value.dtype))
        groups.append(
            {
                "name": str(group.get("group_name", f"group-{index}")),
                "parameter_count": len(group["params"]),
                "state_tensor_count": state_tensor_count,
                "state_dtypes": sorted(state_dtypes),
            }
        )
    return {
        "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "torch_version": torch.__version__,
        "name": spec.name,
        "foreach": spec.foreach,
        "betas": list(spec.betas),
        "eps": spec.eps,
        "quantized": False,
        "groups": groups,
    }


def build_optimizer(spec: OptimizerSpec, selection: ParameterSelection) -> Any:
    if spec.name != "adamw":
        raise SpecError(f"unsupported optimizer: {spec.name}")
    groups = []
    overrides = {override.name: override for override in spec.group_overrides}
    selected_names = {group.name for group in selection.groups}
    unknown_overrides = sorted(set(overrides) - selected_names)
    if unknown_overrides:
        raise SpecError(
            "optimizer overrides unknown parameter groups: "
            + ", ".join(unknown_overrides)
        )
    for group in selection.groups:
        if not group.parameters:
            raise SpecError(f"parameter group {group.name!r} is empty")
        if group.options:
            raise SpecError(
                f"parameter group {group.name!r} attempted to set run-owned "
                "optimizer options"
            )
        payload = {"params": list(group.parameters), "group_name": group.name}
        override = overrides.get(group.name)
        if override is not None:
            if override.learning_rate is not None:
                payload["lr"] = override.learning_rate
            if override.weight_decay is not None:
                payload["weight_decay"] = override.weight_decay
        groups.append(payload)
    if not groups:
        raise SpecError("parameter policy produced no optimizer groups")
    return torch.optim.AdamW(
        groups,
        lr=spec.learning_rate,
        betas=spec.betas,
        eps=spec.eps,
        weight_decay=spec.weight_decay,
        foreach=spec.foreach,
    )

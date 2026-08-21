"""One objective-directed forward/loss micro-step."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from trainomni.contracts.forward import ForwardResult
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.runtime.device.context import DeviceContext


def execute_forward_plan(
    *,
    model: Any,
    objective: Any,
    batch: Any,
    context: ObjectiveContext,
    device: DeviceContext,
):
    requirements = objective.requirements()
    missing_supervision = sorted(
        requirements.supervision_fields - set(batch.supervision)
    )
    if missing_supervision:
        raise ObjectiveError(
            "objective is missing required supervision fields before forward: "
            + ", ".join(missing_supervision)
        )
    plan = objective.plan(batch, context)
    outputs: dict[str, ForwardResult] = {}
    for request in plan.requests:
        inputs = request.inputs
        if not isinstance(inputs, Mapping):
            raise ObjectiveError(f"forward {request.name!r} inputs must be a mapping")
        # ForwardRequest is the semantic boundary between objective state and
        # model inputs. Supervision keeps its declared dtype (for example FP32
        # offline reference log-probs); only tensors actually sent to the model
        # follow true-precision input casting.
        inputs = device.move(inputs)
        gradient_context = torch.enable_grad() if request.requires_grad else torch.no_grad()
        with gradient_context, device.autocast():
            output = model(**inputs)
        result = ForwardResult(name=request.name, output=output)
        for field_name in (
            "logits",
            "hidden_states",
            "attentions",
            "modal_features",
        ):
            if getattr(request.outputs, field_name):
                result.require(field_name)
        outputs[request.name] = result
    loss = objective.compute(batch, outputs, context)
    if not isinstance(loss.total, torch.Tensor) or loss.total.ndim != 0:
        raise ObjectiveError("objective total loss must be a scalar tensor")
    if not bool(torch.isfinite(loss.total.detach()).item()):
        raise ObjectiveError("objective produced a non-finite total loss")
    for name, term in loss.terms.items():
        if not isinstance(term.value, torch.Tensor) or term.value.ndim != 0:
            raise ObjectiveError(f"loss term {name!r} value must be a scalar tensor")
        if not bool(torch.isfinite(term.value.detach()).item()):
            raise ObjectiveError(f"loss term {name!r} is non-finite")
        if not isinstance(term.denominator, torch.Tensor) or term.denominator.ndim != 0:
            raise ObjectiveError(
                f"loss term {name!r} denominator must be a scalar tensor"
            )
        if float(term.denominator.detach().float().item()) <= 0:
            raise ObjectiveError(f"loss term {name!r} denominator must be positive")
    return loss

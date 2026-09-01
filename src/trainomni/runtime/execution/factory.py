"""Execution backend construction and fail-closed capability negotiation."""

from __future__ import annotations

from trainomni.runtime.device.context import DeviceContext

from .deepspeed_backend import build_deepspeed_backend
from .process import ProcessContext
from .torch_backends import build_torch_backend


def build_execution_backend(*, model, selection, run):
    process = ProcessContext.create(run.execution, requested_device=run.device)
    try:
        resolved_device = process.resolve_device(run.device)
        device = DeviceContext(resolved_device, run.precision)
        device.prepare_model(model)
        if run.execution.backend == "deepspeed":
            backend = build_deepspeed_backend(
                model=model,
                selection=selection,
                run=run,
                process=process,
            )
        else:
            backend = build_torch_backend(
                model=model,
                selection=selection,
                run=run,
                process=process,
            )
        backend.device = device
        return backend
    except Exception:
        process.close()
        raise

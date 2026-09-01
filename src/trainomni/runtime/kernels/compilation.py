"""Run-owned torch.compile boundary that leaves checkpoint state unwrapped."""

from __future__ import annotations

import torch

from trainomni.core.errors import SpecError
from trainomni.specs.run import CompileSpec


def compile_forward(model, spec: CompileSpec):
    if not spec.enabled:
        return model
    keyword_arguments = {
        "fullgraph": spec.fullgraph,
        "dynamic": spec.dynamic,
    }
    if spec.backend is not None:
        keyword_arguments["backend"] = spec.backend
    if spec.mode is not None:
        keyword_arguments["mode"] = spec.mode
    try:
        return torch.compile(model, **keyword_arguments)
    except Exception as exc:
        raise SpecError(f"torch.compile initialization failed: {exc}") from exc

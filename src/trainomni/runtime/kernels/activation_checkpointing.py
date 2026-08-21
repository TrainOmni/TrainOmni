"""Component-scoped activation-checkpointing negotiation."""

from __future__ import annotations

from trainomni.core.errors import SpecError
from trainomni.specs.run import ActivationCheckpointSpec


def apply_activation_checkpointing(model, spec: ActivationCheckpointSpec) -> tuple[str, ...]:
    if not spec.enabled:
        return ()
    applied = []
    for component_name in spec.components:
        try:
            component = model.get_submodule(component_name)
        except AttributeError as exc:
            raise SpecError(
                f"activation checkpoint component does not exist: {component_name}"
            ) from exc
        explicit = getattr(component, "enable_activation_checkpointing", None)
        transformers_hook = getattr(component, "gradient_checkpointing_enable", None)
        try:
            if callable(explicit):
                explicit(use_reentrant=spec.use_reentrant)
            elif callable(transformers_hook):
                transformers_hook(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": spec.use_reentrant,
                    }
                )
            else:
                raise SpecError(
                    f"component {component_name!r} has no activation-checkpointing hook"
                )
        except SpecError:
            raise
        except Exception as exc:
            raise SpecError(
                f"activation checkpointing failed for {component_name!r}: {exc}"
            ) from exc
        applied.append(component_name)
    return tuple(applied)

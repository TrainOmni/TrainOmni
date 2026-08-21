"""Apply a run-selected attention implementation to compatible model modules."""

from __future__ import annotations

from trainomni.core.errors import SpecError


def apply_attention_kernel(model, implementation: str) -> tuple[str, ...]:
    if implementation == "auto":
        return ()
    configured = []
    covered_prefixes = []
    for name, module in model.named_modules():
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in covered_prefixes
        ):
            continue
        setter = getattr(module, "set_attn_implementation", None)
        if not callable(setter):
            continue
        try:
            setter(implementation)
        except Exception as exc:
            label = name or "<root>"
            raise SpecError(
                f"attention kernel {implementation!r} rejected by {label}: {exc}"
            ) from exc
        configured.append(name or "<root>")
        if not name:
            break
        covered_prefixes.append(name)
    if not configured:
        raise SpecError(
            f"attention kernel {implementation!r} was requested, but the model exposes "
            "no set_attn_implementation boundary"
        )
    return tuple(configured)

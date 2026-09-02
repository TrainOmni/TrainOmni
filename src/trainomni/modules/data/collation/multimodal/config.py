"""Strict tensor collation configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from trainomni.modules.data._validation import require_int, require_number

_MODES = {"auto", "stack", "pad", "concat", "list"}


@dataclass(frozen=True, slots=True, kw_only=True)
class MultimodalCollatorConfig:
    pad_token_id: int = 0
    label_pad_id: int = -100
    padding_side: str = "right"
    pad_to_multiple_of: int | None = None
    field_modes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    field_pad_values: Mapping[str, int | float] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        require_int(self.pad_token_id, field="pad_token_id")
        require_int(self.label_pad_id, field="label_pad_id")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        if self.pad_to_multiple_of is not None:
            require_int(
                self.pad_to_multiple_of,
                field="pad_to_multiple_of",
                minimum=1,
            )
        if not isinstance(self.field_modes, Mapping):
            raise TypeError("field_modes must be a mapping")
        if not isinstance(self.field_pad_values, Mapping):
            raise TypeError("field_pad_values must be a mapping")
        modes = {}
        for path, mode in self.field_modes.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError("field_modes keys must be non-empty field paths")
            if mode not in _MODES:
                raise ValueError(
                    f"field_modes.{path} must be one of {', '.join(sorted(_MODES))}"
                )
            modes[path] = mode
        pad_values = {}
        for path, value in self.field_pad_values.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError("field_pad_values keys must be non-empty field paths")
            require_number(value, field=f"field_pad_values.{path}")
            pad_values[path] = value
        object.__setattr__(
            self, "field_modes", MappingProxyType(dict(sorted(modes.items())))
        )
        object.__setattr__(
            self,
            "field_pad_values",
            MappingProxyType(dict(sorted(pad_values.items()))),
        )

"""Basic Transformers processor configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from trainomni.contracts._mapping import FrozenDict
from trainomni.core.assets import validate_asset_fields
from trainomni.modules.data._fields import validate_field_paths
from trainomni.modules.data._validation import (
    normalize_string_sequence,
    require_bool,
    require_int,
    require_string,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformersModelIOConfig:
    processor_name_or_path: str
    revision: str | None = None
    asset_manifest_sha256: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    text_separator: str = "\n"
    conversation_mode: str = "auto"
    add_generation_prompt: bool = False
    require_assistant_mask: bool = True
    assistant_mask_fields: tuple[str, ...] = (
        "assistant_masks",
        "assistant_mask",
    )
    loss_mask_field: str = "loss_mask"
    supervision_metadata_key: str | None = None
    batch_axis_fields: tuple[str, ...] = (
        "input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids",
    )
    field_routes: Mapping[str, str] = field(default_factory=FrozenDict)
    discard_fields: tuple[str, ...] = ()
    unmapped_fields: str = "keep"
    modal_token_id: int | None = None
    modal_positions_field: str = "modal_positions"

    def __post_init__(self) -> None:
        if not isinstance(self.field_routes, Mapping):
            raise TypeError("field_routes must map processor field paths to model field paths")
        routes = dict(self.field_routes)
        validate_field_paths(routes)
        validate_field_paths(routes.values())
        discard = normalize_string_sequence(self.discard_fields, field="discard_fields")
        if set(routes) & set(discard):
            raise ValueError("field_routes and discard_fields overlap")
        if "input_ids" in discard or routes.get("input_ids", "input_ids") != "input_ids":
            raise ValueError("input_ids must remain available at the model input root")
        if self.unmapped_fields not in {"keep", "error"}:
            raise ValueError("unmapped_fields must be keep or error")
        if self.modal_token_id is not None:
            require_int(self.modal_token_id, field="modal_token_id", minimum=0)
        require_string(self.modal_positions_field, field="modal_positions_field")
        validate_field_paths((self.modal_positions_field,))
        object.__setattr__(self, "field_routes", FrozenDict(routes))
        object.__setattr__(self, "discard_fields", discard)
        require_string(
            self.processor_name_or_path,
            field="processor_name_or_path",
        )
        require_bool(self.trust_remote_code, field="trust_remote_code")
        require_bool(self.local_files_only, field="local_files_only")
        require_string(self.text_separator, field="text_separator", allow_empty=True)
        require_bool(self.add_generation_prompt, field="add_generation_prompt")
        require_bool(self.require_assistant_mask, field="require_assistant_mask")
        validate_asset_fields(
            revision=self.revision,
            asset_manifest_sha256=self.asset_manifest_sha256,
        )
        if self.conversation_mode not in {"auto", "required", "disabled"}:
            raise ValueError("conversation_mode must be auto, required, or disabled")
        assistant_mask_fields = normalize_string_sequence(
            self.assistant_mask_fields,
            field="assistant_mask_fields",
            allow_empty_sequence=False,
        )
        require_string(self.loss_mask_field, field="loss_mask_field")
        if self.supervision_metadata_key is not None:
            require_string(
                self.supervision_metadata_key,
                field="supervision_metadata_key",
            )
        object.__setattr__(self, "assistant_mask_fields", assistant_mask_fields)
        object.__setattr__(
            self,
            "batch_axis_fields",
            normalize_string_sequence(self.batch_axis_fields, field="batch_axis_fields"),
        )

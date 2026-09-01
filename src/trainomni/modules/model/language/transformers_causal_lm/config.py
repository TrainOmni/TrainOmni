"""Transformers causal-language component configuration."""

from dataclasses import dataclass

from trainomni.core.assets import validate_asset_fields


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformersCausalLMConfig:
    model_name_or_path: str
    revision: str | None = None
    asset_manifest_sha256: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")
        validate_asset_fields(
            revision=self.revision,
            asset_manifest_sha256=self.asset_manifest_sha256,
        )

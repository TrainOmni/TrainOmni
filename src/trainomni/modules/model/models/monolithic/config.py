"""Transformers-compatible monolithic VLM configuration."""

from dataclasses import dataclass

from trainomni.core.assets import validate_asset_fields


@dataclass(frozen=True, slots=True, kw_only=True)
class MonolithicModelConfig:
    model_name_or_path: str
    auto_class: str = "AutoModelForImageTextToText"
    revision: str | None = None
    asset_manifest_sha256: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")
        if self.auto_class not in {
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        }:
            raise ValueError("unsupported Transformers auto_class")
        validate_asset_fields(
            revision=self.revision,
            asset_manifest_sha256=self.asset_manifest_sha256,
        )

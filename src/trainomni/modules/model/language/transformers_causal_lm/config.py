"""Transformers causal-language component configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformersCausalLMConfig:
    model_name_or_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")

"""Strict, versioned user configuration models.

Backend-specific options deliberately live under ``config`` mappings. Public
fields reject unknown keys so typos cannot silently change a training run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUN_SCHEMA_VERSION = "trainomni.run.v1"
STAGE_TYPES = frozenset(
    {
        "vision_preparation",
        "modality_alignment",
        "multimodal_pretraining",
        "capability_curriculum",
        "instruction_sft",
        "reasoning_distillation",
        "reward_verifier",
        "offline_preference",
        "online_rl",
        "agentic_rl",
        "evaluate_export",
    }
)
PRECISIONS = frozenset({"fp32", "tf32", "fp16", "bf16", "fp8"})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(StrictModel):
    plugin: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class DatasetSpec(StrictModel):
    dataset_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    importer: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)
    config: dict[str, Any] = Field(default_factory=dict)


class DataSpec(StrictModel):
    datasets: tuple[DatasetSpec, ...] = ()
    modalities: frozenset[str] = frozenset({"text"})
    content_blocks: frozenset[str] = frozenset({"text"})
    max_media_per_sample: int = Field(default=0, ge=0)
    packing: bool = False
    padding_free: bool = False
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_media_count(self) -> DataSpec:
        media_modalities = self.modalities - {"text"}
        if media_modalities and self.max_media_per_sample < 1:
            raise ValueError(
                "max_media_per_sample must be >= 1 when media modalities are used"
            )
        if not media_modalities and self.max_media_per_sample:
            raise ValueError(
                "max_media_per_sample must be 0 for a text-only data specification"
            )
        return self


class PeftSpec(StrictModel):
    method: Literal["lora", "qlora"]
    rank: int = Field(default=8, gt=0)
    alpha: float = Field(default=16.0, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = ()
    modules_to_save: tuple[str, ...] = ()
    task_type: str = "CAUSAL_LM"
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_targets(self) -> PeftSpec:
        if not self.target_modules or any(not item.strip() for item in self.target_modules):
            raise ValueError("PEFT requires non-blank target_modules")
        return self


class ComponentPolicy(StrictModel):
    trainable: bool
    learning_rate: float | None = Field(default=None, gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    dtype: str | None = None
    gradient_clip: float | None = Field(default=None, gt=0)
    activation_checkpointing: bool | None = None
    peft: PeftSpec | None = None

    @model_validator(mode="after")
    def validate_trainable_options(self) -> ComponentPolicy:
        if not self.trainable and any(
            item is not None
            for item in (
                self.learning_rate,
                self.weight_decay,
                self.gradient_clip,
                self.peft,
            )
        ):
            raise ValueError(
                "frozen components cannot define optimizer or PEFT options"
            )
        return self


class OptimizationSpec(StrictModel):
    optimizer: str = "adamw"
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    max_steps: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_budget(self) -> OptimizationSpec:
        if self.max_steps is None and self.max_tokens is None:
            raise ValueError("one of max_steps or max_tokens is required")
        return self


class EngineSpec(StrictModel):
    backend: str = "torch"
    parallelism: str = "single"
    precision: str = "bf16"
    attention_backend: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_precision(self) -> EngineSpec:
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"unsupported precision {self.precision!r}; "
                f"expected one of {sorted(PRECISIONS)}"
            )
        return self


class CheckpointSpec(StrictModel):
    every_steps: int | None = Field(default=None, gt=0)
    resume_level: Literal["exact", "stage_boundary", "weights_only", "transfer"] = (
        "exact"
    )
    export_formats: frozenset[str] = frozenset({"hf"})
    config: dict[str, Any] = Field(default_factory=dict)


class StageSpec(StrictModel):
    stage_id: str = Field(min_length=1)
    stage_type: str
    # ``objective`` is the semantic sample contract (cpt/sft/preference/
    # prompt_only). ``objective_impl`` selects a loss/algorithm plugin.
    objective: str = Field(min_length=1)
    objective_impl: str | None = Field(default=None, min_length=1)
    data: DataSpec
    component_policy: dict[str, ComponentPolicy] = Field(default_factory=dict)
    optimization: OptimizationSpec
    engine: EngineSpec = EngineSpec()
    checkpoint: CheckpointSpec = CheckpointSpec()
    inputs: dict[str, str] = Field(default_factory=dict)
    evaluations: tuple[dict[str, Any], ...] = ()
    gates: tuple[dict[str, Any], ...] = ()
    outputs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stage_type(self) -> StageSpec:
        if self.stage_type not in STAGE_TYPES:
            raise ValueError(
                f"unsupported stage_type {self.stage_type!r}; "
                f"expected one of {sorted(STAGE_TYPES)}"
            )
        if self.stage_type != "evaluate_export" and not self.data.datasets:
            raise ValueError("a training stage requires at least one dataset")
        return self


class RunSpec(StrictModel):
    schema_version: Literal["trainomni.run.v1"] = RUN_SCHEMA_VERSION
    name: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0)
    model: ModelSpec
    stage: StageSpec
    metadata: dict[str, Any] = Field(default_factory=dict)

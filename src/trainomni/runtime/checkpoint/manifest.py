"""Human-readable checkpoint identity and per-file integrity manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trainomni.core.errors import CheckpointError


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema_version: int
    framework_version: str
    task_digest: str
    run_digest: str
    module_lock: Mapping[str, str]
    global_step: int
    micro_step: int
    model_file: str
    model_sha256: str
    optimizer_file: str
    optimizer_sha256: str
    runtime_file: str
    runtime_sha256: str
    runtime_metadata: Mapping[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "framework_version": self.framework_version,
            "task_digest": self.task_digest,
            "run_digest": self.run_digest,
            "module_lock": dict(sorted(self.module_lock.items())),
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "optimizer_file": self.optimizer_file,
            "optimizer_sha256": self.optimizer_sha256,
            "runtime_file": self.runtime_file,
            "runtime_sha256": self.runtime_sha256,
            "runtime_metadata": self.runtime_metadata,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointManifest:
        allowed = {
            "schema_version",
            "framework_version",
            "task_digest",
            "run_digest",
            "module_lock",
            "global_step",
            "micro_step",
            "model_file",
            "model_sha256",
            "optimizer_file",
            "optimizer_sha256",
            "runtime_file",
            "runtime_sha256",
            "runtime_metadata",
        }
        unknown = sorted(set(value) - allowed)
        missing = sorted(allowed - set(value))
        if unknown or missing:
            raise CheckpointError(
                f"invalid checkpoint manifest keys; missing={missing}, unknown={unknown}"
            )
        if value["schema_version"] != 2:
            raise CheckpointError(
                f"unsupported checkpoint schema: {value['schema_version']!r}"
            )
        raw_lock = value["module_lock"]
        if not isinstance(raw_lock, Mapping):
            raise CheckpointError("checkpoint module_lock must be a mapping")
        raw_runtime_metadata = value["runtime_metadata"]
        if not isinstance(raw_runtime_metadata, Mapping):
            raise CheckpointError("checkpoint runtime_metadata must be a mapping")

        def file_name(field: str) -> str:
            result = value[field]
            if (
                not isinstance(result, str)
                or not result
                or "/" in result
                or "\\" in result
                or result in {".", ".."}
            ):
                raise CheckpointError(f"checkpoint {field} must be a plain file name")
            return result

        def digest(field: str) -> str:
            result = value[field]
            if (
                not isinstance(result, str)
                or len(result) != 64
                or any(character not in "0123456789abcdef" for character in result)
            ):
                raise CheckpointError(f"checkpoint {field} must be a lowercase SHA-256")
            return result

        global_step = int(value["global_step"])
        micro_step = int(value["micro_step"])
        if global_step < 0 or micro_step < 0:
            raise CheckpointError("checkpoint steps must be non-negative")
        return cls(
            schema_version=2,
            framework_version=str(value["framework_version"]),
            task_digest=str(value["task_digest"]),
            run_digest=str(value["run_digest"]),
            module_lock={str(key): str(inner) for key, inner in raw_lock.items()},
            global_step=global_step,
            micro_step=micro_step,
            model_file=file_name("model_file"),
            model_sha256=digest("model_sha256"),
            optimizer_file=file_name("optimizer_file"),
            optimizer_sha256=digest("optimizer_sha256"),
            runtime_file=file_name("runtime_file"),
            runtime_sha256=digest("runtime_sha256"),
            runtime_metadata=dict(raw_runtime_metadata),
        )

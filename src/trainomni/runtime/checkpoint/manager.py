"""Atomic split model/optimizer/runtime checkpoint manager."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from trainomni.core.errors import CheckpointError

from .manifest import CheckpointManifest
from .resume import capture_rng_state, restore_rng_state

_MODEL_FILE = "model.safetensors"
_OPTIMIZER_FILE = "optimizer.pt"
_RUNTIME_FILE = "runtime.pt"
_MANIFEST_FILE = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointManager:
    def __init__(
        self,
        *,
        root: Path,
        task_digest: str,
        run_digest: str,
        module_lock: Mapping[str, str],
        framework_version: str = "0.1.0",
    ) -> None:
        self.root = Path(root)
        self.task_digest = task_digest
        self.run_digest = run_digest
        self.module_lock = dict(sorted(module_lock.items()))
        self.framework_version = framework_version
        self.loaded_runtime_metadata: dict[str, Any] = {}

    def step_path(self, global_step: int) -> Path:
        return self.root / f"step-{global_step:08d}"

    @staticmethod
    def _identity(
        *,
        task_digest: str,
        run_digest: str,
        module_lock: Mapping[str, str],
        global_step: int,
        micro_step: int,
    ) -> dict[str, Any]:
        return {
            "task_digest": task_digest,
            "run_digest": run_digest,
            "module_lock": dict(module_lock),
            "global_step": global_step,
            "micro_step": micro_step,
        }

    def save(
        self,
        *,
        global_step: int,
        micro_step: int,
        model: Any,
        optimizer: Any,
        objective: Any,
        stream: Any,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if global_step < 0:
            raise CheckpointError("global_step must be non-negative")
        if micro_step != 0:
            raise CheckpointError("checkpoints are only valid at optimizer-step boundaries")
        directory = self.step_path(global_step)
        self.root.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            raise CheckpointError(f"refusing to overwrite checkpoint: {directory}")
        staging = self.root / f".{directory.name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(exist_ok=False)
        runtime_metadata = dict(runtime_metadata or {})
        identity = self._identity(
            task_digest=self.task_digest,
            run_digest=self.run_digest,
            module_lock=self.module_lock,
            global_step=global_step,
            micro_step=micro_step,
        )
        try:
            try:
                from safetensors.torch import save_model
            except ImportError as exc:
                raise CheckpointError("checkpointing requires safetensors") from exc
            model_path = staging / _MODEL_FILE
            save_model(
                model,
                str(model_path),
                metadata={
                    "format": "pt",
                    "task_digest": self.task_digest,
                    "run_digest": self.run_digest,
                    "global_step": str(global_step),
                },
            )
            optimizer_path = staging / _OPTIMIZER_FILE
            temporary_optimizer = staging / f".{_OPTIMIZER_FILE}.tmp"
            torch.save(
                {"identity": identity, "optimizer": optimizer.state_dict()},
                temporary_optimizer,
            )
            os.replace(temporary_optimizer, optimizer_path)
            runtime_path = staging / _RUNTIME_FILE
            temporary_runtime = staging / f".{_RUNTIME_FILE}.tmp"
            torch.save(
                {
                    "identity": identity,
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "objective": dict(objective.state_dict()),
                    "stream": dict(stream.state_dict()),
                    "scaler": None if scaler is None else scaler.state_dict(),
                    "rng": capture_rng_state(),
                    "runtime_metadata": runtime_metadata,
                },
                temporary_runtime,
            )
            os.replace(temporary_runtime, runtime_path)
            manifest = CheckpointManifest(
                schema_version=2,
                framework_version=self.framework_version,
                task_digest=self.task_digest,
                run_digest=self.run_digest,
                module_lock=self.module_lock,
                global_step=global_step,
                micro_step=micro_step,
                model_file=_MODEL_FILE,
                model_sha256=_sha256(model_path),
                optimizer_file=_OPTIMIZER_FILE,
                optimizer_sha256=_sha256(optimizer_path),
                runtime_file=_RUNTIME_FILE,
                runtime_sha256=_sha256(runtime_path),
                runtime_metadata=runtime_metadata,
            )
            temporary_manifest = staging / f".{_MANIFEST_FILE}.tmp"
            temporary_manifest.write_text(
                json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, staging / _MANIFEST_FILE)
            os.replace(staging, directory)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return directory

    def _read_manifest(self, checkpoint: Path) -> tuple[Path, CheckpointManifest]:
        directory = Path(checkpoint)
        manifest_path = directory / _MANIFEST_FILE
        if not manifest_path.is_file():
            raise CheckpointError(f"checkpoint manifest is missing: {manifest_path}")
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read checkpoint manifest: {exc}") from exc
        if not isinstance(raw_manifest, Mapping):
            raise CheckpointError("checkpoint manifest root must be a mapping")
        manifest = CheckpointManifest.from_mapping(raw_manifest)
        self._validate_manifest(manifest)
        return directory, manifest

    @staticmethod
    def _validate_file(directory: Path, name: str, expected_digest: str) -> Path:
        path = directory / name
        if not path.is_file():
            raise CheckpointError(f"checkpoint file is missing: {path}")
        actual = _sha256(path)
        if actual != expected_digest:
            raise CheckpointError(
                f"checkpoint file digest mismatch for {name}: "
                f"expected {expected_digest}, got {actual}"
            )
        return path

    @staticmethod
    def _load_torch(path: Path, *, map_location: Any, owner: str) -> Mapping[str, Any]:
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as exc:
            raise CheckpointError(f"cannot load checkpoint {owner}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise CheckpointError(f"checkpoint {owner} root must be a mapping")
        return payload

    @staticmethod
    def _validate_payload_identity(
        payload: Mapping[str, Any],
        manifest: CheckpointManifest,
        *,
        owner: str,
    ) -> None:
        identity = payload.get("identity")
        expected = CheckpointManager._identity(
            task_digest=manifest.task_digest,
            run_digest=manifest.run_digest,
            module_lock=manifest.module_lock,
            global_step=manifest.global_step,
            micro_step=manifest.micro_step,
        )
        if not isinstance(identity, Mapping) or dict(identity) != expected:
            raise CheckpointError(f"checkpoint {owner} identity mismatch")

    def load_model_only(
        self,
        checkpoint: Path,
        *,
        model: Any,
        map_location: Any,
        objective: Any | None = None,
    ) -> CheckpointManifest:
        directory, manifest = self._read_manifest(checkpoint)
        model_path = self._validate_file(
            directory, manifest.model_file, manifest.model_sha256
        )
        try:
            from safetensors.torch import load_model
        except ImportError as exc:
            raise CheckpointError("checkpoint loading requires safetensors") from exc
        try:
            missing, unexpected = load_model(
                model,
                model_path,
                strict=True,
                device=str(map_location),
            )
        except Exception as exc:
            raise CheckpointError(f"checkpoint model is incompatible: {exc}") from exc
        if missing or unexpected:
            raise CheckpointError(
                f"checkpoint model keys differ: missing={missing}, unexpected={unexpected}"
            )
        if objective is not None:
            runtime_path = self._validate_file(
                directory, manifest.runtime_file, manifest.runtime_sha256
            )
            runtime = self._load_torch(
                runtime_path,
                map_location="cpu",
                owner="runtime state",
            )
            self._validate_payload_identity(runtime, manifest, owner="runtime state")
            if "objective" not in runtime:
                raise CheckpointError("checkpoint runtime state has no objective")
            try:
                objective.load_state_dict(runtime["objective"])
            except Exception as exc:
                raise CheckpointError(
                    f"checkpoint objective state is incompatible: {exc}"
                ) from exc
        self.loaded_runtime_metadata = dict(manifest.runtime_metadata)
        return manifest

    def load(
        self,
        checkpoint: Path,
        *,
        model: Any,
        optimizer: Any,
        objective: Any,
        stream: Any,
        map_location: Any,
        scheduler: Any | None = None,
        scaler: Any | None = None,
    ) -> tuple[int, int]:
        directory, manifest = self._read_manifest(checkpoint)
        model_path = self._validate_file(
            directory, manifest.model_file, manifest.model_sha256
        )
        optimizer_path = self._validate_file(
            directory, manifest.optimizer_file, manifest.optimizer_sha256
        )
        runtime_path = self._validate_file(
            directory, manifest.runtime_file, manifest.runtime_sha256
        )
        optimizer_payload = self._load_torch(
            optimizer_path,
            map_location=map_location,
            owner="optimizer state",
        )
        runtime = self._load_torch(
            runtime_path,
            map_location=map_location,
            owner="runtime state",
        )
        self._validate_payload_identity(
            optimizer_payload, manifest, owner="optimizer state"
        )
        self._validate_payload_identity(runtime, manifest, owner="runtime state")
        if set(optimizer_payload) != {"identity", "optimizer"}:
            raise CheckpointError("checkpoint optimizer state keys are invalid")
        required_runtime = {
            "identity",
            "scheduler",
            "objective",
            "stream",
            "scaler",
            "rng",
            "runtime_metadata",
        }
        if set(runtime) != required_runtime:
            raise CheckpointError("checkpoint runtime state keys are invalid")
        if (runtime["scaler"] is None) != (scaler is None):
            raise CheckpointError("checkpoint scaler state is incompatible with this run")
        if (runtime["scheduler"] is None) != (scheduler is None):
            raise CheckpointError("checkpoint scheduler state is incompatible with this run")
        if runtime["runtime_metadata"] != manifest.runtime_metadata:
            raise CheckpointError("checkpoint runtime metadata disagrees with manifest")
        try:
            from safetensors.torch import load_model
        except ImportError as exc:
            raise CheckpointError("checkpoint loading requires safetensors") from exc
        try:
            missing, unexpected = load_model(
                model,
                model_path,
                strict=True,
                device=str(map_location),
            )
            if missing or unexpected:
                raise CheckpointError(
                    f"checkpoint model keys differ: missing={missing}, "
                    f"unexpected={unexpected}"
                )
            optimizer.load_state_dict(optimizer_payload["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(runtime["scheduler"])
            objective.load_state_dict(runtime["objective"])
            stream.load_state_dict(runtime["stream"])
            if scaler is not None:
                scaler.load_state_dict(runtime["scaler"])
            restore_rng_state(runtime["rng"])
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc
        self.loaded_runtime_metadata = dict(manifest.runtime_metadata)
        return manifest.global_step, manifest.micro_step

    def _validate_manifest(self, manifest: CheckpointManifest) -> None:
        mismatches = []
        if manifest.task_digest != self.task_digest:
            mismatches.append("task_digest")
        if manifest.run_digest != self.run_digest:
            mismatches.append("run_digest")
        if dict(manifest.module_lock) != self.module_lock:
            mismatches.append("module_lock")
        if manifest.framework_version != self.framework_version:
            mismatches.append("framework_version")
        if manifest.micro_step != 0:
            mismatches.append("micro_step")
        if mismatches:
            raise CheckpointError(
                "checkpoint identity mismatch: " + ", ".join(sorted(mismatches))
            )

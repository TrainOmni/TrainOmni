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


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(
                _state_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        )
    return left == right

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
        compatible_run_digests: tuple[str, ...] = (),
        framework_version: str = "0.1.2",
        process: Any | None = None,
        state_adapter: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.task_digest = task_digest
        self.run_digest = run_digest
        self.compatible_run_digests = frozenset(compatible_run_digests)
        self.module_lock = dict(sorted(module_lock.items()))
        self.framework_version = framework_version
        self.process = process
        self.state_adapter = state_adapter
        self.loaded_runtime_metadata: dict[str, Any] = {}

    @property
    def _is_primary(self) -> bool:
        return self.process is None or bool(self.process.is_primary)

    @property
    def _world_size(self) -> int:
        return 1 if self.process is None else int(self.process.world_size)

    @property
    def _rank(self) -> int:
        return 0 if self.process is None else int(self.process.rank)

    def _barrier(self) -> None:
        if self.process is not None:
            self.process.barrier()

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
        staging = self.root / f".{directory.name}.{uuid.uuid4().hex}.tmp"
        identity = self._identity(
            task_digest=self.task_digest,
            run_digest=self.run_digest,
            module_lock=self.module_lock,
            global_step=global_step,
            micro_step=micro_step,
        )
        local_runtime = None
        local_failure = None
        try:
            runtime_metadata = dict(runtime_metadata or {})
            local_runtime = {
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "objective": dict(objective.state_dict()),
                "stream": dict(stream.state_dict()),
                "scaler": None if scaler is None else scaler.state_dict(),
                "rng": capture_rng_state(),
                "runtime_metadata": runtime_metadata,
            }
        except Exception as exc:  # noqa: BLE001 - synchronize rank-local capture
            local_failure = exc
        if self.process is not None:
            self.process.propagate_rank_failure(
                local_failure,
                owner="checkpoint runtime capture",
                error_type=CheckpointError,
            )
        elif local_failure is not None:
            raise CheckpointError(
                "checkpoint runtime capture failed on rank 0: "
                f"{type(local_failure).__name__}: {local_failure}"
            ) from local_failure
        if local_runtime is None:  # pragma: no cover - guarded by coordinated failure
            raise CheckpointError("checkpoint runtime capture produced no state")

        captured_model = None
        captured_optimizer = None
        adapter_failure = None
        if self.state_adapter is not None:
            try:
                # FSDP2 capture may itself contain collectives.  All pure local
                # runtime state has already passed the all-rank outcome phase;
                # exceptions raised after adapter capture returns are coordinated
                # below.  A backend-internal collective failure remains the
                # distributed backend's timeout/error boundary.
                captured_model, captured_optimizer = self.state_adapter.capture(
                    model, optimizer
                )
            except Exception as exc:  # noqa: BLE001 - coordinate returned failures
                adapter_failure = exc
            if self.process is not None:
                self.process.propagate_rank_failure(
                    adapter_failure,
                    owner="checkpoint state-adapter capture",
                    error_type=CheckpointError,
                )
            elif adapter_failure is not None:
                raise CheckpointError(
                    "checkpoint state-adapter capture failed on rank 0: "
                    f"{type(adapter_failure).__name__}: {adapter_failure}"
                ) from adapter_failure
        rank_states = None
        if self._world_size > 1:
            if self._is_primary:
                rank_states = [None] * self._world_size
            torch.distributed.gather_object(
                local_runtime,
                object_gather_list=rank_states,
                dst=0,
            )
        if not self._is_primary:
            self.process.propagate_primary_failure(
                None,
                owner="checkpoint save",
                error_type=CheckpointError,
            )
            return directory
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                raise CheckpointError(f"refusing to overwrite checkpoint: {directory}")
            staging.mkdir(exist_ok=False)
        except Exception as exc:
            if self.process is not None:
                self.process.propagate_primary_failure(
                    exc,
                    owner="checkpoint save",
                    error_type=CheckpointError,
                )
            raise
        try:
            try:
                from safetensors.torch import save_file, save_model
            except ImportError as exc:
                raise CheckpointError("checkpointing requires safetensors") from exc
            model_path = staging / _MODEL_FILE
            model_metadata = {
                "format": "pt",
                "task_digest": self.task_digest,
                "run_digest": self.run_digest,
                "global_step": str(global_step),
            }
            if captured_model is None:
                save_model(model, str(model_path), metadata=model_metadata)
            else:
                portable_model = {
                    name: value.detach().cpu().contiguous().clone()
                    for name, value in captured_model.items()
                }
                save_file(portable_model, str(model_path), metadata=model_metadata)
            optimizer_path = staging / _OPTIMIZER_FILE
            temporary_optimizer = staging / f".{_OPTIMIZER_FILE}.tmp"
            torch.save(
                {
                    "identity": identity,
                    "optimizer": (
                        optimizer.state_dict()
                        if captured_optimizer is None
                        else captured_optimizer
                    ),
                },
                temporary_optimizer,
            )
            os.replace(temporary_optimizer, optimizer_path)
            runtime_path = staging / _RUNTIME_FILE
            temporary_runtime = staging / f".{_RUNTIME_FILE}.tmp"
            runtime_payload = (
                {"identity": identity, **local_runtime}
                if rank_states is None
                else {
                    "identity": identity,
                    "distributed_world_size": self._world_size,
                    "rank_states": rank_states,
                    "runtime_metadata": runtime_metadata,
                }
            )
            torch.save(runtime_payload, temporary_runtime)
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
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if self.process is not None:
                self.process.propagate_primary_failure(
                    exc,
                    owner="checkpoint save",
                    error_type=CheckpointError,
                )
            raise
        if self.process is not None:
            self.process.propagate_primary_failure(
                None,
                owner="checkpoint save",
                error_type=CheckpointError,
            )
        return directory

    def _read_manifest(
        self,
        checkpoint: Path,
        *,
        require_run_digest: bool = True,
    ) -> tuple[Path, CheckpointManifest]:
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
        self._validate_manifest(manifest, require_run_digest=require_run_digest)
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
        directory, manifest = self._read_manifest(
            checkpoint,
            require_run_digest=False,
        )
        model_path = self._validate_file(
            directory, manifest.model_file, manifest.model_sha256
        )
        try:
            from safetensors.torch import load_file, load_model
        except ImportError as exc:
            raise CheckpointError("checkpoint loading requires safetensors") from exc
        try:
            if self.state_adapter is None:
                missing, unexpected = load_model(
                    model,
                    model_path,
                    strict=True,
                    device=str(map_location),
                )
            else:
                self.state_adapter.load_model(model, load_file(model_path, device="cpu"))
                missing, unexpected = (), ()
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
            if "rank_states" in runtime:
                rank_states = runtime["rank_states"]
                if (
                    not isinstance(rank_states, (tuple, list))
                    or int(runtime.get("distributed_world_size", -1)) <= 1
                    or len(rank_states)
                    != int(runtime.get("distributed_world_size", -1))
                ):
                    raise CheckpointError(
                        "distributed checkpoint rank states are incompatible"
                    )
                if any(
                    not isinstance(state, Mapping) or "objective" not in state
                    for state in rank_states
                ):
                    raise CheckpointError(
                        "distributed checkpoint objective state is missing"
                    )
                objective_state = rank_states[0]["objective"]
                if any(
                    not _state_equal(objective_state, state["objective"])
                    for state in rank_states[1:]
                ):
                    raise CheckpointError(
                        "distributed objective state is rank-dependent and cannot be "
                        "restored portably"
                    )
                runtime_state = rank_states[0]
            else:
                runtime_state = runtime
            if not isinstance(runtime_state, Mapping) or "objective" not in runtime_state:
                raise CheckpointError("checkpoint runtime state has no objective")
            try:
                objective.load_state_dict(runtime_state["objective"])
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
        distributed_runtime = "rank_states" in runtime
        if distributed_runtime:
            expected_distributed = {
                "identity",
                "distributed_world_size",
                "rank_states",
                "runtime_metadata",
            }
            if set(runtime) != expected_distributed:
                raise CheckpointError("distributed checkpoint runtime keys are invalid")
            if int(runtime["distributed_world_size"]) != self._world_size:
                raise CheckpointError("distributed checkpoint world size changed")
            rank_states = runtime["rank_states"]
            if not isinstance(rank_states, (tuple, list)) or len(rank_states) != self._world_size:
                raise CheckpointError("distributed checkpoint rank states are invalid")
            runtime_state = rank_states[self._rank]
            if not isinstance(runtime_state, Mapping):
                raise CheckpointError("distributed checkpoint local rank state is invalid")
        else:
            if set(runtime) != required_runtime:
                raise CheckpointError("checkpoint runtime state keys are invalid")
            runtime_state = runtime
        if (runtime_state["scaler"] is None) != (scaler is None):
            raise CheckpointError("checkpoint scaler state is incompatible with this run")
        if (runtime_state["scheduler"] is None) != (scheduler is None):
            raise CheckpointError("checkpoint scheduler state is incompatible with this run")
        if runtime_state["runtime_metadata"] != manifest.runtime_metadata:
            raise CheckpointError("checkpoint runtime metadata disagrees with manifest")
        try:
            from safetensors.torch import load_file, load_model
        except ImportError as exc:
            raise CheckpointError("checkpoint loading requires safetensors") from exc
        try:
            if self.state_adapter is None:
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
            else:
                self.state_adapter.load_training(
                    model,
                    optimizer,
                    load_file(model_path, device="cpu"),
                    optimizer_payload["optimizer"],
                )
            if scheduler is not None:
                scheduler.load_state_dict(runtime_state["scheduler"])
            objective.load_state_dict(runtime_state["objective"])
            stream.load_state_dict(runtime_state["stream"])
            if scaler is not None:
                scaler.load_state_dict(runtime_state["scaler"])
            restore_rng_state(runtime_state["rng"])
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc
        self.loaded_runtime_metadata = dict(manifest.runtime_metadata)
        return manifest.global_step, manifest.micro_step

    def _validate_manifest(
        self,
        manifest: CheckpointManifest,
        *,
        require_run_digest: bool = True,
    ) -> None:
        mismatches = []
        if manifest.task_digest != self.task_digest:
            mismatches.append("task_digest")
        if require_run_digest and manifest.run_digest not in {
            self.run_digest,
            *self.compatible_run_digests,
        }:
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

"""PyTorch Distributed Checkpoint adapter for FSDP2/DTensor state."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .local import CheckpointError
from .state import StateRegistry

DCP_CHECKPOINT_VERSION = "trainomni.dcp-checkpoint.v1"


class DCPApplicationState:
    """Canonical model/optimizer FQNs across DDP/FSDP2 parallelisms."""

    def __init__(self, model: Any, optimizer: Any) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Mapping[str, Any]:
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optimizer": optimizer_state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import set_state_dict

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )


class DCPModelState:
    def __init__(self, model: Any) -> None:
        self.model = model

    def state_dict(self) -> Mapping[str, Any]:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        return get_model_state_dict(self.model)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        set_model_state_dict(self.model, dict(state))


class DCPCheckpointManager:
    def __init__(self, root: Path, torch: Any) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.torch = torch

    def save(
        self,
        name: str,
        *,
        model: Any,
        optimizer: Any,
        runtime: StateRegistry,
        metadata: Mapping[str, Any],
    ) -> Path:
        dcp = _dcp()
        target = self.root / name
        if target.exists():
            raise CheckpointError(f"checkpoint already exists: {target}")
        temporary = self.root / f".{name}.incomplete-{self._shared_uuid()}"
        if self._rank() == 0:
            temporary.mkdir()
            (temporary / "INCOMPLETE").write_text("not committed\n", encoding="utf-8")
        self._barrier()
        dcp.save(
            {
                "application": DCPApplicationState(model, optimizer),
                "model_only": DCPModelState(model),
            },
            checkpoint_id=temporary,
        )
        runtime_path = temporary / f"runtime-rank-{self._rank():05d}.pkl"
        with runtime_path.open("wb") as handle:
            pickle.dump(runtime.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        self._barrier()
        if self._rank() == 0:
            runtime_files = {
                path.name: {"size": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(temporary.glob("runtime-rank-*.pkl"))
            }
            if len(runtime_files) != self._world_size():
                raise CheckpointError("DCP checkpoint is missing rank-local runtime state")
            manifest = {
                "schema_version": DCP_CHECKPOINT_VERSION,
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "complete": True,
                "world_size": self._world_size(),
                "trusted_runtime_serialization": "python-pickle",
                "runtime_files": runtime_files,
                "metadata": dict(metadata),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            (temporary / "INCOMPLETE").unlink()
            os.replace(temporary, target)
        self._barrier()
        return target

    def load(
        self,
        name: str,
        *,
        model: Any,
        optimizer: Any,
        runtime: StateRegistry,
        trusted: bool,
    ) -> Mapping[str, Any]:
        dcp = _dcp()
        target = self.root / name
        manifest_path = target / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid DCP checkpoint manifest: {target}") from exc
        if (
            manifest.get("schema_version") != DCP_CHECKPOINT_VERSION
            or manifest.get("complete") is not True
            or (target / "INCOMPLETE").exists()
        ):
            raise CheckpointError("DCP checkpoint is incomplete or incompatible")
        if not trusted:
            raise CheckpointError(
                "DCP exact resume contains rank-local pickle state; pass trusted=True "
                "only for a trusted run"
            )
        if manifest.get("world_size") != self._world_size():
            raise CheckpointError(
                "exact DCP runtime resume currently requires the same world size"
            )
        dcp.load(
            {"application": DCPApplicationState(model, optimizer)},
            checkpoint_id=target,
        )
        runtime_path = target / f"runtime-rank-{self._rank():05d}.pkl"
        expected = manifest.get("runtime_files", {}).get(runtime_path.name, {})
        if (
            not runtime_path.is_file()
            or runtime_path.stat().st_size != expected.get("size")
            or _sha256(runtime_path) != expected.get("sha256")
        ):
            raise CheckpointError("DCP rank-local runtime state is corrupt")
        try:
            with runtime_path.open("rb") as handle:
                runtime_state = pickle.load(handle)
        except Exception as exc:
            raise CheckpointError("cannot deserialize trusted DCP runtime state") from exc
        if not isinstance(runtime_state, Mapping):
            raise CheckpointError("DCP runtime state root must be a mapping")
        runtime.load_state_dict(runtime_state)
        metadata = manifest.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    def load_model(self, name: str, model: Any) -> None:
        target = self.root / name
        dcp = _dcp()
        dcp.load({"model_only": DCPModelState(model)}, checkpoint_id=target)

    def list_complete(self) -> tuple[str, ...]:
        values = []
        for path in self.root.iterdir():
            if path.name.startswith(".") or not path.is_dir():
                continue
            manifest = path / "manifest.json"
            if manifest.is_file() and not (path / "INCOMPLETE").exists():
                values.append(path.name)
        return tuple(sorted(values))

    def _rank(self) -> int:
        if self.torch.distributed.is_initialized():
            return int(self.torch.distributed.get_rank())
        return 0

    def _world_size(self) -> int:
        if self.torch.distributed.is_initialized():
            return int(self.torch.distributed.get_world_size())
        return 1

    def _barrier(self) -> None:
        if self.torch.distributed.is_initialized():
            self.torch.distributed.barrier()

    def _shared_uuid(self) -> str:
        value = [uuid.uuid4().hex if self._rank() == 0 else ""]
        if self.torch.distributed.is_initialized():
            self.torch.distributed.broadcast_object_list(value, src=0)
        return value[0]


def _dcp() -> Any:
    try:
        import torch.distributed.checkpoint as dcp
    except ImportError as exc:  # pragma: no cover - optional/version-specific
        raise CheckpointError("PyTorch Distributed Checkpoint is unavailable") from exc
    return dcp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

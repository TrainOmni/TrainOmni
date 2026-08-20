"""Atomic trusted local checkpoints for single-process development.

Scale engines use DCP/backend-native storage. This manager is the correctness
oracle for local state and explicitly requires trust before unpickling.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import StateRegistry

LOCAL_CHECKPOINT_VERSION = "trainomni.local-checkpoint.v1"
_CHECKPOINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CheckpointError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalCheckpointManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise CheckpointError(f"checkpoint root is not a directory: {self.root}")

    def save(
        self,
        name: str,
        registry: StateRegistry,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        self._validate_name(name)
        target = self._target(name)
        if target.exists():
            raise CheckpointError(f"checkpoint already exists: {target}")
        temporary = self.root / f".{name}.incomplete-{uuid.uuid4().hex}"
        temporary.mkdir()
        incomplete = temporary / "INCOMPLETE"
        incomplete.write_text("checkpoint is not committed\n", encoding="utf-8")
        state_path = temporary / "state.pkl"
        with state_path.open("wb") as handle:
            pickle.dump(registry.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "schema_version": LOCAL_CHECKPOINT_VERSION,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "trusted_serialization": "python-pickle",
            "files": {
                "state.pkl": {
                    "size": state_path.stat().st_size,
                    "sha256": _sha256(state_path),
                }
            },
            "metadata": dict(metadata or {}),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        # If a write above fails, the incomplete marker is deliberately retained.
        incomplete.unlink()
        os.replace(temporary, target)
        return target

    def load(
        self,
        name: str,
        registry: StateRegistry,
        *,
        trusted: bool,
        strict: bool = True,
    ) -> Mapping[str, Any]:
        self._validate_name(name)
        if not trusted:
            raise CheckpointError(
                "local checkpoint uses pickle; pass trusted=True only for a trusted run"
            )
        target = self._target(name)
        manifest_path = target / "manifest.json"
        state_path = target / "state.pkl"
        if not target.is_dir() or (target / "INCOMPLETE").exists():
            raise CheckpointError(f"checkpoint is missing or incomplete: {target}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid checkpoint manifest: {manifest_path}") from exc
        if (
            manifest.get("schema_version") != LOCAL_CHECKPOINT_VERSION
            or manifest.get("complete") is not True
        ):
            raise CheckpointError("checkpoint manifest version/completion mismatch")
        expected = manifest.get("files", {}).get("state.pkl", {})
        if not state_path.is_file() or state_path.stat().st_size != expected.get("size"):
            raise CheckpointError("checkpoint state size mismatch")
        if _sha256(state_path) != expected.get("sha256"):
            raise CheckpointError("checkpoint state SHA-256 mismatch")
        try:
            with state_path.open("rb") as handle:
                state = pickle.load(handle)
        except Exception as exc:
            raise CheckpointError("cannot deserialize trusted checkpoint state") from exc
        if not isinstance(state, Mapping):
            raise CheckpointError("checkpoint state root must be a mapping")
        registry.load_state_dict(state, strict=strict)
        metadata = manifest.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    def list_complete(self) -> tuple[str, ...]:
        names = []
        for path in self.root.iterdir():
            if path.name.startswith(".") or not path.is_dir():
                continue
            if (path / "INCOMPLETE").exists():
                continue
            manifest = path / "manifest.json"
            if manifest.is_file():
                names.append(path.name)
        return tuple(sorted(names))

    def _validate_name(self, name: str) -> None:
        if not _CHECKPOINT_NAME.fullmatch(name):
            raise CheckpointError(f"invalid checkpoint name {name!r}")

    def _target(self, name: str) -> Path:
        target = (self.root / name).resolve()
        if target.parent != self.root:
            raise CheckpointError("checkpoint target escapes checkpoint root")
        return target

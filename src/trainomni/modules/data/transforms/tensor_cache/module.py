"""Load per-sample tensors from immutable safetensors sidecars before forward."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from trainomni.contracts.sample import OmniSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TensorCacheConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TensorCacheTransform:
    def __init__(self, config: TensorCacheConfig, *, task_root: Path | None) -> None:
        self.config = config
        path = Path(config.index_path)
        if not path.is_absolute():
            if task_root is None:
                raise SpecError("relative tensor-cache index requires a task root")
            path = task_root / path
        self.index_path = path.resolve()
        if not self.index_path.is_file():
            raise SpecError(f"tensor-cache index does not exist: {self.index_path}")
        if _sha256(self.index_path) != config.index_sha256:
            raise SpecError("tensor-cache index digest mismatch")
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SpecError(f"cannot read tensor-cache index: {exc}") from exc
        if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "samples"}:
            raise SpecError("tensor-cache index root is invalid")
        if raw["schema_version"] != 1 or not isinstance(raw["samples"], Mapping):
            raise SpecError("unsupported tensor-cache index schema")
        self.entries = {
            str(sample_id): self._validate_entry(str(sample_id), entry)
            for sample_id, entry in raw["samples"].items()
        }
        self._loaded = {}

    def _validate_entry(self, sample_id: str, entry):
        if not sample_id or not isinstance(entry, Mapping) or set(entry) != {
            "file",
            "sha256",
            "tensors",
        }:
            raise SpecError(f"tensor-cache entry {sample_id!r} is invalid")
        filename = entry["file"]
        digest = entry["sha256"]
        tensors = entry["tensors"]
        if not isinstance(filename, str) or not filename:
            raise SpecError(f"tensor-cache entry {sample_id!r} has invalid file")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise SpecError("tensor-cache files must stay below the index directory")
        path = (self.index_path.parent / relative).resolve()
        try:
            path.relative_to(self.index_path.parent)
        except ValueError as exc:
            raise SpecError("tensor-cache file escapes the index directory") from exc
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SpecError(f"tensor-cache entry {sample_id!r} has invalid digest")
        if (
            not isinstance(tensors, Mapping)
            or not tensors
            or any(
                not isinstance(output, str)
                or not output
                or not isinstance(source, str)
                or not source
                for output, source in tensors.items()
            )
        ):
            raise SpecError(f"tensor-cache entry {sample_id!r} has invalid tensors")
        return path, digest, dict(tensors)

    def _load(self, path: Path, digest: str):
        key = (path, digest)
        if key in self._loaded:
            return self._loaded[key]
        if not path.is_file() or _sha256(path) != digest:
            raise SpecError(f"tensor-cache file digest mismatch: {path}")
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise SpecError("tensor-cache loading requires safetensors") from exc
        tensors = load_file(path, device="cpu")
        self._loaded[key] = tensors
        return tensors

    def apply(self, sample: OmniSample) -> OmniSample:
        if self.config.metadata_key in sample.metadata:
            raise SpecError(
                f"sample metadata already contains {self.config.metadata_key!r}"
            )
        try:
            path, digest, names = self.entries[sample.sample_id]
        except KeyError as exc:
            raise SpecError(
                f"tensor-cache index has no sample {sample.sample_id!r}"
            ) from exc
        source = self._load(path, digest)
        missing = sorted(set(names.values()) - set(source))
        if missing:
            raise SpecError(
                f"tensor-cache sample {sample.sample_id!r} is missing tensors: "
                + ", ".join(missing)
            )
        metadata = dict(sample.metadata)
        metadata[self.config.metadata_key] = {
            output: source[tensor_name] for output, tensor_name in names.items()
        }
        return OmniSample(
            sample.sample_id,
            sample.content,
            metadata,
            sample.messages,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("sample_transform:trainomni/tensor_cache@1"),
        config_type=TensorCacheConfig,
        factory=lambda config, context: TensorCacheTransform(
            config, task_root=context.task_root
        ),
        provides=CapabilitySet.of({"sample.cache.tensors"}),
        requires=CapabilitySet.of({"data.sample.omni"}),
    )

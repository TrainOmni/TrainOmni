"""Dependency-free static inspection of Hugging Face checkpoint assets.

The probe reads configuration and SafeTensors headers only. It never loads
tensor payloads or imports model code, so it is safe to use in recipe dry-runs.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trainomni.data.model import freeze_json, thaw_json

MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


class ProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    filename: str


@dataclass(frozen=True, slots=True)
class CheckpointProbe:
    path: Path
    architecture: str
    model_type: str
    hidden_size: int | None
    vocab_size: int | None
    max_position_embeddings: int | None
    dtype: str | None
    config: Mapping[str, Any]
    tokenizer_config: Mapping[str, Any]
    processor_config: Mapping[str, Any]
    remote_code_required: bool
    python_files: tuple[str, ...]
    tensors: Mapping[str, TensorInfo]
    prefix_counts: tuple[tuple[str, int], ...]

    def tensor(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise ProbeError(f"tensor {name!r} is not present in {self.path}") from exc


@dataclass(frozen=True, slots=True)
class CompositeCompatibility:
    compatible: bool
    vision_output_dim: int | None
    language_hidden_dim: int | None
    connector_in_dim: int | None
    connector_out_dim: int | None
    source_vocab_size: int | None
    target_vocab_size: int | None
    reserved_placeholder_token: str | None
    reserved_placeholder_id: int | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


def _load_json(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ProbeError(f"required file is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot parse JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"JSON root must be an object: {path}")
    return value


def _read_safetensors_header(path: Path) -> dict[str, TensorInfo]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ProbeError(f"truncated SafeTensors length prefix: {path}")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
                raise ProbeError(
                    f"unsafe SafeTensors header length {header_length} in {path}"
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ProbeError(f"truncated SafeTensors header: {path}")
    except OSError as exc:
        raise ProbeError(f"cannot read SafeTensors file {path}: {exc}") from exc
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"invalid SafeTensors header JSON in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ProbeError(f"SafeTensors header root must be an object: {path}")

    data_bytes = file_size - 8 - header_length
    tensors: dict[str, TensorInfo] = {}
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ProbeError(f"invalid tensor entry {name!r} in {path}")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(size, int) and size >= 0 for size in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
        ):
            raise ProbeError(f"invalid tensor metadata for {name!r} in {path}")
        start, end = offsets
        if start < 0 or end < start or end > data_bytes:
            raise ProbeError(f"tensor {name!r} has out-of-range offsets in {path}")
        tensors[name] = TensorInfo(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            data_offsets=(start, end),
            filename=path.name,
        )
    return tensors


def _load_tensor_headers(root: Path) -> dict[str, TensorInfo]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path, required=True)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ProbeError(f"weight_map is missing or empty: {index_path}")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
            raise ProbeError(f"weight_map must map strings to strings: {index_path}")
        tensors: dict[str, TensorInfo] = {}
        for filename in sorted(set(weight_map.values())):
            shard = root / filename
            for name, info in _read_safetensors_header(shard).items():
                if name in tensors:
                    raise ProbeError(f"tensor {name!r} appears in multiple shards")
                tensors[name] = info
        index_names = set(weight_map)
        header_names = set(tensors)
        if index_names != header_names:
            missing = sorted(index_names - header_names)[:5]
            extra = sorted(header_names - index_names)[:5]
            raise ProbeError(
                f"index/header tensor mismatch; missing={missing}, extra={extra}"
            )
        for name, filename in weight_map.items():
            if tensors[name].filename != filename:
                raise ProbeError(
                    f"index sends {name!r} to {filename!r}, header uses "
                    f"{tensors[name].filename!r}"
                )
        return tensors

    candidates = sorted(root.glob("*.safetensors"))
    candidates = [path for path in candidates if not path.name.endswith(".index.safetensors")]
    if len(candidates) != 1:
        raise ProbeError(
            f"expected one SafeTensors file or an index under {root}, found {len(candidates)}"
        )
    return _read_safetensors_header(candidates[0])


def _prefix_counts(names: list[str]) -> tuple[tuple[str, int], ...]:
    counts = Counter(".".join(name.split(".")[:2]) for name in names)
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def probe_checkpoint(path: str | Path) -> CheckpointProbe:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ProbeError(f"checkpoint path is not a directory: {root}")
    config = _load_json(root / "config.json", required=True)
    tokenizer_config = _load_json(root / "tokenizer_config.json", required=False)
    processor_config = _load_json(root / "preprocessor_config.json", required=False)
    architectures = config.get("architectures")
    architecture = (
        architectures[0]
        if isinstance(architectures, list)
        and architectures
        and isinstance(architectures[0], str)
        else "unknown"
    )
    primary = config.get("text_config")
    if not isinstance(primary, dict):
        primary = config
    tensors = _load_tensor_headers(root)
    python_files = tuple(path.name for path in sorted(root.glob("*.py")))
    auto_map = config.get("auto_map") or tokenizer_config.get("auto_map")
    remote_code_required = bool(auto_map or python_files)
    frozen_tensors = MappingProxyType(dict(tensors))
    return CheckpointProbe(
        path=root,
        architecture=architecture,
        model_type=str(config.get("model_type", "unknown")),
        hidden_size=(
            primary.get("hidden_size")
            if isinstance(primary.get("hidden_size"), int)
            else None
        ),
        vocab_size=(
            primary.get("vocab_size")
            if isinstance(primary.get("vocab_size"), int)
            else None
        ),
        max_position_embeddings=(
            primary.get("max_position_embeddings")
            if isinstance(primary.get("max_position_embeddings"), int)
            else None
        ),
        dtype=(
            primary.get("dtype")
            or primary.get("torch_dtype")
            or config.get("torch_dtype")
        ),
        config=freeze_json(config),
        tokenizer_config=freeze_json(tokenizer_config),
        processor_config=freeze_json(processor_config),
        remote_code_required=remote_code_required,
        python_files=python_files,
        tensors=frozen_tensors,
        prefix_counts=_prefix_counts(list(tensors)),
    )


def _reserved_token(tokenizer_config: Mapping[str, Any]) -> tuple[str | None, int | None]:
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, Mapping):
        return None, None
    candidates: list[tuple[int, str]] = []
    for raw_id, value in decoder.items():
        if not isinstance(value, Mapping):
            continue
        content = value.get("content")
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if isinstance(content, str) and content.startswith("<unused_token_"):
            candidates.append((token_id, content))
    if not candidates:
        return None, None
    token_id, content = min(candidates)
    return content, token_id


def analyze_composite(
    vision_checkpoint: CheckpointProbe, language_checkpoint: CheckpointProbe
) -> CompositeCompatibility:
    """Derive connector/tokenizer constraints without importing Transformers."""

    issues: list[str] = []
    warnings: list[str] = []
    vision_config = vision_checkpoint.config.get("vision_config")
    if not isinstance(vision_config, Mapping):
        issues.append("vision checkpoint has no vision_config")
        vision_output_dim = None
    else:
        raw_output_dim = vision_config.get("out_hidden_size") or vision_config.get(
            "hidden_size"
        )
        vision_output_dim = raw_output_dim if isinstance(raw_output_dim, int) else None
        if vision_output_dim is None:
            issues.append("vision output dimension is not declared")
    language_hidden_dim = language_checkpoint.hidden_size
    if language_hidden_dim is None:
        issues.append("language hidden size is not declared")
    if not any(name.startswith("model.visual.") for name in vision_checkpoint.tensors):
        warnings.append("vision tensor prefix model.visual.* was not found")
    if language_checkpoint.vocab_size is None:
        issues.append("language vocabulary size is not declared")
    token, token_id = _reserved_token(language_checkpoint.tokenizer_config)
    if token is None:
        warnings.append("language tokenizer has no reserved <unused_token_N> placeholder")
    if vision_checkpoint.remote_code_required:
        warnings.append("vision checkpoint requires remote/custom code")
    if language_checkpoint.remote_code_required:
        warnings.append("language checkpoint requires remote/custom code")
    if vision_checkpoint.vocab_size != language_checkpoint.vocab_size:
        warnings.append(
            "vision and language tokenizers have different vocabularies; "
            "the composite must use the language tokenizer and its own media token profile"
        )
    return CompositeCompatibility(
        compatible=not issues,
        vision_output_dim=vision_output_dim,
        language_hidden_dim=language_hidden_dim,
        connector_in_dim=vision_output_dim,
        connector_out_dim=language_hidden_dim,
        source_vocab_size=vision_checkpoint.vocab_size,
        target_vocab_size=language_checkpoint.vocab_size,
        reserved_placeholder_token=token,
        reserved_placeholder_id=token_id,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def probe_to_dict(probe: CheckpointProbe) -> dict[str, Any]:
    return {
        "path": str(probe.path),
        "architecture": probe.architecture,
        "model_type": probe.model_type,
        "hidden_size": probe.hidden_size,
        "vocab_size": probe.vocab_size,
        "max_position_embeddings": probe.max_position_embeddings,
        "dtype": probe.dtype,
        "remote_code_required": probe.remote_code_required,
        "python_files": list(probe.python_files),
        "tensor_count": len(probe.tensors),
        "prefix_counts": [list(item) for item in probe.prefix_counts],
        "processor_config": thaw_json(probe.processor_config),
    }


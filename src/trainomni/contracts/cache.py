"""Stable value identity used by offline supervision cache producers/consumers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch


def _digest_bytes(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _update_model_inputs_digest(digest, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided or value.is_quantized:
            raise ValueError("cache model-input identity requires dense tensors")
        normalized = value.detach().to(device="cpu").contiguous()
        digest.update(b"tensor")
        _digest_bytes(digest, str(normalized.dtype).encode("ascii"))
        _digest_bytes(digest, repr(tuple(normalized.shape)).encode("ascii"))
        raw = normalized.reshape(-1).view(torch.uint8).numpy().tobytes()
        _digest_bytes(digest, raw)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("cache model-input identity requires string mapping keys")
        digest.update(b"mapping")
        digest.update(len(value).to_bytes(8, "big"))
        for key in sorted(value):
            _digest_bytes(digest, key.encode("utf-8"))
            _update_model_inputs_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"tuple" if isinstance(value, tuple) else b"list")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_model_inputs_digest(digest, item)
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"bool1" if value else b"bool0")
        return
    if isinstance(value, int):
        digest.update(b"int")
        _digest_bytes(digest, str(value).encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"float")
        _digest_bytes(digest, value.hex().encode("ascii"))
        return
    if isinstance(value, str):
        digest.update(b"str")
        _digest_bytes(digest, value.encode("utf-8"))
        return
    raise ValueError(
        "cache model-input identity supports tensors, nested mappings/sequences, "
        "and scalar values only"
    )


def model_inputs_digest(inputs: Mapping[str, Any]) -> str:
    """Hash one uncollated sample's complete model-input mapping."""

    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("cache model-input identity requires a non-empty mapping")
    digest = hashlib.sha256()
    digest.update(b"trainomni.model-inputs.v1\0")
    _update_model_inputs_digest(digest, inputs)
    return digest.hexdigest()


def digest_tensor(value: str, *, device=None) -> torch.Tensor:
    """Represent one lowercase SHA-256 digest as 32 uint8 values."""

    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("cache identity must be a lowercase SHA-256 digest")
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8, device=device)


def current_model_inputs_field(cache_field: str) -> str:
    """Return the reserved supervision field for a current-input digest."""

    if not cache_field:
        raise ValueError("cache field must not be empty")
    return f"__current_model_inputs__{cache_field}__sha256"

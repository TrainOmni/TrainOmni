"""Human- and machine-readable canonical data inspection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .canonical import sample_to_dict
from .importers import ImportedSample
from .model import CanonicalSample


def sample_summary(sample: CanonicalSample) -> dict[str, Any]:
    blocks: dict[str, int] = {}
    roles: dict[str, int] = {}
    for message in sample.messages:
        roles[message.role] = roles.get(message.role, 0) + 1
        for block in message.content:
            blocks[block.type] = blocks.get(block.type, 0) + 1
    return {
        "sample_id": sample.id,
        "objective": sample.objective,
        "messages": len(sample.messages),
        "roles": dict(sorted(roles.items())),
        "blocks": dict(sorted(blocks.items())),
        "assets": [
            {
                "id": asset.id,
                "modality": asset.modality,
                "uri": asset.uri,
                "width": asset.width,
                "height": asset.height,
                "num_frames": asset.num_frames,
            }
            for asset in sample.assets
        ],
        "has_preference": sample.preference is not None,
        "has_rollout": sample.rollout is not None,
    }


def inspect_imported_sample(
    imported: ImportedSample, *, include_canonical: bool = False
) -> dict[str, Any]:
    value = {
        "trace": imported.trace.to_dict(),
        "summary": sample_summary(imported.sample),
    }
    if include_canonical:
        value["canonical"] = sample_to_dict(imported.sample)
    return value


def take_round_robin(
    streams: Iterable[Iterable[ImportedSample]], limit: int
) -> tuple[ImportedSample, ...]:
    if limit < 1:
        raise ValueError("inspection limit must be positive")
    iterators = [iter(stream) for stream in streams]
    results: list[ImportedSample] = []
    while iterators and len(results) < limit:
        remaining = []
        for iterator in iterators:
            if len(results) >= limit:
                break
            try:
                results.append(next(iterator))
                remaining.append(iterator)
            except StopIteration:
                continue
        iterators = remaining
    return tuple(results)

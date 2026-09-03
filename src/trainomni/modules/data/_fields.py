"""Explicit dotted paths for nested processor and packing fields."""

from collections.abc import Mapping

from trainomni.core.errors import SpecError


def flatten_fields(value: Mapping, *, leaves=(), prefix="") -> dict:
    output = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or "." in key:
            raise SpecError(f"field keys must be non-empty strings without dots: {key!r}")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping) and path not in leaves:
            if not item:
                raise SpecError(f"nested field {path!r} must not be an empty mapping")
            output.update(flatten_fields(item, leaves=leaves, prefix=path))
        else:
            output[path] = item
    return output


def unflatten_fields(value: Mapping) -> dict:
    output = {}
    for path, item in value.items():
        parts = path.split(".")
        if any(not part for part in parts):
            raise SpecError(f"invalid field path: {path!r}")
        parent = output
        for part in parts[:-1]:
            if part in parent and not isinstance(parent[part], dict):
                raise SpecError(f"field path collides with another field: {path!r}")
            parent = parent.setdefault(part, {})
        if parts[-1] in parent:
            raise SpecError(f"field path collides with another field: {path!r}")
        parent[parts[-1]] = item
    return output


def validate_field_paths(paths) -> None:
    paths = tuple(paths)
    if any(not isinstance(path, str) or not path or any(not p for p in path.split("."))
           for path in paths):
        raise ValueError("field paths must contain non-empty dot-separated names")
    ordered = sorted(paths)
    for index, path in enumerate(ordered):
        if any(other == path or other.startswith(path + ".") for other in ordered[index + 1:]):
            raise ValueError(f"overlapping field paths: {path!r}")

"""Stable digest of the installed builtin Python implementation."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def builtin_source_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = tuple(sorted(package_root.rglob("*.py"), key=lambda path: path.as_posix()))
    if not files:  # pragma: no cover - an imported package necessarily has Python files
        raise RuntimeError("TrainOmni builtin source tree is empty")
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        # Git may materialize CRLF on Windows and LF on Linux. Provenance binds
        # Python semantics rather than checkout newline policy.
        payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()

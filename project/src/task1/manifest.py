"""Hashing and artifact validation helpers for the Task 1 audit manifest."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_required_artifacts(root: str | Path, relative_paths: Sequence[str]) -> list[Path]:
    base = Path(root)
    resolved = [base / relative for relative in relative_paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Task 1 artifacts: " + ", ".join(missing))
    return resolved

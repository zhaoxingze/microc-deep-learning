from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.task1.manifest import sha256_file, verify_required_artifacts


def test_sha256_file_matches_known_bytes(tmp_path: Path) -> None:
    path = tmp_path / "known.bin"
    path.write_bytes(b"micro-c")
    assert sha256_file(path) == hashlib.sha256(b"micro-c").hexdigest()


def test_required_artifact_verification_reports_missing(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("ok", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        verify_required_artifacts(tmp_path, ["present.txt", "missing.txt"])

"""Validate standalone .cool and .cool.gz files with cooler."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import cooler


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_cool_files(raw_dir: Path) -> list[Path]:
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.name.casefold().endswith((".cool", ".cool.gz"))
    )


@contextlib.contextmanager
def readable_cool_path(path: Path) -> Iterator[Path]:
    """Yield an HDF5 path, temporarily decompressing a real gzip if needed."""
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic != b"\x1f\x8b":
        yield path
        return

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cool", delete=False) as output:
            temporary = Path(output.name)
            with gzip.open(path, "rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        yield temporary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inspect(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "filepath": str(path.resolve()),
        "binsize": "",
        "shape": "",
        "chromnames": "",
        "chromsizes": "",
        "has_weight": False,
        "readable": False,
        "error": "",
    }
    try:
        with readable_cool_path(path) as open_path:
            contact_map = cooler.Cooler(str(open_path))
            row.update(
                {
                    "binsize": int(contact_map.binsize) if contact_map.binsize else "variable",
                    "shape": f"{int(contact_map.shape[0])}x{int(contact_map.shape[1])}",
                    "chromnames": json.dumps(list(contact_map.chromnames), ensure_ascii=False),
                    "chromsizes": json.dumps(
                        {str(k): int(v) for k, v in contact_map.chromsizes.items()},
                        ensure_ascii=False,
                    ),
                    "has_weight": "weight" in contact_map.bins().columns,
                    "readable": True,
                }
            )
    except Exception as exc:  # each file must produce a diagnostic row
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    _configure_stdout()
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=project_root.parent / "micro-c数据")
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "cool_file_summary.csv")
    args = parser.parse_args()

    files = find_cool_files(args.raw_dir.resolve())
    rows = [inspect(path) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "binsize", "shape", "chromnames", "chromsizes", "has_weight", "readable", "error"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cooler summary written: {args.output.resolve()}")
    print(f"Standalone Cooler files: {len(rows)}; readable: {sum(bool(row['readable']) for row in rows)}")
    for row in rows:
        print(
            f"- {row['filepath']} | binsize={row['binsize']} | shape={row['shape']} | "
            f"chromsizes={row['chromsizes']} | weight={row['has_weight']} | "
            f"readable={row['readable']} | error={row['error'] or '-'}"
        )
    if not rows:
        print("No standalone .cool/.cool.gz files found. Archive members are not extracted automatically.")


if __name__ == "__main__":
    main()

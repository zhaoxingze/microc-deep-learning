"""Inventory raw Micro-C inputs without changing or extracting them."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tarfile
from pathlib import Path


TARGET_SUFFIXES = (".cool", ".cool.gz", ".csv", ".xlsx", ".xls", ".bed", ".gff", ".gff3")
KEYWORDS = (
    "WT", "37C", "rep1", "rep2", "HNS", "FIS", "muk", "stpA", "bleo",
    "novo", "cipro", "net", "NoLig", "RT", "fix", "CHIN", "OPCID", "CHID",
)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _file_type(name: str) -> str:
    lowered = name.casefold()
    return ".cool.gz" if lowered.endswith(".cool.gz") else Path(name).suffix.casefold()


def _is_target(name: str) -> bool:
    lowered = name.casefold()
    return any(lowered.endswith(suffix) for suffix in TARGET_SUFFIXES)


def _classify(name: str, file_type: str) -> tuple[str, str, str]:
    lowered = name.casefold()
    keywords = ";".join(word for word in KEYWORDS if word.casefold() in lowered)
    category = "other"
    reason = ""
    if file_type in {".csv", ".xlsx", ".xls", ".bed", ".gff", ".gff3"}:
        category = "structure_annotation_candidate"
        reason = "tabular/annotation file; inspect contents"
    if any(token in lowered for token in ("hns", "fis", "muk", "stpa", "dstpa")):
        category = "mutation_condition"
        reason = "mutation keyword in name"
    elif any(token in lowered for token in ("bleo", "novo", "cipro", "net")):
        category = "drug_condition"
        reason = "drug keyword in name"
    elif any(token in lowered for token in ("nolig", "rt_fix", "37c_fix")):
        category = "technical_control"
        reason = "NoLig/RT/fix keyword in name"
    elif re.search(r"37c[_-]?rep[12]", lowered) and "fix" not in lowered:
        category = "likely_wt_candidate"
        reason = "standalone 37C biological replicate; WT not explicit in filename"
    elif "wt" in lowered:
        category = "wt_named_candidate"
        reason = "WT keyword in name"
    return keywords, category, reason


def inventory(raw_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        if _is_target(path.name):
            file_type = _file_type(path.name)
            keywords, category, reason = _classify(path.name, file_type)
            rows.append(
                {
                    "full_path": str(path.resolve()),
                    "file_name": path.name,
                    "file_size_bytes": path.stat().st_size,
                    "file_type": file_type,
                    "location_kind": "filesystem",
                    "container_path": "",
                    "matched_keywords": keywords,
                    "condition_category": category,
                    "classification_reason": reason,
                }
            )

        if path.suffix.casefold() == ".tar":
            try:
                with tarfile.open(path, mode="r") as archive:
                    for member in archive.getmembers():
                        if not member.isfile() or not _is_target(member.name):
                            continue
                        member_name = Path(member.name).name
                        file_type = _file_type(member_name)
                        keywords, category, reason = _classify(member_name, file_type)
                        rows.append(
                            {
                                "full_path": f"{path.resolve()}::{member.name}",
                                "file_name": member_name,
                                "file_size_bytes": member.size,
                                "file_type": file_type,
                                "location_kind": "archive_member",
                                "container_path": str(path.resolve()),
                                "matched_keywords": keywords,
                                "condition_category": category,
                                "classification_reason": reason,
                            }
                        )
            except (tarfile.TarError, OSError) as exc:
                print(f"WARNING: could not inspect archive {path}: {exc}")
    return rows


def main() -> None:
    _configure_stdout()
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=project_root.parent / "micro-c数据")
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "data_inventory.csv")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw data directory does not exist: {raw_dir}")
    rows = inventory(raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "full_path", "file_name", "file_size_bytes", "file_type", "location_kind",
        "container_path", "matched_keywords", "condition_category", "classification_reason",
    ]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cool_rows = [row for row in rows if row["file_type"] in {".cool", ".cool.gz"}]
    standalone = [row for row in cool_rows if row["location_kind"] == "filesystem"]
    archived = [row for row in cool_rows if row["location_kind"] == "archive_member"]
    print(f"Inventory written: {args.output.resolve()}")
    print(f"Cooler files: {len(cool_rows)} known ({len(standalone)} standalone, {len(archived)} inside archives)")
    for category, title in (
        ("likely_wt_candidate", "Most likely WT candidates"),
        ("wt_named_candidate", "Explicitly WT-named candidates"),
        ("mutation_condition", "Mutation conditions"),
        ("drug_condition", "Drug conditions"),
        ("technical_control", "Technical controls"),
        ("structure_annotation_candidate", "Possible structure annotation files"),
    ):
        matches = [row["full_path"] for row in rows if row["condition_category"] == category]
        print(f"{title} ({len(matches)}):")
        for match in matches:
            print(f"  - {match}")
    print("WT identity note: 37C rep1/rep2 are candidates, but 'WT' is not explicit in their filenames.")


if __name__ == "__main__":
    main()

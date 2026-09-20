"""Discover CHIN/OPCID/CHID annotations and build a unified coordinate table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls", ".bed"}
OUTPUT_COLUMNS = [
    "type", "chrom", "start", "end", "center", "length_bp", "source_file", "source_sheet"
]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _normalized(value: object) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum() or ch == "_")


def _detect_type(columns: list[object]) -> str | None:
    normalized = {_normalized(column) for column in columns}
    if "opcid_id" in normalized:
        return "OPCID"
    if "chin_id" in normalized:
        return "CHIN"
    if "chid_id" in normalized:
        return "CHID"
    return None


def _find_column(frame: pd.DataFrame, *names: str) -> str:
    lookup = {_normalized(column): str(column) for column in frame.columns}
    for name in names:
        if _normalized(name) in lookup:
            return lookup[_normalized(name)]
    raise KeyError(f"None of the required columns {names} were found in {list(frame.columns)}")


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("; ".join(errors))


def _standardize(
    frame: pd.DataFrame, structure_type: str, source_file: Path, source_sheet: str
) -> tuple[pd.DataFrame, int]:
    chrom_col = _find_column(frame, "Chr", "chrom", "chromosome")
    start_col = _find_column(frame, "Start", "start")
    end_col = _find_column(frame, "End", "end")
    id_col = _find_column(frame, f"{structure_type}_ID")

    relevant = frame[[id_col, chrom_col, start_col, end_col]].copy()
    relevant = relevant.loc[~relevant.isna().all(axis=1)]
    starts = pd.to_numeric(relevant[start_col], errors="coerce")
    ends = pd.to_numeric(relevant[end_col], errors="coerce")
    centers = (starts + ends) / 2.0

    supplied_center_mismatches = 0
    try:
        center_col = _find_column(frame, "Center", "center")
    except KeyError:
        center_col = ""
    if center_col:
        supplied = pd.to_numeric(frame.loc[relevant.index, center_col], errors="coerce")
        comparable = supplied.notna() & centers.notna()
        supplied_center_mismatches = int((~np.isclose(supplied[comparable], centers[comparable])).sum())

    output = pd.DataFrame(
        {
            "type": structure_type,
            "chrom": relevant[chrom_col].astype("string").str.strip(),
            "start": starts,
            "end": ends,
            "center": centers,
            "length_bp": ends - starts,
            "source_file": str(source_file.resolve()),
            "source_sheet": source_sheet,
        }
    )
    return output, supplied_center_mismatches


def _discover_tables(path: Path) -> list[tuple[str, str, pd.DataFrame]]:
    found: list[tuple[str, str, pd.DataFrame]] = []
    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        print(f"Excel sheets in {path.resolve()}:")
        for sheet in workbook.sheet_names:
            print(f"  - {sheet}")
            header = pd.read_excel(workbook, sheet_name=sheet, nrows=0)
            structure_type = _detect_type(list(header.columns))
            if structure_type:
                found.append((structure_type, sheet, pd.read_excel(workbook, sheet_name=sheet)))
    elif suffix == ".csv":
        header = _read_csv(path, nrows=0)
        structure_type = _detect_type(list(header.columns))
        if structure_type:
            found.append((structure_type, "", _read_csv(path)))
    elif suffix == ".bed":
        inferred = next((kind for kind in ("CHIN", "OPCID", "CHID") if kind.casefold() in path.name.casefold()), None)
        if inferred:
            frame = pd.read_csv(path, sep="\t", comment="#", header=None)
            if frame.shape[1] < 3:
                raise ValueError(f"BED requires at least 3 columns: {path}")
            frame = frame.iloc[:, :4]
            frame.columns = ["Chr", "Start", "End", f"{inferred}_ID"][: frame.shape[1]]
            if f"{inferred}_ID" not in frame:
                frame[f"{inferred}_ID"] = [f"{inferred}_{i + 1}" for i in range(len(frame))]
            found.append((inferred, "", frame))
    return found


def main() -> None:
    _configure_stdout()
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=project_root.parent / "micro-c数据")
    parser.add_argument("--output", type=Path, default=project_root / "data" / "processed" / "structures.csv")
    parser.add_argument(
        "--summary-output", type=Path,
        default=project_root / "outputs" / "structure_annotation_summary.csv",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    candidates = sorted(
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.casefold() in TABULAR_SUFFIXES
    )
    outputs: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    for path in candidates:
        try:
            tables = _discover_tables(path)
            for structure_type, sheet, frame in tables:
                standardized, center_mismatches = _standardize(frame, structure_type, path, sheet)
                outputs.append(standardized)
                source_rows.append(
                    {
                        "type": structure_type,
                        "source_file": str(path.resolve()),
                        "source_sheet": sheet,
                        "record_count": len(standardized),
                        "supplied_center_mismatches": center_mismatches,
                        "error": "",
                    }
                )
        except Exception as exc:
            source_rows.append(
                {
                    "type": "",
                    "source_file": str(path.resolve()),
                    "source_sheet": "",
                    "record_count": 0,
                    "supplied_center_mismatches": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"WARNING: could not inspect {path}: {type(exc).__name__}: {exc}")

    if not outputs:
        raise SystemExit("No CHIN/OPCID/CHID annotation tables were found.")
    structures = pd.concat(outputs, ignore_index=True)[OUTPUT_COLUMNS]
    type_order = pd.Categorical(structures["type"], categories=["CHIN", "OPCID", "CHID"], ordered=True)
    structures = structures.assign(_type_order=type_order).sort_values(
        ["_type_order", "start", "end"], kind="stable", na_position="last"
    ).drop(columns="_type_order").reset_index(drop=True)

    missing_coordinates = int(structures[["chrom", "start", "end"]].isna().any(axis=1).sum())
    invalid_intervals = int(((structures["end"] <= structures["start"]) & structures[["start", "end"]].notna().all(axis=1)).sum())
    duplicate_coordinates = int(structures.duplicated(["type", "chrom", "start", "end"], keep=False).sum())
    type_counts = structures["type"].value_counts().to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    structures.to_csv(args.output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    print(f"Unified structures written: {args.output.resolve()}")
    print(f"Source summary written: {args.summary_output.resolve()}")
    print(f"Counts: {json.dumps(type_counts, ensure_ascii=False)}; total={len(structures)}")
    print(f"Duplicate coordinate rows: {duplicate_coordinates}")
    print(f"Missing coordinate rows: {missing_coordinates}")
    print(f"Invalid intervals (end <= start): {invalid_intervals}")
    if len(structures) != 344:
        print(f"DIAGNOSTIC: expected approximately 344 structures, found {len(structures)}; no rows were altered to force the target.")


if __name__ == "__main__":
    main()

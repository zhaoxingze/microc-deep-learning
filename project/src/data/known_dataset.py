"""Reusable builders for aligned known-structure Micro-C replicate datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .load_microc import load_local_matrix
from .normalize import compute_oe, log1p_normalize


LABELS = {"CHIN": 0, "OPCID": 1, "CHID": 2}
PAIR_COLUMNS = [
    "structure_id",
    "genomic_group_id",
    "replicate",
    "type",
    "label_id",
    "chrom",
    "start",
    "end",
    "center",
    "array_index",
    "has_zero_axis",
    "zero_row_count",
    "zero_col_count",
    "padded",
    "available",
    "cool_path",
]


@dataclass(frozen=True)
class ReplicateBuild:
    arrays: dict[str, np.ndarray]
    labels: np.ndarray
    metadata: pd.DataFrame


def _format_center(value: object) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def stable_structure_id(row: Mapping[str, object] | pd.Series) -> str:
    """Return a row-order-independent ID derived from stable genomic fields."""
    structure_type = str(row["type"]).strip().upper()
    key = "|".join(
        [
            structure_type,
            str(row["chrom"]).strip(),
            str(int(row["start"])),
            str(int(row["end"])),
            _format_center(row["center"]),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{structure_type}_{digest}"


def attach_structure_ids(structures: pd.DataFrame) -> pd.DataFrame:
    required = {"type", "chrom", "start", "end", "center"}
    missing = sorted(required - set(structures.columns))
    if missing:
        raise ValueError(f"Missing structure columns: {missing}")
    output = structures.copy()
    output.insert(0, "structure_id", [stable_structure_id(row) for _, row in output.iterrows()])
    duplicates = output.loc[output["structure_id"].duplicated(keep=False), "structure_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate stable structure_id values: {sorted(set(duplicates))}")
    return output


def build_replicate(
    structures: pd.DataFrame,
    cool_path: str | Path,
    replicate: str,
    window_bp: int = 6400,
    target_bin_size: int = 100,
    balance: bool = False,
) -> ReplicateBuild:
    """Build aligned raw/log1p/OE arrays for one biological replicate."""
    if replicate not in {"rep1", "rep2"}:
        raise ValueError(f"replicate must be rep1 or rep2, got {replicate!r}")
    identified = structures if "structure_id" in structures else attach_structure_ids(structures)
    if identified["structure_id"].duplicated().any():
        raise ValueError("Duplicate stable structure_id values are not allowed")

    raw_matrices: list[np.ndarray] = []
    log_matrices: list[np.ndarray] = []
    oe_matrices: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []
    failures: list[str] = []
    resolved_cool = Path(cool_path).expanduser().resolve()

    for array_index, (_, row) in enumerate(identified.iterrows()):
        try:
            structure_type = str(row["type"]).upper()
            if structure_type not in LABELS:
                raise ValueError(f"Unknown structure type {structure_type!r}")
            raw, window_meta = load_local_matrix(
                cool_path=resolved_cool,
                chrom=str(row["chrom"]),
                center=float(row["center"]),
                window_bp=window_bp,
                target_bin_size=target_bin_size,
                balance=balance,
            )
            expected_bins = window_bp // target_bin_size
            if raw.shape != (expected_bins, expected_bins):
                raise ValueError(f"Expected {(expected_bins, expected_bins)}, got {raw.shape}")
            log_matrix = log1p_normalize(raw).astype(np.float32)
            oe_matrix = compute_oe(raw).astype(np.float32)
            if not all(np.isfinite(matrix).all() for matrix in (raw, log_matrix, oe_matrix)):
                raise ValueError("Non-finite value in raw/log1p/OE matrix")

            zero_row_count = int(np.all(raw == 0, axis=1).sum())
            zero_col_count = int(np.all(raw == 0, axis=0).sum())
            raw_matrices.append(raw.astype(np.float32))
            log_matrices.append(log_matrix)
            oe_matrices.append(oe_matrix)
            labels.append(LABELS[structure_type])
            metadata_rows.append(
                {
                    "structure_id": str(row["structure_id"]),
                    "replicate": replicate,
                    "type": structure_type,
                    "label_id": LABELS[structure_type],
                    "chrom": str(row["chrom"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "center": float(row["center"]),
                    "array_index": array_index,
                    "has_zero_axis": bool(zero_row_count or zero_col_count),
                    "zero_row_count": zero_row_count,
                    "zero_col_count": zero_col_count,
                    "padded": bool(window_meta["padded"]),
                    "available": True,
                    "cool_path": str(resolved_cool),
                }
            )
        except Exception as exc:
            failures.append(f"{row.get('structure_id', array_index)}: {type(exc).__name__}: {exc}")

    if failures:
        detail = "\n".join(failures[:20])
        raise RuntimeError(f"{replicate} build failed for {len(failures)} structures:\n{detail}")
    if not raw_matrices:
        raise RuntimeError(f"{replicate} build produced no matrices")

    arrays = {
        "raw": np.stack(raw_matrices).astype(np.float32)[:, np.newaxis, :, :],
        "log1p": np.stack(log_matrices).astype(np.float32)[:, np.newaxis, :, :],
        "oe": np.stack(oe_matrices).astype(np.float32)[:, np.newaxis, :, :],
    }
    return ReplicateBuild(
        arrays=arrays,
        labels=np.asarray(labels, dtype=np.int64),
        metadata=pd.DataFrame(metadata_rows),
    )


def combine_replicate_metadata(
    rep1: pd.DataFrame,
    rep2: pd.DataFrame,
    groups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate one-to-one replicate alignment and return long paired metadata."""
    ids1 = rep1["structure_id"].astype(str).tolist()
    ids2 = rep2["structure_id"].astype(str).tolist()
    if ids1 != ids2:
        raise ValueError("rep1/rep2 structure_id order does not match")
    if "type" in rep1 and "type" in rep2 and rep1["type"].astype(str).tolist() != rep2["type"].astype(str).tolist():
        raise ValueError("rep1/rep2 labels do not match")

    paired = pd.concat([rep1.copy(), rep2.copy()], ignore_index=True)
    if groups is not None:
        if groups["structure_id"].duplicated().any():
            raise ValueError("groups contains duplicate structure_id values")
        paired = paired.drop(columns=["genomic_group_id"], errors="ignore").merge(
            groups[["structure_id", "genomic_group_id"]],
            on="structure_id",
            how="left",
            validate="many_to_one",
        )
        if paired["genomic_group_id"].isna().any():
            raise ValueError("Some structures are missing genomic_group_id")
    elif "genomic_group_id" not in paired:
        paired["genomic_group_id"] = pd.NA

    for column in PAIR_COLUMNS:
        if column not in paired:
            paired[column] = pd.NA
    return paired[PAIR_COLUMNS]


def write_replicate_build(build: ReplicateBuild, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_type, array in build.arrays.items():
        np.save(output_dir / f"known_windows_{input_type}.npy", array, allow_pickle=False)
    np.save(output_dir / "known_labels.npy", build.labels, allow_pickle=False)
    build.metadata.to_csv(output_dir / "known_metadata.csv", index=False, encoding="utf-8-sig")

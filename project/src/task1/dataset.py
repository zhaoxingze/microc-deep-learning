"""Paired-replicate PyTorch dataset and train-only weighting helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABELS = {"CHIN": 0, "OPCID": 1, "CHID": 2}


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def replicate_balance_weights(structure_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(str(value) for value in structure_ids)
    if not counts:
        return np.asarray([], dtype=np.float32)
    return np.asarray([1.0 / counts[str(value)] for value in structure_ids], dtype=np.float32)


def compute_structure_class_weights(train_records: pd.DataFrame) -> torch.Tensor:
    """Compute inverse-frequency weights from unique train structures only."""
    required = {"structure_id", "type"}
    missing = required - set(train_records.columns)
    if missing:
        raise ValueError(f"Missing class-weight columns: {sorted(missing)}")
    unique = train_records[["structure_id", "type"]].drop_duplicates("structure_id")
    if unique.groupby("structure_id")["type"].nunique().max() > 1:
        raise ValueError("A structure has conflicting class labels")
    counts = unique["type"].value_counts().reindex(LABELS, fill_value=0).to_numpy(dtype=np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Train split is missing at least one class: {counts.tolist()}")
    weights = len(unique) / (len(LABELS) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_records(
    paired_metadata: pd.DataFrame,
    group_split: pd.DataFrame,
    split_name: str,
    train_replicates: Sequence[str] | str = ("rep1", "rep2"),
    exclude_zero_axis: bool = False,
    structure_balanced: bool = False,
) -> pd.DataFrame:
    """Select available replicate records without changing the fixed split."""
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split {split_name!r}")
    if isinstance(train_replicates, str):
        allowed_replicates = tuple(part.strip() for part in train_replicates.split(",") if part.strip())
    else:
        allowed_replicates = tuple(train_replicates)
    if not allowed_replicates or not set(allowed_replicates) <= {"rep1", "rep2"}:
        raise ValueError(f"Invalid replicate selection: {allowed_replicates}")

    split_columns = ["structure_id", "genomic_group_id", "split"]
    merged = paired_metadata.drop(columns=["genomic_group_id", "split"], errors="ignore").merge(
        group_split[split_columns], on="structure_id", how="inner", validate="many_to_one"
    )
    available = _boolean_series(merged["available"])
    keep = (merged["split"] == split_name) & available & merged["replicate"].isin(allowed_replicates)
    if exclude_zero_axis:
        keep &= ~_boolean_series(merged["has_zero_axis"])
    records = merged.loc[keep].copy().reset_index(drop=True)
    if structure_balanced:
        records["sample_weight"] = replicate_balance_weights(records["structure_id"].astype(str).tolist())
    else:
        records["sample_weight"] = np.float32(1.0)
    return records


class PairedMicroCDataset(Dataset[dict[str, object]]):
    """Load selected replicate matrices by array_index from paired .npy stores."""

    def __init__(self, records: pd.DataFrame, processed_root: str | Path, input_type: str) -> None:
        if input_type not in {"raw", "log1p", "oe"}:
            raise ValueError(f"Unknown input_type {input_type!r}")
        self.records = records.reset_index(drop=True).copy()
        self.processed_root = Path(processed_root).resolve()
        self.input_type = input_type
        self.arrays: dict[str, np.ndarray] = {}
        for replicate in sorted(self.records["replicate"].unique()):
            path = self.processed_root / str(replicate) / f"known_windows_{input_type}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 4 or array.shape[1:] != (1, 64, 64):
                raise ValueError(f"Expected (N,1,64,64) in {path}, got {array.shape}")
            self.arrays[str(replicate)] = array

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records.iloc[index]
        matrix = np.asarray(self.arrays[str(row["replicate"])][int(row["array_index"])], dtype=np.float32).copy()
        return {
            "matrix": torch.from_numpy(matrix),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "structure_id": str(row["structure_id"]),
            "replicate": str(row["replicate"]),
            "sample_weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
        }

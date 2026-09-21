"""Density, reconstruction, robust normalization, and paired candidate scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.task2.expected import apply_expected


def off_diagonal_mask(size: int, *, exclude_band: int = 2) -> np.ndarray:
    if size <= 0 or exclude_band < 0:
        raise ValueError("invalid score mask geometry")
    rows, cols = np.indices((size, size))
    return (cols > rows) & ((cols - rows) > exclude_band)


def density_features(matrix_oe: np.ndarray, *, exclude_band: int = 2) -> dict[str, float]:
    matrix = np.asarray(matrix_oe, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density features require a square matrix")
    values = matrix[off_diagonal_mask(matrix.shape[0], exclude_band=exclude_band)]
    if values.size == 0:
        raise ValueError("density mask selects no pixels")
    return {
        "density_raw": float(values.mean()),
        "oe_mean": float(values.mean()),
        "oe_std": float(values.std()),
        "oe_max": float(values.max()),
        "oe_p95": float(np.percentile(values, 95)),
        "oe_total": float(values.sum()),
    }


@dataclass(frozen=True)
class RobustScale:
    median: float
    mad: float
    eps: float = 1e-8

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.median) / (
            1.4826 * self.mad + self.eps
        )


def fit_robust_scale(values: np.ndarray) -> RobustScale:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("robust reference must be non-empty and finite")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return RobustScale(median, mad)


def candidate_score(
    density_z: np.ndarray,
    shape_z: np.ndarray,
    *,
    density_weight: float = 0.5,
    shape_weight: float = 0.5,
) -> np.ndarray:
    if density_weight < 0 or shape_weight < 0 or not np.isclose(
        density_weight + shape_weight, 1.0
    ):
        raise ValueError("density_weight and shape_weight must be non-negative and sum to one")
    return density_weight * np.asarray(density_z) + shape_weight * np.asarray(shape_z)


def score_windows(
    metadata: pd.DataFrame,
    array_paths: Mapping[str, str | Path],
    expected: Mapping[str, np.ndarray],
    reconstruction_scores: pd.DataFrame,
    background_window_ids: set[str],
    *,
    density_weight: float = 0.5,
    shape_weight: float = 0.5,
    exclude_band: int = 2,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    required = {
        "window_id", "chrom", "start", "end", "center", "replicate",
        "array_index", "zero_axis",
    }
    if not required.issubset(metadata.columns):
        raise ValueError("scan metadata is incomplete")
    arrays = {
        replicate: np.load(Path(path), mmap_mode="r", allow_pickle=False)
        for replicate, path in array_paths.items()
    }
    long_rows: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        replicate = str(row.replicate)
        raw = np.asarray(arrays[replicate][int(row.array_index), 0], dtype=np.float32)
        features = density_features(
            apply_expected(raw, expected[replicate]), exclude_band=exclude_band
        )
        long_rows.append(
            {
                "window_id": str(row.window_id),
                "replicate": replicate,
                "zero_axis": bool(row.zero_axis),
                **features,
            }
        )
    long = pd.DataFrame(long_rows).merge(
        reconstruction_scores,
        on=["window_id", "replicate"],
        how="left",
        validate="one_to_one",
    )
    if long["shape_raw"].isna().any():
        raise ValueError("some windows are missing reconstruction scores")
    statistics: dict[str, dict[str, float]] = {}
    for replicate in ("rep1", "rep2"):
        replicate_rows = long["replicate"].eq(replicate)
        reference = replicate_rows & long["window_id"].isin(background_window_ids)
        if not reference.any():
            raise ValueError(f"no background reference scores for {replicate}")
        density_scale = fit_robust_scale(long.loc[reference, "density_raw"].to_numpy())
        shape_scale = fit_robust_scale(long.loc[reference, "shape_raw"].to_numpy())
        long.loc[replicate_rows, "density_z"] = density_scale.transform(
            long.loc[replicate_rows, "density_raw"].to_numpy()
        )
        long.loc[replicate_rows, "shape_z"] = shape_scale.transform(
            long.loc[replicate_rows, "shape_raw"].to_numpy()
        )
        long.loc[replicate_rows, "candidate_score"] = candidate_score(
            long.loc[replicate_rows, "density_z"].to_numpy(),
            long.loc[replicate_rows, "shape_z"].to_numpy(),
            density_weight=density_weight,
            shape_weight=shape_weight,
        )
        statistics[replicate] = {
            "density_median": density_scale.median,
            "density_mad": density_scale.mad,
            "shape_median": shape_scale.median,
            "shape_mad": shape_scale.mad,
        }

    coordinates = metadata.loc[metadata["replicate"].eq("rep1"), [
        "window_id", "chrom", "start", "end", "center"
    ]].copy()
    metric_columns = [
        "density_raw", "density_z", "shape_raw", "shape_z", "candidate_score",
        "oe_mean", "oe_std", "oe_max", "oe_p95", "oe_total", "zero_axis",
    ]
    wide = coordinates
    for replicate in ("rep1", "rep2"):
        subset = long.loc[long["replicate"].eq(replicate), ["window_id", *metric_columns]].copy()
        subset = subset.rename(columns={column: f"{replicate}_{column}" for column in metric_columns})
        wide = wide.merge(subset, on="window_id", how="left", validate="one_to_one")
    wide["paired_density_z"] = (
        wide["rep1_density_z"] + wide["rep2_density_z"]
    ) / 2.0
    wide["paired_shape_z"] = (
        wide["rep1_shape_z"] + wide["rep2_shape_z"]
    ) / 2.0
    wide["paired_candidate_score"] = (
        wide["rep1_candidate_score"] + wide["rep2_candidate_score"]
    ) / 2.0
    wide["score_difference"] = (
        wide["rep1_candidate_score"] - wide["rep2_candidate_score"]
    ).abs()
    wide["score_min"] = wide[["rep1_candidate_score", "rep2_candidate_score"]].min(axis=1)
    wide["score_max"] = wide[["rep1_candidate_score", "rep2_candidate_score"]].max(axis=1)
    wide = wide.rename(
        columns={"rep1_zero_axis": "zero_axis_rep1", "rep2_zero_axis": "zero_axis_rep2"}
    )
    wide = wide.sort_values("paired_candidate_score", ascending=False).reset_index(drop=True)
    wide["rank"] = np.arange(1, len(wide) + 1)
    wide["percentile"] = 1.0 - (wide["rank"] - 1) / len(wide)
    return wide, statistics

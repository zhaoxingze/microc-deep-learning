"""Background-fitted shape representation for the refined Task 2A detector."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.task2.expected import apply_expected


def valid_upper_values(matrix: np.ndarray, *, exclude_band: int = 2) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("shape preprocessing requires a square matrix")
    rows, columns = np.indices(array.shape)
    return array[(columns > rows) & ((columns - rows) > exclude_band)]


def preprocess_shape_window(
    raw: np.ndarray,
    expected: np.ndarray,
    clip_value: float,
    *,
    exclude_band: int = 2,
    eps: float = 1e-8,
) -> np.ndarray:
    if not np.isfinite(clip_value) or clip_value <= 0:
        raise ValueError("O/E clip value must be finite and positive")
    if eps <= 0:
        raise ValueError("normalization epsilon must be positive")
    oe = apply_expected(raw, expected)
    finite = np.nan_to_num(oe, nan=0.0, posinf=clip_value, neginf=0.0)
    logged = np.log1p(np.clip(finite, 0.0, clip_value))
    valid = valid_upper_values(logged, exclude_band=exclude_band)
    if valid.size == 0:
        raise ValueError("shape normalization mask selects no pixels")
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    normalized = (logged - median) / (1.4826 * mad + eps)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_background_oe_clips(
    rows: pd.DataFrame,
    array_paths: Mapping[str, str | Path],
    expected: Mapping[str, np.ndarray],
    *,
    percentile: float = 99.5,
    exclude_band: int = 2,
) -> dict[str, float]:
    required = {"replicate", "array_index", "background_split"}
    if not required.issubset(rows.columns):
        raise ValueError("clip-fitting rows are incomplete")
    if not 0 < percentile <= 100:
        raise ValueError("clip percentile must be in (0, 100]")
    arrays = {
        replicate: np.load(Path(path), mmap_mode="r", allow_pickle=False)
        for replicate, path in array_paths.items()
    }
    clips: dict[str, float] = {}
    for replicate in ("rep1", "rep2"):
        reference = rows.loc[
            rows["replicate"].eq(replicate) & rows["background_split"].eq("train")
        ]
        if reference.empty:
            raise ValueError(f"no background-train windows for {replicate} clip fitting")
        chunks = []
        for row in reference.itertuples(index=False):
            raw = np.asarray(arrays[replicate][int(row.array_index), 0], dtype=np.float32)
            values = valid_upper_values(apply_expected(raw, expected[replicate]), exclude_band=exclude_band)
            values = values[np.isfinite(values) & (values >= 0)]
            if values.size:
                chunks.append(values)
        if not chunks:
            raise ValueError(f"no finite background O/E pixels for {replicate}")
        clip = float(np.percentile(np.concatenate(chunks), percentile))
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError(f"invalid background O/E clip for {replicate}: {clip}")
        clips[replicate] = clip
    return clips

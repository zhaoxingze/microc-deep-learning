"""Train-only intensity-feature baseline for the supervised Task 1 audit."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BrightnessModel:
    scaler: StandardScaler
    model: LogisticRegression


def extract_intensity_features(matrix: np.ndarray) -> pd.Series:
    values = np.asarray(matrix, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Intensity matrix must be non-empty and finite")
    return pd.Series(
        {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
            "total_contact": float(values.sum()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    )


def fit_brightness_model(features: np.ndarray, labels: np.ndarray, seed: int) -> BrightnessModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Expected a 2D feature matrix and aligned 1D labels")
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    ).fit(scaler.transform(x), y)
    return BrightnessModel(scaler=scaler, model=model)


def aggregate_structure_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    structure_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if score_array.ndim != 2 or label_array.ndim != 1:
        raise ValueError("Scores must be 2D and labels must be 1D")
    if len(score_array) != len(label_array) or len(score_array) != len(structure_ids):
        raise ValueError("Scores, labels, and structure IDs must be aligned")
    order = list(dict.fromkeys(str(value) for value in structure_ids))
    aggregated: list[np.ndarray] = []
    structure_labels: list[int] = []
    ids = np.asarray([str(value) for value in structure_ids], dtype=object)
    for structure_id in order:
        mask = ids == structure_id
        unique_labels = np.unique(label_array[mask])
        if len(unique_labels) != 1:
            raise ValueError(f"Conflicting labels for structure {structure_id}")
        aggregated.append(score_array[mask].mean(axis=0))
        structure_labels.append(int(unique_labels[0]))
    return np.stack(aggregated), np.asarray(structure_labels, dtype=np.int64), order

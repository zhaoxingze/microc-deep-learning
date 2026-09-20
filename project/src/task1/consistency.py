"""Rep1/rep2 matrix correlation utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def triangle_masks(size: int, diagonal_band: int = 2) -> dict[str, np.ndarray]:
    if size <= 0 or diagonal_band < 0:
        raise ValueError("size must be positive and diagonal_band non-negative")
    rows, cols = np.indices((size, size))
    upper = rows <= cols
    return {
        "upper": upper,
        "offdiag": upper & (cols > rows),
        f"exclude_band{diagonal_band}": upper & ((cols - rows) > diagonal_band),
    }


def _safe_correlations(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    if first.size < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(first, second).statistic), float(spearmanr(first, second).statistic)


def matrix_correlations(
    rep1: np.ndarray,
    rep2: np.ndarray,
    diagonal_band: int = 2,
) -> dict[str, float]:
    first = np.asarray(rep1, dtype=np.float64)
    second = np.asarray(rep2, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError(f"Expected equal square matrices, got {first.shape} and {second.shape}")
    masks = triangle_masks(first.shape[0], diagonal_band)
    result: dict[str, float] = {}
    for key, mask in masks.items():
        pearson, spearman = _safe_correlations(first[mask], second[mask])
        suffix = "upper" if key == "upper" else ("offdiag" if key == "offdiag" else key)
        result[f"pearson_{suffix}"] = pearson
        result[f"spearman_{suffix}"] = spearman
    return result

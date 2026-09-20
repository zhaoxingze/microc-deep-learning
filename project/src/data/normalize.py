"""Numerically safe normalization functions for local contact matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Matrix contains NaN or infinite values.")
    return array


def log1p_normalize(matrix: ArrayLike) -> NDArray[np.float64]:
    """Apply element-wise log(1 + x) to a non-negative matrix."""
    array = _as_float_matrix(matrix)
    if np.any(array < 0):
        raise ValueError("log1p normalization requires non-negative values.")
    return np.log1p(array)


def percentile_clip(
    matrix: ArrayLike, percentile: float = 99.5
) -> NDArray[np.float64]:
    """Clip only the upper tail at the requested finite-value percentile."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100].")
    array = _as_float_matrix(matrix)
    upper = float(np.percentile(array, percentile))
    return np.clip(array, a_min=None, a_max=upper)


def minmax_normalize(matrix: ArrayLike) -> NDArray[np.float64]:
    """Scale a matrix to [0, 1]; a constant matrix becomes all zeros."""
    array = _as_float_matrix(matrix)
    minimum = float(array.min())
    span = float(array.max() - minimum)
    if span == 0:
        return np.zeros_like(array)
    return (array - minimum) / span


def compute_oe(matrix: ArrayLike, eps: float = 1e-8) -> NDArray[np.float64]:
    """Compute observed/expected values by genomic diagonal distance.

    E(d) is the mean of every entry with |i-j| == d. Both triangles are
    included so the definition also behaves sensibly for slightly asymmetric
    inputs. Zero-expectation diagonals remain zero when observations are zero.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    array = _as_float_matrix(matrix)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"O/E requires a square matrix, got {array.shape}.")

    size = array.shape[0]
    result = np.zeros_like(array)
    for distance in range(size):
        upper_rows = np.arange(0, size - distance)
        upper_cols = upper_rows + distance
        values = array[upper_rows, upper_cols]
        if distance:
            lower_values = array[upper_cols, upper_rows]
            values = np.concatenate((values, lower_values))
        expected = float(values.mean()) if values.size else 0.0
        denominator = expected + eps
        result[upper_rows, upper_cols] = array[upper_rows, upper_cols] / denominator
        if distance:
            result[upper_cols, upper_rows] = array[upper_cols, upper_rows] / denominator
    return result

from __future__ import annotations

import numpy as np
import pytest

from src.task1.consistency import matrix_correlations, triangle_masks


def test_correlation_masks_have_hand_derived_counts() -> None:
    masks = triangle_masks(4, diagonal_band=2)
    assert int(masks["upper"].sum()) == 10
    assert int(masks["offdiag"].sum()) == 6
    assert int(masks["exclude_band2"].sum()) == 1


def test_identical_nonconstant_matrices_have_unit_correlations() -> None:
    matrix = np.arange(64, dtype=float).reshape(8, 8)
    correlations = matrix_correlations(matrix, matrix)
    for value in correlations.values():
        assert value == pytest.approx(1.0)


def test_constant_vectors_return_nan_instead_of_warning_value() -> None:
    matrix = np.ones((4, 4), dtype=float)
    correlations = matrix_correlations(matrix, matrix)
    assert all(np.isnan(value) for value in correlations.values())

import numpy as np
import pytest

from src.data.normalize import compute_oe, log1p_normalize, minmax_normalize, percentile_clip


def test_log1p_and_minmax_are_finite() -> None:
    matrix = np.array([[0.0, 1.0], [1.0, 3.0]])
    assert np.allclose(log1p_normalize(matrix), np.log1p(matrix))
    scaled = minmax_normalize(matrix)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


def test_constant_minmax_becomes_zero() -> None:
    assert np.array_equal(minmax_normalize(np.ones((3, 3))), np.zeros((3, 3)))


def test_percentile_clip_and_validation() -> None:
    matrix = np.arange(100, dtype=float).reshape(10, 10)
    clipped = percentile_clip(matrix, 90)
    assert clipped.max() == pytest.approx(np.percentile(matrix, 90))
    with pytest.raises(ValueError, match="percentile"):
        percentile_clip(matrix, 0)


def test_compute_oe_uses_absolute_diagonals() -> None:
    matrix = np.array([[2.0, 4.0, 8.0], [4.0, 2.0, 4.0], [8.0, 4.0, 2.0]])
    oe = compute_oe(matrix)
    assert np.allclose(oe, np.ones_like(matrix), atol=1e-7)


def test_invalid_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        log1p_normalize(np.array([[-2.0]]))
    with pytest.raises(ValueError, match="square"):
        compute_oe(np.ones((2, 3)))
    with pytest.raises(ValueError, match="NaN"):
        minmax_normalize(np.array([[np.nan]]))

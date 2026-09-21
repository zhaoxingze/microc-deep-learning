import numpy as np
import pandas as pd

from src.task2.expected import apply_expected, expected_from_target_pixels


def test_expected_uses_all_possible_target_pairs_including_zeros() -> None:
    pixels = pd.DataFrame(
        {
            "target1": [0, 0, 1, 2],
            "target2": [0, 1, 1, 2],
            "count": [6.0, 4.0, 3.0, 3.0],
        }
    )
    expected = expected_from_target_pixels([pixels], n_target_bins=3, max_distance_bins=2)
    np.testing.assert_allclose(expected, [4.0, 2.0, 0.0])


def test_apply_expected_uses_target_bin_distance_and_keeps_zero_expected_zero() -> None:
    matrix = np.array([[4.0, 4.0, 9.0], [4.0, 8.0, 2.0], [9.0, 2.0, 12.0]])
    actual = apply_expected(matrix, np.array([4.0, 2.0, 0.0]))
    np.testing.assert_allclose(
        actual,
        [[1.0, 2.0, 0.0], [2.0, 2.0, 1.0], [0.0, 1.0, 3.0]],
    )

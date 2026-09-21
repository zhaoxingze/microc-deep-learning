import numpy as np
import pandas as pd
import pytest

from src.task2.scoring import (
    candidate_score,
    density_features,
    fit_robust_scale,
    off_diagonal_mask,
    score_windows,
)


def test_density_uses_upper_triangle_beyond_excluded_band() -> None:
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    mask = off_diagonal_mask(4, exclude_band=2)
    assert np.argwhere(mask).tolist() == [[0, 3]]
    features = density_features(matrix, exclude_band=2)
    assert features["density_raw"] == 3.0
    assert features["oe_total"] == 3.0


def test_robust_scale_zero_mad_remains_finite() -> None:
    scale = fit_robust_scale(np.array([2.0, 2.0, 2.0]))
    transformed = scale.transform(np.array([2.0, 3.0]))
    assert transformed[0] == 0.0
    assert np.isfinite(transformed).all()


def test_candidate_score_uses_fixed_weighted_sum_and_validates_weights() -> None:
    np.testing.assert_allclose(
        candidate_score(np.array([2.0]), np.array([4.0]), density_weight=0.5),
        [3.0],
    )
    with pytest.raises(ValueError, match="sum to one"):
        candidate_score(np.array([2.0]), np.array([4.0]), density_weight=0.7, shape_weight=0.7)


def test_score_windows_emits_paired_component_scores(tmp_path) -> None:
    rep1_path = tmp_path / "rep1.npy"
    rep2_path = tmp_path / "rep2.npy"
    np.save(rep1_path, np.stack([np.ones((1, 4, 4)), np.full((1, 4, 4), 2.0)]))
    np.save(rep2_path, np.stack([np.full((1, 4, 4), 3.0), np.full((1, 4, 4), 5.0)]))
    metadata = pd.DataFrame(
        [
            {"window_id": window_id, "chrom": "chr", "start": index * 100,
             "end": index * 100 + 400, "center": index * 100 + 200,
             "replicate": replicate, "array_index": index, "zero_axis": False}
            for replicate in ("rep1", "rep2")
            for index, window_id in enumerate(("w1", "w2"))
        ]
    )
    reconstruction = pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w1", "w2"],
            "replicate": ["rep1", "rep1", "rep2", "rep2"],
            "shape_raw": [1.0, 2.0, 3.0, 5.0],
        }
    )

    scores, _ = score_windows(
        metadata,
        {"rep1": rep1_path, "rep2": rep2_path},
        {"rep1": np.ones(4), "rep2": np.ones(4)},
        reconstruction,
        {"w1", "w2"},
    )

    assert {"paired_density_z", "paired_shape_z", "paired_candidate_score"}.issubset(scores)
    np.testing.assert_allclose(
        scores["paired_candidate_score"],
        0.5 * scores["paired_density_z"] + 0.5 * scores["paired_shape_z"],
    )

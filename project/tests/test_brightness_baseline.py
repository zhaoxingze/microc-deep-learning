from __future__ import annotations

import numpy as np
import pytest

from src.task1.brightness import (
    aggregate_structure_scores,
    extract_intensity_features,
    fit_brightness_model,
)


def test_intensity_features_are_hand_computed() -> None:
    matrix = np.array([[0.0, 2.0], [4.0, 6.0]])
    features = extract_intensity_features(matrix)
    assert features[["mean", "max", "total_contact"]].tolist() == pytest.approx([3.0, 6.0, 12.0])
    assert features["std"] == pytest.approx(np.std(matrix))


def test_scaler_is_fit_to_train_rows_only() -> None:
    train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]])
    labels = np.array([0, 1, 2, 0, 1, 2])
    fitted = fit_brightness_model(train, labels, seed=7)
    assert fitted.scaler.mean_.tolist() == pytest.approx(train.mean(axis=0).tolist())


def test_structure_scores_average_available_replicates() -> None:
    scores = np.array([[3.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    aggregated, labels, order = aggregate_structure_scores(scores, np.array([0, 0, 2]), ["A", "A", "B"])
    assert order == ["A", "B"]
    np.testing.assert_allclose(aggregated, [[2.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
    assert labels.tolist() == [0, 2]

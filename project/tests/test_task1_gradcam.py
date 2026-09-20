from __future__ import annotations

import numpy as np
import pytest
import torch

from src.task1.gradcam import GradCAM, diagonal_attention_ratio, zero_axis_attention_ratio
from src.task1.model import SmallMicroCCNN


def test_gradcam_is_finite_and_matches_input_shape() -> None:
    model = SmallMicroCCNN()
    explainer = GradCAM(model, model.target_layer)
    heatmap = explainer.compute(torch.zeros(1, 1, 64, 64), target_class=0)
    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()
    assert np.all((0 <= heatmap) & (heatmap <= 1))
    explainer.close()


def test_diagonal_attention_ratio_has_literal_denominator() -> None:
    assert diagonal_attention_ratio(np.ones((5, 5)), band=0) == pytest.approx(5 / 25)


def test_zero_axis_attention_ratio_uses_zero_rows_and_columns() -> None:
    raw = np.ones((4, 4))
    raw[1, :] = 0
    raw[:, 3] = 0
    cam = np.ones((4, 4))
    assert zero_axis_attention_ratio(cam, raw) == pytest.approx(7 / 16)

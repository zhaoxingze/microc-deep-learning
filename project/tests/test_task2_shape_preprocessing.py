from pathlib import Path

import numpy as np
import pandas as pd

from src.task2.shape_preprocessing import (
    fit_background_oe_clips,
    preprocess_shape_window,
    valid_upper_values,
)


def test_shape_preprocessing_clips_then_log_normalizes_valid_pixels() -> None:
    raw = np.zeros((5, 5), dtype=float)
    raw[0, 3], raw[0, 4], raw[1, 4] = 0.0, 3.0, 100.0
    transformed = preprocess_shape_window(raw, np.ones(5), clip_value=8.0)
    values = valid_upper_values(transformed)

    assert np.isfinite(transformed).all()
    assert abs(float(np.median(values))) < 1e-7
    assert np.isclose(np.median(np.abs(values)), 1 / 1.4826, atol=1e-6)
    clipped_raw = raw.copy()
    clipped_raw[1, 4] = 8.0
    np.testing.assert_allclose(
        transformed,
        preprocess_shape_window(clipped_raw, np.ones(5), clip_value=8.0),
    )


def test_shape_preprocessing_zero_mad_and_nonfinite_values_remain_finite() -> None:
    raw = np.full((5, 5), 4.0)
    raw[0, 3] = np.inf
    raw[0, 4] = np.nan
    transformed = preprocess_shape_window(raw, np.ones(5), clip_value=10.0)

    assert np.isfinite(transformed).all()


def test_clip_fitting_uses_only_background_train_rows(tmp_path: Path) -> None:
    paths = {}
    for replicate in ("rep1", "rep2"):
        path = tmp_path / f"{replicate}.npy"
        np.save(
            path,
            np.stack(
                [np.full((1, 5, 5), 2.0, dtype=np.float32),
                 np.full((1, 5, 5), 1_000.0, dtype=np.float32)]
            ),
        )
        paths[replicate] = path
    rows = pd.DataFrame(
        [
            {"window_id": f"{replicate}_background", "replicate": replicate,
             "array_index": 0, "background_split": "train"}
            for replicate in ("rep1", "rep2")
        ]
        + [
            {"window_id": f"{replicate}_known", "replicate": replicate,
             "array_index": 1, "background_split": "non_background"}
            for replicate in ("rep1", "rep2")
        ]
    )

    clips = fit_background_oe_clips(
        rows, paths, {"rep1": np.ones(5), "rep2": np.ones(5)}, percentile=99.5
    )

    assert clips == {"rep1": 2.0, "rep2": 2.0}

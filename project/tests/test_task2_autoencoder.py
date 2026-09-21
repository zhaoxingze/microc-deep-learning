from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.task2.autoencoder import (
    BackgroundAutoencoder,
    WindowDataset,
    masked_reconstruction_error,
    train_autoencoder,
)


def test_autoencoder_output_shape_matches_input() -> None:
    model = BackgroundAutoencoder()
    inputs = torch.zeros((2, 1, 64, 64))
    assert model(inputs).shape == inputs.shape


def test_masked_reconstruction_error_uses_upper_off_diagonal_pixels() -> None:
    original = torch.zeros((1, 1, 4, 4))
    reconstruction = original.clone()
    reconstruction[0, 0, 0, 3] = 2.0
    assert masked_reconstruction_error(original, reconstruction, exclude_band=2).item() == 4.0


def test_window_dataset_applies_log1p_and_uses_replicate_array(tmp_path: Path) -> None:
    for replicate, value in (("rep1", 3.0), ("rep2", 8.0)):
        np.save(tmp_path / f"{replicate}.npy", np.full((1, 1, 4, 4), value, dtype=np.float32))
    rows = pd.DataFrame(
        {"replicate": ["rep1", "rep2"], "array_index": [0, 0], "window_id": ["w", "w"]}
    )
    dataset = WindowDataset(
        rows,
        {"rep1": tmp_path / "rep1.npy", "rep2": tmp_path / "rep2.npy"},
        input_type="log1p",
    )
    np.testing.assert_allclose(dataset[0]["matrix"].numpy(), np.log(4.0))
    np.testing.assert_allclose(dataset[1]["matrix"].numpy(), np.log(9.0))


def test_train_autoencoder_writes_background_validation_checkpoint(tmp_path: Path) -> None:
    train = TensorDataset(torch.zeros((2, 1, 64, 64)))
    val = TensorDataset(torch.zeros((2, 1, 64, 64)))
    result = train_autoencoder(
        train,
        val,
        tmp_path,
        epochs=2,
        patience=1,
        batch_size=2,
        seed=7,
        device=torch.device("cpu"),
    )
    assert (tmp_path / "best_model.pth").is_file()
    assert (tmp_path / "training_history.csv").is_file()
    assert result.best_epoch >= 1
    assert "val_loss" in result.history


"""Background convolutional autoencoder and validation-only training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.task1.trainer import set_reproducible_seed
from src.task2.expected import apply_expected


class BackgroundAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class WindowDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        array_paths: Mapping[str, str | Path],
        *,
        input_type: str = "log1p",
        expected: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if input_type not in {"log1p", "genome_oe"}:
            raise ValueError("input_type must be log1p or genome_oe")
        required = {"replicate", "array_index", "window_id"}
        if not required.issubset(rows.columns):
            raise ValueError("dataset rows are incomplete")
        self.rows = rows.reset_index(drop=True).copy()
        self.arrays = {
            name: np.load(Path(path), mmap_mode="r", allow_pickle=False)
            for name, path in array_paths.items()
        }
        self.input_type = input_type
        self.expected = expected
        if input_type == "genome_oe" and expected is None:
            raise ValueError("genome_oe input requires replicate expected vectors")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        replicate = str(row["replicate"])
        matrix = np.asarray(
            self.arrays[replicate][int(row["array_index"]), 0], dtype=np.float32
        )
        if self.input_type == "log1p":
            transformed = np.log1p(matrix).astype(np.float32)
        else:
            assert self.expected is not None
            transformed = apply_expected(matrix, self.expected[replicate]).astype(np.float32)
        return {
            "matrix": torch.from_numpy(transformed[np.newaxis, :, :].copy()),
            "window_id": str(row["window_id"]),
            "replicate": replicate,
            "array_index": int(row["array_index"]),
        }


def _off_diagonal_mask(size: int, exclude_band: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(size, device=device)
    distances = indices[None, :] - indices[:, None]
    return distances > exclude_band


def masked_reconstruction_error(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    exclude_band: int = 2,
) -> torch.Tensor:
    if original.shape != reconstruction.shape or original.ndim != 4:
        raise ValueError("reconstruction tensors must have identical NCHW shape")
    mask = _off_diagonal_mask(original.shape[-1], exclude_band, original.device)
    squared = (original - reconstruction).pow(2)[:, 0]
    return squared[:, mask].mean(dim=1)


@dataclass(frozen=True)
class AutoencoderTrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: pd.DataFrame


def _extract_matrix(batch: object) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["matrix"]  # type: ignore[return-value]
    if isinstance(batch, (tuple, list)):
        return batch[0]  # type: ignore[return-value]
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError(f"unsupported autoencoder batch type: {type(batch).__name__}")


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = _extract_matrix(batch).to(device=device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs)
            loss = nn.functional.mse_loss(reconstruction, inputs)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
    if count == 0:
        raise ValueError("autoencoder loader produced no batches")
    return total / count


def train_autoencoder(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str | Path,
    *,
    epochs: int = 60,
    patience: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 20_260_920,
    device: torch.device | None = None,
) -> AutoencoderTrainingResult:
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_reproducible_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = BackgroundAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = root / "best_model.pth"
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _epoch_loss(model, train_loader, device, optimizer)
        val_loss = _epoch_loss(model, val_loader, device, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": val_loss,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    frame = pd.DataFrame(history)
    frame.to_csv(root / "training_history.csv", index=False, encoding="utf-8-sig")
    return AutoencoderTrainingResult(best_epoch, best_loss, frame)


def load_autoencoder(checkpoint_path: str | Path, device: torch.device) -> BackgroundAutoencoder:
    model = BackgroundAutoencoder().to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score_reconstruction(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 32,
    device: torch.device | None = None,
    exclude_band: int = 2,
) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, object]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = _extract_matrix(batch).to(device=device, dtype=torch.float32)
            reconstruction = model(inputs)
            offdiag = masked_reconstruction_error(
                inputs, reconstruction, exclude_band=exclude_band
            ).cpu().numpy()
            all_pixel = (inputs - reconstruction).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
            for local_index in range(len(inputs)):
                source = dataset[offset + local_index]  # type: ignore[index]
                rows.append(
                    {
                        "window_id": source["window_id"],
                        "replicate": source["replicate"],
                        "shape_raw": float(offdiag[local_index]),
                        "shape_all_pixels": float(all_pixel[local_index]),
                    }
                )
            offset += len(inputs)
    return pd.DataFrame(rows)


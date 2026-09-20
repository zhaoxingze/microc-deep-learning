"""Deterministic CNN training using structure-level validation only."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import EvaluationResult, evaluate_structure_level


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation: dict[str, float]
    history: pd.DataFrame


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def choose_better_epoch(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    candidate_f1 = float(candidate["val_structure_macro_f1"])
    incumbent_f1 = float(incumbent["val_structure_macro_f1"])
    if candidate_f1 > incumbent_f1 and not math.isclose(candidate_f1, incumbent_f1, abs_tol=1e-12):
        return True
    if math.isclose(candidate_f1, incumbent_f1, abs_tol=1e-12):
        return float(candidate["val_structure_loss"]) < float(incumbent["val_structure_loss"])
    return False


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    for batch in loader:
        matrices = batch["matrix"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)
        sample_weights = batch["sample_weight"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(matrices)
        losses = F.cross_entropy(logits, labels, weight=class_weights.to(device), reduction="none")
        loss = (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
        loss.backward()
        optimizer.step()
        weighted_loss_sum += float((losses.detach() * sample_weights).sum().cpu())
        weight_sum += float(sample_weights.sum().cpu())
        labels_all.append(labels.detach().cpu().numpy())
        predictions_all.append(logits.detach().argmax(dim=1).cpu().numpy())
    if not labels_all:
        raise ValueError("Training loader produced no batches")
    true = np.concatenate(labels_all)
    predicted = np.concatenate(predictions_all)
    return {
        "train_loss": weighted_loss_sum / weight_sum,
        "train_accuracy": float(accuracy_score(true, predicted)),
        "train_macro_f1": float(f1_score(true, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
    }


def train_experiment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    device: torch.device,
) -> TrainingResult:
    """Train without any test-set object; select solely on validation structures."""
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best: dict[str, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, optimizer, train_loader, class_weights, device)
        validation: EvaluationResult = evaluate_structure_level(model, val_loader, class_weights, device)
        candidate = {
            "val_structure_macro_f1": float(validation.summary["macro_f1"]),
            "val_structure_loss": float(validation.summary["structure_loss"]),
            "val_structure_accuracy": float(validation.summary["accuracy"]),
        }
        history_rows.append({"epoch": epoch, **train_stats, **candidate})
        if best is None or choose_better_epoch(candidate, best):
            best = candidate
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": candidate,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:03d} train_loss={train_stats['train_loss']:.5f} "
            f"train_f1={train_stats['train_macro_f1']:.4f} "
            f"val_loss={candidate['val_structure_loss']:.5f} "
            f"val_f1={candidate['val_structure_macro_f1']:.4f}",
            flush=True,
        )
        if stale_epochs >= patience:
            break
    if best is None:
        raise RuntimeError("No checkpoint was selected")
    return TrainingResult(best_epoch, best, pd.DataFrame(history_rows))

"""Structure-level evaluation and validation-only model/input selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .aggregation import aggregate_structure_logits
from .metrics import MetricResult, classification_metrics


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float]
    primary: MetricResult
    predictions: pd.DataFrame
    secondary: dict[str, MetricResult]


@torch.no_grad()
def collect_replicate_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    structure_ids: list[str] = []
    replicates: list[str] = []
    for batch in loader:
        matrices = batch["matrix"].to(device=device, dtype=torch.float32)
        logits_parts.append(model(matrices).detach().cpu())
        label_parts.append(batch["label"].detach().cpu().long())
        structure_ids.extend(str(value) for value in batch["structure_id"])
        replicates.extend(str(value) for value in batch["replicate"])
    if not logits_parts:
        raise ValueError("Evaluation loader produced no available replicates")
    return torch.cat(logits_parts), torch.cat(label_parts), structure_ids, replicates


def evaluate_structure_level(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> EvaluationResult:
    replicate_logits, replicate_labels, structure_ids, replicates = collect_replicate_logits(model, loader, device)
    structure_logits, structure_labels, structure_order = aggregate_structure_logits(
        replicate_logits, replicate_labels, structure_ids
    )
    structure_loss = F.cross_entropy(structure_logits, structure_labels, weight=class_weights.cpu()).item()
    structure_predictions = structure_logits.argmax(dim=1).numpy()
    primary = classification_metrics(structure_labels.numpy(), structure_predictions)

    replicate_frame = pd.DataFrame(
        {
            "structure_id": structure_ids,
            "replicate": replicates,
            "label": replicate_labels.numpy(),
            "prediction": replicate_logits.argmax(dim=1).numpy(),
        }
    )
    secondary: dict[str, MetricResult] = {
        "replicate_level": classification_metrics(
            replicate_frame["label"].to_numpy(), replicate_frame["prediction"].to_numpy()
        )
    }
    for replicate in ("rep1", "rep2"):
        subset = replicate_frame.loc[replicate_frame["replicate"] == replicate]
        if not subset.empty:
            secondary[f"{replicate}_only"] = classification_metrics(
                subset["label"].to_numpy(), subset["prediction"].to_numpy()
            )

    replicate_lookup = replicate_frame.groupby("structure_id")["replicate"].agg(lambda values: ";".join(sorted(values)))
    predictions = pd.DataFrame(
        {
            "structure_id": structure_order,
            "true_label_id": structure_labels.numpy(),
            "predicted_label_id": structure_predictions,
            "replicates_used": [replicate_lookup[structure_id] for structure_id in structure_order],
            "missing_replicate": [len(replicate_lookup[structure_id].split(";")) < 2 for structure_id in structure_order],
            **{
                f"logit_{class_id}": structure_logits[:, class_id].numpy()
                for class_id in range(structure_logits.shape[1])
            },
        }
    )
    summary = {
        **primary.summary,
        "structure_loss": float(structure_loss),
        "n_structures": float(len(structure_order)),
        "n_replicates": float(len(replicate_frame)),
    }
    return EvaluationResult(summary=summary, primary=primary, predictions=predictions, secondary=secondary)


def evaluate_locked_checkpoint(
    checkpoint_path: str | Path,
    model_factory: Callable[[], nn.Module],
    eval_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> EvaluationResult:
    """Load a locked checkpoint and run an evaluation outside the trainer."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model = model_factory().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate_structure_level(model, eval_loader, class_weights, device)


def select_best_input(comparison: pd.DataFrame) -> str:
    required = {"input_type", "best_val_macro_f1", "best_val_loss"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"Missing input selection columns: {sorted(missing)}")
    ordered = comparison.sort_values(
        ["best_val_macro_f1", "best_val_loss", "input_type"],
        ascending=[False, True, True],
        kind="stable",
    )
    return str(ordered.iloc[0]["input_type"])

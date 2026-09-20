"""Structure-level aggregation shared by validation and test evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def aggregate_structure_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    structure_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Average available replicate logits in first-appearance structure order."""
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got {tuple(logits.shape)}")
    if len(logits) != len(labels) or len(logits) != len(structure_ids):
        raise ValueError("logits, labels, and structure_ids must have equal length")
    if not structure_ids:
        raise ValueError("Cannot aggregate an empty collection")

    order = list(dict.fromkeys(str(value) for value in structure_ids))
    aggregated_logits: list[torch.Tensor] = []
    aggregated_labels: list[torch.Tensor] = []
    ids = [str(value) for value in structure_ids]
    for structure_id in order:
        positions = [index for index, value in enumerate(ids) if value == structure_id]
        index_tensor = torch.tensor(positions, dtype=torch.long, device=logits.device)
        structure_labels = labels.to(logits.device).index_select(0, index_tensor)
        if torch.unique(structure_labels).numel() != 1:
            raise ValueError(f"Conflicting labels for structure {structure_id}")
        aggregated_logits.append(logits.index_select(0, index_tensor).mean(dim=0))
        aggregated_labels.append(structure_labels[0])
    return torch.stack(aggregated_logits), torch.stack(aggregated_labels).long(), order


def structure_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    structure_ids: Sequence[str],
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    structure_logits, structure_labels, _ = aggregate_structure_logits(logits, labels, structure_ids)
    weights = class_weights.to(structure_logits.device) if class_weights is not None else None
    return F.cross_entropy(structure_logits, structure_labels, weight=weights)

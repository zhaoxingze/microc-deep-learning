from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.task1.aggregation import aggregate_structure_logits, structure_cross_entropy
from src.task1.dataset import (
    PairedMicroCDataset,
    build_records,
    compute_structure_class_weights,
    replicate_balance_weights,
)
from src.task1.metrics import classification_metrics


def _paired_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "structure_id": ["A", "B", "A", "B"],
            "replicate": ["rep1", "rep1", "rep2", "rep2"],
            "type": ["CHIN", "OPCID", "CHIN", "OPCID"],
            "label_id": [0, 1, 0, 1],
            "array_index": [0, 1, 0, 1],
            "available": [True, True, True, True],
            "has_zero_axis": [False, False, False, True],
        }
    )


def _split() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "structure_id": ["A", "B"],
            "type": ["CHIN", "OPCID"],
            "genomic_group_id": ["group_0001", "group_0002"],
            "split": ["train", "train"],
        }
    )


def test_class_weights_use_unique_train_structures_not_replicate_rows() -> None:
    train = pd.DataFrame(
        {
            "structure_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "type": ["CHIN", "CHIN", "CHIN", "CHIN", "OPCID", "OPCID", "CHID", "CHID"],
        }
    )
    weights = compute_structure_class_weights(train)
    assert torch.allclose(weights, torch.tensor([2 / 3, 4 / 3, 4 / 3], dtype=torch.float32))


def test_structure_balanced_replicate_weights_sum_to_one() -> None:
    weights = replicate_balance_weights(["A", "A", "B"])
    assert weights.tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_zero_axis_exclusion_keeps_other_replicate_available() -> None:
    records = build_records(_paired_metadata(), _split(), "train", exclude_zero_axis=True, structure_balanced=True)
    assert records[["structure_id", "replicate"]].values.tolist() == [["A", "rep1"], ["B", "rep1"], ["A", "rep2"]]
    assert records.loc[records.structure_id == "B", "sample_weight"].item() == 1.0


def test_dataset_returns_one_by_64_by_64_and_valid_label(tmp_path: Path) -> None:
    for replicate in ("rep1", "rep2"):
        directory = tmp_path / replicate
        directory.mkdir()
        np.save(directory / "known_windows_raw.npy", np.ones((2, 1, 64, 64), dtype=np.float32))
    records = build_records(_paired_metadata(), _split(), "train")
    dataset = PairedMicroCDataset(records, tmp_path, "raw")
    item = dataset[0]
    assert item["matrix"].shape == (1, 64, 64)
    assert item["label"].item() in {0, 1, 2}


def test_aggregate_logits_uses_available_replicates_only() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    labels = torch.tensor([0, 0, 2])
    aggregated, structure_labels, order = aggregate_structure_logits(logits, labels, ["A", "A", "B"])
    assert order == ["A", "B"]
    assert torch.equal(aggregated, torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]]))
    assert torch.equal(structure_labels, torch.tensor([0, 2]))


def test_structure_validation_loss_is_ce_after_logit_aggregation() -> None:
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    labels = torch.tensor([0, 0, 2])
    class_weights = torch.tensor([1.0, 2.0, 3.0])
    actual = structure_cross_entropy(logits, labels, ["A", "A", "B"], class_weights)
    expected = F.cross_entropy(torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]]), torch.tensor([0, 2]), weight=class_weights)
    assert torch.allclose(actual, expected)


def test_aggregation_rejects_conflicting_labels_within_structure() -> None:
    with pytest.raises(ValueError, match="Conflicting labels"):
        aggregate_structure_logits(torch.zeros(2, 3), torch.tensor([0, 1]), ["A", "A"])


def test_classification_metrics_use_fixed_class_order() -> None:
    result = classification_metrics(np.array([0, 1, 2, 2]), np.array([0, 0, 2, 1]))
    assert result.summary["accuracy"] == pytest.approx(0.5)
    assert result.confusion.tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 1]]
    assert result.report["class"].tolist() == ["CHIN", "OPCID", "CHID", "macro avg", "weighted avg", "accuracy"]

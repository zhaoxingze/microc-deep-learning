from __future__ import annotations

import inspect

import pandas as pd
import torch
import numpy as np

from src.task1.evaluation import select_best_input
from src.task1.model import SmallMicroCCNN
from src.task1.metrics import classification_metrics
from src.task1.evaluation import EvaluationResult
from src.task1.reporting import write_evaluation_outputs, write_training_history
from src.task1.trainer import choose_better_epoch, set_reproducible_seed, train_experiment


def test_train_signature_has_no_test_loader() -> None:
    assert "test_loader" not in inspect.signature(train_experiment).parameters


def test_epoch_selection_prefers_higher_validation_structure_macro_f1() -> None:
    incumbent = {"val_structure_macro_f1": 0.60, "val_structure_loss": 0.2}
    candidate = {"val_structure_macro_f1": 0.61, "val_structure_loss": 9.0}
    assert choose_better_epoch(candidate, incumbent)


def test_epoch_tie_break_uses_lower_structure_validation_loss() -> None:
    incumbent = {"val_structure_macro_f1": 0.60, "val_structure_loss": 0.8}
    candidate = {"val_structure_macro_f1": 0.60, "val_structure_loss": 0.7}
    assert choose_better_epoch(candidate, incumbent)


def test_epoch_selection_does_not_accept_test_metrics() -> None:
    candidate = {"val_structure_macro_f1": 0.5, "val_structure_loss": 0.7, "test_macro_f1": 1.0}
    incumbent = {"val_structure_macro_f1": 0.6, "val_structure_loss": 0.8, "test_macro_f1": 0.0}
    assert not choose_better_epoch(candidate, incumbent)


def test_input_selection_uses_validation_not_test() -> None:
    comparison = pd.DataFrame(
        [
            {"input_type": "raw", "best_val_macro_f1": 0.8, "best_val_loss": 0.5, "test_macro_f1": 0.0},
            {"input_type": "oe", "best_val_macro_f1": 0.7, "best_val_loss": 0.4, "test_macro_f1": 1.0},
        ]
    )
    assert select_best_input(comparison) == "raw"


def test_same_seed_reinitializes_identical_model_parameters() -> None:
    set_reproducible_seed(20260920)
    first = SmallMicroCCNN()
    first_state = {name: value.detach().clone() for name, value in first.state_dict().items()}
    set_reproducible_seed(20260920)
    second = SmallMicroCCNN()
    assert all(torch.equal(first_state[name], second.state_dict()[name]) for name in first_state)


def test_reporting_writes_primary_secondary_and_curve_outputs(tmp_path) -> None:
    metric = classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]))
    evaluation = EvaluationResult(
        summary={**metric.summary, "structure_loss": 0.1, "n_structures": 3.0, "n_replicates": 6.0},
        primary=metric,
        predictions=pd.DataFrame({"structure_id": ["A", "B", "C"], "true_label_id": [0, 1, 2], "predicted_label_id": [0, 1, 2]}),
        secondary={"rep1_only": metric, "rep2_only": metric, "replicate_level": metric},
    )
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [1.0, 0.8],
            "train_accuracy": [0.4, 0.6],
            "train_macro_f1": [0.3, 0.5],
            "val_structure_loss": [1.1, 0.9],
            "val_structure_accuracy": [0.3, 0.6],
            "val_structure_macro_f1": [0.2, 0.5],
        }
    )
    write_evaluation_outputs(evaluation, tmp_path)
    write_training_history(history, tmp_path)
    expected = {
        "classification_report.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "normalized_confusion_matrix.png",
        "primary_structure_level_metrics.json",
        "secondary_replicate_level_metrics.csv",
        "training_history.csv",
        "training_curve.png",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}

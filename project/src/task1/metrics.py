"""Fixed-order multiclass metrics for structure and replicate predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


CLASS_NAMES = ("CHIN", "OPCID", "CHID")
CLASS_IDS = (0, 1, 2)


@dataclass(frozen=True)
class MetricResult:
    summary: dict[str, float]
    report: pd.DataFrame
    confusion: np.ndarray
    normalized_confusion: np.ndarray


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if true.shape != predicted.shape or true.ndim != 1:
        raise ValueError(f"Expected equal 1D arrays, got {true.shape} and {predicted.shape}")
    if true.size == 0:
        raise ValueError("Cannot compute metrics for zero samples")

    raw_report = classification_report(
        true,
        predicted,
        labels=list(CLASS_IDS),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict[str, object]] = []
    for name in (*CLASS_NAMES, "macro avg", "weighted avg"):
        values = raw_report[name]
        rows.append(
            {
                "class": name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    accuracy = float(accuracy_score(true, predicted))
    rows.append({"class": "accuracy", "precision": np.nan, "recall": np.nan, "f1": accuracy, "support": int(len(true))})
    report = pd.DataFrame(rows)

    confusion = confusion_matrix(true, predicted, labels=list(CLASS_IDS))
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals != 0,
    )
    summary: dict[str, float] = {
        "accuracy": accuracy,
        "macro_precision": float(raw_report["macro avg"]["precision"]),
        "macro_recall": float(raw_report["macro avg"]["recall"]),
        "macro_f1": float(raw_report["macro avg"]["f1-score"]),
        "weighted_f1": float(raw_report["weighted avg"]["f1-score"]),
        "n_samples": float(len(true)),
    }
    for name in CLASS_NAMES:
        summary[f"{name}_precision"] = float(raw_report[name]["precision"])
        summary[f"{name}_recall"] = float(raw_report[name]["recall"])
        summary[f"{name}_f1"] = float(raw_report[name]["f1-score"])
    return MetricResult(summary, report, confusion, normalized)

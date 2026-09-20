"""CSV, JSON, confusion-matrix, and training-curve outputs for Task 1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluation import EvaluationResult
from .metrics import CLASS_NAMES


def _plot_confusion(matrix: np.ndarray, output: Path, normalized: bool) -> None:
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalized else "d",
        cmap="Blues",
        vmin=0,
        vmax=1 if normalized else None,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Normalized confusion matrix" if normalized else "Confusion matrix")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_evaluation_outputs(result: EvaluationResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.primary.report.to_csv(output / "classification_report.csv", index=False, encoding="utf-8-sig")
    confusion_frame = pd.DataFrame(result.primary.confusion, index=CLASS_NAMES, columns=CLASS_NAMES)
    confusion_frame.index.name = "True"
    confusion_frame.to_csv(output / "confusion_matrix.csv", encoding="utf-8-sig")
    normalized_frame = pd.DataFrame(result.primary.normalized_confusion, index=CLASS_NAMES, columns=CLASS_NAMES)
    normalized_frame.index.name = "True"
    normalized_frame.to_csv(output / "normalized_confusion_matrix.csv", encoding="utf-8-sig")
    _plot_confusion(result.primary.confusion, output / "confusion_matrix.png", normalized=False)
    _plot_confusion(result.primary.normalized_confusion, output / "normalized_confusion_matrix.png", normalized=True)
    result.predictions.to_csv(output / "structure_predictions.csv", index=False, encoding="utf-8-sig")
    (output / "primary_structure_level_metrics.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    secondary_rows: list[dict[str, object]] = []
    for scope, metrics in result.secondary.items():
        secondary_rows.append({"scope": scope, **metrics.summary})
        metrics.report.to_csv(output / f"{scope}_classification_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(secondary_rows).to_csv(
        output / "secondary_replicate_level_metrics.csv", index=False, encoding="utf-8-sig"
    )


def write_training_history(history: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history.to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_structure_loss"], label="Validation structure")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_macro_f1"], label="Train replicate")
    axes[1].plot(history["epoch"], history["val_structure_macro_f1"], label="Validation structure")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Macro-F1")
    axes[1].legend()
    fig.savefig(output / "training_curve.png", dpi=180)
    plt.close(fig)

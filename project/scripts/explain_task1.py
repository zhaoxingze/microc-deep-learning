"""Generate Grad-CAM examples and quantitative summaries for the selected Task 1 CNN."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.gradcam import (
    GradCAM,
    attention_intensity_correlations,
    diagonal_attention_ratio,
    zero_axis_attention_ratio,
)
from src.task1.metrics import CLASS_NAMES
from src.task1.model import SmallMicroCCNN


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _choose_examples(predictions: pd.DataFrame, correct_per_class: int, incorrect_per_class: int) -> pd.DataFrame:
    pieces = []
    for class_id in range(len(CLASS_NAMES)):
        class_rows = predictions.loc[predictions["true_label_id"] == class_id]
        correct = class_rows.loc[class_rows["predicted_label_id"] == class_id].head(correct_per_class)
        incorrect = class_rows.loc[class_rows["predicted_label_id"] != class_id].head(incorrect_per_class)
        pieces.extend([correct, incorrect])
    return pd.concat(pieces, ignore_index=True).drop_duplicates("structure_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs" / "task1")
    parser.add_argument("--correct-per-class", type=int, default=5)
    parser.add_argument("--incorrect-per-class", type=int, default=3)
    parser.add_argument("--diagonal-band", type=int, default=2)
    args = parser.parse_args()

    outputs_root = args.outputs_root.resolve()
    comparison = pd.read_csv(outputs_root / "input_comparison.csv")
    best_rows = comparison.loc[comparison["best_input_type"].astype(str).str.casefold() == "true"]
    if len(best_rows) != 1:
        raise SystemExit("Expected exactly one validation-selected input type")
    input_type = str(best_rows.iloc[0]["input_type"])
    experiment_dir = outputs_root / input_type
    predictions = pd.read_csv(experiment_dir / "structure_predictions.csv")
    selected = _choose_examples(predictions, args.correct_per_class, args.incorrect_per_class)
    paired = pd.read_csv(args.processed_root.resolve() / "paired_known_metadata.csv")
    paired = paired.loc[paired["structure_id"].isin(selected["structure_id"])].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False)
    model = SmallMicroCCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    explainer = GradCAM(model, model.target_layer)
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for replicate in ("rep1", "rep2"):
        arrays[(replicate, input_type)] = np.load(
            args.processed_root.resolve() / replicate / f"known_windows_{input_type}.npy",
            mmap_mode="r", allow_pickle=False,
        )
        arrays[(replicate, "raw")] = np.load(
            args.processed_root.resolve() / replicate / "known_windows_raw.npy",
            mmap_mode="r", allow_pickle=False,
        )

    output = outputs_root / "gradcam"
    output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    for prediction in selected.itertuples(index=False):
        structure_rows = paired.loc[paired["structure_id"] == prediction.structure_id]
        figure, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        for column, replicate in enumerate(("rep1", "rep2")):
            matches = structure_rows.loc[structure_rows["replicate"] == replicate]
            if matches.empty or not bool(matches.iloc[0]["available"]):
                axes[0, column].axis("off")
                axes[1, column].axis("off")
                continue
            row = matches.iloc[0]
            index = int(row["array_index"])
            matrix = np.asarray(arrays[(replicate, input_type)][index, 0], dtype=np.float32)
            raw = np.asarray(arrays[(replicate, "raw")][index, 0], dtype=np.float32)
            inputs = torch.from_numpy(matrix.copy())[None, None].to(device)
            cam = explainer.compute(inputs, int(prediction.predicted_label_id))
            pearson, spearman = attention_intensity_correlations(cam, matrix)
            metrics.append(
                {
                    "structure_id": prediction.structure_id,
                    "replicate": replicate,
                    "true_class": CLASS_NAMES[int(prediction.true_label_id)],
                    "predicted_class": CLASS_NAMES[int(prediction.predicted_label_id)],
                    "correct": bool(prediction.true_label_id == prediction.predicted_label_id),
                    "diagonal_attention_ratio": diagonal_attention_ratio(cam, args.diagonal_band),
                    "zero_axis_attention_ratio": zero_axis_attention_ratio(cam, raw),
                    "cam_input_pearson": pearson,
                    "cam_input_spearman": spearman,
                    "has_zero_axis": bool(row["has_zero_axis"]),
                }
            )
            axes[0, column].imshow(matrix, cmap="magma")
            axes[0, column].set_title(f"{replicate} {input_type}")
            axes[1, column].imshow(matrix, cmap="gray")
            axes[1, column].imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[1, column].set_title(f"{replicate} Grad-CAM")
            for axis in (axes[0, column], axes[1, column]):
                axis.set_xticks([])
                axis.set_yticks([])
        true_name = CLASS_NAMES[int(prediction.true_label_id)]
        predicted_name = CLASS_NAMES[int(prediction.predicted_label_id)]
        status = "correct" if true_name == predicted_name else "incorrect"
        figure.suptitle(f"{prediction.structure_id} | true={true_name}, predicted={predicted_name}")
        figure.savefig(output / f"{true_name}_{status}_{_safe_name(str(prediction.structure_id))}.png", dpi=180)
        plt.close(figure)
    explainer.close()

    detail = pd.DataFrame(metrics)
    detail.to_csv(output / "gradcam_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby(["true_class", "correct"], observed=True)
        .agg(
            N=("structure_id", "nunique"),
            replicate_N=("replicate", "size"),
            diagonal_attention_mean=("diagonal_attention_ratio", "mean"),
            zero_axis_attention_mean=("zero_axis_attention_ratio", "mean"),
            cam_input_pearson_mean=("cam_input_pearson", "mean"),
            cam_input_spearman_mean=("cam_input_spearman", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "gradcam_summary.csv", index=False, encoding="utf-8-sig")
    (output / "gradcam_config.json").write_text(
        json.dumps(
            {
                "input_type": input_type,
                "checkpoint": str((experiment_dir / "best_model.pth").resolve()),
                "selection": {"correct_per_class": args.correct_per_class, "incorrect_per_class": args.incorrect_per_class},
                "diagonal_band": args.diagonal_band,
                "n_structures": int(detail["structure_id"].nunique()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

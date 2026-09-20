"""Run a train-only brightness/intensity baseline with structure-level evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.brightness import aggregate_structure_scores, extract_intensity_features, fit_brightness_model
from src.task1.evaluation import EvaluationResult
from src.task1.metrics import classification_metrics
from src.task1.reporting import write_evaluation_outputs


FEATURE_NAMES = ["mean", "std", "max", "total_contact", "median", "p95"]


def _build_feature_frame(processed_root: Path, split_path: Path) -> pd.DataFrame:
    paired = pd.read_csv(processed_root / "paired_known_metadata.csv")
    split = pd.read_csv(split_path)[["structure_id", "split"]]
    records = paired.merge(split, on="structure_id", how="inner", validate="many_to_one")
    records = records.loc[records["available"].astype(str).str.casefold().isin({"true", "1"})].copy()
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for replicate in sorted(records["replicate"].unique()):
        arrays[str(replicate)] = np.load(
            processed_root / str(replicate) / "known_windows_raw.npy", mmap_mode="r", allow_pickle=False
        )
    for row in records.itertuples(index=False):
        matrix = arrays[str(row.replicate)][int(row.array_index), 0]
        rows.append(
            {
                "structure_id": str(row.structure_id),
                "replicate": str(row.replicate),
                "type": str(row.type),
                "label_id": int(row.label_id),
                "split": str(row.split),
                **extract_intensity_features(matrix).to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _evaluate(frame: pd.DataFrame, scores: np.ndarray) -> EvaluationResult:
    structure_scores, structure_labels, order = aggregate_structure_scores(
        scores, frame["label_id"].to_numpy(), frame["structure_id"].astype(str).tolist()
    )
    structure_predictions = structure_scores.argmax(axis=1)
    primary = classification_metrics(structure_labels, structure_predictions)
    replicate_predictions = scores.argmax(axis=1)
    secondary = {
        "replicate_level": classification_metrics(frame["label_id"].to_numpy(), replicate_predictions)
    }
    for replicate in ("rep1", "rep2"):
        mask = frame["replicate"].to_numpy() == replicate
        if mask.any():
            secondary[f"{replicate}_only"] = classification_metrics(
                frame.loc[mask, "label_id"].to_numpy(), replicate_predictions[mask]
            )
    replicate_lookup = frame.groupby("structure_id")["replicate"].agg(lambda x: ";".join(sorted(x)))
    predictions = pd.DataFrame(
        {
            "structure_id": order,
            "true_label_id": structure_labels,
            "predicted_label_id": structure_predictions,
            "replicates_used": [replicate_lookup.loc[value] for value in order],
            "missing_replicate": [len(replicate_lookup.loc[value].split(";")) < 2 for value in order],
            **{f"score_{index}": structure_scores[:, index] for index in range(structure_scores.shape[1])},
        }
    )
    summary = {**primary.summary, "n_structures": float(len(order)), "n_replicates": float(len(frame))}
    return EvaluationResult(summary=summary, primary=primary, predictions=predictions, secondary=secondary)


def _plot_features(frame: pd.DataFrame, output: Path) -> None:
    long = frame.melt(id_vars=["split", "type"], value_vars=FEATURE_NAMES, var_name="feature", value_name="value")
    long["plot_value"] = np.sign(long["value"]) * np.log1p(np.abs(long["value"]))
    grid = sns.catplot(data=long, x="type", y="plot_value", col="feature", col_wrap=3, kind="box", sharey=False)
    grid.set_axis_labels("Class", "signed log1p(value)")
    grid.fig.suptitle("Intensity feature distributions", y=1.02)
    grid.savefig(output, dpi=180)
    plt.close(grid.fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--split-path", type=Path, default=PROJECT_ROOT / "data" / "splits" / "group_split.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "task1" / "brightness_baseline")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = _build_feature_frame(args.processed_root.resolve(), args.split_path.resolve())
    frame.to_csv(output / "feature_distribution.csv", index=False, encoding="utf-8-sig")
    _plot_features(frame, output / "feature_distribution.png")
    train = frame.loc[frame["split"] == "train"]
    fitted = fit_brightness_model(train[FEATURE_NAMES].to_numpy(), train["label_id"].to_numpy(), args.seed)

    summaries: dict[str, dict[str, float]] = {}
    for split_name in ("val", "test"):
        subset = frame.loc[frame["split"] == split_name].reset_index(drop=True)
        scores = fitted.model.decision_function(fitted.scaler.transform(subset[FEATURE_NAMES].to_numpy()))
        result = _evaluate(subset, scores)
        write_evaluation_outputs(result, output / "validation" if split_name == "val" else output)
        summaries[split_name] = result.summary
    (output / "baseline_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

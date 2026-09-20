"""Create reproducible quality-control plots and statistics for known windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LABEL_ORDER = ["CHIN", "OPCID", "CHID"]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _symmetry_errors(array: np.ndarray) -> np.ndarray:
    return np.max(np.abs(array - np.swapaxes(array, -1, -2)), axis=(-1, -2))


def _plot_examples(
    matrices: np.ndarray,
    indices: np.ndarray,
    title: str,
    output: Path,
    vmax: float,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    flat_axes = axes.ravel()
    image = None
    for axis, sample_index in zip(flat_axes, indices, strict=False):
        image = axis.imshow(matrices[sample_index], cmap="magma", vmin=0, vmax=vmax, origin="lower")
        axis.set_title(f"sample {sample_index}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat_axes[len(indices):]:
        axis.axis("off")
    fig.suptitle(f"{title} examples (shared color scale)")
    if image is not None:
        fig.colorbar(image, ax=flat_axes.tolist(), shrink=0.75, label="log1p contact count")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_mean(matrix: np.ndarray, title: str, output: Path, vmax: float) -> None:
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=vmax, origin="lower")
    axis.set_title(f"{title} mean (log1p)")
    axis.set_xlabel("100 bp bin")
    axis.set_ylabel("100 bp bin")
    fig.colorbar(image, ax=axis, label="mean log1p contact count")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    _configure_stdout()
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "qc")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--examples-per-class", type=int, default=10)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(data_dir / "known_windows_raw.npy", allow_pickle=False)
    log1p = np.load(data_dir / "known_windows_log1p.npy", allow_pickle=False)
    oe = np.load(data_dir / "known_windows_oe.npy", allow_pickle=False)
    labels = np.load(data_dir / "known_labels.npy", allow_pickle=False)
    metadata = pd.read_csv(data_dir / "known_metadata.csv")

    if not (raw.shape == log1p.shape == oe.shape):
        raise SystemExit(f"Matrix version shapes differ: raw={raw.shape}, log1p={log1p.shape}, oe={oe.shape}")
    if raw.ndim != 4 or raw.shape[1:] != (1, 64, 64):
        raise SystemExit(f"Expected (N, 1, 64, 64), got {raw.shape}")
    if len(raw) != len(labels) or len(raw) != len(metadata):
        raise SystemExit(f"Length mismatch: X={len(raw)}, y={len(labels)}, metadata={len(metadata)}")

    matrices = log1p[:, 0].astype(np.float64)
    raw_2d = raw[:, 0].astype(np.float64)
    symmetry = _symmetry_errors(raw_2d)
    has_nan = np.isnan(raw).any(axis=(1, 2, 3)) | np.isnan(log1p).any(axis=(1, 2, 3)) | np.isnan(oe).any(axis=(1, 2, 3))
    has_inf = np.isinf(raw).any(axis=(1, 2, 3)) | np.isinf(log1p).any(axis=(1, 2, 3)) | np.isinf(oe).any(axis=(1, 2, 3))
    all_zero = np.count_nonzero(raw, axis=(1, 2, 3)) == 0
    zero_row_count = np.all(raw_2d == 0, axis=2).sum(axis=1)
    zero_col_count = np.all(raw_2d == 0, axis=1).sum(axis=1)
    has_zero_axis = (zero_row_count > 0) | (zero_col_count > 0)
    rng = np.random.default_rng(args.seed)
    shared_vmax = float(np.percentile(matrices, 99.5))
    if shared_vmax <= 0:
        shared_vmax = 1.0

    summary_rows: list[dict[str, object]] = []
    sample_metrics = metadata[["sample_index", "label", "chrom", "center", "padded"]].copy()
    sample_metrics["all_zero"] = all_zero
    sample_metrics["zero_row_count"] = zero_row_count
    sample_metrics["zero_col_count"] = zero_col_count
    sample_metrics["has_zero_axis"] = has_zero_axis
    sample_metrics["has_nan"] = has_nan
    sample_metrics["has_inf"] = has_inf
    sample_metrics["symmetry_max_abs_error"] = symmetry
    sample_metrics["raw_min"] = raw_2d.min(axis=(1, 2))
    sample_metrics["raw_max"] = raw_2d.max(axis=(1, 2))
    sample_metrics["raw_mean"] = raw_2d.mean(axis=(1, 2))
    sample_metrics["raw_zero_fraction"] = (raw_2d == 0).mean(axis=(1, 2))

    class_means: dict[str, np.ndarray] = {}
    for label in LABEL_ORDER:
        class_indices = np.flatnonzero(metadata["label"].to_numpy() == label)
        if not len(class_indices):
            print(f"WARNING: no {label} samples found")
            continue
        selected = np.sort(rng.choice(class_indices, size=min(args.examples_per_class, len(class_indices)), replace=False))
        _plot_examples(matrices, selected, label, output_dir / f"{label}_examples.png", shared_vmax)
        class_means[label] = matrices[class_indices].mean(axis=0)

        values = raw_2d[class_indices].ravel()
        summary_rows.append(
            {
                "class": label,
                "n_samples": len(class_indices),
                "shape_valid_count": int(sum(raw[index].shape == (1, 64, 64) for index in class_indices)),
                "all_zero_count": int(all_zero[class_indices].sum()),
                "zero_axis_sample_count": int(has_zero_axis[class_indices].sum()),
                "nan_count": int(has_nan[class_indices].sum()),
                "inf_count": int(has_inf[class_indices].sum()),
                "padded_count": int(metadata.iloc[class_indices]["padded"].astype(bool).sum()),
                "symmetry_error_mean": float(symmetry[class_indices].mean()),
                "symmetry_error_max": float(symmetry[class_indices].max()),
                "raw_min": float(values.min()),
                "raw_q25": float(np.percentile(values, 25)),
                "raw_median": float(np.median(values)),
                "raw_q75": float(np.percentile(values, 75)),
                "raw_q99": float(np.percentile(values, 99)),
                "raw_max": float(values.max()),
                "raw_mean": float(values.mean()),
                "raw_zero_fraction": float((values == 0).mean()),
            }
        )

    mean_vmax = max(float(np.max(matrix)) for matrix in class_means.values()) if class_means else 1.0
    for label, matrix in class_means.items():
        _plot_mean(matrix, label, output_dir / f"{label}_mean.png", mean_vmax)

    counts = metadata["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=axis, palette="colorblind")
    axis.set_title("Known structure class distribution")
    axis.set_xlabel("Structure class")
    axis.set_ylabel("Sample count")
    for position, value in enumerate(counts.values):
        axis.text(position, value, str(int(value)), ha="center", va="bottom")
    fig.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    overall_values = raw_2d.ravel()
    overall = {
        "class": "ALL",
        "n_samples": len(raw),
        "shape_valid_count": int(sum(sample.shape == (1, 64, 64) for sample in raw)),
        "all_zero_count": int(all_zero.sum()),
        "zero_axis_sample_count": int(has_zero_axis.sum()),
        "nan_count": int(has_nan.sum()),
        "inf_count": int(has_inf.sum()),
        "padded_count": int(metadata["padded"].astype(bool).sum()),
        "symmetry_error_mean": float(symmetry.mean()),
        "symmetry_error_max": float(symmetry.max()),
        "raw_min": float(overall_values.min()),
        "raw_q25": float(np.percentile(overall_values, 25)),
        "raw_median": float(np.median(overall_values)),
        "raw_q75": float(np.percentile(overall_values, 75)),
        "raw_q99": float(np.percentile(overall_values, 99)),
        "raw_max": float(overall_values.max()),
        "raw_mean": float(overall_values.mean()),
        "raw_zero_fraction": float((overall_values == 0).mean()),
    }
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    summary.to_csv(output_dir / "qc_summary.csv", index=False, encoding="utf-8-sig")
    sample_metrics.to_csv(output_dir / "qc_sample_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"QC outputs written: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

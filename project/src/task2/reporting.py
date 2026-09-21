"""Auditable tables, figures, manifest, and conservative Task 2A summary."""

from __future__ import annotations

import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.task1.manifest import sha256_file


def _prepare(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def write_csv_utf8(table: pd.DataFrame, path: str | Path) -> Path:
    destination = _prepare(path)
    table.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination


def plot_recall_curve(
    recall_table: pd.DataFrame,
    random_table: pd.DataFrame,
    path: str | Path,
) -> Path:
    destination = _prepare(path)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x = recall_table["candidate_fraction"].to_numpy(dtype=float) * 100
    for label, column in (
        ("Overall", "overall_recall"),
        ("CHIN", "CHIN_recall"),
        ("OPCID", "OPCID_recall"),
        ("CHID", "CHID_recall"),
    ):
        if column in recall_table:
            axis.plot(x, recall_table[column], marker="o", label=label)
    if "random_overall_recall_mean" in random_table:
        axis.plot(
            random_table["candidate_fraction"].to_numpy(dtype=float) * 100,
            random_table["random_overall_recall_mean"],
            linestyle="--",
            color="0.35",
            label="Random overall mean",
        )
    axis.set(xlabel="Top candidate windows (%)", ylabel="Known-structure recall", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_training_curve(history: pd.DataFrame, path: str | Path) -> Path:
    destination = _prepare(path)
    required = {"epoch", "train_loss", "val_loss"}
    if not required.issubset(history.columns):
        raise ValueError("training history is incomplete")
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    axis.plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
    axis.set(xlabel="Epoch", ylabel="MSE loss")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_branch_scatter(scores: pd.DataFrame, path: str | Path) -> Path:
    destination = _prepare(path)
    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    known = scores.get("known_overlap", pd.Series(False, index=scores.index)).astype(bool)
    axis.scatter(
        scores.loc[~known, "paired_density_z"],
        scores.loc[~known, "paired_shape_z"],
        s=8, alpha=0.25, color="#4477AA", label="Other windows",
    )
    if known.any():
        axis.scatter(
            scores.loc[known, "paired_density_z"],
            scores.loc[known, "paired_shape_z"],
            s=18, alpha=0.7, color="#CC6677", label="Known overlap",
        )
    axis.axhline(0, color="0.5", linewidth=0.8)
    axis.axvline(0, color="0.5", linewidth=0.8)
    axis.set(xlabel="Paired DensityZ", ylabel="Paired ShapeZ")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_detector_recall_curves(table: pd.DataFrame, path: str | Path) -> Path:
    destination = _prepare(path)
    labels = {
        "density_only": "Density", "shape_only": "Shape",
        "legacy_mean": "Legacy Mean", "or_max": "OR-Max",
        "positive_sum": "Positive-Sum",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for method, rows in table.groupby("method", sort=False):
        ordered = rows.sort_values("candidate_fraction")
        axis.plot(
            ordered["candidate_fraction"] * 100,
            ordered["overall_recall"], marker="o", label=labels.get(str(method), str(method)),
        )
    axis.set(xlabel="Top candidate windows (%)", ylabel="Known-structure recall", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_or_max_vs_random(
    detector_table: pd.DataFrame, random_table: pd.DataFrame, path: str | Path
) -> Path:
    destination = _prepare(path)
    detector = detector_table.loc[detector_table["method"].eq("or_max")].sort_values(
        "candidate_fraction"
    )
    random = random_table.loc[random_table["method"].eq("or_max")].sort_values(
        "candidate_fraction"
    )
    x = detector["candidate_fraction"].to_numpy(dtype=float) * 100
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(x, detector["overall_recall"], marker="o", label="OR-Max")
    axis.plot(x, random["window_count_random_mean"], linestyle="--", label="Window-count random")
    axis.plot(x, random["coverage_random_mean"], linestyle="--", label="Coverage-matched random")
    if {"coverage_random_ci_low", "coverage_random_ci_high"}.issubset(random.columns):
        axis.fill_between(
            x,
            random["coverage_random_ci_low"].to_numpy(dtype=float),
            random["coverage_random_ci_high"].to_numpy(dtype=float),
            alpha=0.15, color="0.4", label="Coverage random 95% interval",
        )
    axis.set(xlabel="Top candidate windows (%)", ylabel="Known-structure recall", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def plot_reconstruction_examples(
    examples: Sequence[Mapping[str, object]], output_dir: str | Path
) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, example in enumerate(examples, start=1):
        original = np.asarray(example["original"], dtype=float)
        reconstruction = np.asarray(example["reconstruction"], dtype=float)
        residual = np.abs(original - reconstruction)
        figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True)
        for axis, matrix, title in zip(
            axes,
            (original, reconstruction, residual),
            ("Input", "Reconstruction", "Absolute residual"),
        ):
            image = axis.imshow(matrix, cmap="magma", origin="lower")
            axis.set_title(title)
            axis.set_axis_off()
            figure.colorbar(image, ax=axis, fraction=0.046)
        name = _safe_name(example.get("name", f"example_{index:02d}"))
        path = destination / f"{index:02d}_{name}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)
    return outputs


def plot_top_candidates(
    scores: pd.DataFrame,
    matrices: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path,
    *,
    limit: int = 50,
    score_column: str = "paired_candidate_score",
) -> list[Path]:
    if limit <= 0:
        raise ValueError("candidate plot limit must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    ranked = scores.sort_values(score_column, ascending=False).head(limit)
    for row in ranked.itertuples(index=False):
        window_id = str(row.window_id)
        if window_id not in matrices:
            raise ValueError(f"missing matrices for candidate {window_id}")
        rep1, rep2 = matrices[window_id]
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        for axis, matrix, title in zip(axes, (rep1, rep2), ("rep1", "rep2")):
            image = axis.imshow(np.asarray(matrix), cmap="magma", origin="lower")
            axis.set_title(title)
            axis.set_axis_off()
            figure.colorbar(image, ax=axis, fraction=0.046)
        figure.suptitle(
            f"Rank {int(row.rank)} | {window_id} | score={float(getattr(row, score_column)):.3f}"
        )
        path = destination / f"rank_{int(row.rank):04d}_{_safe_name(window_id)}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)
    return outputs


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def write_manifest(
    path: str | Path,
    *,
    configuration: Mapping[str, object],
    input_paths: Sequence[str | Path],
    output_paths: Sequence[str | Path],
    statistics: Mapping[str, object],
) -> Path:
    destination = _prepare(path)
    inputs = {str(Path(item).resolve()): _file_record(Path(item)) for item in input_paths}
    outputs = {str(Path(item).resolve()): _file_record(Path(item)) for item in output_paths}
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": dict(configuration),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "statistics": dict(statistics),
        "inputs": inputs,
        "outputs": outputs,
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def write_summary(
    path: str | Path,
    *,
    mode: str,
    counts: Mapping[str, object],
    correlations: Mapping[str, object],
    zero_axis: Mapping[str, object],
    recall_table: pd.DataFrame,
    artifact_paths: Sequence[str | Path],
) -> Path:
    destination = _prepare(path)
    smoke = mode == "smoke"
    lines = ["# Task 2A candidate detector summary", ""]
    if smoke:
        lines += [
            "> **SMOKE 运行警告：** 本次仅扫描小范围并缩短训练，结果不能作为科学结论。",
            "",
        ]
    lines += ["## Run scope", "", f"- Mode: `{mode}`"]
    lines += [f"- {key}: {value}" for key, value in counts.items()]
    lines += ["", "## Replicate consistency", ""]
    lines += [f"- {key}: {value}" for key, value in correlations.items()]
    lines += ["", "## Zero-axis audit", ""]
    lines += [f"- {key}: {value}" for key, value in zero_axis.items()]
    lines += [
        "",
        "zero-axis 富集只作为数据质量警告；本流程不自动宣称任何新型结构或生物学发现。",
        "",
        "## Known-structure recall",
        "",
        recall_table.to_markdown(index=False),
        "",
        "## Artifacts",
        "",
    ]
    lines += [f"- `{Path(item)}`" for item in artifact_paths]
    lines += [
        "",
        "## Readiness boundary",
        "",
        "输出仅支持候选排序和 Task 2A 内部验证；未进行聚类、UMAP、HDBSCAN、novel class 或 Task 2B/2C 分析。",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return destination


def write_refined_summary(
    path: str | Path,
    *,
    mode: str,
    detector_table: pd.DataFrame,
    random_table: pd.DataFrame,
    branch_table: pd.DataFrame,
    legacy_branch_correlations: Mapping[str, float],
    training: Mapping[str, object],
    zero_axis: Mapping[str, float],
    replicate_consistency: Mapping[str, float],
    readiness: Mapping[str, object],
    artifact_paths: Sequence[str | Path],
) -> Path:
    destination = _prepare(path)

    def detector_row(method: str, fraction: float = 0.20) -> pd.Series:
        rows = detector_table.loc[
            detector_table["method"].eq(method)
            & np.isclose(detector_table["candidate_fraction"], fraction)
        ]
        return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)

    def metric(method: str, column: str) -> str:
        row = detector_row(method)
        return "NA" if row.empty else f"{float(row[column]):.6f}"

    all_branch = branch_table.loc[branch_table["subset"].eq("all")]
    refined_pearson = float(all_branch.iloc[0]["pearson"]) if not all_branch.empty else float("nan")
    refined_spearman = float(all_branch.iloc[0]["spearman"]) if not all_branch.empty else float("nan")
    old_pearson = float(legacy_branch_correlations.get("pearson", float("nan")))
    old_spearman = float(legacy_branch_correlations.get("spearman", float("nan")))
    or_random = random_table.loc[
        random_table["method"].eq("or_max")
        & np.isclose(random_table["candidate_fraction"], 0.20)
    ]
    window_random = coverage_random = coverage_high = float("nan")
    if not or_random.empty:
        window_random = float(or_random.iloc[0]["window_count_random_mean"])
        coverage_random = float(or_random.iloc[0]["coverage_random_mean"])
        coverage_high = float(or_random.iloc[0]["coverage_random_ci_high"])
    ready = bool(readiness.get("ready", False)) and mode == "full"
    conclusion = (
        "Task2A detector meets the current evidence criteria for entering Task2B."
        if ready else "Task2A detector still needs refinement."
    )
    lines = ["# Task 2A.1 refined candidate detector summary", ""]
    if mode == "smoke":
        lines += [
            "> **SMOKE warning:** this reduced run validates plumbing only and cannot support scientific conclusions.",
            "",
        ]
    lines += [
        "- primary_detector = OR_MAX",
        "- shape_input_mode = oe_log_robust",
        "- Fusion rules were fixed before known-recall evaluation; no alpha/beta search was performed.",
        "",
        "## Requested questions", "",
        "1. **Why change the old score?** DensityZ and ShapeZ were strongly anticorrelated, so their mean suppressed one-branch anomalies.",
        "2. **Old Full issue:** the legacy paired branches had "
        f"Pearson {old_pearson:.6f} and Spearman {old_spearman:.6f}.",
        "3. **Why mean is unsuitable:** the scientific target is Density anomaly OR Shape anomaly; a negative branch must not cancel a positive branch.",
        "4. **Shape normalization:** genome O/E → background-train clip → log1p → per-window valid-pixel median/MAD normalization.",
        "5. **Does AE focus more on shape?** This is diagnosed, not assumed, by the branch-correlation change and the separate Shape-only recall.",
        f"6. **Correlation change:** refined Pearson {refined_pearson:.6f}; refined Spearman {refined_spearman:.6f}.",
        f"7. **OR-Max Top-20% recall:** {metric('or_max', 'overall_recall')}.",
        f"8. **Legacy Mean Top-20% recall with refined shape input:** {metric('legacy_mean', 'overall_recall')}.",
        f"9. **Positive-Sum Top-20% recall:** {metric('positive_sum', 'overall_recall')}.",
        f"10. **Density-only / Shape-only Top-20% recall:** {metric('density_only', 'overall_recall')} / {metric('shape_only', 'overall_recall')}.",
        f"11. **Window-count random Top-20% mean:** {window_random:.6f}.",
        f"12. **Coverage-matched random Top-20% mean / 95% high:** {coverage_random:.6f} / {coverage_high:.6f}.",
        f"13. **CHIN OR-Max Top-20% recall:** {metric('or_max', 'CHIN_recall')}.",
        f"14. **OPCID OR-Max Top-20% recall:** {metric('or_max', 'OPCID_recall')}.",
        f"15. **CHID OR-Max Top-20% recall:** {metric('or_max', 'CHID_recall')}.",
        "16. **Zero-axis audit:** "
        f"all={float(zero_axis.get('all_rate', float('nan'))):.6f}, "
        f"top={float(zero_axis.get('top_rate', float('nan'))):.6f}, "
        f"enrichment={float(zero_axis.get('enrichment', float('nan'))):.6f}.",
        "17. **Replicate consistency:** "
        f"Pearson={float(replicate_consistency.get('pearson', float('nan'))):.6f}, "
        f"Spearman={float(replicate_consistency.get('spearman', float('nan'))):.6f}.",
        f"18. **Task2B readiness:** {conclusion}",
        "",
        "## AutoEncoder checkpoint", "",
        f"- best_epoch: {training.get('best_epoch')}",
        f"- best_validation_loss: {training.get('best_validation_loss')}",
        "- Checkpoint selection used background validation loss only.",
        "",
        "## Readiness checks", "",
    ]
    checks = readiness.get("checks", {})
    if isinstance(checks, Mapping):
        lines += [f"- {key}: {value}" for key, value in checks.items()]
    lines += [
        "", "## Detector comparison", "", detector_table.to_markdown(index=False),
        "", "## Branch correlations", "", branch_table.to_markdown(index=False),
        "", "## Artifacts", "",
    ]
    lines += [f"- `{Path(item)}`" for item in artifact_paths]
    lines += [
        "", "## Scope boundary", "",
        "No PCA, UMAP, HDBSCAN, clustering, latent clustering, novel-class claim, biological candidate interpretation, or Task2B–2D analysis was performed.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return destination


def validate_task2a_outputs(manifest_path: str | Path) -> dict[str, object]:
    source = Path(manifest_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    for section in ("inputs", "outputs"):
        records = payload.get(section)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"manifest {section} section is missing")
        for raw_path, record in records.items():
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_file(path) != record.get("sha256"):
                raise ValueError(f"hash mismatch for {path}")
    return payload

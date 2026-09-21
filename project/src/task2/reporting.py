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
) -> list[Path]:
    if limit <= 0:
        raise ValueError("candidate plot limit must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    ranked = scores.sort_values("paired_candidate_score", ascending=False).head(limit)
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
            f"Rank {int(row.rank)} | {window_id} | score={float(row.paired_candidate_score):.3f}"
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

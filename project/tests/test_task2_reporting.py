import json

import numpy as np
import pandas as pd
import pytest

from src.task1.manifest import sha256_file
from src.task2.reporting import (
    plot_branch_scatter,
    plot_detector_recall_curves,
    plot_or_max_vs_random,
    plot_recall_curve,
    plot_reconstruction_examples,
    plot_top_candidates,
    validate_task2a_outputs,
    write_csv_utf8,
    write_manifest,
    write_refined_summary,
    write_summary,
)


def _recall_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_fraction": [0.01, 0.10],
            "overall_recall": [0.1, 0.5],
            "CHIN_recall": [0.1, 0.6],
            "OPCID_recall": [0.0, 0.4],
            "CHID_recall": [0.0, 0.2],
        }
    )


def test_report_figures_and_utf8_csv_are_created(tmp_path) -> None:
    recall = _recall_table()
    random = pd.DataFrame(
        {"candidate_fraction": [0.01, 0.10], "random_overall_recall_mean": [0.01, 0.1]}
    )
    recall_path = plot_recall_curve(recall, random, tmp_path / "recall.png")
    examples = [
        {"name": "CHIN_example", "original": np.eye(8), "reconstruction": np.eye(8) * 0.8}
    ]
    recon_paths = plot_reconstruction_examples(examples, tmp_path / "recon")
    matrices = {"w1": (np.eye(8), np.fliplr(np.eye(8)))}
    candidate_paths = plot_top_candidates(
        pd.DataFrame({"window_id": ["w1"], "rank": [1], "paired_candidate_score": [3.0]}),
        matrices,
        tmp_path / "top",
    )
    csv_path = write_csv_utf8(recall, tmp_path / "recall.csv")

    assert recall_path.stat().st_size > 0
    assert recon_paths[0].stat().st_size > 0
    assert candidate_paths[0].stat().st_size > 0
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_refined_diagnostic_figures_and_summary_are_created(tmp_path) -> None:
    scores = pd.DataFrame(
        {
            "paired_density_z": [-1.0, 1.0, 2.0],
            "paired_shape_z": [2.0, 1.0, -1.0],
            "known_overlap": [False, True, False],
        }
    )
    detector = pd.DataFrame(
        [
            {"method": method, "candidate_fraction": fraction, "overall_recall": value,
             "CHIN_recall": value, "OPCID_recall": value, "CHID_recall": value,
             "n_windows": 1, "union_coverage_bp": 6400}
            for method, value in (("density_only", 0.1), ("shape_only", 0.2),
                                  ("legacy_mean", 0.25), ("or_max", 0.4),
                                  ("positive_sum", 0.35))
            for fraction in (0.1, 0.2)
        ]
    )
    random = detector[["method", "candidate_fraction"]].copy()
    random["window_count_random_mean"] = 0.1
    random["coverage_random_mean"] = 0.12
    random["coverage_random_ci_low"] = 0.05
    random["coverage_random_ci_high"] = 0.2
    branch = pd.DataFrame(
        [{"subset": "all", "n_windows": 3, "pearson": -0.5, "spearman": -0.5}]
    )

    figures = [
        plot_branch_scatter(scores, tmp_path / "branch.png"),
        plot_detector_recall_curves(detector, tmp_path / "detectors.png"),
        plot_or_max_vs_random(detector, random, tmp_path / "random.png"),
    ]
    summary = write_refined_summary(
        tmp_path / "summary.md", mode="full", detector_table=detector,
        random_table=random, branch_table=branch,
        legacy_branch_correlations={"pearson": -0.8, "spearman": -0.9},
        legacy_top20_recall={"overall_recall": 0.5, "CHIN_recall": 0.4,
                             "OPCID_recall": 0.7, "CHID_recall": 0.6},
        training={"best_epoch": 47, "best_validation_loss": 0.25},
        zero_axis={"all_rate": 0.1, "top_rate": 0.1, "enrichment": 1.0},
        replicate_consistency={"pearson": 0.9, "spearman": 0.8},
        readiness={"ready": False, "checks": {"beats_random": False}},
        artifact_paths=figures,
    )

    assert all(path.stat().st_size > 0 for path in figures)
    text = summary.read_text(encoding="utf-8-sig")
    assert "primary_detector = OR_MAX" in text
    assert "Original legacy Full Top-20%" in text
    assert "Task2A detector still needs refinement." in text


def test_manifest_hashes_inputs_outputs_and_validator_detects_damage(tmp_path) -> None:
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.csv"
    input_path.write_bytes(b"source")
    output_path.write_bytes(b"result")
    manifest_path = write_manifest(
        tmp_path / "manifest.json",
        configuration={"mode": "smoke"},
        input_paths=[input_path],
        output_paths=[output_path],
        statistics={"windows": 3},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["inputs"][str(input_path.resolve())]["sha256"] == sha256_file(input_path)
    assert manifest["outputs"][str(output_path.resolve())]["sha256"] == sha256_file(output_path)
    validate_task2a_outputs(manifest_path)
    output_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_task2a_outputs(manifest_path)


def test_smoke_summary_is_conservative_and_contains_zero_axis_warning(tmp_path) -> None:
    path = write_summary(
        tmp_path / "summary.md",
        mode="smoke",
        counts={"windows": 118, "background_train": 20, "background_val": 10},
        correlations={"pearson": 0.4, "spearman": 0.3},
        zero_axis={"all_rate": 0.10, "top_rate": 0.40, "enrichment": 4.0},
        recall_table=_recall_table(),
        artifact_paths=[tmp_path / "candidate_scores.csv"],
    )
    text = path.read_text(encoding="utf-8-sig")

    assert "SMOKE" in text
    assert "不能作为科学结论" in text
    assert "zero-axis" in text
    assert "不自动宣称" in text

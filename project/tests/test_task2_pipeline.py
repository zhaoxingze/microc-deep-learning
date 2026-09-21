from pathlib import Path

import pytest

from scripts.run_task2a import parse_args
from src.task1.pipeline import PipelineValidationError
from src.task2.pipeline import (
    Task2AConfig,
    Task2APaths,
    build_task2a_plan,
    execute_task2a,
    refined_manifest_configuration,
)


def _config(tmp_path: Path, mode: str = "smoke", **kwargs) -> Task2AConfig:
    project = tmp_path / "project"
    raw = tmp_path / "raw"
    project.mkdir(parents=True)
    raw.mkdir(parents=True)
    rep1 = raw / "rep1.cool"
    rep2 = raw / "rep2.cool"
    rep1.write_bytes(b"cool")
    rep2.write_bytes(b"cool")
    structures = project / "data" / "processed" / "structures.csv"
    structures.parent.mkdir(parents=True)
    structures.write_text("type,chrom,start,end,center\nCHIN,MG1655,1,2,1\n", encoding="utf-8")
    return Task2AConfig(
        project_root=project,
        raw_root=raw,
        rep1_cool=rep1,
        rep2_cool=rep2,
        mode=mode,
        **kwargs,
    )


def test_smoke_and_full_paths_and_defaults_are_isolated(tmp_path) -> None:
    smoke = _config(tmp_path / "smoke")
    full = _config(tmp_path / "full", mode="full")
    smoke_paths = Task2APaths.from_config(smoke)
    full_paths = Task2APaths.from_config(full)

    assert smoke.scan_end == 100_000
    assert smoke.epochs == 3
    assert smoke.block_bp == 25_600
    assert "task2a_smoke" in str(smoke_paths.workspace_root)
    assert full.scan_end is None
    assert full.epochs == 60
    assert full.block_bp == 128_000
    assert full_paths.workspace_root == full.project_root.resolve()
    assert smoke_paths.data_root != full_paths.data_root


def test_refined_version_has_separate_outputs_and_longer_full_training(tmp_path) -> None:
    smoke = _config(tmp_path / "smoke_refined", detector_version="refined")
    full = _config(tmp_path / "full_refined", mode="full", detector_version="refined")
    smoke_paths = Task2APaths.from_config(smoke)
    full_paths = Task2APaths.from_config(full)

    assert smoke.epochs == 3 and smoke.patience == 2
    assert full.epochs == 120 and full.patience == 15
    assert smoke_paths.workspace_root.name == "task2a_refined_smoke"
    assert full_paths.output_root.name == "task2a_refined"
    assert full_paths.data_root == full.project_root / "data" / "task2"


def test_refined_dry_run_preserves_legacy_results(tmp_path) -> None:
    config = _config(tmp_path, detector_version="refined", dry_run=True)
    legacy = config.project_root / "outputs" / "task2a" / "legacy.txt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"immutable legacy")

    execute_task2a(config)

    assert legacy.read_bytes() == b"immutable legacy"
    assert not Task2APaths.from_config(config).workspace_root.exists()


def test_refined_manifest_configuration_records_fixed_science_rules(tmp_path) -> None:
    config = _config(tmp_path, mode="full", detector_version="refined")
    payload = refined_manifest_configuration(
        config,
        {"rep1": 12.0, "rep2": 13.0},
        best_epoch=47,
        best_validation_loss=0.25,
        branch_correlations=[{"subset": "all", "pearson": 0.1, "spearman": 0.2}],
    )

    assert payload["shape_input_mode"] == "oe_log_robust"
    assert payload["oe_clip_source"] == "background_train"
    assert payload["oe_clip_percentile"] == 99.5
    assert payload["primary_fusion"] == "OR_MAX"
    assert payload["fusion_methods"] == ["MEAN", "OR_MAX", "POSITIVE_SUM"]
    assert payload["epochs"] == 120 and payload["patience"] == 15
    assert payload["best_epoch"] == 47


def test_plan_contains_all_scientific_stages_and_dry_run_writes_nothing(tmp_path) -> None:
    config = _config(tmp_path, dry_run=True)
    paths = Task2APaths.from_config(config)
    names = [step.name for step in build_task2a_plan(config, paths)]

    assert names == [
        "validate_inputs", "scan_windows", "genome_expected", "background_split",
        "train_autoencoder", "score_candidates", "evaluate_and_report",
    ]
    result = execute_task2a(config)
    assert result.dry_run is True
    assert not paths.workspace_root.exists()


def test_missing_input_fails_before_creating_scientific_outputs(tmp_path) -> None:
    config = _config(tmp_path)
    config.rep2_cool.unlink()
    paths = Task2APaths.from_config(config)
    with pytest.raises(PipelineValidationError, match="rep2"):
        execute_task2a(config)
    assert not paths.data_root.exists()


def test_cli_rejects_invalid_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--mode", "invalid"])

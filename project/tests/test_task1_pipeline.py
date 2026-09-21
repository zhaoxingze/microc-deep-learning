from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.task1.pipeline import (
    PipelineConfig,
    PipelineLogger,
    PipelinePaths,
    PipelineStepError,
    PipelineValidationError,
    build_pipeline_steps,
    check_environment,
    execute_pipeline,
    read_validation_selected_input,
    should_resume_step,
    validate_base_dataset,
    validate_experiments,
    validate_finalization,
    validate_gradcam,
    validate_group_split,
    validate_paired_dataset,
    validate_qc,
    validate_structures,
)


def make_config(tmp_path: Path, mode: str = "smoke", **overrides: object) -> PipelineConfig:
    project = tmp_path / "project with spaces"
    raw = tmp_path / "raw data"
    project.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {
        "project_root": project,
        "raw_root": raw,
        "rep1_cool": raw / "rep 1.cool",
        "rep2_cool": raw / "rep 2.cool",
        "mode": mode,
        "python_executable": Path(sys.executable),
    }
    values.update(overrides)
    return PipelineConfig(**values)


def step_by_number(config: PipelineConfig, number: int):
    return {step.number: step for step in build_pipeline_steps(config)}[number]


def option_value(command: tuple[str, ...], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def write_comparison(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames = ["input_type", "best_input_type", "test_accuracy", "test_macro_f1"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_smoke_commands_use_three_epochs_and_patience_two(tmp_path: Path) -> None:
    config = make_config(tmp_path, mode="smoke")
    for number in (8, 13):
        command = step_by_number(config, number).command
        assert command is not None
        assert option_value(command, "--epochs") == "3"
        assert option_value(command, "--patience") == "2"


def test_full_commands_do_not_duplicate_formal_training_defaults(tmp_path: Path) -> None:
    config = make_config(tmp_path, mode="full")
    for number in (8, 13):
        command = step_by_number(config, number).command
        assert command is not None
        assert "--epochs" not in command
        assert "--patience" not in command


def test_smoke_paths_are_isolated_from_formal_outputs(tmp_path: Path) -> None:
    config = make_config(tmp_path, mode="smoke")
    command = step_by_number(config, 8).command
    assert command is not None
    output_root = Path(option_value(command, "--output-root"))
    assert output_root == config.project_root / "outputs" / "pipeline_runs" / "smoke" / "outputs" / "task1"
    assert output_root != config.project_root / "outputs" / "task1"


def test_command_arguments_preserve_paths_with_spaces(tmp_path: Path) -> None:
    config = make_config(tmp_path, mode="smoke")
    command = step_by_number(config, 6).command
    assert command is not None
    assert option_value(command, "--rep1-cool") == str(config.rep1_cool)
    assert option_value(command, "--rep2-cool") == str(config.rep2_cool)


def test_best_input_uses_only_explicit_validation_selection(tmp_path: Path) -> None:
    path = write_comparison(
        tmp_path / "comparison.csv",
        [
            {"input_type": "raw", "best_input_type": False, "test_accuracy": 1.0, "test_macro_f1": 1.0},
            {"input_type": "log1p", "best_input_type": True, "test_accuracy": 0.0, "test_macro_f1": 0.0},
            {"input_type": "oe", "best_input_type": False, "test_accuracy": 0.9, "test_macro_f1": 0.9},
        ],
    )
    assert read_validation_selected_input(path) == "log1p"

    path = write_comparison(
        path,
        [
            {"input_type": "raw", "best_input_type": False, "test_accuracy": 0.0, "test_macro_f1": 0.0},
            {"input_type": "log1p", "best_input_type": True, "test_accuracy": 1.0, "test_macro_f1": 1.0},
            {"input_type": "oe", "best_input_type": False, "test_accuracy": 0.1, "test_macro_f1": 0.1},
        ],
    )
    assert read_validation_selected_input(path) == "log1p"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"input_type": "raw", "best_input_type": False, "test_accuracy": 1, "test_macro_f1": 1},
                {"input_type": "oe", "best_input_type": False, "test_accuracy": 0, "test_macro_f1": 0},
            ],
            "exactly one",
        ),
        (
            [
                {"input_type": "raw", "best_input_type": True, "test_accuracy": 0, "test_macro_f1": 0},
                {"input_type": "oe", "best_input_type": True, "test_accuracy": 1, "test_macro_f1": 1},
            ],
            "exactly one",
        ),
        (
            [{"input_type": "invalid", "best_input_type": True, "test_accuracy": 0, "test_macro_f1": 0}],
            "invalid input type",
        ),
    ],
)
def test_best_input_rejects_invalid_selection(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    path = write_comparison(tmp_path / "comparison.csv", rows)
    with pytest.raises(PipelineValidationError, match=message):
        read_validation_selected_input(path)


def test_best_input_requires_selection_columns(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    path.write_text("input_type,test_accuracy\nraw,1.0\n", encoding="utf-8")
    with pytest.raises(PipelineValidationError, match="best_input_type"):
        read_validation_selected_input(path)


def artifact_paths(tmp_path: Path) -> PipelinePaths:
    return PipelinePaths.from_config(make_config(tmp_path, mode="smoke"))


def write_structures(paths: PipelinePaths) -> None:
    paths.processed_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "type": ["CHIN", "OPCID", "CHID"],
            "chrom": ["MG1655"] * 3,
            "start": [100, 200, 300],
            "end": [120, 220, 320],
            "center": [110, 210, 310],
        }
    ).to_csv(paths.processed_root / "structures.csv", index=False)


def write_base_dataset(paths: PipelinePaths, shape: tuple[int, ...] = (3, 1, 64, 64)) -> None:
    paths.processed_root.mkdir(parents=True, exist_ok=True)
    for input_type in ("raw", "log1p", "oe"):
        np.save(paths.processed_root / f"known_windows_{input_type}.npy", np.zeros(shape, dtype=np.float32))
    np.save(paths.processed_root / "known_labels.npy", np.array([0, 1, 2], dtype=np.int64))
    pd.DataFrame({"structure_id": ["A", "B", "C"]}).to_csv(
        paths.processed_root / "known_metadata.csv", index=False
    )


def write_paired_dataset(paths: PipelinePaths, rep2_shape: tuple[int, ...] = (3, 1, 64, 64)) -> None:
    for replicate, shape in (("rep1", (3, 1, 64, 64)), ("rep2", rep2_shape)):
        root = paths.processed_root / replicate
        root.mkdir(parents=True, exist_ok=True)
        for input_type in ("raw", "log1p", "oe"):
            np.save(root / f"known_windows_{input_type}.npy", np.zeros(shape, dtype=np.float32))
        np.save(root / "known_labels.npy", np.array([0, 1, 2], dtype=np.int64))
        pd.DataFrame({"structure_id": ["A", "B", "C"]}).to_csv(root / "known_metadata.csv", index=False)
    pd.DataFrame(
        {
            "structure_id": ["A", "A", "B", "B", "C", "C"],
            "replicate": ["rep1", "rep2"] * 3,
        }
    ).to_csv(paths.processed_root / "paired_known_metadata.csv", index=False)


def write_split(paths: PipelinePaths, leak: bool = False) -> None:
    paths.split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "structure_id": ["A", "B", "C"],
            "type": ["CHIN", "OPCID", "CHID"],
            "genomic_group_id": ["g1", "g1" if leak else "g2", "g3"],
            "split": ["train", "val", "test"],
        }
    ).to_csv(paths.split_path, index=False)


def test_data_artifact_validators_accept_aligned_toy_outputs(tmp_path: Path) -> None:
    paths = artifact_paths(tmp_path)
    write_structures(paths)
    write_base_dataset(paths)
    write_paired_dataset(paths)
    write_split(paths)
    paths.qc_root.mkdir(parents=True)
    (paths.qc_root / "qc_summary.csv").write_text("metric,value\ncount,3\n", encoding="utf-8")
    (paths.qc_root / "class_distribution.png").write_bytes(b"png")

    assert "Total=3" in validate_structures(paths)
    validate_base_dataset(paths)
    validate_qc(paths)
    validate_paired_dataset(paths)
    validate_group_split(paths)


def test_paired_validator_rejects_misaligned_replicate_shape(tmp_path: Path) -> None:
    paths = artifact_paths(tmp_path)
    write_paired_dataset(paths, rep2_shape=(2, 1, 64, 64))
    with pytest.raises(PipelineValidationError, match="shape"):
        validate_paired_dataset(paths)


def test_group_split_validator_rejects_genomic_group_leakage(tmp_path: Path) -> None:
    paths = artifact_paths(tmp_path)
    write_split(paths, leak=True)
    with pytest.raises(PipelineValidationError, match="leakage"):
        validate_group_split(paths)


def test_resume_skips_only_valid_resumable_artifacts(tmp_path: Path) -> None:
    paths = artifact_paths(tmp_path)
    write_base_dataset(paths)
    step = step_by_number(make_config(tmp_path), 4)
    assert should_resume_step(step, paths) is True

    np.save(paths.processed_root / "known_windows_raw.npy", np.zeros((2, 64, 64), dtype=np.float32))
    assert should_resume_step(step, paths) is False
    assert should_resume_step(step_by_number(make_config(tmp_path), 8), paths) is False


def test_analysis_artifact_validators_require_core_outputs(tmp_path: Path) -> None:
    paths = artifact_paths(tmp_path)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for input_type in ("raw", "log1p", "oe"):
        root = paths.outputs_root / input_type
        root.mkdir()
        (root / "best_model.pth").write_bytes(b"checkpoint")
        (root / "primary_structure_level_metrics.json").write_text("{}", encoding="utf-8")
        rows.append({"input_type": input_type, "best_input_type": input_type == "raw"})
    pd.DataFrame(rows).to_csv(paths.outputs_root / "input_comparison.csv", index=False)
    validate_experiments(paths)

    gradcam = paths.outputs_root / "gradcam"
    gradcam.mkdir()
    (gradcam / "gradcam_metrics.csv").write_text("structure_id\nA\n", encoding="utf-8")
    (gradcam / "gradcam_summary.csv").write_text("group,N\nCHIN,1\n", encoding="utf-8")
    (gradcam / "example.png").write_bytes(b"png")
    validate_gradcam(paths)

    (paths.outputs_root / "task1_summary.md").write_text("summary", encoding="utf-8")
    (paths.outputs_root / "task1_manifest.json").write_text(json.dumps({"task": "task1"}), encoding="utf-8")
    validate_finalization(paths)


def test_nonzero_return_stops_later_steps(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(step, logger):
        calls.append(step.number)
        return 7 if step.number == 3 else 0

    config = make_config(tmp_path)
    with pytest.raises(PipelineStepError, match="Step 3"):
        execute_pipeline(
            config,
            runner=runner,
            environment_checker=lambda *_: None,
        )
    assert calls == [0, 1, 2, 3]


def test_dry_run_executes_no_runner_and_creates_no_log(tmp_path: Path) -> None:
    calls: list[object] = []
    config = make_config(tmp_path, dry_run=True)

    result = execute_pipeline(
        config,
        runner=lambda *args: calls.append(args),
        environment_checker=lambda *_: None,
    )

    assert calls == []
    assert result.log_path is None
    assert not (config.project_root / "outputs").exists()


def test_skip_tests_omits_only_test_steps_before_failure(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(step, logger):
        calls.append(step.number)
        return 9 if step.number == 1 else 0

    config = make_config(tmp_path, skip_tests=True)
    with pytest.raises(PipelineStepError):
        execute_pipeline(
            config,
            runner=runner,
            environment_checker=lambda *_: None,
        )
    assert calls == [1]


def test_environment_check_reports_missing_exact_cooler_and_candidates(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    candidate = config.raw_root / "different.cool"
    candidate.write_bytes(b"candidate")
    logger = PipelineLogger(None, config.project_root)

    with pytest.raises(PipelineValidationError, match="different.cool"):
        check_environment(config, logger, required_modules=())


def test_logger_writes_utf8_to_terminal_and_file(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "运行.log"
    logger = PipelineLogger(log_path, tmp_path)
    logger.write("中文日志")

    assert "中文日志" in capsys.readouterr().out
    assert log_path.read_text(encoding="utf-8") == "中文日志\n"


def load_run_all_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("task1_run_all", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_smoke_mode() -> None:
    module = load_run_all_module()
    args = module.parse_args([])
    assert args.mode == "smoke"
    assert args.skip_tests is False
    assert args.dry_run is False
    assert args.resume is False


def test_cli_parses_full_and_control_flags() -> None:
    module = load_run_all_module()
    args = module.parse_args(["--mode", "full", "--skip-tests", "--dry-run", "--resume"])
    assert (args.mode, args.skip_tests, args.dry_run, args.resume) == (
        "full",
        True,
        True,
        True,
    )


def test_cli_rejects_invalid_mode() -> None:
    module = load_run_all_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--mode", "abc"])


def test_powershell_wrapper_maps_only_pipeline_flags() -> None:
    wrapper = (Path(__file__).resolve().parents[1] / "run_all.ps1").read_text(encoding="utf-8")
    for flag in ("--mode", "--skip-tests", "--dry-run", "--resume"):
        assert flag in wrapper
    for scientific_script in (
        "build_known_structure_dataset.py",
        "run_task1_experiments.py",
        "train_task1.py",
    ):
        assert scientific_script not in wrapper

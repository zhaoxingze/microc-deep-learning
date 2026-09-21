from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from src.task1.pipeline import (
    PipelineConfig,
    PipelineValidationError,
    build_pipeline_steps,
    read_validation_selected_input,
)


def make_config(tmp_path: Path, mode: str = "smoke", **overrides: object) -> PipelineConfig:
    project = tmp_path / "project with spaces"
    raw = tmp_path / "raw data"
    project.mkdir(parents=True)
    raw.mkdir(parents=True)
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

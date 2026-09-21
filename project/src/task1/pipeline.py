"""Fail-fast orchestration primitives for the existing Task 1 workflow."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


INPUT_TYPES = ("raw", "log1p", "oe")


class PipelineError(RuntimeError):
    """Base error for one-click pipeline failures."""


class PipelineValidationError(PipelineError):
    """Raised when an input or generated artifact violates its contract."""


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    raw_root: Path
    rep1_cool: Path
    rep2_cool: Path
    mode: Literal["smoke", "full"] = "smoke"
    skip_tests: bool = False
    dry_run: bool = False
    resume: bool = False
    python_executable: Path = field(default_factory=lambda: Path(sys.executable))

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError(f"Invalid pipeline mode: {self.mode}")


@dataclass(frozen=True)
class PipelinePaths:
    source_root: Path
    workspace_root: Path
    processed_root: Path
    split_path: Path
    outputs_root: Path
    qc_root: Path
    inventory_path: Path
    cool_summary_path: Path
    annotation_summary_path: Path

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PipelinePaths":
        source = config.project_root.resolve()
        workspace = (
            source
            if config.mode == "full"
            else source / "outputs" / "pipeline_runs" / "smoke"
        )
        return cls(
            source_root=source,
            workspace_root=workspace,
            processed_root=workspace / "data" / "processed",
            split_path=workspace / "data" / "splits" / "group_split.csv",
            outputs_root=workspace / "outputs" / "task1",
            qc_root=workspace / "outputs" / "qc",
            inventory_path=workspace / "outputs" / "data_inventory.csv",
            cool_summary_path=workspace / "outputs" / "cool_file_summary.csv",
            annotation_summary_path=workspace / "outputs" / "structure_annotation_summary.csv",
        )


@dataclass(frozen=True)
class PipelineStep:
    number: int
    name: str
    command: tuple[str, ...] | None
    validator_name: str | None = None
    resumable: bool = False


def _script(paths: PipelinePaths, name: str) -> str:
    return str(paths.source_root / "scripts" / name)


def _python(config: PipelineConfig) -> str:
    return str(config.python_executable)


def build_zero_axis_command(
    config: PipelineConfig,
    paths: PipelinePaths,
    selected_input: str = "<validation-selected-input>",
) -> tuple[str, ...]:
    command = [
        _python(config),
        _script(paths, "train_task1.py"),
        "--input-type",
        selected_input,
        "--mode",
        "both",
        "--processed-root",
        str(paths.processed_root),
        "--split-path",
        str(paths.split_path),
        "--output-dir",
        str(paths.outputs_root / "zero_axis_sensitivity"),
        "--exclude-zero-axis",
    ]
    if config.mode == "smoke":
        command.extend(["--epochs", "3", "--patience", "2"])
    return tuple(command)


def build_pipeline_steps(config: PipelineConfig) -> list[PipelineStep]:
    paths = PipelinePaths.from_config(config)
    python = _python(config)
    processed = str(paths.processed_root)
    split = str(paths.split_path)
    outputs = str(paths.outputs_root)
    structures = str(paths.processed_root / "structures.csv")
    structures_with_ids = str(paths.processed_root / "structures_with_ids.csv")
    paired_metadata = str(paths.processed_root / "paired_known_metadata.csv")

    experiment_command = [
        python,
        _script(paths, "run_task1_experiments.py"),
        "--processed-root",
        processed,
        "--split-path",
        split,
        "--output-root",
        outputs,
    ]
    if config.mode == "smoke":
        experiment_command.extend(["--epochs", "3", "--patience", "2"])

    return [
        PipelineStep(0, "Initial pytest", (python, "-m", "pytest", "-q")),
        PipelineStep(
            1,
            "Inspect raw data",
            (python, _script(paths, "inspect_data.py"), "--raw-dir", str(config.raw_root), "--output", str(paths.inventory_path)),
        ),
        PipelineStep(
            2,
            "Check Cooler files",
            (python, _script(paths, "check_cool_files.py"), "--raw-dir", str(config.raw_root), "--output", str(paths.cool_summary_path)),
        ),
        PipelineStep(
            3,
            "Extract structure annotations",
            (
                python,
                _script(paths, "find_structure_annotations.py"),
                "--raw-dir",
                str(config.raw_root),
                "--output",
                structures,
                "--summary-output",
                str(paths.annotation_summary_path),
            ),
            "structures",
            True,
        ),
        PipelineStep(
            4,
            "Build rep1 base dataset",
            (
                python,
                _script(paths, "build_known_structure_dataset.py"),
                "--structures",
                structures,
                "--raw-dir",
                str(config.raw_root),
                "--cool-path",
                str(config.rep1_cool),
                "--output-dir",
                processed,
            ),
            "base_dataset",
            True,
        ),
        PipelineStep(
            5,
            "Quality control",
            (python, _script(paths, "qc_known_structures.py"), "--data-dir", processed, "--output-dir", str(paths.qc_root)),
            "qc",
            True,
        ),
        PipelineStep(
            6,
            "Build paired rep1/rep2 dataset",
            (
                python,
                _script(paths, "build_paired_known_dataset.py"),
                "--structures",
                structures,
                "--rep1-cool",
                str(config.rep1_cool),
                "--rep2-cool",
                str(config.rep2_cool),
                "--output-root",
                processed,
            ),
            "paired_dataset",
            True,
        ),
        PipelineStep(
            7,
            "Build genomic-group split",
            (
                python,
                _script(paths, "build_group_split.py"),
                "--structures",
                structures_with_ids,
                "--paired-metadata",
                paired_metadata,
                "--output",
                split,
                "--summary",
                str(paths.outputs_root / "split_summary.csv"),
            ),
            "group_split",
            True,
        ),
        PipelineStep(8, "Run raw/log1p/OE CNN experiments", tuple(experiment_command), "experiments"),
        PipelineStep(
            9,
            "Analyze replicate consistency",
            (python, _script(paths, "analyze_replicate_consistency.py"), "--processed-root", processed, "--output-root", outputs),
        ),
        PipelineStep(
            10,
            "Run brightness baseline",
            (
                python,
                _script(paths, "run_brightness_baseline.py"),
                "--processed-root",
                processed,
                "--split-path",
                split,
                "--output-dir",
                str(paths.outputs_root / "brightness_baseline"),
            ),
        ),
        PipelineStep(
            11,
            "Generate Grad-CAM",
            (python, _script(paths, "explain_task1.py"), "--processed-root", processed, "--outputs-root", outputs),
            "gradcam",
        ),
        PipelineStep(12, "Read validation-selected input", None),
        PipelineStep(13, "Run zero-axis sensitivity", build_zero_axis_command(config, paths)),
        PipelineStep(
            14,
            "Finalize Task 1",
            (
                python,
                _script(paths, "finalize_task1.py"),
                "--project-root",
                str(paths.workspace_root),
                "--raw-data-root",
                str(config.raw_root),
                "--verify",
            ),
            "finalization",
        ),
        PipelineStep(15, "Final pytest", (python, "-m", "pytest", "-q")),
    ]


def _parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise PipelineValidationError(f"Invalid best_input_type boolean: {value!r}")


def read_validation_selected_input(path: str | Path) -> str:
    try:
        import pandas as pd

        comparison = pd.read_csv(Path(path), usecols=["input_type", "best_input_type"])
    except (FileNotFoundError, ValueError) as exc:
        raise PipelineValidationError(f"Cannot read input_type/best_input_type from {path}: {exc}") from exc
    selected = comparison.loc[comparison["best_input_type"].map(_parse_boolean)]
    if len(selected) != 1:
        raise PipelineValidationError(
            f"Expected exactly one best_input_type=True row, found {len(selected)}"
        )
    input_type = str(selected.iloc[0]["input_type"]).strip().casefold()
    if input_type not in INPUT_TYPES:
        raise PipelineValidationError(f"Selected invalid input type: {input_type!r}")
    return input_type

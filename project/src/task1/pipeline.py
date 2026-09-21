"""Fail-fast orchestration primitives for the existing Task 1 workflow."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


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


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise PipelineValidationError(f"Required artifact is missing: {path}")
    return path


def _require_columns(frame: object, columns: set[str], label: str) -> None:
    available = set(getattr(frame, "columns", ()))
    missing = columns - available
    if missing:
        raise PipelineValidationError(
            f"{label} is missing required columns: {', '.join(sorted(missing))}"
        )


def _load_window_arrays(root: Path) -> tuple[dict[str, object], object, object]:
    import numpy as np
    import pandas as pd

    arrays = {
        input_type: np.load(
            _require_file(root / f"known_windows_{input_type}.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        for input_type in INPUT_TYPES
    }
    labels = np.load(
        _require_file(root / "known_labels.npy"), mmap_mode="r", allow_pickle=False
    )
    metadata = pd.read_csv(_require_file(root / "known_metadata.csv"))
    return arrays, labels, metadata


def _validate_window_set(root: Path, label: str) -> tuple[tuple[int, ...], object, object]:
    arrays, labels, metadata = _load_window_arrays(root)
    shapes = {input_type: tuple(array.shape) for input_type, array in arrays.items()}
    expected_shape = next(iter(shapes.values()))
    if any(shape != expected_shape for shape in shapes.values()):
        raise PipelineValidationError(f"{label} input arrays have inconsistent shape: {shapes}")
    if len(expected_shape) != 4 or expected_shape[1:] != (1, 64, 64):
        raise PipelineValidationError(
            f"{label} arrays must have shape (N, 1, 64, 64), found {expected_shape}"
        )
    count = expected_shape[0]
    if tuple(labels.shape) != (count,):
        raise PipelineValidationError(
            f"{label} labels shape {tuple(labels.shape)} does not match array count {count}"
        )
    if len(metadata) != count:
        raise PipelineValidationError(
            f"{label} metadata row count {len(metadata)} does not match array count {count}"
        )
    _require_columns(metadata, {"structure_id"}, f"{label} metadata")
    return expected_shape, labels, metadata


def validate_structures(paths: PipelinePaths) -> str:
    import pandas as pd

    structures = pd.read_csv(_require_file(paths.processed_root / "structures.csv"))
    _require_columns(
        structures, {"type", "chrom", "start", "end", "center"}, "structures.csv"
    )
    if structures.empty:
        raise PipelineValidationError("structures.csv is empty")
    counts = structures["type"].astype(str).value_counts()
    return " ".join(
        [*(f"{label}={int(counts.get(label, 0))}" for label in ("CHIN", "OPCID", "CHID")), f"Total={len(structures)}"]
    )


def validate_base_dataset(paths: PipelinePaths) -> str:
    shape, _, _ = _validate_window_set(paths.processed_root, "base dataset")
    return f"N={shape[0]} shape={shape}"


def validate_qc(paths: PipelinePaths) -> str:
    _require_file(paths.qc_root / "qc_summary.csv")
    figures = list(paths.qc_root.glob("*.png"))
    if not figures:
        raise PipelineValidationError(f"QC directory has no PNG figures: {paths.qc_root}")
    return f"figures={len(figures)}"


def validate_paired_dataset(paths: PipelinePaths) -> str:
    import numpy as np
    import pandas as pd

    rep1_shape, rep1_labels, rep1_metadata = _validate_window_set(
        paths.processed_root / "rep1", "rep1 dataset"
    )
    rep2_shape, rep2_labels, rep2_metadata = _validate_window_set(
        paths.processed_root / "rep2", "rep2 dataset"
    )
    if rep1_shape != rep2_shape:
        raise PipelineValidationError(
            f"replicate array shape mismatch: rep1={rep1_shape}, rep2={rep2_shape}"
        )
    if not np.array_equal(rep1_labels, rep2_labels):
        raise PipelineValidationError("replicate labels are not aligned")
    rep1_ids = rep1_metadata["structure_id"].astype(str).tolist()
    rep2_ids = rep2_metadata["structure_id"].astype(str).tolist()
    if rep1_ids != rep2_ids:
        raise PipelineValidationError("replicate structure_id order is not aligned")

    paired = pd.read_csv(
        _require_file(paths.processed_root / "paired_known_metadata.csv")
    )
    _require_columns(paired, {"structure_id", "replicate"}, "paired metadata")
    if set(paired["replicate"].astype(str)) != {"rep1", "rep2"}:
        raise PipelineValidationError("paired metadata must contain rep1 and rep2")
    counts = paired.groupby("structure_id")["replicate"].nunique()
    if len(counts) != rep1_shape[0] or not counts.eq(2).all():
        raise PipelineValidationError("paired metadata does not contain two replicates per structure")
    return f"structures={rep1_shape[0]} replicate_rows={len(paired)}"


def validate_group_split(paths: PipelinePaths) -> str:
    import pandas as pd

    split = pd.read_csv(_require_file(paths.split_path))
    _require_columns(
        split,
        {"structure_id", "type", "genomic_group_id", "split"},
        "group split",
    )
    if split.empty:
        raise PipelineValidationError("group split is empty")
    if split["structure_id"].astype(str).duplicated().any():
        raise PipelineValidationError("group split contains duplicate structure_id values")
    split_names = set(split["split"].astype(str))
    if split_names != {"train", "val", "test"}:
        raise PipelineValidationError(
            f"group split must contain train, val, and test; found {sorted(split_names)}"
        )
    group_span = split.groupby("genomic_group_id")["split"].nunique()
    if (group_span > 1).any():
        leaking = group_span[group_span > 1].index.astype(str).tolist()
        raise PipelineValidationError(
            f"genomic group leakage detected: {', '.join(leaking[:5])}"
        )
    return f"structures={len(split)} groups={split['genomic_group_id'].nunique()} leakage=none"


def validate_experiments(paths: PipelinePaths) -> str:
    import pandas as pd

    comparison_path = _require_file(paths.outputs_root / "input_comparison.csv")
    comparison = pd.read_csv(comparison_path)
    _require_columns(comparison, {"input_type", "best_input_type"}, "input comparison")
    input_types = comparison["input_type"].astype(str).str.casefold()
    if len(comparison) != len(INPUT_TYPES) or set(input_types) != set(INPUT_TYPES):
        raise PipelineValidationError("input comparison must contain raw, log1p, and oe exactly once")
    if input_types.duplicated().any():
        raise PipelineValidationError("input comparison contains duplicate input types")
    selected = read_validation_selected_input(comparison_path)
    for input_type in INPUT_TYPES:
        experiment = paths.outputs_root / input_type
        _require_file(experiment / "best_model.pth")
        _require_file(experiment / "primary_structure_level_metrics.json")
    return f"input_types=3 validation_selected={selected}"


def validate_gradcam(paths: PipelinePaths) -> str:
    gradcam = paths.outputs_root / "gradcam"
    _require_file(gradcam / "gradcam_metrics.csv")
    _require_file(gradcam / "gradcam_summary.csv")
    figures = list(gradcam.glob("*.png"))
    if not figures:
        raise PipelineValidationError(f"Grad-CAM directory has no PNG figures: {gradcam}")
    return f"figures={len(figures)}"


def validate_finalization(paths: PipelinePaths) -> str:
    import json

    summary = _require_file(paths.outputs_root / "task1_summary.md")
    manifest = _require_file(paths.outputs_root / "task1_manifest.json")
    if not summary.read_text(encoding="utf-8").strip():
        raise PipelineValidationError("Task 1 summary is empty")
    try:
        content = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineValidationError(f"Task 1 manifest is invalid JSON: {exc}") from exc
    if not isinstance(content, dict):
        raise PipelineValidationError("Task 1 manifest must be a JSON object")
    return "summary=ok manifest=ok"


Validator = Callable[[PipelinePaths], str]

VALIDATORS: dict[str, Validator] = {
    "structures": validate_structures,
    "base_dataset": validate_base_dataset,
    "qc": validate_qc,
    "paired_dataset": validate_paired_dataset,
    "group_split": validate_group_split,
    "experiments": validate_experiments,
    "gradcam": validate_gradcam,
    "finalization": validate_finalization,
}


def should_resume_step(step: PipelineStep, paths: PipelinePaths) -> bool:
    if not step.resumable or not step.validator_name:
        return False
    validator = VALIDATORS.get(step.validator_name)
    if validator is None:
        return False
    try:
        validator(paths)
    except (PipelineValidationError, FileNotFoundError, OSError, ValueError):
        return False
    return True

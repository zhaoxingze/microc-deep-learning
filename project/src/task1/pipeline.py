"""Fail-fast orchestration primitives for the existing Task 1 workflow."""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, TextIO


INPUT_TYPES = ("raw", "log1p", "oe")


class PipelineError(RuntimeError):
    """Base error for one-click pipeline failures."""


class PipelineValidationError(PipelineError):
    """Raised when an input or generated artifact violates its contract."""


class PipelineStepError(PipelineError):
    """Raised when a pipeline subprocess fails."""


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


@dataclass(frozen=True)
class PipelineResult:
    mode: str
    selected_input: str | None
    test_accuracy: float | None
    test_macro_f1: float | None
    summary_path: Path | None
    manifest_path: Path | None
    gradcam_dir: Path | None
    checkpoint_path: Path | None
    log_path: Path | None
    elapsed_seconds: float
    dry_run: bool = False


class PipelineLogger:
    """Write one UTF-8 stream to the terminal and, optionally, a log file."""

    def __init__(self, log_path: Path | None, cwd: Path) -> None:
        self.log_path = log_path
        self.cwd = cwd
        self._handle: TextIO | None = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = log_path.open("a", encoding="utf-8", newline="")

    def write(self, message: str) -> None:
        line = message.rstrip("\r\n")
        print(line, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "PipelineLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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


def _validate_window_set(
    root: Path,
    label: str,
    *,
    require_structure_id: bool = True,
) -> tuple[tuple[int, ...], object, object]:
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
    if require_structure_id:
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
    shape, _, _ = _validate_window_set(
        paths.processed_root,
        "base dataset",
        require_structure_id=False,
    )
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


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "cooler",
    "matplotlib",
    "pytest",
)


def check_environment(
    config: PipelineConfig,
    logger: PipelineLogger,
    required_modules: tuple[str, ...] = REQUIRED_MODULES,
) -> None:
    """Validate exact inputs and imports without modifying the experiment workspace."""

    metadata = (
        ("Python executable", config.python_executable),
        ("Python version", platform.python_version()),
        ("ProjectRoot", config.project_root.resolve()),
        ("RawRoot", config.raw_root.resolve()),
        ("Rep1", config.rep1_cool.resolve()),
        ("Rep2", config.rep2_cool.resolve()),
        ("Mode", config.mode),
        ("DryRun", config.dry_run),
        ("Resume", config.resume),
        ("SkipTests", config.skip_tests),
    )
    logger.write("[START] Environment Check")
    for key, value in metadata:
        logger.write(f"{key}: {value}")

    problems: list[str] = []
    if not config.project_root.is_dir():
        problems.append(f"Project root is missing: {config.project_root}")
    if not config.raw_root.is_dir():
        problems.append(f"Raw-data root is missing: {config.raw_root}")

    missing_coolers = [
        path for path in (config.rep1_cool, config.rep2_cool) if not path.is_file()
    ]
    if missing_coolers:
        problems.extend(f"Required Cooler is missing: {path}" for path in missing_coolers)
        candidates = sorted(config.raw_root.glob("*.cool*")) if config.raw_root.is_dir() else []
        if candidates:
            problems.append(
                "Available Cooler candidates (not auto-selected): "
                + ", ".join(str(path) for path in candidates)
            )
        else:
            problems.append("Available Cooler candidates: none")

    missing_scripts = []
    for step in build_pipeline_steps(config):
        if step.command and len(step.command) > 1 and step.command[1].endswith(".py"):
            script = Path(step.command[1])
            if not script.is_file():
                missing_scripts.append(script)
    problems.extend(f"Pipeline script is missing: {path}" for path in missing_scripts)

    imported: dict[str, object] = {}
    for module_name in required_modules:
        try:
            imported[module_name] = importlib.import_module(module_name)
        except ImportError:
            problems.append(f"Missing dependency: {module_name}")

    if problems:
        for problem in problems:
            logger.write(problem)
        if any(problem.startswith("Missing dependency:") for problem in problems):
            logger.write("Install with: python -m pip install -r requirements.txt")
        raise PipelineValidationError("Environment check failed:\n" + "\n".join(problems))

    torch = imported.get("torch")
    subprocess_environment = build_subprocess_environment()
    logger.write(
        "CUBLAS_WORKSPACE_CONFIG: "
        + subprocess_environment["CUBLAS_WORKSPACE_CONFIG"]
    )
    logger.write(
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD: "
        + subprocess_environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]
    )
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        logger.write(f"torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            logger.write(f"GPU name: {torch.cuda.get_device_name(0)}")
        else:
            logger.write("CUDA unavailable; Task1 CNN will run on CPU.")
    logger.write("[PASS] Environment Check")


def _display_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def build_subprocess_environment() -> dict[str, str]:
    """Return an inherited environment safe for deterministic CUDA and pytest."""

    environment = os.environ.copy()
    if environment.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def run_subprocess_streaming(step: PipelineStep, logger: PipelineLogger) -> int:
    """Run one command without a shell and stream merged output to the logger."""

    if step.command is None:
        raise PipelineStepError(f"Step {step.number} has no subprocess command")
    try:
        process = subprocess.Popen(
            list(step.command),
            cwd=logger.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            env=build_subprocess_environment(),
        )
    except OSError as exc:
        raise PipelineStepError(f"Step {step.number} could not start: {exc}") from exc
    assert process.stdout is not None
    for line in process.stdout:
        logger.write(line)
    return process.wait()


Runner = Callable[[PipelineStep, PipelineLogger], int]
EnvironmentChecker = Callable[[PipelineConfig, PipelineLogger], None]


def _new_log_path(project_root: Path) -> Path:
    log_dir = project_root / "outputs" / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"run_all_{timestamp}.log"
    suffix = 1
    while path.exists():
        path = log_dir / f"run_all_{timestamp}_{suffix}.log"
        suffix += 1
    return path


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_final_metrics(paths: PipelinePaths, selected_input: str) -> tuple[float, float]:
    metrics_path = _require_file(
        paths.outputs_root / selected_input / "primary_structure_level_metrics.json"
    )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics["accuracy"]), float(metrics["macro_f1"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PipelineValidationError(f"Invalid primary metrics file {metrics_path}: {exc}") from exc


def _print_final_summary(
    logger: PipelineLogger,
    result: PipelineResult,
) -> None:
    logger.write("=" * 60)
    logger.write("TASK 1 PIPELINE COMPLETED SUCCESSFULLY")
    logger.write("=" * 60)
    logger.write(f"Mode: {result.mode}")
    logger.write(f"Selected input type: {result.selected_input}")
    logger.write(f"Test Accuracy: {result.test_accuracy:.6f}")
    logger.write(f"Test Macro-F1: {result.test_macro_f1:.6f}")
    logger.write(f"Task1 summary: {result.summary_path}")
    logger.write(f"Task1 manifest: {result.manifest_path}")
    logger.write(f"Grad-CAM directory: {result.gradcam_dir}")
    logger.write(f"Main model checkpoint: {result.checkpoint_path}")
    logger.write(f"Log file: {result.log_path}")
    logger.write(f"Elapsed time: {result.elapsed_seconds:.1f} seconds")
    if result.mode == "smoke":
        logger.write("=" * 60)
        logger.write("SMOKE RESULT - NOT FOR FINAL REPORTING")
        logger.write("Results are for pipeline verification only.")
        logger.write("Do NOT use smoke results for final scientific reporting.")
        logger.write("=" * 60)


def execute_pipeline(
    config: PipelineConfig,
    *,
    runner: Runner = run_subprocess_streaming,
    environment_checker: EnvironmentChecker = check_environment,
) -> PipelineResult:
    """Execute the complete ordered Task 1 pipeline or print its dry-run plan."""

    started = time.monotonic()
    paths = PipelinePaths.from_config(config)
    log_path = None if config.dry_run else _new_log_path(config.project_root)
    with PipelineLogger(log_path, config.project_root) as logger:
        logger.write(f"Run timestamp: {_timestamp()}")
        environment_checker(config, logger)
        steps = build_pipeline_steps(config)

        if config.dry_run:
            for step in steps:
                if config.skip_tests and step.number in {0, 15}:
                    logger.write(f"[DRY RUN] Step {step.number} — {step.name}: SKIPPED by --skip-tests")
                elif step.command is None:
                    logger.write(f"[DRY RUN] Step {step.number} — {step.name}: internal validation-only selection")
                else:
                    logger.write(f"[DRY RUN] Step {step.number} — {step.name}")
                    logger.write(f"Command: {_display_command(step.command)}")
            return PipelineResult(
                mode=config.mode,
                selected_input=None,
                test_accuracy=None,
                test_macro_f1=None,
                summary_path=None,
                manifest_path=None,
                gradcam_dir=None,
                checkpoint_path=None,
                log_path=None,
                elapsed_seconds=time.monotonic() - started,
                dry_run=True,
            )

        selected_input: str | None = None
        for original_step in steps:
            step = original_step
            if config.skip_tests and step.number in {0, 15}:
                logger.write(f"[SKIP] Step {step.number} — {step.name}: SKIPPED by --skip-tests")
                continue
            if config.resume and should_resume_step(step, paths):
                logger.write(f"[SKIP] Step {step.number} — {step.name}: validated by --resume")
                continue

            step_started = time.monotonic()
            logger.write(f"[START] Step {step.number} — {step.name}")
            logger.write(f"Step start time: {_timestamp()}")
            if step.number == 12:
                try:
                    selected_input = read_validation_selected_input(
                        paths.outputs_root / "input_comparison.csv"
                    )
                    logger.write(f"Validation-selected input type: {selected_input}")
                except PipelineError:
                    logger.write(f"Step end time: {_timestamp()}")
                    logger.write(f"[FAIL] Step {step.number} — {step.name}")
                    raise
                logger.write(f"Step end time: {_timestamp()}")
                logger.write(f"[PASS] Step {step.number} — {step.name}")
                continue
            if step.number == 13:
                if selected_input is None:
                    raise PipelineValidationError(
                        "Zero-axis sensitivity cannot run before validation-selected input resolution"
                    )
                step = replace(
                    step,
                    command=build_zero_axis_command(config, paths, selected_input),
                )
            assert step.command is not None
            logger.write(f"Command: {_display_command(step.command)}")
            try:
                return_code = runner(step, logger)
                if return_code != 0:
                    raise PipelineStepError(
                        f"Step {step.number} failed with return code {return_code}: {step.name}"
                    )
                if step.validator_name:
                    validator = VALIDATORS[step.validator_name]
                    validation_summary = validator(paths)
                    logger.write(f"Artifact validation: {validation_summary}")
                    if step.validator_name == "structures":
                        total = int(validation_summary.rsplit("Total=", 1)[1])
                        if total != 344:
                            logger.write(f"WARNING: expected 344 structures, found {total}")
            except PipelineError as exc:
                logger.write(f"Step end time: {_timestamp()}")
                logger.write(f"[FAIL] Step {step.number} — {step.name}: {exc}")
                raise
            elapsed = time.monotonic() - step_started
            logger.write(f"Step end time: {_timestamp()}")
            logger.write(f"[PASS] Step {step.number} — {step.name} ({elapsed:.1f}s)")

        if selected_input is None:
            raise PipelineValidationError("Pipeline completed steps without selecting an input type")
        test_accuracy, test_macro_f1 = _read_final_metrics(paths, selected_input)
        elapsed = time.monotonic() - started
        result = PipelineResult(
            mode=config.mode,
            selected_input=selected_input,
            test_accuracy=test_accuracy,
            test_macro_f1=test_macro_f1,
            summary_path=paths.outputs_root / "task1_summary.md",
            manifest_path=paths.outputs_root / "task1_manifest.json",
            gradcam_dir=paths.outputs_root / "gradcam",
            checkpoint_path=paths.outputs_root / selected_input / "best_model.pth",
            log_path=log_path,
            elapsed_seconds=elapsed,
        )
        _print_final_summary(logger, result)
        return result

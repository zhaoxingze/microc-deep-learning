"""One-command orchestration for the bounded Task 2A candidate detector."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import cooler
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.data.load_microc import _resolve_chromosome
from src.task1.pipeline import (
    PipelineLogger,
    PipelineStepError,
    PipelineValidationError,
)
from src.task2.autoencoder import (
    WindowDataset,
    load_autoencoder,
    score_reconstruction,
    train_autoencoder,
)
from src.task2.background import (
    assert_no_background_overlap,
    blocked_background_split,
    build_exclusion_intervals,
    mark_background_candidates,
)
from src.task2.expected import compute_genome_expected
from src.task2.recall import (
    annotate_known_overlap,
    merge_candidate_regions,
    random_recall_baseline,
    recall_at_fractions,
    score_ablation,
)
from src.task2.reporting import (
    plot_recall_curve,
    plot_reconstruction_examples,
    plot_top_candidates,
    plot_training_curve,
    validate_task2a_outputs,
    write_csv_utf8,
    write_manifest,
    write_summary,
)
from src.task2.scoring import score_windows
from src.task2.window_scan import scan_replicates


@dataclass(frozen=True)
class Task2AConfig:
    project_root: Path
    raw_root: Path
    rep1_cool: Path
    rep2_cool: Path
    mode: Literal["smoke", "full"] = "smoke"
    dry_run: bool = False
    resume: bool = False
    chrom: str = "MG1655"
    seed: int = 20_260_920
    window_bp: int = 6_400
    step_bp: int = 800
    target_bin_size: int = 100
    exclusion_margin_bp: int = 3_200
    density_weight: float = 0.5
    shape_weight: float = 0.5
    random_repeats: int = 100

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError(f"Invalid Task 2A mode: {self.mode}")

    @property
    def scan_end(self) -> int | None:
        return 100_000 if self.mode == "smoke" else None

    @property
    def epochs(self) -> int:
        return 3 if self.mode == "smoke" else 60

    @property
    def patience(self) -> int:
        return 2 if self.mode == "smoke" else 10

    @property
    def block_bp(self) -> int:
        return 25_600 if self.mode == "smoke" else 128_000


@dataclass(frozen=True)
class Task2APaths:
    source_root: Path
    workspace_root: Path
    structures_path: Path
    data_root: Path
    output_root: Path
    autoencoder_root: Path
    figures_root: Path
    metadata_path: Path
    rep1_array_path: Path
    rep2_array_path: Path
    expected_path: Path
    background_path: Path
    candidate_scores_path: Path
    candidate_regions_path: Path
    recall_path: Path
    random_path: Path
    ablation_path: Path
    consistency_path: Path
    zero_axis_path: Path
    summary_path: Path
    manifest_path: Path
    log_dir: Path

    @classmethod
    def from_config(cls, config: Task2AConfig) -> "Task2APaths":
        source = config.project_root.resolve()
        workspace = (
            source
            if config.mode == "full"
            else source / "outputs" / "pipeline_runs" / "task2a_smoke"
        )
        data_root = workspace / "data" / "task2"
        output_root = workspace / "outputs" / "task2a"
        return cls(
            source_root=source,
            workspace_root=workspace,
            structures_path=source / "data" / "processed" / "structures.csv",
            data_root=data_root,
            output_root=output_root,
            autoencoder_root=output_root / "background_autoencoder",
            figures_root=output_root / "figures",
            metadata_path=data_root / "genome_windows_metadata.csv",
            rep1_array_path=data_root / "genome_windows_rep1.npy",
            rep2_array_path=data_root / "genome_windows_rep2.npy",
            expected_path=data_root / "genome_expected.npz",
            background_path=data_root / "background_split.csv",
            candidate_scores_path=output_root / "candidate_scores.csv",
            candidate_regions_path=output_root / "top_candidate_regions.csv",
            recall_path=output_root / "known_structure_recall.csv",
            random_path=output_root / "random_recall_baseline.csv",
            ablation_path=output_root / "score_ablation.csv",
            consistency_path=output_root / "replicate_consistency.csv",
            zero_axis_path=output_root / "zero_axis_audit.csv",
            summary_path=output_root / "task2a_summary.md",
            manifest_path=output_root / "task2a_manifest.json",
            log_dir=source / "outputs" / "logs",
        )


@dataclass(frozen=True)
class Task2AStep:
    number: int
    name: str
    resumable: bool = False


@dataclass(frozen=True)
class Task2AResult:
    mode: str
    summary_path: Path | None
    manifest_path: Path | None
    log_path: Path | None
    elapsed_seconds: float
    dry_run: bool = False


def build_task2a_plan(config: Task2AConfig, paths: Task2APaths) -> list[Task2AStep]:
    del config, paths
    return [
        Task2AStep(1, "validate_inputs"),
        Task2AStep(2, "scan_windows", True),
        Task2AStep(3, "genome_expected", True),
        Task2AStep(4, "background_split", True),
        Task2AStep(5, "train_autoencoder"),
        Task2AStep(6, "score_candidates"),
        Task2AStep(7, "evaluate_and_report"),
    ]


def _validate_inputs(config: Task2AConfig, paths: Task2APaths) -> None:
    required = {
        "project root": config.project_root,
        "raw root": config.raw_root,
        "rep1 Cooler": config.rep1_cool,
        "rep2 Cooler": config.rep2_cool,
        "known structures": paths.structures_path,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise PipelineValidationError("Missing Task 2A input(s): " + "; ".join(missing))
    if config.rep1_cool.resolve() == config.rep2_cool.resolve():
        raise PipelineValidationError("rep1 and rep2 must be different Cooler files")


def _geometry(config: Task2AConfig) -> tuple[int, str]:
    values = []
    for path in (config.rep1_cool, config.rep2_cool):
        contact_map = cooler.Cooler(str(path.resolve()))
        chrom = _resolve_chromosome(contact_map, config.chrom)
        values.append((int(contact_map.chromsizes[chrom]), int(contact_map.binsize or 0), chrom))
    if values[0][:2] != values[1][:2]:
        raise PipelineValidationError(f"replicate Cooler geometry differs: {values}")
    if values[0][1] <= 0 or config.target_bin_size % values[0][1]:
        raise PipelineValidationError("target bin size is incompatible with Cooler binsize")
    return values[0][0], config.chrom


def _array_paths(paths: Task2APaths) -> dict[str, Path]:
    return {"rep1": paths.rep1_array_path, "rep2": paths.rep2_array_path}


def _validate_scan(paths: Task2APaths, config: Task2AConfig) -> pd.DataFrame:
    required = {paths.metadata_path, paths.rep1_array_path, paths.rep2_array_path}
    if not all(path.is_file() for path in required):
        raise PipelineValidationError("resumable scan artifacts are incomplete")
    metadata = pd.read_csv(paths.metadata_path, encoding="utf-8-sig")
    columns = {"window_id", "replicate", "array_index", "chrom", "start", "end", "center"}
    if not columns.issubset(metadata.columns):
        raise PipelineValidationError("resumable scan metadata has an invalid schema")
    ids = []
    for replicate, array_path in _array_paths(paths).items():
        rows = metadata.loc[metadata["replicate"].eq(replicate)].sort_values("array_index")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (len(rows), 1, config.window_bp // config.target_bin_size,
                          config.window_bp // config.target_bin_size)
        if array.shape != expected_shape:
            raise PipelineValidationError(f"resumable {replicate} array shape is invalid")
        ids.append(rows["window_id"].astype(str).tolist())
    if ids[0] != ids[1] or not ids[0]:
        raise PipelineValidationError("resumable replicate window alignment is invalid")
    return metadata


def _validate_expected(paths: Task2APaths, config: Task2AConfig) -> dict[str, np.ndarray]:
    if not paths.expected_path.is_file():
        raise PipelineValidationError("resumable genome expected artifact is missing")
    with np.load(paths.expected_path, allow_pickle=False) as stored:
        expected = {replicate: stored[replicate].copy() for replicate in ("rep1", "rep2")}
    needed = config.window_bp // config.target_bin_size
    if any(vector.shape != (needed,) or not np.isfinite(vector).all() for vector in expected.values()):
        raise PipelineValidationError("resumable genome expected vectors are invalid")
    return expected


def _validate_background(paths: Task2APaths) -> pd.DataFrame:
    if not paths.background_path.is_file():
        raise PipelineValidationError("resumable background split is missing")
    split = pd.read_csv(paths.background_path, encoding="utf-8-sig")
    assert_no_background_overlap(split)
    return split


def _log_plan(config: Task2AConfig, paths: Task2APaths, logger: PipelineLogger) -> None:
    logger.write(f"Task 2A mode={config.mode} workspace={paths.workspace_root}")
    logger.write(
        f"window={config.window_bp}bp step={config.step_bp}bp target={config.target_bin_size}bp "
        f"scan_end={config.scan_end or 'chromosome_end'} epochs={config.epochs} patience={config.patience}"
    )
    for step in build_task2a_plan(config, paths):
        logger.write(f"[{step.number}/7] {step.name}" + (" (resumable)" if step.resumable else ""))


def _selected_examples(
    model: torch.nn.Module,
    dataset_rows: pd.DataFrame,
    array_paths: dict[str, Path],
    scores: pd.DataFrame,
    device: torch.device,
) -> list[dict[str, object]]:
    selected: list[tuple[str, str]] = []
    for label in ("CHIN", "OPCID", "CHID"):
        rows = scores.loc[scores["known_types"].fillna("").str.contains(label, regex=False)]
        if not rows.empty:
            selected.append((str(rows.iloc[0]["window_id"]), label))
    background = dataset_rows.loc[dataset_rows["background_split"].eq("val")]
    if not background.empty:
        selected.append((str(background.iloc[0]["window_id"]), "background"))
    if not scores.empty:
        selected.append((str(scores.iloc[0]["window_id"]), "top_candidate"))
    examples = []
    seen = set()
    for window_id, label in selected:
        if window_id in seen:
            continue
        seen.add(window_id)
        row = dataset_rows.loc[
            dataset_rows["window_id"].eq(window_id) & dataset_rows["replicate"].eq("rep1")
        ].head(1)
        if row.empty:
            continue
        dataset = WindowDataset(row, array_paths, input_type="log1p")
        matrix = dataset[0]["matrix"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            reconstruction = model(matrix).cpu().numpy()[0, 0]
        examples.append(
            {"name": f"{label}_{window_id}", "original": matrix.cpu().numpy()[0, 0],
             "reconstruction": reconstruction}
        )
    return examples


def _candidate_matrices(
    scores: pd.DataFrame, paths: Task2APaths, metadata: pd.DataFrame, limit: int = 50
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False)
              for name, path in _array_paths(paths).items()}
    index_by_id = (
        metadata.loc[metadata["replicate"].eq("rep1")]
        .set_index("window_id")["array_index"].astype(int).to_dict()
    )
    matrices = {}
    for window_id in scores.head(limit)["window_id"].astype(str):
        index = index_by_id[window_id]
        matrices[window_id] = (
            np.log1p(np.asarray(arrays["rep1"][index, 0])),
            np.log1p(np.asarray(arrays["rep2"][index, 0])),
        )
    return matrices


def execute_task2a(config: Task2AConfig) -> Task2AResult:
    started = time.monotonic()
    paths = Task2APaths.from_config(config)
    _validate_inputs(config, paths)
    if config.dry_run:
        with PipelineLogger(None, config.project_root) as logger:
            _log_plan(config, paths, logger)
            logger.write("DRY RUN: inputs exist; no directories, logs, or scientific artifacts were created.")
        return Task2AResult(config.mode, None, None, None, time.monotonic() - started, True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paths.log_dir / f"run_task2a_{config.mode}_{timestamp}.log"
    try:
        with PipelineLogger(log_path, config.project_root) as logger:
            _log_plan(config, paths, logger)
            paths.data_root.mkdir(parents=True, exist_ok=True)
            paths.output_root.mkdir(parents=True, exist_ok=True)
            chrom_size, scan_chrom = _geometry(config)
            known = pd.read_csv(paths.structures_path, encoding="utf-8-sig")

            logger.write("[2/7] scanning aligned replicate windows")
            if config.resume:
                metadata = _validate_scan(paths, config)
                logger.write("resume: validated and reused scan arrays")
            else:
                metadata = scan_replicates(
                    config.rep1_cool, config.rep2_cool, scan_chrom, chrom_size,
                    paths.data_root, window_bp=config.window_bp, step_bp=config.step_bp,
                    target_bin_size=config.target_bin_size, scan_end=config.scan_end,
                )
            logger.write(f"scan complete: {metadata['window_id'].nunique()} paired windows")

            logger.write("[3/7] computing streamed genome-wide expected vectors")
            if config.resume:
                expected = _validate_expected(paths, config)
                logger.write("resume: validated and reused genome expected vectors")
            else:
                expected = {
                    "rep1": compute_genome_expected(
                        config.rep1_cool, scan_chrom, target_bin_size=config.target_bin_size,
                        max_distance_bins=config.window_bp // config.target_bin_size - 1,
                    ),
                    "rep2": compute_genome_expected(
                        config.rep2_cool, scan_chrom, target_bin_size=config.target_bin_size,
                        max_distance_bins=config.window_bp // config.target_bin_size - 1,
                    ),
                }
                np.savez(paths.expected_path, **expected)

            logger.write("[4/7] building leakage-safe background split")
            if config.resume:
                background = _validate_background(paths)
                logger.write("resume: validated and reused background split")
            else:
                unique_windows = metadata.loc[metadata["replicate"].eq("rep1")].copy()
                exclusions = build_exclusion_intervals(
                    known, window_bp=config.window_bp, margin_bp=config.exclusion_margin_bp
                )
                candidates = mark_background_candidates(unique_windows, exclusions)
                background = blocked_background_split(
                    candidates, block_bp=config.block_bp, seed=config.seed
                )
                write_csv_utf8(background, paths.background_path)
            split_counts = background["background_split"].value_counts().to_dict()
            logger.write(f"background split counts: {split_counts}")

            rows = metadata.merge(
                background[["window_id", "background_split", "background_block"]],
                on="window_id", how="left", validate="many_to_one",
            )
            array_paths = _array_paths(paths)
            train_rows = rows.loc[rows["background_split"].eq("train")]
            val_rows = rows.loc[rows["background_split"].eq("val")]
            train_dataset = WindowDataset(train_rows, array_paths, input_type="log1p")
            val_dataset = WindowDataset(val_rows, array_paths, input_type="log1p")

            logger.write("[5/7] training background autoencoder (validation-only checkpoint)")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            training = train_autoencoder(
                train_dataset, val_dataset, paths.autoencoder_root,
                epochs=config.epochs, patience=config.patience, seed=config.seed,
            )
            logger.write(
                f"best checkpoint: epoch={training.best_epoch} val_loss={training.best_validation_loss:.6f}"
            )

            logger.write("[6/7] scoring density and reconstruction shape")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint_path = paths.autoencoder_root / "best_model.pth"
            model = load_autoencoder(checkpoint_path, device)
            all_dataset = WindowDataset(rows, array_paths, input_type="log1p")
            reconstruction = score_reconstruction(model, all_dataset, device=device)
            background_ids = set(
                background.loc[
                    background["background_split"].isin(["train", "val"]), "window_id"
                ].astype(str)
            )
            scores, robust_stats = score_windows(
                metadata, array_paths, expected, reconstruction, background_ids,
                density_weight=config.density_weight, shape_weight=config.shape_weight,
            )
            scores = annotate_known_overlap(scores, known)
            write_csv_utf8(scores, paths.candidate_scores_path)

            logger.write("[7/7] evaluating known recall, random baseline, and artifacts")
            recall = recall_at_fractions(scores, known)
            random = random_recall_baseline(
                scores, known, repeats=config.random_repeats, seed=config.seed
            )
            ablation = score_ablation(scores, known)
            regions = annotate_known_overlap(merge_candidate_regions(scores), known)
            write_csv_utf8(recall, paths.recall_path)
            write_csv_utf8(random, paths.random_path)
            write_csv_utf8(ablation, paths.ablation_path)
            write_csv_utf8(regions, paths.candidate_regions_path)

            rep1_values = scores["rep1_candidate_score"].to_numpy(dtype=float)
            rep2_values = scores["rep2_candidate_score"].to_numpy(dtype=float)
            pearson = float(pearsonr(rep1_values, rep2_values).statistic)
            spearman = float(spearmanr(rep1_values, rep2_values).statistic)
            consistency = pd.DataFrame([{"pearson": pearson, "spearman": spearman,
                                         "window_count": len(scores)}])
            write_csv_utf8(consistency, paths.consistency_path)

            zero_any = scores[["zero_axis_rep1", "zero_axis_rep2"]].any(axis=1)
            top_count = max(1, int(np.ceil(len(scores) * 0.10)))
            top_rate = float(zero_any.head(top_count).mean())
            all_rate = float(zero_any.mean())
            enrichment = float(top_rate / all_rate) if all_rate else 0.0
            zero_audit = pd.DataFrame(
                [
                    {"subset": "all", "window_count": len(scores), "zero_axis_rate": all_rate},
                    {"subset": "top_10_percent", "window_count": top_count,
                     "zero_axis_rate": top_rate},
                ]
            )
            write_csv_utf8(zero_audit, paths.zero_axis_path)

            figure_paths = [
                plot_training_curve(training.history, paths.figures_root / "training_curve.png"),
                plot_recall_curve(recall, random, paths.figures_root / "recall_curve.png"),
            ]
            examples = _selected_examples(model, rows, array_paths, scores, device)
            figure_paths += plot_reconstruction_examples(
                examples, paths.figures_root / "reconstruction_examples"
            )
            matrices = _candidate_matrices(scores, paths, metadata)
            figure_paths += plot_top_candidates(
                scores, matrices, paths.figures_root / "top_candidates", limit=50
            )

            output_paths = [
                paths.metadata_path, paths.rep1_array_path, paths.rep2_array_path,
                paths.expected_path, paths.background_path, checkpoint_path,
                paths.autoencoder_root / "training_history.csv", paths.candidate_scores_path,
                paths.candidate_regions_path, paths.recall_path, paths.random_path,
                paths.ablation_path, paths.consistency_path, paths.zero_axis_path,
                *figure_paths,
            ]
            counts = {
                "windows": len(scores),
                "background_train": int(split_counts.get("train", 0)),
                "background_val": int(split_counts.get("val", 0)),
                "known_structures": len(known),
                "candidate_regions": len(regions),
            }
            correlations = {"pearson": pearson, "spearman": spearman}
            zero_stats = {"all_rate": all_rate, "top_rate": top_rate, "enrichment": enrichment}
            write_summary(
                paths.summary_path, mode=config.mode, counts=counts,
                correlations=correlations, zero_axis=zero_stats,
                recall_table=recall, artifact_paths=output_paths,
            )
            output_paths.append(paths.summary_path)
            write_manifest(
                paths.manifest_path,
                configuration={
                    "mode": config.mode, "chrom": config.chrom, "window_bp": config.window_bp,
                    "step_bp": config.step_bp, "target_bin_size": config.target_bin_size,
                    "scan_end": config.scan_end, "block_bp": config.block_bp,
                    "epochs": config.epochs, "patience": config.patience, "seed": config.seed,
                    "density_weight": config.density_weight, "shape_weight": config.shape_weight,
                    "random_repeats": config.random_repeats,
                },
                input_paths=[config.rep1_cool, config.rep2_cool, paths.structures_path],
                output_paths=output_paths,
                statistics={**counts, "robust_scales": robust_stats,
                            "best_epoch": training.best_epoch,
                            "best_validation_loss": training.best_validation_loss,
                            "zero_axis": zero_stats},
            )
            validate_task2a_outputs(paths.manifest_path)
            logger.write(f"Task 2A complete: {paths.summary_path}")
    except PipelineValidationError:
        raise
    except Exception as exc:
        raise PipelineStepError(f"Task 2A failed: {exc}. See log: {log_path}") from exc
    return Task2AResult(
        config.mode, paths.summary_path, paths.manifest_path, log_path,
        time.monotonic() - started,
    )

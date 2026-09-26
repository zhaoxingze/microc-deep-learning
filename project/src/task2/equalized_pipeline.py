"""Task2A.4 post-processing pipeline for equalized cross-scale calibration."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.data.known_dataset import attach_structure_ids
from src.task1.manifest import sha256_file
from src.task2.calibrated_evaluation import BUDGETS, coverage_curve, random_baselines, zero_axis_audit
from src.task2.equalized import (
    assess_equalized_readiness,
    build_equalized_regions,
    common_p_floor,
    equalize_window_scores,
    rank_equalized_windows,
    scale_contribution,
)
from src.task2.reporting import validate_task2a_outputs, write_csv_utf8, write_manifest
from src.task2.region_detection import select_coverage_budget


EQUALIZED_OUTPUT_NAMES = {
    "equalized_window_scores.csv",
    "equalized_candidate_regions.csv",
    "region_membership.csv",
    "scale_contribution.csv",
    "coverage_recall.csv",
    "coverage_random_baseline.csv",
    "region_random_baseline.csv",
    "random_baseline_draws.csv",
    "replicate_consistency.csv",
    "zero_axis_audit.csv",
    "readiness.json",
    "task2a_equalized_summary.md",
    "protected_artifacts_audit.json",
    "task2a_equalized_manifest.json",
}


@dataclass(frozen=True)
class EqualizedPaths:
    project_root: Path

    @property
    def prior_root(self) -> Path:
        return self.project_root / "outputs" / "task2a_multiscale"

    @property
    def protected_data_root(self) -> Path:
        return self.project_root / "data" / "task2_multiscale"

    @property
    def output_root(self) -> Path:
        return self.project_root / "outputs" / "task2a_multiscale_equalized"

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "task2a_equalized_manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.output_root / "task2a_equalized_summary.md"


@dataclass(frozen=True)
class EqualizedResult:
    output_root: Path
    summary_path: Path
    manifest_path: Path
    readiness_path: Path
    elapsed_seconds: float


def resolve_equalized_inputs(paths: EqualizedPaths) -> dict[str, Path]:
    root = paths.prior_root
    return {
        "prior_manifest": root / "task2a_multiscale_manifest.json",
        "prior_protocol": root / "frozen_multiscale_protocol.json",
        "scale_summary": root / "scale_summary.csv",
        "prior_regions": root / "multiscale_candidate_regions.csv",
        "leakage_audit": root / "background_split_leakage_audit.csv",
        "known_overlap_audit": root / "background_known_interval_overlap_audit.csv",
        "small_scores": root / "small" / "window_scores.csv",
        "medium_scores": root / "medium" / "window_scores.csv",
        "large_scores": root / "large" / "window_scores.csv",
        "structures": paths.project_root / "data" / "processed" / "structures.csv",
    }


def protected_tree_hashes(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root)
    files: list[Path] = []
    for protected in (
        root / "outputs" / "task2a_multiscale",
        root / "data" / "task2_multiscale",
    ):
        if protected.is_dir():
            files.extend(path for path in protected.rglob("*") if path.is_file())
    return {str(path.resolve()): sha256_file(path) for path in sorted(files)}


def assert_protected_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    if before != after:
        added = sorted(set(after).difference(before))
        removed = sorted(set(before).difference(after))
        changed = sorted(path for path in set(before).intersection(after) if before[path] != after[path])
        raise ValueError(
            "protected Task2A.3 artifacts changed: "
            f"added={added}, removed={removed}, changed={changed}"
        )


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _require_inputs(inputs: dict[str, Path]) -> None:
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Task2A.4 inputs missing: {missing}")
    forbidden = {".cool", ".npy", ".pt", ".pth", ".ckpt"}
    if any(path.suffix.lower() in forbidden for path in inputs.values()):
        raise ValueError("equalized pipeline input contract includes a prohibited raw/model artifact")


def validate_prior_direct_inputs(manifest_path: Path, direct_paths: list[Path]) -> dict:
    """Verify only files consumed here, without touching transitive raw/model inputs."""
    payload = _load_json(manifest_path)
    records: dict[str, dict] = {}
    for section in ("inputs", "outputs"):
        section_records = payload.get(section)
        if not isinstance(section_records, dict):
            raise ValueError(f"prior manifest {section} section is invalid")
        for raw_path, record in section_records.items():
            records[str(Path(raw_path).resolve()).casefold()] = record
    for path in direct_paths:
        resolved = path.resolve()
        record = records.get(str(resolved).casefold())
        if not isinstance(record, dict) or "sha256" not in record:
            raise ValueError(f"direct input is not covered by prior manifest: {resolved}")
        if sha256_file(resolved) != record["sha256"]:
            raise ValueError(f"prior manifest hash mismatch for direct input: {resolved}")
    return payload


def _reject_unexpected_outputs(output_root: Path) -> None:
    if not output_root.exists():
        return
    unexpected = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and str(path.relative_to(output_root)).replace("\\", "/") not in EQUALIZED_OUTPUT_NAMES
    )
    if unexpected:
        raise ValueError(f"equalized output directory contains unmanifested stale files: {unexpected}")


def _calibration_counts(scale_summary: pd.DataFrame) -> dict[str, int]:
    required = {"scale", "calibration_background"}
    if not required.issubset(scale_summary.columns) or scale_summary.scale.duplicated().any():
        raise ValueError("invalid Task2A.3 scale summary")
    counts = {
        str(row.scale): int(row.calibration_background)
        for row in scale_summary.itertuples(index=False)
    }
    floor = common_p_floor(counts)
    if min(counts.values()) != 147 or not np.isclose(floor, 1 / 148, rtol=0, atol=1e-15):
        raise ValueError(f"frozen Task2A.4 calibration basis changed: {counts}")
    return counts


def _load_known(path: Path) -> pd.DataFrame:
    known = pd.read_csv(path, encoding="utf-8-sig")
    if "structure_id" not in known.columns:
        known = attach_structure_ids(known)
    required = {"structure_id", "type", "chrom", "start", "end", "center"}
    if not required.issubset(known.columns) or known.structure_id.duplicated().any():
        raise ValueError("known structure table is invalid")
    return known


def _audit_zero(path: Path, label: str) -> bool:
    table = pd.read_csv(path, encoding="utf-8-sig")
    if "overlap_count" not in table.columns or table.empty:
        raise ValueError(f"{label} audit is invalid")
    values = pd.to_numeric(table.overlap_count, errors="coerce")
    if values.isna().any() or (values < 0).any():
        raise ValueError(f"{label} audit contains invalid counts")
    return bool(values.sum() == 0)


def frozen_evaluation_domain(protocol: dict, known: pd.DataFrame) -> tuple[str, int]:
    """Resolve the frozen linear domain without assuming a protocol object shape."""
    domain = int(protocol.get("evaluation_domain_size", 0))
    chromosomes = sorted(set(known.chrom.astype(str))) if "chrom" in known else []
    if domain <= 0 or len(chromosomes) != 1:
        raise ValueError("invalid frozen evaluation domain")
    marker = protocol.get("random_domain")
    if marker not in (None, "linear") and not isinstance(marker, dict):
        raise ValueError("unsupported frozen random-domain marker")
    if isinstance(marker, dict):
        declared = marker.get("chromosome")
        if declared is not None and str(declared) != chromosomes[0]:
            raise ValueError("frozen random-domain chromosome mismatch")
    return chromosomes[0], domain


def _write_summary(
    path: Path,
    *,
    counts: dict[str, int],
    floor: float,
    ceiling: float,
    contribution: pd.DataFrame,
    curve: pd.DataFrame,
    coverage_random: pd.DataFrame,
    region_random: pd.DataFrame,
    replicate: dict[str, object],
    zero: dict[str, object],
    readiness: dict[str, object],
) -> Path:
    at = lambda frame, budget: frame.loc[np.isclose(frame.budget, budget)].iloc[0]
    c10, c20 = at(curve, .10), at(curve, .20)
    r10, r20 = at(coverage_random, .10), at(coverage_random, .20)
    g10, g20 = at(region_random, .10), at(region_random, .20)
    ready = bool(readiness["ready_for_task2b"])
    if ready:
        conclusion = ["TASK2A FINALIZED", "", "READY FOR TASK2B"]
    else:
        conclusion = [
            "TASK2A NOT READY FOR TASK2B",
            "",
            "在修复 cross-scale calibration-size bias 后，pure anomaly detector 仍不能稳定富集 known structures。",
            "下一阶段应改变 proposal-generation paradigm；不再增加 scale、替换 AE 或调整 threshold。",
        ]
    lines = [
        "# Task2A.4 cross-scale calibration equalization summary", "",
        "> 本轮只复用现有 empirical p-values 做共同下限校准、排序、NMS 和冻结评估；未扫描 Micro-C、未计算 E(d)、未训练或加载 AE、未执行 Task2B。", "",
        "## Frozen calibration", "",
        f"- Calibration counts: `{counts}`",
        f"- common_p_floor: `{floor:.15f}` (`1/148`)",
        f"- equalized maximum anomaly: `{ceiling:.15f}`", "",
        "## Scale contribution before and after", "",
        contribution.to_markdown(index=False), "",
        "## Coverage results", "",
        f"- 10% overall recall: `{float(c10.overall_recall):.9f}`",
        f"- 10% coverage-random 97.5%: `{float(r10.overall_ci_high):.9f}`",
        f"- 10% region-random 97.5%: `{float(g10.overall_ci_high):.9f}`",
        f"- 20% overall recall: `{float(c20.overall_recall):.9f}`",
        f"- 20% coverage-random 97.5%: `{float(r20.overall_ci_high):.9f}`",
        f"- 20% region-random 97.5%: `{float(g20.overall_ci_high):.9f}`",
        f"- 20% CHIN recall: `{float(c20.CHIN_recall):.9f}`",
        f"- 20% OPCID recall: `{float(c20.OPCID_recall):.9f}`",
        f"- 20% CHID recall: `{float(c20.CHID_recall):.9f}`", "",
        "## Stability and data quality", "",
        f"- Rep1/Rep2 Spearman: `{float(replicate['spearman']):.9f}`",
        f"- Zero-axis enrichment: `{float(zero['enrichment']):.9f}`", "",
        "## Frozen readiness gate", "",
        pd.DataFrame([
            {"check": key, "passed": value} for key, value in readiness["checks"].items()
        ]).to_markdown(index=False), "",
        f"- ready_for_task2b: `{str(ready).lower()}`",
        "- task2b_executed: `false`", "",
        *conclusion, "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def execute_equalized(
    project_root: str | Path,
    *,
    random_repeats: int = 100,
    seed: int = 20_260_920,
) -> EqualizedResult:
    """Execute Task2A.4 only; no scan, expected calculation, AE, or Task2B."""
    started = time.monotonic()
    if random_repeats != 100 or seed != 20_260_920:
        raise ValueError("Task2A.4 random protocol is frozen at repeats=100 and seed=20260920")
    paths = EqualizedPaths(Path(project_root).resolve())
    inputs = resolve_equalized_inputs(paths)
    _require_inputs(inputs)
    _reject_unexpected_outputs(paths.output_root)
    before = protected_tree_hashes(paths.project_root)
    if not before:
        raise ValueError("protected Task2A.3 trees are empty")
    validate_prior_direct_inputs(
        inputs["prior_manifest"],
        [path for name, path in inputs.items() if name != "prior_manifest"],
    )
    protocol = _load_json(inputs["prior_protocol"])
    scale_summary = pd.read_csv(inputs["scale_summary"], encoding="utf-8-sig")
    counts = _calibration_counts(scale_summary)
    floor = common_p_floor(counts)
    ceiling = float(-np.log10(floor))

    score_tables = []
    for scale in ("small", "medium", "large"):
        table = pd.read_csv(inputs[f"{scale}_scores"], encoding="utf-8-sig")
        if not table.scale.astype(str).eq(scale).all():
            raise ValueError(f"{scale} score file contains another scale")
        score_tables.append(table)
    original_pool = pd.concat(score_tables, ignore_index=True)
    equalized = equalize_window_scores(original_pool, floor)
    ranked = rank_equalized_windows(equalized)
    observed_common = ranked[
        [f"{rep}_{branch}_common_anomaly" for rep in ("rep1", "rep2") for branch in ("density", "shape")]
    ].to_numpy(dtype=float)
    if observed_common.max() > ceiling + 1e-12:
        raise ValueError("equalized anomaly exceeds frozen common ceiling")
    regions, membership = build_equalized_regions(ranked)

    prior_regions = pd.read_csv(inputs["prior_regions"], encoding="utf-8-sig")
    contribution = scale_contribution(prior_regions, regions)
    known = _load_known(inputs["structures"])
    chrom, domain = frozen_evaluation_domain(protocol, known)
    curve = coverage_curve(regions, known, domain, budgets=BUDGETS)
    coverage_random, region_random, random_draws = random_baselines(
        ranked, regions, known, {chrom: domain},
        budgets=BUDGETS, repeats=random_repeats, seed=seed,
    )
    if len(regions) < 2:
        raise ValueError("at least two regions are required for replicate consistency")
    replicate = {
        "pearson": float(pearsonr(regions.rep1_peak_score, regions.rep2_peak_score).statistic),
        "spearman": float(spearmanr(regions.rep1_peak_score, regions.rep2_peak_score).statistic),
        "unit": "equalized multiscale peak windows",
        "n": len(regions),
    }
    selected20, _ = select_coverage_budget(regions, .20, domain)
    zero = zero_axis_audit(ranked, selected20)
    no_leakage = _audit_zero(inputs["leakage_audit"], "split leakage")
    known_overlap_zero = _audit_zero(inputs["known_overlap_audit"], "background-known overlap")
    mode = str(protocol.get("mode", "full"))
    readiness = assess_equalized_readiness(
        curve, coverage_random,
        zero_enrichment=float(zero["enrichment"]),
        replicate_spearman=float(replicate["spearman"]),
        no_leakage=no_leakage,
        known_overlap_zero=known_overlap_zero,
        mode=mode,
    )

    paths.output_root.mkdir(parents=True, exist_ok=True)
    outputs = [
        write_csv_utf8(ranked, paths.output_root / "equalized_window_scores.csv"),
        write_csv_utf8(regions, paths.output_root / "equalized_candidate_regions.csv"),
        write_csv_utf8(membership, paths.output_root / "region_membership.csv"),
        write_csv_utf8(contribution, paths.output_root / "scale_contribution.csv"),
        write_csv_utf8(curve, paths.output_root / "coverage_recall.csv"),
        write_csv_utf8(coverage_random, paths.output_root / "coverage_random_baseline.csv"),
        write_csv_utf8(region_random, paths.output_root / "region_random_baseline.csv"),
        write_csv_utf8(random_draws, paths.output_root / "random_baseline_draws.csv"),
        write_csv_utf8(pd.DataFrame([replicate]), paths.output_root / "replicate_consistency.csv"),
        write_csv_utf8(pd.DataFrame([zero]), paths.output_root / "zero_axis_audit.csv"),
        _write_json(paths.output_root / "readiness.json", readiness),
    ]
    outputs.append(_write_summary(
        paths.summary_path,
        counts=counts, floor=floor, ceiling=ceiling,
        contribution=contribution, curve=curve,
        coverage_random=coverage_random, region_random=region_random,
        replicate=replicate, zero=zero, readiness=readiness,
    ))
    after = protected_tree_hashes(paths.project_root)
    assert_protected_unchanged(before, after)
    outputs.append(_write_json(paths.output_root / "protected_artifacts_audit.json", {
        "unchanged": True,
        "file_count": len(before),
        "before": before,
        "after": after,
    }))
    configuration = {
        "version": "Task2A.4-cross-scale-equalized",
        "source_run": "Task2A.3-MultiScale-Full",
        "mode": mode,
        "calibration_counts": counts,
        "min_calibration_n": min(counts.values()),
        "common_p_floor": floor,
        "equalized_max_anomaly": ceiling,
        "paired_branch_aggregation": "mean of available rep1 and rep2 common anomalies",
        "within_scale_score": "max(paired density common anomaly, paired shape common anomaly)",
        "cross_scale_score": "max across scale candidates via unchanged NMS",
        "tie_break": [
            "replicate_min_score descending",
            "original_uncapped_tail_score descending",
            "chrom/start/end/window_id ascending",
        ],
        "known_labels_used_for_ranking": False,
        "threshold_searched": False,
        "window_scanning_executed": False,
        "genome_expected_recomputed": False,
        "autoencoder_retrained": False,
        "autoencoder_checkpoint_loaded": False,
        "existing_empirical_p_values_reused": True,
        "new_window_scale_added": False,
        "known_recall_definition_changed": False,
        "random_repetitions": random_repeats,
        "random_seed": seed,
        "task2b_executed": False,
    }
    statistics = {
        "n_windows": len(ranked),
        "n_candidate_regions": len(regions),
        "observed_max_common_anomaly": float(observed_common.max()),
        "scale_contribution": contribution.to_dict("records"),
        "replicate_consistency": replicate,
        "zero_axis": zero,
        "readiness": readiness,
    }
    direct_inputs = list(inputs.values()) + [Path(__file__), Path(__file__).with_name("equalized.py")]
    write_manifest(
        paths.manifest_path,
        configuration=configuration,
        input_paths=direct_inputs,
        output_paths=outputs,
        statistics=statistics,
    )
    validate_task2a_outputs(paths.manifest_path)
    assert_protected_unchanged(before, protected_tree_hashes(paths.project_root))
    return EqualizedResult(
        output_root=paths.output_root,
        summary_path=paths.summary_path,
        manifest_path=paths.manifest_path,
        readiness_path=paths.output_root / "readiness.json",
        elapsed_seconds=time.monotonic() - started,
    )

"""Finalize Task2A as a frozen, label-independent proposal generator."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.known_dataset import attach_structure_ids
from src.task1.manifest import sha256_file
from src.task2.calibrated_evaluation import recall_metrics
from src.task2.equalized_pipeline import (
    assert_protected_unchanged, protected_tree_hashes, validate_prior_direct_inputs,
)
from src.task2.final_proposals import (
    CHANNELS, PROPOSAL_P_THRESHOLD, SCALE_GEOMETRY, final_engineering_gate,
    generate_channel_proposals, merge_proposals,
)
from src.task2.refined_evaluation import union_coverage_bp
from src.task2.reporting import validate_task2a_outputs, write_csv_utf8, write_manifest


@dataclass(frozen=True)
class FinalResult:
    output_root: Path
    candidate_path: Path
    protocol_path: Path
    finalization_path: Path
    manifest_path: Path
    summary_path: Path


def annotate_candidates(regions: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Append known labels after all proposal selection and geometry are fixed."""
    out = regions.copy()
    annotations = []
    for region in out.itertuples(index=False):
        same = known.loc[known.chrom.astype(str).eq(str(region.chrom))]
        center_hit = same.loc[same.center.ge(region.start) & same.center.lt(region.end)]
        interval_hit = same.loc[same.start.lt(region.end) & same.end.gt(region.start)]
        annotations.append({
            "known_center_overlap": not center_hit.empty,
            "known_interval_overlap": not interval_hit.empty,
            "known_types": ";".join(sorted(interval_hit.type.astype(str).unique())),
            "known_structure_ids": ";".join(sorted(interval_hit.structure_id.astype(str).unique())),
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(annotations)], axis=1)


def final_recall_table(candidates: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    values = recall_metrics(candidates, known)
    rows = []
    for label in ("overall", "CHIN", "OPCID", "CHID"):
        rows.append({
            "type": label,
            "n_known": len(known) if label == "overall" else int(known.type.eq(label).sum()),
            "center_recall": values[f"{label}_recall"],
            "interval_overlap_recall": values[f"{label}_overlap_recall"],
        })
    return pd.DataFrame(rows)


def calibration_tail_audit(scores: pd.DataFrame) -> pd.DataFrame:
    """Audit the fixed paired p-equivalent threshold on prior calibration rows."""
    required = {"scale", "background_split"} | {
        f"{rep}_{branch}_p" for rep in ("rep1", "rep2") for branch in ("density", "shape")
    }
    if missing := required.difference(scores.columns):
        raise ValueError(f"calibration audit columns missing: {sorted(missing)}")
    rows = []
    for scale, branch in CHANNELS:
        calibration = scores.loc[scores.scale.eq(scale) & scores.background_split.eq("calibration")]
        if calibration.empty:
            raise ValueError(f"missing {scale} calibration windows")
        p1 = calibration[f"rep1_{branch}_p"].to_numpy(dtype=float)
        p2 = calibration[f"rep2_{branch}_p"].to_numpy(dtype=float)
        if not np.isfinite(p1).all() or not np.isfinite(p2).all() or (
            (p1 <= 0) | (p1 > 1) | (p2 <= 0) | (p2 > 1)
        ).any():
            raise ValueError(f"invalid {scale}_{branch} calibration p-values")
        n_tail = int((np.sqrt(p1 * p2) <= PROPOSAL_P_THRESHOLD).sum())
        rows.append({"channel": f"{scale}_{branch}",
                     "n_calibration_windows": len(calibration),
                     "n_below_fixed_threshold": n_tail,
                     "observed_tail_fraction": n_tail / len(calibration),
                     "fixed_threshold": PROPOSAL_P_THRESHOLD})
    return pd.DataFrame(rows)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_audit_zero(path: Path) -> bool:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty or "overlap_count" not in frame:
        raise ValueError(f"incomplete Task2A audit: {path}")
    counts = pd.to_numeric(frame.overlap_count, errors="coerce")
    if counts.isna().any() or (counts < 0).any():
        raise ValueError(f"invalid Task2A audit: {path}")
    return bool(counts.sum() == 0)


def _protocol(project_root: Path, scores: dict[str, Path], prior_manifest: Path) -> dict:
    return {
        "version": "Task2A-Final-independent-six-channel-v1",
        "purpose": "high-recall proposal generation; no novelty determination",
        "scales": [{"scale": s, "window_bp": bp, "step_bp": step, "target_bin_size": 100}
                   for s, (bp, step) in SCALE_GEOMETRY.items()],
        "channels": [f"{scale}_{branch}" for scale, branch in CHANNELS],
        "channel_p": "sqrt(rep1_empirical_p * rep2_empirical_p); equivalent to mean replicate -log10(p)",
        "proposal_threshold": PROPOSAL_P_THRESHOLD,
        "channel_nms": "per channel; ascending p; center distance strictly < window_bp/2 suppressed",
        "cross_channel_merge": "transitive connected components of positive interval overlap",
        "representative_tie_break": ["lowest empirical p", "highest replicate-min anomaly",
                                     "chrom/start/end/window_id/channel ascending"],
        "known_labels_used_for_selection": False,
        "known_recall_used_as_performance_gate": False,
        "source_task2a_manifest_sha256": sha256_file(prior_manifest),
        "source_score_sha256": {name: sha256_file(path) for name, path in scores.items()},
        "proposal_code_sha256": sha256_file(Path(__file__).with_name("final_proposals.py")),
        "seed": 20_260_920,
    }


def _write_frozen_protocol(path: Path, payload: dict) -> None:
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != payload:
            raise ValueError("frozen Task2A proposal protocol differs from current inputs/code")
    else:
        _write_json(path, payload)


def _summary(path: Path, protocol: dict, counts: dict[str, int], proposals: pd.DataFrame,
             candidates: pd.DataFrame, recall: pd.DataFrame, coverage_bp: int,
             domain_bp: int, finalization: dict, calibration_audit: pd.DataFrame) -> Path:
    lines = [
        "# Task2A Final: frozen high-recall proposal generator", "",
        "Task2A now proposes regions for representation and grouping; it does not determine novelty. The fixed threshold and rules were frozen before final known recall.", "",
        "## Frozen protocol and output", "",
        f"- Six independent scale × branch channels: `{protocol['channels']}`",
        f"- Fixed paired p-equivalent score threshold: `<= {PROPOSAL_P_THRESHOLD}`",
        "Each replicate p-value is empirical on background calibration data. Their geometric mean is a fixed paired ranking score, **not** a separately calibrated paired empirical p-value; the threshold must not be interpreted as a formal paired 5% test.",
        "Observed fraction of calibration-background windows below this fixed score threshold (before NMS):", "",
        calibration_audit.to_markdown(index=False), "",
        f"- Channel peaks: `{counts}`",
        f"- Proposals before cross-channel merge: `{len(proposals)}`",
        f"- Candidate regions after transitive overlap merge: `{len(candidates)}`",
        f"- Candidate genomic union coverage: `{coverage_bp}` bp / `{domain_bp}` bp = `{coverage_bp/domain_bp:.9f}`",
        f"- Regions with a known center: `{int(candidates.known_center_overlap.sum())}`",
        f"- Regions with a known interval overlap: `{int(candidates.known_interval_overlap.sum())}`",
        "These overlap fractions describe the candidate pool and are not calibrated precision, because unlabeled candidates cannot be declared false positives.", "",
        "## Final known-structure recall (center-based primary)", "",
        recall.to_markdown(index=False), "",
        "## Historical experiments and limitation", "",
        "Single-scale anomaly, mean fusion, OR fusion, calibrated OR, multi-scale winner-takes-all, and equalized cross-scale ranking were tested. Global winner-takes-all exposed branch competition, scale multiplicity bias, NMS order dependence, unequal calibration resolution, and candidate-selection bias.",
        "The same 344 known structures have been reused during method development. Further tuning to raise their recall would weaken their value as validation. This final recall is an experimental outcome and limitation, not a threshold to optimize or a Task2B performance gate.", "",
        "## Engineering readiness", "",
        pd.DataFrame([{"check": k, "passed": v} for k, v in finalization["checks"].items()]).to_markdown(index=False), "",
        f"- ready_for_task2b: `{str(finalization['ready_for_task2b']).lower()}`", "",
        "No Task2C clustering or novelty claim is made here.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def execute_task2a_final(project_root: str | Path) -> FinalResult:
    root = Path(project_root).resolve()
    prior = root / "outputs" / "task2a_multiscale"
    out = root / "outputs" / "task2a_final"
    scores = {name: prior / name / "window_scores.csv" for name in SCALE_GEOMETRY}
    prior_manifest = prior / "task2a_multiscale_manifest.json"
    structures_path = root / "data" / "processed" / "structures.csv"
    leakage_path = prior / "background_split_leakage_audit.csv"
    overlap_path = prior / "background_known_interval_overlap_audit.csv"
    direct = [*scores.values(), structures_path, leakage_path, overlap_path]
    if not prior_manifest.is_file() or any(not path.is_file() for path in direct):
        raise FileNotFoundError("Task2A.3 score, structure, or audit input is missing")
    before = protected_tree_hashes(root)
    validate_prior_direct_inputs(prior_manifest, direct)
    frames = []
    for scale, path in scores.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not frame.scale.astype(str).eq(scale).all():
            raise ValueError(f"invalid {scale} score scale")
        frames.append(frame)
    all_scores = pd.concat(frames, ignore_index=True)
    proposals = generate_channel_proposals(all_scores)
    calibration_audit = calibration_tail_audit(all_scores)
    candidates = merge_proposals(proposals)
    payload = _protocol(root, scores, prior_manifest)
    protocol_path = out / "frozen_proposal_protocol.json"
    _write_frozen_protocol(protocol_path, payload)  # Before reading known labels or computing recall.

    known = pd.read_csv(structures_path, encoding="utf-8-sig")
    known = known if "structure_id" in known else attach_structure_ids(known)
    if known.structure_id.duplicated().any() or len(known) != 344:
        raise ValueError("frozen reference known set must have 344 unique structures")
    candidates = annotate_candidates(candidates, known)
    recall = final_recall_table(candidates, known)
    protocol_a3 = json.loads((prior / "frozen_multiscale_protocol.json").read_text(encoding="utf-8"))
    domain_bp = int(protocol_a3["evaluation_domain_size"])
    coverage_bp = int(union_coverage_bp(candidates))
    raw_root = root.parent / "micro-c数据"
    raw_paths = [raw_root / f"GSE272159_37C_{rep}.mapq_30.10.cool" for rep in ("rep1", "rep2")]
    no_leakage = _read_audit_zero(leakage_path)
    background_clean = _read_audit_zero(overlap_path)
    finalization = final_engineering_gate(
        candidates, protocol_frozen=protocol_path.is_file(),
        artifacts_complete=bool(background_clean and before),
        replicates_accessible=all(path.is_file() for path in raw_paths),
        no_leakage=no_leakage, known_label_tuning=False,
    )
    finalization["background_known_overlap_zero"] = background_clean
    finalization["task2a_known_recall_is_performance_gate"] = False
    finalization["task2b_executed"] = False
    out.mkdir(parents=True, exist_ok=True)
    counts = {f"{scale}_{branch}": int(((proposals.scale == scale) & (proposals.branch == branch)).sum())
              for scale, branch in CHANNELS}
    output_paths = [
        protocol_path,
        write_csv_utf8(proposals, out / "channel_proposals.csv"),
        write_csv_utf8(candidates, out / "task2a_final_candidates.csv"),
        write_csv_utf8(recall, out / "task2a_final_recall.csv"),
        write_csv_utf8(calibration_audit, out / "calibration_tail_audit.csv"),
        _write_json(out / "task2a_finalization.json", finalization),
        _summary(out / "task2a_final_summary.md", payload, counts, proposals,
                 candidates, recall, coverage_bp, domain_bp, finalization, calibration_audit),
    ]
    assert_protected_unchanged(before, protected_tree_hashes(root))
    manifest_path = out / "task2a_final_manifest.json"
    write_manifest(
        manifest_path,
        configuration=payload,
        input_paths=[prior_manifest, prior / "frozen_multiscale_protocol.json", *direct,
                     Path(__file__), Path(__file__).with_name("final_proposals.py")],
        output_paths=output_paths,
        statistics={"channel_peak_counts": counts, "n_proposals": len(proposals),
                    "n_candidates": len(candidates), "coverage_bp": coverage_bp,
                    "coverage_fraction": coverage_bp / domain_bp,
                    "calibration_tail_audit": calibration_audit.to_dict("records"),
                    "recall": recall.to_dict("records"), "finalization": finalization},
    )
    validate_task2a_outputs(manifest_path)
    if not finalization["ready_for_task2b"]:
        raise ValueError("Task2A Final engineering readiness failed; see task2a_finalization.json")
    return FinalResult(out, out / "task2a_final_candidates.csv", protocol_path,
                       out / "task2a_finalization.json", manifest_path,
                       out / "task2a_final_summary.md")

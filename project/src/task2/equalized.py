"""Frozen Task2A.4 cross-scale calibration equalization and ranking."""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.task2.refined_evaluation import _coverage_index
from src.task2.region_detection import dominant_branch


SCALES = ("small", "medium", "large")
BRANCHES = ("density", "shape")
REPLICATES = ("rep1", "rep2")


def common_p_floor(calibration_counts: Mapping[str, int]) -> float:
    """Return the pre-registered common empirical-p floor."""
    if set(calibration_counts) != set(SCALES):
        raise ValueError("calibration counts must contain exactly small, medium, and large")
    counts = [int(calibration_counts[name]) for name in SCALES]
    if any(value <= 0 for value in counts):
        raise ValueError("calibration counts must be positive")
    return 1.0 / (min(counts) + 1)


def _validate_score_pool(pool: pd.DataFrame, floor: float) -> None:
    required = {
        "window_id", "chrom", "start", "end", "center", "scale", "window_bp",
        "scale_or_score", "zero_axis_rep1", "zero_axis_rep2",
        *(f"{rep}_{branch}_p" for rep in REPLICATES for branch in BRANCHES),
    }
    if not 0 < float(floor) <= 1:
        raise ValueError("common p floor must be in (0, 1]")
    missing = required.difference(pool.columns)
    if missing:
        raise ValueError(f"score table is missing required columns: {sorted(missing)}")
    if pool.empty or pool.window_id.astype(str).duplicated().any():
        raise ValueError("score table must contain unique windows")
    if not set(pool.scale.astype(str)).issubset(SCALES):
        raise ValueError("unknown scale in score table")
    numeric = ["start", "end", "center", "window_bp", "scale_or_score"] + [
        f"{rep}_{branch}_p" for rep in REPLICATES for branch in BRANCHES
    ]
    values = pool[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("score table contains non-finite numeric values")
    if (values.end <= values.start).any() or (values.window_bp <= 0).any():
        raise ValueError("score table contains invalid window geometry")
    p_columns = [f"{rep}_{branch}_p" for rep in REPLICATES for branch in BRANCHES]
    p_values = values[p_columns].to_numpy(dtype=float)
    if ((p_values <= 0) | (p_values > 1)).any():
        raise ValueError("empirical p-values must be in (0, 1]")


def equalize_window_scores(pool: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Apply the common p floor and recompute the frozen paired OR score."""
    _validate_score_pool(pool, floor)
    out = pool.copy()
    for rep in REPLICATES:
        common_anomalies = []
        for branch in BRANCHES:
            p_column = f"{rep}_{branch}_p"
            common_p_column = f"{rep}_{branch}_p_common"
            anomaly_column = f"{rep}_{branch}_common_anomaly"
            out[common_p_column] = np.maximum(out[p_column].to_numpy(dtype=float), float(floor))
            out[anomaly_column] = -np.log10(out[common_p_column].to_numpy(dtype=float))
            common_anomalies.append(out[anomaly_column].to_numpy(dtype=float))
        out[f"{rep}_common_or_score"] = np.maximum(*common_anomalies)
    for branch in BRANCHES:
        out[f"paired_{branch}_common_anomaly"] = (
            out[f"rep1_{branch}_common_anomaly"].to_numpy(dtype=float)
            + out[f"rep2_{branch}_common_anomaly"].to_numpy(dtype=float)
        ) / 2.0
    out["equalized_scale_score"] = np.maximum(
        out["paired_density_common_anomaly"].to_numpy(dtype=float),
        out["paired_shape_common_anomaly"].to_numpy(dtype=float),
    )
    out["replicate_min_score"] = np.minimum(
        out["rep1_common_or_score"].to_numpy(dtype=float),
        out["rep2_common_or_score"].to_numpy(dtype=float),
    )
    out["original_uncapped_tail_score"] = out["scale_or_score"].to_numpy(dtype=float)
    out["common_p_floor"] = float(floor)
    out["equalized_max_anomaly"] = -math.log10(float(floor))
    return out


def rank_equalized_windows(pool: pd.DataFrame) -> pd.DataFrame:
    """Rank with the fixed label-free Task2A.4 tie-break hierarchy."""
    required = {
        "equalized_scale_score", "replicate_min_score", "original_uncapped_tail_score",
        "chrom", "start", "end", "window_id",
    }
    if missing := required.difference(pool.columns):
        raise ValueError(f"equalized ranking columns missing: {sorted(missing)}")
    out = pool.copy()
    out = out.sort_values(
        ["equalized_scale_score", "replicate_min_score", "original_uncapped_tail_score",
         "chrom", "start", "end", "window_id"],
        ascending=[False, False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["equalized_rank"] = np.arange(1, len(out) + 1)
    out["equalized_percentile"] = 1.0 - (out["equalized_rank"] - 1) / len(out)
    return out


def build_equalized_regions(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the unchanged cross-scale NMS using the equalized deterministic rank."""
    if pool.empty:
        raise ValueError("empty equalized score pool")
    ranked = rank_equalized_windows(pool)
    selected = []
    centers: list[tuple[int, int]] = []
    owners: list[int] = []
    for row in ranked.itertuples(index=False):
        left = bisect_left(centers, (int(row.center) - 6_400, -1))
        right = bisect_right(centers, (int(row.center) + 6_400, 10**12))
        candidates = []
        for _, index in centers[left:right]:
            peak = selected[index]
            if (
                peak.chrom == row.chrom
                and abs(int(row.center) - int(peak.center))
                < max(int(row.window_bp), int(peak.window_bp)) / 2
            ):
                candidates.append(index)
        if candidates:
            owner = min(candidates)
        else:
            owner = len(selected)
            selected.append(row)
            insort(centers, (int(row.center), owner))
        owners.append(owner)
    ranked["owner"] = owners
    regions = []
    members = []
    for index, peak in enumerate(selected):
        support = ranked.loc[ranked.owner.eq(index)]
        region_id = "EQREG_" + str(peak.window_id).removeprefix("WIN_")
        counts = support.scale.value_counts()
        regions.append({
            "region_id": region_id,
            "chrom": peak.chrom,
            "start": int(peak.start),
            "end": int(peak.end),
            "center": float(peak.center),
            "peak_scale": peak.scale,
            "peak_window_id": peak.window_id,
            "peak_score": float(peak.equalized_scale_score),
            "peak_window_bp": int(peak.window_bp),
            "density_anomaly": float(peak.paired_density_common_anomaly),
            "shape_anomaly": float(peak.paired_shape_common_anomaly),
            "dominant_branch": dominant_branch(
                peak.paired_density_common_anomaly, peak.paired_shape_common_anomaly
            ),
            "rep1_peak_score": float(peak.rep1_common_or_score),
            "rep2_peak_score": float(peak.rep2_common_or_score),
            "replicate_min_score": float(peak.replicate_min_score),
            "original_uncapped_tail_score": float(peak.original_uncapped_tail_score),
            "support_small_count": int(counts.get("small", 0)),
            "support_medium_count": int(counts.get("medium", 0)),
            "support_large_count": int(counts.get("large", 0)),
            "support_union_start": int(support.start.min()),
            "support_union_end": int(support.end.max()),
            "member_window_count": len(support),
            "union_length_bp": int(peak.end - peak.start),
        })
        for row in support.itertuples(index=False):
            members.append({
                "region_id": region_id,
                "peak_window_id": peak.window_id,
                "window_id": row.window_id,
                "scale": row.scale,
                "window_score": float(row.equalized_scale_score),
            })
    frame = pd.DataFrame(regions)
    coverage_index = _coverage_index(frame)
    occupied = np.zeros(len(coverage_index.segment_lengths), dtype=bool)
    total = 0
    cumulative = []
    for left, right in coverage_index.slices:
        total += int(coverage_index.segment_lengths[left:right][~occupied[left:right]].sum())
        occupied[left:right] = True
        cumulative.append(total)
    frame["cumulative_union_bp"] = cumulative
    frame["region_rank"] = np.arange(1, len(frame) + 1)
    return frame, pd.DataFrame(members)


def scale_contribution(
    original_regions: pd.DataFrame, equalized_regions: pd.DataFrame, *, top_n: int = 20
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    records = []
    for phase, frame in (("before", original_regions), ("after", equalized_regions)):
        if "peak_scale" not in frame:
            raise ValueError("candidate regions require peak_scale")
        counts = frame.peak_scale.value_counts()
        top = frame.head(top_n).peak_scale.value_counts()
        for scale in SCALES:
            count = int(counts.get(scale, 0))
            records.append({
                "phase": phase,
                "scale": scale,
                "peak_count": count,
                "peak_fraction": count / len(frame) if len(frame) else float("nan"),
                "top20_count": int(top.get(scale, 0)),
                "total_regions": len(frame),
            })
    return pd.DataFrame(records)


def assess_equalized_readiness(
    curve: pd.DataFrame,
    coverage_random: pd.DataFrame,
    *,
    zero_enrichment: float,
    replicate_spearman: float,
    no_leakage: bool,
    known_overlap_zero: bool,
    mode: str = "full",
) -> dict[str, object]:
    """Evaluate exactly the seven latest user-frozen readiness checks."""
    def at(frame: pd.DataFrame, budget: float) -> pd.Series:
        rows = frame.loc[np.isclose(frame.budget, budget)]
        if len(rows) != 1:
            raise ValueError(f"expected one row at coverage budget {budget}")
        return rows.iloc[0]

    def valid(budget: float) -> bool:
        detector = at(curve, budget)
        random = at(coverage_random, budget)
        return bool(
            detector.budget_reached
            and random.repeats >= 100
            and random.matched_repeats == random.repeats
        )

    c20 = at(curve, 0.20)
    checks = {
        "beats_coverage_random_at_10pct": bool(
            valid(0.10)
            and at(curve, 0.10).overall_recall > at(coverage_random, 0.10).overall_ci_high
        ),
        "beats_coverage_random_at_20pct": bool(
            valid(0.20) and c20.overall_recall > at(coverage_random, 0.20).overall_ci_high
        ),
        "all_classes_positive_at_20pct": bool(
            all(c20[f"{label}_recall"] > 0 for label in ("CHIN", "OPCID", "CHID"))
        ),
        "zero_axis_enrichment_le_one": bool(
            np.isfinite(zero_enrichment) and zero_enrichment <= 1
        ),
        "replicate_spearman_positive": bool(
            np.isfinite(replicate_spearman) and replicate_spearman > 0
        ),
        "same_and_cross_scale_no_leakage": bool(no_leakage),
        "all_background_known_overlap_zero": bool(known_overlap_zero),
    }
    return {
        "checks": checks,
        "ready_for_task2b": bool(mode == "full" and all(checks.values())),
        "task2b_executed": False,
        "coverage_evidence_valid_10pct": valid(0.10),
        "coverage_evidence_valid_20pct": valid(0.20),
    }

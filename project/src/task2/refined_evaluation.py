"""Fixed-detector comparisons and spatially fair random baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.task2.recall import LABELS, _recall_values, _selection_count


METHOD_COLUMNS = {
    "density_only": "paired_density_z",
    "shape_only": "paired_shape_z",
    "legacy_mean": "score_mean",
    "or_max": "score_or_max",
    "positive_sum": "score_positive_sum",
}


def union_coverage_bp(intervals: pd.DataFrame) -> int:
    required = {"chrom", "start", "end"}
    if not required.issubset(intervals.columns):
        raise ValueError("intervals are missing chrom/start/end")
    total = 0
    for _, rows in intervals.sort_values(["chrom", "start", "end"]).groupby("chrom"):
        current_start: int | None = None
        current_end: int | None = None
        for row in rows.itertuples(index=False):
            start, end = int(row.start), int(row.end)
            if end <= start:
                raise ValueError("interval end must be greater than start")
            if current_start is None or start > int(current_end):
                if current_start is not None:
                    total += int(current_end) - current_start
                current_start, current_end = start, end
            else:
                current_end = max(int(current_end), end)
        if current_start is not None:
            total += int(current_end) - current_start
    return total


@dataclass(frozen=True)
class _CoverageIndex:
    segment_lengths: np.ndarray
    slices: tuple[tuple[int, int], ...]


def _coverage_index(windows: pd.DataFrame) -> _CoverageIndex:
    lengths: list[int] = []
    slices: list[tuple[int, int] | None] = [None] * len(windows)
    offset = 0
    reset = windows.reset_index(drop=True)
    for chrom, indices in reset.groupby("chrom", sort=False).groups.items():
        group = reset.loc[indices]
        endpoints = np.unique(
            np.concatenate([group["start"].to_numpy(dtype=int), group["end"].to_numpy(dtype=int)])
        )
        if len(endpoints) < 2:
            raise ValueError(f"no usable coverage segments for {chrom}")
        local_lengths = np.diff(endpoints)
        lengths.extend(local_lengths.tolist())
        for index in indices:
            row = reset.loc[index]
            left = int(np.searchsorted(endpoints, int(row["start"])))
            right = int(np.searchsorted(endpoints, int(row["end"])))
            slices[int(index)] = (offset + left, offset + right)
        offset += len(local_lengths)
    return _CoverageIndex(np.asarray(lengths, dtype=np.int64), tuple(item for item in slices if item))


def sample_coverage_matched(
    windows: pd.DataFrame,
    target_bp: int,
    rng: np.random.Generator,
    *,
    tolerance: float = 0.02,
    attempts: int = 32,
) -> tuple[pd.DataFrame, int, bool]:
    if windows.empty or target_bp <= 0 or tolerance < 0 or attempts <= 0:
        raise ValueError("invalid coverage-matched sampling request")
    reset = windows.reset_index(drop=True)
    index = _coverage_index(reset)
    return _sample_coverage_with_index(
        reset, target_bp, rng, index, tolerance=tolerance, attempts=attempts
    )


def _sample_coverage_with_index(
    reset: pd.DataFrame,
    target_bp: int,
    rng: np.random.Generator,
    index: _CoverageIndex,
    *,
    tolerance: float,
    attempts: int = 32,
) -> tuple[pd.DataFrame, int, bool]:
    best_indices: list[int] = []
    best_coverage = 0
    best_gap = abs(target_bp)
    for _ in range(attempts):
        occupied = np.zeros(len(index.segment_lengths), dtype=bool)
        chosen: list[int] = []
        coverage = 0
        for row_index in rng.permutation(len(reset)):
            left, right = index.slices[int(row_index)]
            uncovered = ~occupied[left:right]
            increment = int(index.segment_lengths[left:right][uncovered].sum())
            if increment <= 0 or coverage + increment > target_bp:
                continue
            occupied[left:right] = True
            chosen.append(int(row_index))
            coverage += increment
            gap = abs(target_bp - coverage)
            if gap < best_gap:
                best_indices, best_coverage, best_gap = chosen.copy(), coverage, gap
            if coverage == target_bp:
                break
        if best_gap == 0:
            break
    if not best_indices:
        one = int(rng.integers(0, len(reset)))
        best_indices = [one]
        best_coverage = int(reset.loc[one, "end"] - reset.loc[one, "start"])
    matched = abs(best_coverage - target_bp) <= tolerance * target_bp
    return reset.iloc[best_indices].copy().reset_index(drop=True), best_coverage, bool(matched)


def detector_comparison(
    scores: pd.DataFrame,
    structures: pd.DataFrame,
    *,
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, column in METHOD_COLUMNS.items():
        if column not in scores:
            raise ValueError(f"scores are missing {column}")
        ranked = scores.sort_values(column, ascending=False)
        for fraction in fractions:
            count = _selection_count(len(ranked), fraction)
            selected = ranked.head(count)
            rows.append(
                {
                    "method": method,
                    "candidate_fraction": fraction,
                    "n_windows": count,
                    "union_coverage_bp": union_coverage_bp(selected),
                    **_recall_values(selected, structures),
                }
            )
    return pd.DataFrame(rows)


def _summary(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=0)),
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
    )


def random_baseline_comparison(
    scores: pd.DataFrame,
    structures: pd.DataFrame,
    detector_table: pd.DataFrame,
    *,
    repeats: int = 100,
    seed: int = 20_260_920,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("random baseline repeats must be positive")
    rng = np.random.default_rng(seed)
    metrics = ["overall_recall", *(f"{label}_recall" for label in LABELS)]
    rows = []
    universe = scores.reset_index(drop=True)
    coverage_index = _coverage_index(universe)
    for detector in detector_table.itertuples(index=False):
        count_values = {metric: [] for metric in metrics}
        coverage_values = {metric: [] for metric in metrics}
        achieved: list[int] = []
        matched: list[bool] = []
        for _ in range(repeats):
            count_indices = rng.choice(len(universe), size=int(detector.n_windows), replace=False)
            count_recall = _recall_values(universe.iloc[count_indices], structures)
            coverage_sample, coverage, success = _sample_coverage_with_index(
                universe,
                int(detector.union_coverage_bp),
                rng,
                coverage_index,
                tolerance=tolerance,
            )
            coverage_recall = _recall_values(coverage_sample, structures)
            for metric in metrics:
                count_values[metric].append(count_recall[metric])
                coverage_values[metric].append(coverage_recall[metric])
            achieved.append(coverage)
            matched.append(success)
        count_overall = _summary(count_values["overall_recall"])
        coverage_overall = _summary(coverage_values["overall_recall"])
        row: dict[str, object] = {
            "method": detector.method,
            "candidate_fraction": float(detector.candidate_fraction),
            "detector_recall": float(detector.overall_recall),
            "n_windows": int(detector.n_windows),
            "target_union_coverage_bp": int(detector.union_coverage_bp),
            "window_count_random_mean": count_overall[0],
            "window_count_random_std": count_overall[1],
            "window_count_random_ci_low": count_overall[2],
            "window_count_random_ci_high": count_overall[3],
            "coverage_random_mean": coverage_overall[0],
            "coverage_random_std": coverage_overall[1],
            "coverage_random_ci_low": coverage_overall[2],
            "coverage_random_ci_high": coverage_overall[3],
            "coverage_achieved_bp_mean": float(np.mean(achieved)),
            "coverage_match_rate": float(np.mean(matched)),
            "random_repeats": repeats,
        }
        for metric in metrics[1:]:
            label = metric.removesuffix("_recall")
            count_stats = _summary(count_values[metric])
            coverage_stats = _summary(coverage_values[metric])
            row[f"{label}_window_count_random_mean"] = count_stats[0]
            row[f"{label}_coverage_random_mean"] = coverage_stats[0]
            row[f"{label}_coverage_random_ci_low"] = coverage_stats[2]
            row[f"{label}_coverage_random_ci_high"] = coverage_stats[3]
        rows.append(row)
    return pd.DataFrame(rows)


def _correlation_row(name: str, rows: pd.DataFrame) -> dict[str, object]:
    density = rows["paired_density_z"].to_numpy(dtype=float)
    shape = rows["paired_shape_z"].to_numpy(dtype=float)
    finite = np.isfinite(density) & np.isfinite(shape)
    density, shape = density[finite], shape[finite]
    if len(density) < 2 or np.ptp(density) == 0 or np.ptp(shape) == 0:
        pearson = spearman = float("nan")
    else:
        pearson = float(pearsonr(density, shape).statistic)
        spearman = float(spearmanr(density, shape).statistic)
    return {"subset": name, "n_windows": len(density), "pearson": pearson, "spearman": spearman}


def branch_correlations(
    scores: pd.DataFrame,
    background_window_ids: set[str],
    *,
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    top_count = max(1, int(math.ceil(len(scores) * top_fraction)))
    subsets = {
        "all": scores,
        "background": scores.loc[scores["window_id"].astype(str).isin(background_window_ids)],
        "known_overlap": scores.loc[scores["known_overlap"].astype(bool)],
        "top_candidates": scores.nlargest(top_count, "score_or_max"),
    }
    return pd.DataFrame([_correlation_row(name, rows) for name, rows in subsets.items()])

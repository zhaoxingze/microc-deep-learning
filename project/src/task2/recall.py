"""Known-structure recall, random baselines, ablations, and region merging."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.data.known_dataset import attach_structure_ids


LABELS = ("CHIN", "OPCID", "CHID")


def _identified(structures: pd.DataFrame) -> pd.DataFrame:
    return structures.copy() if "structure_id" in structures else attach_structure_ids(structures)


def _matching_structures(windows: pd.DataFrame, structures: pd.DataFrame) -> set[str]:
    recalled: set[str] = set()
    for window in windows.itertuples(index=False):
        match = structures.loc[
            (structures["chrom"].astype(str) == str(window.chrom))
            & (structures["center"].astype(float) >= float(window.start))
            & (structures["center"].astype(float) < float(window.end))
        ]
        recalled.update(match["structure_id"].astype(str))
    return recalled


def annotate_known_overlap(scores: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    known = _identified(structures)
    output = scores.copy()
    known_overlap: list[bool] = []
    interval_overlap: list[bool] = []
    known_types: list[str] = []
    known_ids: list[str] = []
    for row in output.itertuples(index=False):
        same_chrom = known["chrom"].astype(str) == str(row.chrom)
        center_match = known.loc[
            same_chrom
            & (known["center"].astype(float) >= float(row.start))
            & (known["center"].astype(float) < float(row.end))
        ]
        interval_match = known.loc[
            same_chrom
            & (known["start"].astype(float) < float(row.end))
            & (known["end"].astype(float) > float(row.start))
        ]
        known_overlap.append(not center_match.empty)
        interval_overlap.append(not interval_match.empty)
        known_types.append(";".join(sorted(center_match["type"].astype(str).unique())))
        known_ids.append(";".join(sorted(center_match["structure_id"].astype(str).unique())))
    output["known_overlap"] = known_overlap
    output["known_interval_overlap"] = interval_overlap
    output["known_types"] = known_types
    output["known_structure_ids"] = known_ids
    return output


def _recall_values(selected: pd.DataFrame, structures: pd.DataFrame) -> dict[str, float]:
    known = _identified(structures)
    recalled = _matching_structures(selected, known)
    values = {"overall_recall": len(recalled) / len(known) if len(known) else 0.0}
    for label in LABELS:
        ids = set(known.loc[known["type"].astype(str) == label, "structure_id"].astype(str))
        values[f"{label}_recall"] = len(ids & recalled) / len(ids) if ids else 0.0
    return values


def _selection_count(total: int, fraction: float) -> int:
    if total <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid top-percent selection")
    return min(total, max(1, int(math.ceil(total * fraction))))


def recall_at_fractions(
    scores: pd.DataFrame,
    structures: pd.DataFrame,
    *,
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
    score_column: str = "paired_candidate_score",
) -> pd.DataFrame:
    ranked = scores.sort_values(score_column, ascending=False)
    rows = []
    for fraction in fractions:
        count = _selection_count(len(ranked), fraction)
        selected = ranked.head(count)
        rows.append(
            {
                "candidate_fraction": fraction,
                "candidate_windows": count,
                **_recall_values(selected, structures),
            }
        )
    return pd.DataFrame(rows)


def random_recall_baseline(
    scores: pd.DataFrame,
    structures: pd.DataFrame,
    *,
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
    repeats: int = 100,
    seed: int = 20_260_920,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("random baseline repeats must be positive")
    rng = np.random.default_rng(seed)
    metrics = ["overall_recall", *(f"{label}_recall" for label in LABELS)]
    rows = []
    for fraction in fractions:
        count = _selection_count(len(scores), fraction)
        values = {metric: [] for metric in metrics}
        for _ in range(repeats):
            indices = rng.choice(len(scores), size=count, replace=False)
            recalls = _recall_values(scores.iloc[indices], structures)
            for metric in metrics:
                values[metric].append(recalls[metric])
        row: dict[str, float | int] = {
            "candidate_fraction": fraction,
            "candidate_windows": count,
            "random_repeats": repeats,
        }
        for metric in metrics:
            array = np.asarray(values[metric], dtype=float)
            row[f"random_{metric}_mean"] = float(array.mean())
            row[f"random_{metric}_std"] = float(array.std(ddof=0))
            row[f"random_{metric}_q025"] = float(np.quantile(array, 0.025))
            row[f"random_{metric}_q975"] = float(np.quantile(array, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def merge_candidate_regions(
    scores: pd.DataFrame,
    *,
    fraction: float = 0.10,
    merge_distance_bp: int = 3_200,
    score_column: str = "paired_candidate_score",
) -> pd.DataFrame:
    count = _selection_count(len(scores), fraction)
    selected = scores.nlargest(count, score_column).sort_values(["chrom", "center"])
    groups: list[list[object]] = []
    current: list[object] = []
    previous = None
    for row in selected.itertuples(index=False):
        if (
            previous is None
            or str(row.chrom) != str(previous.chrom)
            or int(row.center) - int(previous.center) > merge_distance_bp
        ):
            if current:
                groups.append(current)
            current = [row]
        else:
            current.append(row)
        previous = row
    if current:
        groups.append(current)
    rows = []
    for index, group in enumerate(groups, start=1):
        peak = max(group, key=lambda item: float(getattr(item, score_column)))
        rows.append(
            {
                "region_id": f"REGION_{index:05d}",
                "chrom": str(group[0].chrom),
                "start": min(int(item.start) for item in group),
                "end": max(int(item.end) for item in group),
                "center": int(round(np.mean([int(item.center) for item in group]))),
                "member_count": len(group),
                "peak_window_id": str(peak.window_id),
                "peak_score": float(getattr(peak, score_column)),
            }
        )
    return pd.DataFrame(rows)


def score_ablation(
    scores: pd.DataFrame,
    structures: pd.DataFrame,
    *,
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    methods = {
        "density_only": "paired_density_z",
        "shape_only": "paired_shape_z",
        "combined": "paired_candidate_score",
    }
    tables = []
    for method, column in methods.items():
        table = recall_at_fractions(
            scores, structures, fractions=fractions, score_column=column
        )
        table.insert(0, "method", method)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


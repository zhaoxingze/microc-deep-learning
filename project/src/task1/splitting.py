"""Connected genomic grouping and MILP group-aware data splitting."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


SPLITS = ("train", "val", "test")
CLASSES = ("CHIN", "OPCID", "CHID")


def assign_genomic_groups(frame: pd.DataFrame, window_bp: int = 6400) -> pd.DataFrame:
    """Assign deterministic connected-component IDs to overlapping windows."""
    required = {"structure_id", "type", "chrom", "center"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")
    if window_bp <= 0:
        raise ValueError("window_bp must be positive")
    if frame["structure_id"].duplicated().any():
        raise ValueError("structure_id values must be unique")

    working = frame.copy()
    working["window_start"] = pd.to_numeric(working["center"]) - window_bp / 2
    working["window_end"] = pd.to_numeric(working["center"]) + window_bp / 2
    working["_original_order"] = np.arange(len(working))
    ordered = working.sort_values(
        ["chrom", "window_start", "window_end", "structure_id"], kind="stable"
    ).copy()

    group_ids: list[str] = []
    current_chrom: str | None = None
    current_end: float | None = None
    group_number = 0
    for row in ordered.itertuples(index=False):
        row_chrom = str(row.chrom)
        row_start = float(row.window_start)
        row_end = float(row.window_end)
        if row_chrom != current_chrom or current_end is None or row_start >= current_end:
            group_number += 1
            current_chrom = row_chrom
            current_end = row_end
        else:
            current_end = max(current_end, row_end)
        group_ids.append(f"group_{group_number:04d}")
    ordered["genomic_group_id"] = group_ids
    return ordered.sort_values("_original_order", kind="stable").drop(columns="_original_order").reset_index(drop=True)


def _group_counts(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_id, group in frame.groupby("genomic_group_id", sort=True):
        counts = group["type"].value_counts()
        records.append(
            {
                "genomic_group_id": group_id,
                "total": int(len(group)),
                **{label: int(counts.get(label, 0)) for label in CLASSES},
            }
        )
    return pd.DataFrame(records)


def solve_group_milp(
    frame: pd.DataFrame,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 20260920,
) -> pd.DataFrame:
    """Assign whole groups to splits while minimizing count deviations."""
    if len(ratios) != len(SPLITS) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain three values summing to 1")
    groups = _group_counts(frame)
    if groups.empty:
        raise ValueError("Cannot split an empty dataset")
    for label in CLASSES:
        if int(groups[label].sum()) < len(SPLITS):
            raise ValueError(f"Not enough {label} structures to place one in every split")

    n_groups = len(groups)
    metrics = ("total", *CLASSES)
    n_assign = n_groups * len(SPLITS)
    n_deviation = len(metrics) * len(SPLITS)
    n_variables = n_assign + n_deviation

    objective = np.zeros(n_variables, dtype=np.float64)
    for group_index, group_id in enumerate(groups["genomic_group_id"]):
        for split_index, split_name in enumerate(SPLITS):
            digest = hashlib.sha256(f"{seed}|{group_id}|{split_name}".encode("utf-8")).digest()
            objective[group_index * len(SPLITS) + split_index] = int.from_bytes(digest[:4], "big") / (2**32) * 1e-9
    totals = {metric: float(groups[metric].sum()) for metric in metrics}
    for metric_index, metric in enumerate(metrics):
        for split_index, split_name in enumerate(SPLITS):
            target = totals[metric] * float(ratios[split_index])
            weight = 1.0 / max(target, 1.0)
            if metric == "CHID" and split_name in {"val", "test"}:
                weight *= 2.0
            deviation_index = n_assign + metric_index * len(SPLITS) + split_index
            objective[deviation_index] = weight

    integrality = np.zeros(n_variables, dtype=np.int32)
    integrality[:n_assign] = 1
    lower_bounds = np.zeros(n_variables, dtype=np.float64)
    upper_bounds = np.full(n_variables, np.inf, dtype=np.float64)
    upper_bounds[:n_assign] = 1.0

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for group_index in range(n_groups):
        row = np.zeros(n_variables, dtype=np.float64)
        start = group_index * len(SPLITS)
        row[start : start + len(SPLITS)] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    for split_index in range(len(SPLITS)):
        for label in CLASSES:
            row = np.zeros(n_variables, dtype=np.float64)
            for group_index in range(n_groups):
                row[group_index * len(SPLITS) + split_index] = float(groups.iloc[group_index][label])
            rows.append(row)
            lower.append(1.0)
            upper.append(np.inf)

    for metric_index, metric in enumerate(metrics):
        for split_index in range(len(SPLITS)):
            target = totals[metric] * float(ratios[split_index])
            deviation_index = n_assign + metric_index * len(SPLITS) + split_index
            positive = np.zeros(n_variables, dtype=np.float64)
            negative = np.zeros(n_variables, dtype=np.float64)
            for group_index in range(n_groups):
                coefficient = float(groups.iloc[group_index][metric])
                variable_index = group_index * len(SPLITS) + split_index
                positive[variable_index] = coefficient
                negative[variable_index] = -coefficient
            positive[deviation_index] = -1.0
            negative[deviation_index] = -1.0
            rows.extend([positive, negative])
            lower.extend([-np.inf, -np.inf])
            upper.extend([target, -target])

    matrix = lil_matrix(np.vstack(rows)).tocsr()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")

    assignment: dict[str, str] = {}
    decisions = result.x[:n_assign].reshape(n_groups, len(SPLITS))
    for group_index, group_id in enumerate(groups["genomic_group_id"]):
        assignment[str(group_id)] = SPLITS[int(np.argmax(decisions[group_index]))]
    output = frame.copy()
    output["split"] = output["genomic_group_id"].map(assignment)
    output.attrs["milp_status"] = str(result.message)
    output.attrs["milp_objective"] = float(result.fun)
    return output


def validate_group_split(frame: pd.DataFrame, require_all_classes: bool = True) -> None:
    """Independently validate group integrity, coverage, and class presence."""
    required = {"structure_id", "type", "genomic_group_id", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing split columns: {missing}")
    if frame["structure_id"].duplicated().any():
        raise ValueError("Each structure_id must occur exactly once in group split")
    if frame["split"].isna().any() or set(frame["split"]) != set(SPLITS):
        raise ValueError("Expected nonempty train, val, and test splits")
    leaking = frame.groupby("genomic_group_id")["split"].nunique()
    leaking = leaking[leaking > 1]
    if not leaking.empty:
        raise ValueError(f"Genomic group leakage detected: {leaking.index.tolist()}")
    if require_all_classes:
        for split_name in SPLITS:
            found = set(frame.loc[frame["split"] == split_name, "type"])
            missing_classes = set(CLASSES) - found
            if missing_classes:
                raise ValueError(f"Split {split_name} is missing classes: {sorted(missing_classes)}")


def split_summary(frame: pd.DataFrame, ratios: Sequence[float] = (0.70, 0.15, 0.15)) -> pd.DataFrame:
    rows = []
    total = len(frame)
    class_totals = frame["type"].value_counts()
    for split_index, split_name in enumerate(SPLITS):
        subset = frame.loc[frame["split"] == split_name]
        counts = subset["type"].value_counts()
        actual_ratio = len(subset) / total
        row: dict[str, object] = {
            "split": split_name,
            "CHIN": int(counts.get("CHIN", 0)),
            "OPCID": int(counts.get("OPCID", 0)),
            "CHID": int(counts.get("CHID", 0)),
            "total": int(len(subset)),
            "groups": int(subset["genomic_group_id"].nunique()),
            "actual_ratio": actual_ratio,
            "target_ratio": float(ratios[split_index]),
            "ratio_deviation": actual_ratio - float(ratios[split_index]),
        }
        for label in CLASSES:
            class_ratio = int(counts.get(label, 0)) / int(class_totals[label])
            row[f"{label}_ratio"] = class_ratio
            row[f"{label}_ratio_deviation"] = class_ratio - float(ratios[split_index])
        rows.append(row)
    return pd.DataFrame(rows)

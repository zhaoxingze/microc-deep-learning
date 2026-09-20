from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.task1.splitting import (
    assign_genomic_groups,
    solve_group_milp,
    split_summary,
    validate_group_split,
)


def _frame(centers: list[float], labels: list[str] | None = None) -> pd.DataFrame:
    if labels is None:
        labels = ["CHIN"] * len(centers)
    return pd.DataFrame(
        {
            "structure_id": [f"S{i:03d}" for i in range(len(centers))],
            "type": labels,
            "chrom": "MG1655",
            "start": np.asarray(centers) - 10,
            "end": np.asarray(centers) + 10,
            "center": centers,
        }
    )


def test_transitive_overlap_merges_connected_component() -> None:
    grouped = assign_genomic_groups(_frame([3200, 9599, 15998]), window_bp=6400)
    assert grouped["genomic_group_id"].nunique() == 1


def test_endpoint_only_touching_does_not_overlap() -> None:
    grouped = assign_genomic_groups(_frame([3200, 9600]), window_bp=6400)
    assert grouped["genomic_group_id"].tolist() == ["group_0001", "group_0002"]


def test_group_ids_are_stable_under_row_reordering() -> None:
    original = _frame([3200, 9599, 20000, 30000])
    first = assign_genomic_groups(original, window_bp=6400).set_index("structure_id")["genomic_group_id"]
    second = assign_genomic_groups(original.sample(frac=1, random_state=7), window_bp=6400).set_index("structure_id")["genomic_group_id"]
    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())


def test_milp_split_has_no_leakage_and_each_split_has_all_classes() -> None:
    labels = ["CHIN"] * 4 + ["OPCID"] * 4 + ["CHID"] * 4
    centers = [500 + i * 10000 for i in range(12)]
    grouped = assign_genomic_groups(_frame(centers, labels), window_bp=6400)
    split = solve_group_milp(grouped, ratios=(0.70, 0.15, 0.15), seed=20260920)

    validate_group_split(split)

    assert split.groupby("genomic_group_id")["split"].nunique().max() == 1
    for split_name in ("train", "val", "test"):
        assert set(split.loc[split.split == split_name, "type"]) == {"CHIN", "OPCID", "CHID"}


def test_validator_rejects_group_leakage() -> None:
    leaked = pd.DataFrame(
        {
            "structure_id": ["A", "B", "C", "D"],
            "type": ["CHIN", "OPCID", "CHID", "CHIN"],
            "genomic_group_id": ["group_0001", "group_0001", "group_0002", "group_0003"],
            "split": ["train", "test", "val", "test"],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_group_split(leaked, require_all_classes=False)


def test_split_summary_reports_each_class_ratio_and_deviation() -> None:
    labels = ["CHIN"] * 4 + ["OPCID"] * 4 + ["CHID"] * 4
    grouped = assign_genomic_groups(_frame([500 + i * 10000 for i in range(12)], labels))
    split = solve_group_milp(grouped)
    summary = split_summary(split)
    for label in ("CHIN", "OPCID", "CHID"):
        assert f"{label}_ratio" in summary
        assert f"{label}_ratio_deviation" in summary
        assert summary[f"{label}_ratio"].sum() == pytest.approx(1.0)

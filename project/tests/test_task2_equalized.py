import math

import numpy as np
import pandas as pd
import pytest

from src.task2.equalized import (
    assess_equalized_readiness,
    build_equalized_regions,
    common_p_floor,
    equalize_window_scores,
    rank_equalized_windows,
    scale_contribution,
)


def _row(window_id, start, scale="small", window_bp=3200, **overrides):
    base = dict(
        window_id=window_id,
        chrom="MG1655",
        start=start,
        end=start + window_bp,
        center=start + window_bp // 2,
        scale=scale,
        window_bp=window_bp,
        rep1_density_p=0.01,
        rep1_shape_p=0.02,
        rep2_density_p=0.03,
        rep2_shape_p=0.04,
        scale_or_score=3.0,
        zero_axis_rep1=False,
        zero_axis_rep2=False,
    )
    base.update(overrides)
    return base


def test_common_floor_and_equalized_formula_have_shared_ceiling():
    floor = common_p_floor({"small": 1054, "medium": 431, "large": 147})
    assert floor == pytest.approx(1 / 148)
    rows = pd.DataFrame([
        _row("s", 0, rep1_density_p=1 / 1055, rep1_shape_p=0.5,
             rep2_density_p=1 / 1055, rep2_shape_p=0.5),
        _row("l", 5000, scale="large", window_bp=12800,
             rep1_density_p=1 / 148, rep1_shape_p=0.5,
             rep2_density_p=1 / 148, rep2_shape_p=0.5),
    ])
    out = equalize_window_scores(rows, floor)
    ceiling = -math.log10(1 / 148)
    assert out["rep1_density_p_common"].tolist() == pytest.approx([floor, floor])
    assert out["paired_density_common_anomaly"].tolist() == pytest.approx([ceiling, ceiling])
    assert out["equalized_scale_score"].max() == pytest.approx(ceiling)
    assert out.loc[0, "replicate_min_score"] == pytest.approx(ceiling)
    assert out.loc[0, "original_uncapped_tail_score"] == 3.0


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "range"])
def test_equalization_rejects_invalid_score_tables(mutation):
    rows = pd.DataFrame([_row("a", 0), _row("b", 4000)])
    if mutation == "duplicate":
        rows.loc[1, "window_id"] = "a"
    elif mutation == "missing":
        rows = rows.drop(columns="rep1_density_p")
    else:
        rows.loc[0, "rep2_shape_p"] = 1.1
    with pytest.raises(ValueError):
        equalize_window_scores(rows, 1 / 148)


def test_ranking_uses_all_frozen_tie_breakers_in_order():
    rows = pd.DataFrame([
        _row("coordinate_later", 400, rep1_density_p=.1, rep2_density_p=.1, scale_or_score=2),
        _row("tail_wins", 1200, rep1_density_p=.1, rep2_density_p=.1, scale_or_score=4),
        _row("replicate_wins", 800, rep1_density_p=.01, rep2_density_p=.01, scale_or_score=1),
        _row("coordinate_first", 0, rep1_density_p=.1, rep2_density_p=.1, scale_or_score=2),
    ])
    equalized = equalize_window_scores(rows, 1 / 148)
    # Force the primary score tie while retaining replicate and original-tail differences.
    equalized["equalized_scale_score"] = 1.0
    ranked = rank_equalized_windows(equalized)
    assert ranked.window_id.tolist() == [
        "replicate_wins", "tail_wins", "coordinate_first", "coordinate_later"
    ]
    assert ranked["equalized_rank"].tolist() == [1, 2, 3, 4]


def test_equalized_nms_preserves_strict_half_larger_window_boundary():
    rows = pd.DataFrame([
        _row("peak", 0, scale_or_score=4),
        _row("suppressed", 1200, scale_or_score=3),  # center distance 1200 < 1600
        _row("boundary_kept", 1600, scale_or_score=2),  # exactly 1600 is retained
    ])
    equalized = equalize_window_scores(rows, 1 / 148)
    regions, members = build_equalized_regions(equalized)
    assert regions.peak_window_id.tolist() == ["peak", "boundary_kept"]
    assert set(members.loc[members.region_id.eq(regions.iloc[0].region_id), "window_id"]) == {
        "peak", "suppressed"
    }
    assert regions.iloc[0].start == 0 and regions.iloc[0].end == 3200


def test_scale_contribution_reports_before_after_and_top20():
    old = pd.DataFrame({"peak_scale": ["small", "small", "medium"]})
    new = pd.DataFrame({"peak_scale": ["large", "medium", "small"]})
    table = scale_contribution(old, new, top_n=2)
    lookup = table.set_index(["phase", "scale"])
    assert lookup.loc[("before", "small"), "peak_count"] == 2
    assert lookup.loc[("before", "small"), "top20_count"] == 2
    assert lookup.loc[("after", "large"), "peak_count"] == 1


def test_readiness_is_exactly_the_frozen_seven_checks():
    curve = pd.DataFrame([
        dict(budget=.1, budget_reached=True, overall_recall=.2,
             CHIN_recall=.1, OPCID_recall=.1, CHID_recall=.1),
        dict(budget=.2, budget_reached=True, overall_recall=.3,
             CHIN_recall=.1, OPCID_recall=.2, CHID_recall=.3),
    ])
    random = pd.DataFrame([
        dict(budget=.1, repeats=100, matched_repeats=100, overall_ci_high=.19),
        dict(budget=.2, repeats=100, matched_repeats=100, overall_ci_high=.29),
    ])
    result = assess_equalized_readiness(
        curve, random, zero_enrichment=1.0, replicate_spearman=.01,
        no_leakage=True, known_overlap_zero=True, mode="full"
    )
    assert len(result["checks"]) == 7
    assert all(result["checks"].values())
    assert result["ready_for_task2b"] is True
    assert result["task2b_executed"] is False
    failed = curve.copy()
    failed.loc[np.isclose(failed.budget, .2), "CHID_recall"] = 0
    assert assess_equalized_readiness(
        failed, random, zero_enrichment=1, replicate_spearman=.01,
        no_leakage=True, known_overlap_zero=True, mode="full"
    )["ready_for_task2b"] is False

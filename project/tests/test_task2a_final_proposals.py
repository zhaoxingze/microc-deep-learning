import pandas as pd
import pytest

from src.task2.final_proposals import (
    PROPOSAL_P_THRESHOLD,
    final_engineering_gate,
    generate_channel_proposals,
    merge_proposals,
)
from src.task2.final_pipeline import annotate_candidates, calibration_tail_audit, final_recall_table


def score(window_id, scale, start, window_bp, density_p, shape_p, *, chrom="MG1655"):
    return dict(
        window_id=window_id, scale=scale, chrom=chrom, start=start,
        end=start + window_bp, center=start + window_bp // 2,
        window_bp=window_bp,
        rep1_density_p=density_p, rep2_density_p=density_p,
        rep1_shape_p=shape_p, rep2_shape_p=shape_p,
        zero_axis_rep1=False, zero_axis_rep2=False,
    )


def test_six_channels_use_fixed_independent_p_threshold():
    rows = pd.DataFrame([
        score("s", "small", 0, 3200, .04, .06),
        score("m", "medium", 10000, 6400, .06, .03),
        score("l", "large", 30000, 12800, .05, .05),
    ])
    proposals = generate_channel_proposals(rows)
    assert PROPOSAL_P_THRESHOLD == .05
    assert set(zip(proposals.scale, proposals.branch)) == {
        ("small", "density"), ("medium", "shape"),
        ("large", "density"), ("large", "shape"),
    }


def test_nms_is_channel_local_and_cross_scale_peaks_survive():
    rows = pd.DataFrame([
        score("s1", "small", 0, 3200, .001, .7),
        score("s2", "small", 400, 3200, .002, .7),
        score("m1", "medium", 0, 6400, .001, .7),
        score("l1", "large", 0, 12800, .001, .7),
    ])
    proposals = generate_channel_proposals(rows)
    assert set(proposals.window_id) == {"s1", "m1", "l1"}


def test_transitive_overlap_merges_connected_components():
    rows = pd.DataFrame([
        score("a", "small", 0, 3200, .01, .8),
        score("b", "small", 3000, 3200, .02, .8),
        score("c", "small", 6000, 3200, .03, .8),
    ])
    proposals = generate_channel_proposals(rows)
    regions = merge_proposals(proposals)
    assert len(regions) == 1
    assert regions.iloc[0].start == 0
    assert regions.iloc[0].end == 9200
    assert regions.iloc[0].n_support_proposals == 3


def test_known_columns_do_not_change_proposal_inclusion_or_merge():
    rows = pd.DataFrame([
        score("a", "small", 0, 3200, .01, .7),
        score("b", "medium", 1000, 6400, .03, .7),
    ])
    first = merge_proposals(generate_channel_proposals(rows))
    altered = rows.assign(known_center_overlap=[True, False], known_types=["CHID", ""])
    second = merge_proposals(generate_channel_proposals(altered))
    assert first[["region_id", "start", "end", "representative_window_id"]].equals(
        second[["region_id", "start", "end", "representative_window_id"]]
    )


def test_geometric_mean_p_matches_paired_anomaly_and_replicate_tie_break():
    rows = pd.DataFrame([
        score("unstable", "small", 0, 3200, .01, .8),
        score("stable", "small", 4000, 3200, .01, .8),
    ])
    rows.loc[0, ["rep1_density_p", "rep2_density_p"]] = [.0001, 1.0]
    proposals = generate_channel_proposals(rows)
    assert proposals.loc[proposals.window_id.eq("unstable"), "best_p"].iloc[0] == pytest.approx(.01)
    assert proposals.loc[proposals.window_id.eq("stable"), "best_p"].iloc[0] == pytest.approx(.01)
    assert proposals.window_id.tolist() == ["stable", "unstable"]


def test_engineering_gate_does_not_consume_recall():
    valid = pd.DataFrame([dict(region_id="R", chrom="MG1655", start=0, end=3200, center=1600)])
    gate = final_engineering_gate(
        valid, protocol_frozen=True, artifacts_complete=True,
        replicates_accessible=True, no_leakage=True, known_label_tuning=False,
    )
    assert gate["ready_for_task2b"] is True
    assert "recall" not in str(gate).lower()


def test_annotation_is_post_merge_and_reports_center_vs_interval():
    regions = merge_proposals(generate_channel_proposals(pd.DataFrame([
        score("a", "small", 0, 3200, .01, .8),
    ])))
    known = pd.DataFrame([
        dict(structure_id="K1", type="CHIN", chrom="MG1655", start=100, end=200, center=150),
        dict(structure_id="K2", type="CHID", chrom="MG1655", start=3100, end=3400, center=3250),
    ])
    annotated = annotate_candidates(regions, known)
    assert bool(annotated.iloc[0].known_center_overlap)
    assert bool(annotated.iloc[0].known_interval_overlap)
    assert annotated.iloc[0].known_structure_ids == "K1;K2"
    assert annotated.iloc[0].known_types == "CHID;CHIN"
    recall = final_recall_table(annotated, known)
    assert recall.set_index("type").loc["CHIN", "center_recall"] == 1
    assert recall.set_index("type").loc["CHID", "center_recall"] == 0


def test_calibration_tail_audit_uses_existing_p_without_changing_selection():
    rows = pd.DataFrame([
        {**score("a", "small", 0, 3200, .01, .8), "background_split": "calibration"},
        {**score("b", "small", 4000, 3200, .1, .8), "background_split": "calibration"},
        {**score("c", "small", 8000, 3200, .001, .8), "background_split": "train"},
        {**score("d", "medium", 12000, 6400, .1, .8), "background_split": "calibration"},
        {**score("e", "large", 24000, 12800, .1, .8), "background_split": "calibration"},
    ])
    before = rows.copy(deep=True)
    audit = calibration_tail_audit(rows)
    density = audit.set_index("channel").loc["small_density"]
    assert density.n_calibration_windows == 2
    assert density.n_below_fixed_threshold == 1
    assert density.observed_tail_fraction == pytest.approx(.5)
    assert rows.equals(before)

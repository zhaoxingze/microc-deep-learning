import pandas as pd
import pandas.testing as pdt

from src.task2.recall import (
    annotate_known_overlap,
    merge_candidate_regions,
    random_recall_baseline,
    recall_at_fractions,
)


def toy_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w3", "w4"],
            "chrom": ["chr"] * 4,
            "start": [0, 100, 500, 2_000],
            "end": [400, 500, 900, 2_400],
            "center": [200, 300, 700, 2_200],
            "paired_candidate_score": [4.0, 2.0, 3.0, 1.0],
        }
    )


def toy_structures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "structure_id": ["s1", "s2", "s3"],
            "type": ["CHIN", "OPCID", "CHID"],
            "chrom": ["chr"] * 3,
            "start": [190, 690, 2_490],
            "end": [210, 710, 2_510],
            "center": [200, 700, 2_500],
        }
    )


def test_annotation_and_top_fraction_recall_use_known_center() -> None:
    annotated = annotate_known_overlap(toy_scores(), toy_structures())
    assert annotated.set_index("window_id").loc["w1", "known_types"] == "CHIN"
    table = recall_at_fractions(
        annotated, toy_structures(), fractions=(0.25, 0.5), score_column="paired_candidate_score"
    )
    quarter = table.loc[table.candidate_fraction == 0.25].iloc[0]
    half = table.loc[table.candidate_fraction == 0.5].iloc[0]
    assert quarter.overall_recall == 1 / 3
    assert half.overall_recall == 2 / 3
    assert half.CHIN_recall == 1.0 and half.OPCID_recall == 1.0 and half.CHID_recall == 0.0


def test_random_baseline_is_seed_deterministic() -> None:
    first = random_recall_baseline(toy_scores(), toy_structures(), fractions=(0.5,), repeats=20, seed=9)
    second = random_recall_baseline(toy_scores(), toy_structures(), fractions=(0.5,), repeats=20, seed=9)
    pdt.assert_frame_equal(first, second)


def test_candidate_region_merge_uses_center_distance() -> None:
    regions = merge_candidate_regions(
        toy_scores(), fraction=0.75, merge_distance_bp=150
    )
    assert len(regions) == 2
    assert regions.iloc[0]["member_count"] == 2
    assert regions.iloc[0]["start"] == 0 and regions.iloc[0]["end"] == 500

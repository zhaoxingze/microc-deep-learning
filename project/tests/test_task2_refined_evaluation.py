import numpy as np
import pandas as pd

from src.task2.refined_evaluation import (
    branch_correlations,
    detector_comparison,
    random_baseline_comparison,
    sample_coverage_matched,
    union_coverage_bp,
)


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w3", "w4"],
            "chrom": ["chr", "chr", "chr", "chr"],
            "start": [0, 5, 20, 25],
            "end": [10, 15, 30, 35],
            "center": [5, 10, 25, 30],
            "paired_density_z": [4.0, 1.0, 3.0, 0.0],
            "paired_shape_z": [-2.0, 5.0, 2.0, 0.0],
            "score_mean": [1.0, 3.0, 2.5, 0.0],
            "score_or_max": [4.0, 5.0, 3.0, 0.0],
            "score_positive_sum": [4.0, 6.0, 5.0, 0.0],
            "known_overlap": [True, True, False, False],
        }
    )


def _known() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": ["CHIN", "OPCID", "CHID"],
            "chrom": ["chr", "chr", "chr"],
            "start": [4, 24, 32],
            "end": [6, 26, 34],
            "center": [5, 25, 33],
        }
    )


def test_union_coverage_merges_overlaps_per_chromosome() -> None:
    intervals = pd.DataFrame(
        {"chrom": ["a", "a", "b"], "start": [0, 5, 0], "end": [10, 15, 4]}
    )
    assert union_coverage_bp(intervals) == 19


def test_coverage_sampler_matches_target_and_is_seed_reproducible() -> None:
    windows = _windows()[["window_id", "chrom", "start", "end"]]
    first, coverage1, matched1 = sample_coverage_matched(
        windows, 15, np.random.default_rng(7), tolerance=0.02
    )
    second, coverage2, matched2 = sample_coverage_matched(
        windows, 15, np.random.default_rng(7), tolerance=0.02
    )

    assert coverage1 == coverage2 == 15
    assert matched1 and matched2
    assert first["window_id"].tolist() == second["window_id"].tolist()


def test_detector_comparison_contains_five_fixed_methods_and_union_coverage() -> None:
    table = detector_comparison(_windows(), _known(), fractions=(0.5,))

    assert set(table["method"]) == {
        "density_only", "shape_only", "legacy_mean", "or_max", "positive_sum"
    }
    assert table["n_windows"].eq(2).all()
    assert (table["union_coverage_bp"] > 0).all()
    assert {"overall_recall", "CHIN_recall", "OPCID_recall", "CHID_recall"}.issubset(table)


def test_random_baseline_comparison_is_reproducible() -> None:
    scores = _windows()
    detectors = detector_comparison(scores, _known(), fractions=(0.5,))
    first = random_baseline_comparison(
        scores, _known(), detectors, repeats=5, seed=11, tolerance=0.02
    )
    second = random_baseline_comparison(
        scores, _known(), detectors, repeats=5, seed=11, tolerance=0.02
    )

    pd.testing.assert_frame_equal(first, second)
    assert {
        "window_count_random_mean", "window_count_random_std",
        "coverage_random_mean", "coverage_random_std",
        "coverage_random_ci_low", "coverage_random_ci_high",
    }.issubset(first)


def test_branch_correlations_report_empty_subset_without_crashing() -> None:
    scores = _windows()
    scores["known_overlap"] = False
    table = branch_correlations(scores, {"w1", "w2"}, top_fraction=0.5)
    known = table.loc[table["subset"].eq("known_overlap")].iloc[0]

    assert known["n_windows"] == 0
    assert np.isnan(known["pearson"])
    assert np.isnan(known["spearman"])

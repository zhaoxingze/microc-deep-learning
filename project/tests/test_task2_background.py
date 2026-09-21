import pandas as pd
import pytest

from src.task2.background import (
    assert_no_background_overlap,
    blocked_background_split,
    build_exclusion_intervals,
    mark_background_candidates,
)


def test_known_exclusion_uses_center_plus_half_window_and_margin() -> None:
    structures = pd.DataFrame(
        {"structure_id": ["s1"], "chrom": ["chr"], "center": [10_000]}
    )
    result = build_exclusion_intervals(
        structures, window_bp=6_400, margin_bp=3_200
    )
    assert result[["exclusion_start", "exclusion_end"]].iloc[0].tolist() == [3_600, 16_400]


def test_background_exclusion_uses_strict_half_open_overlap() -> None:
    windows = pd.DataFrame(
        {
            "window_id": ["touch", "overlap", "after"],
            "chrom": ["chr"] * 3,
            "start": [0, 1, 16_400],
            "end": [3_600, 3_601, 20_000],
            "center": [1_800, 1_801, 18_200],
        }
    )
    exclusions = pd.DataFrame(
        {"chrom": ["chr"], "exclusion_start": [3_600], "exclusion_end": [16_400]}
    )
    marked = mark_background_candidates(windows, exclusions)
    assert marked.set_index("window_id")["background_candidate"].to_dict() == {
        "touch": True,
        "overlap": False,
        "after": True,
    }


def test_blocked_split_drops_crossing_windows_and_has_no_overlap() -> None:
    windows = pd.DataFrame(
        {
            "window_id": ["a", "cross", "b", "c"],
            "chrom": ["chr"] * 4,
            "start": [0, 8_000, 10_000, 20_000],
            "end": [4_000, 12_000, 14_000, 24_000],
            "center": [2_000, 10_000, 12_000, 22_000],
            "background_candidate": [True] * 4,
        }
    )
    split = blocked_background_split(
        windows, block_bp=10_000, train_fraction=0.5, seed=7
    )
    assert split.set_index("window_id").loc["cross", "background_split"] == "boundary_excluded"
    assert set(split.background_split) == {"train", "val", "boundary_excluded"}
    assert_no_background_overlap(split)


def test_blocked_split_requires_two_eligible_blocks() -> None:
    windows = pd.DataFrame(
        {
            "window_id": ["a"], "chrom": ["chr"], "start": [0], "end": [4_000],
            "center": [2_000], "background_candidate": [True]
        }
    )
    with pytest.raises(ValueError, match="two eligible genomic blocks"):
        blocked_background_split(windows, block_bp=10_000)

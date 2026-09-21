"""Known-structure exclusion and leakage-safe background splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.known_dataset import attach_structure_ids


def build_exclusion_intervals(
    structures: pd.DataFrame,
    *,
    window_bp: int = 6_400,
    margin_bp: int = 3_200,
) -> pd.DataFrame:
    if window_bp <= 0 or margin_bp < 0:
        raise ValueError("invalid exclusion geometry")
    identified = structures.copy()
    if "structure_id" not in identified:
        identified = attach_structure_ids(identified)
    required = {"structure_id", "chrom", "center"}
    missing = required - set(identified.columns)
    if missing:
        raise ValueError(f"structures missing columns: {sorted(missing)}")
    radius = window_bp // 2 + margin_bp
    result = identified[["structure_id", "chrom", "center"]].copy()
    result["exclusion_start"] = np.floor(result["center"] - radius).clip(lower=0).astype(int)
    result["exclusion_end"] = np.ceil(result["center"] + radius).astype(int)
    return result


def mark_background_candidates(
    windows: pd.DataFrame, exclusions: pd.DataFrame
) -> pd.DataFrame:
    required_windows = {"window_id", "chrom", "start", "end", "center"}
    required_exclusions = {"chrom", "exclusion_start", "exclusion_end"}
    if not required_windows.issubset(windows.columns):
        raise ValueError("window metadata is incomplete")
    if not required_exclusions.issubset(exclusions.columns):
        raise ValueError("exclusion metadata is incomplete")
    output = windows.drop_duplicates("window_id").copy().reset_index(drop=True)
    is_background = np.ones(len(output), dtype=bool)
    for chrom, indices in output.groupby("chrom").groups.items():
        known = exclusions.loc[exclusions["chrom"].astype(str) == str(chrom)]
        if known.empty:
            continue
        starts = output.loc[indices, "start"].to_numpy(dtype=np.int64)[:, None]
        ends = output.loc[indices, "end"].to_numpy(dtype=np.int64)[:, None]
        exclusion_starts = known["exclusion_start"].to_numpy(dtype=np.int64)[None, :]
        exclusion_ends = known["exclusion_end"].to_numpy(dtype=np.int64)[None, :]
        overlaps = (starts < exclusion_ends) & (ends > exclusion_starts)
        is_background[np.asarray(list(indices), dtype=np.int64)] = ~overlaps.any(axis=1)
    output["background_candidate"] = is_background
    return output


def blocked_background_split(
    windows: pd.DataFrame,
    *,
    block_bp: int = 128_000,
    train_fraction: float = 0.8,
    seed: int = 20_260_920,
) -> pd.DataFrame:
    if block_bp <= 0 or not 0 < train_fraction < 1:
        raise ValueError("invalid blocked split configuration")
    required = {"window_id", "chrom", "start", "end", "background_candidate"}
    if not required.issubset(windows.columns):
        raise ValueError("background windows are missing required columns")
    output = windows.copy().reset_index(drop=True)
    block_index = (output["start"].astype(int) // block_bp).astype(int)
    block_end = (block_index + 1) * block_bp
    inside_block = output["end"].astype(int) <= block_end
    output["background_block"] = [
        f"{chrom}:{index}" for chrom, index in zip(output["chrom"], block_index, strict=True)
    ]
    output["background_split"] = "non_background"
    boundary = output["background_candidate"].astype(bool) & ~inside_block
    output.loc[boundary, "background_split"] = "boundary_excluded"
    eligible = output["background_candidate"].astype(bool) & inside_block
    blocks = sorted(output.loc[eligible, "background_block"].unique().tolist())
    if len(blocks) < 2:
        raise ValueError("background split requires at least two eligible genomic blocks")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(blocks, dtype=object)[rng.permutation(len(blocks))])
    train_count = int(round(len(blocks) * train_fraction))
    train_count = min(max(train_count, 1), len(blocks) - 1)
    train_blocks = set(shuffled[:train_count])
    output.loc[eligible, "background_split"] = np.where(
        output.loc[eligible, "background_block"].isin(train_blocks), "train", "val"
    )
    assert_no_background_overlap(output)
    return output


def assert_no_background_overlap(split: pd.DataFrame) -> None:
    train = split.loc[split["background_split"] == "train"]
    val = split.loc[split["background_split"] == "val"]
    if train.empty or val.empty:
        raise ValueError("background train and validation must both be non-empty")
    for chrom in sorted(set(train["chrom"].astype(str)) & set(val["chrom"].astype(str))):
        train_rows = train.loc[train["chrom"].astype(str) == chrom]
        val_rows = val.loc[val["chrom"].astype(str) == chrom]
        for train_row in train_rows.itertuples(index=False):
            overlap = (val_rows["start"].astype(int) < int(train_row.end)) & (
                val_rows["end"].astype(int) > int(train_row.start)
            )
            if overlap.any():
                raise ValueError(
                    f"background train/val overlap leakage at {chrom}:{train_row.start}-{train_row.end}"
                )


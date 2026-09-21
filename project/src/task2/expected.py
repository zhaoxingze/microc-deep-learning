"""Streaming genome-wide expected contact values on the target-bin scale."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

from src.data.load_microc import _resolve_chromosome


def expected_from_target_pixels(
    pixel_chunks: Iterable[pd.DataFrame],
    *,
    n_target_bins: int,
    max_distance_bins: int,
) -> np.ndarray:
    if n_target_bins <= 0 or max_distance_bins < 0:
        raise ValueError("invalid expected dimensions")
    sums = np.zeros(max_distance_bins + 1, dtype=np.float64)
    for pixels in pixel_chunks:
        required = {"target1", "target2", "count"}
        if not required.issubset(pixels.columns):
            raise ValueError(f"pixel chunk is missing columns: {sorted(required - set(pixels.columns))}")
        distances = np.abs(
            pixels["target2"].to_numpy(dtype=np.int64)
            - pixels["target1"].to_numpy(dtype=np.int64)
        )
        counts = pixels["count"].to_numpy(dtype=np.float64)
        keep = (distances >= 0) & (distances <= max_distance_bins)
        np.add.at(sums, distances[keep], counts[keep])
    denominators = np.asarray(
        [max(n_target_bins - distance, 0) for distance in range(max_distance_bins + 1)],
        dtype=np.float64,
    )
    return np.divide(sums, denominators, out=np.zeros_like(sums), where=denominators > 0)


def compute_genome_expected(
    cool_path: str | Path,
    chrom: str,
    *,
    target_bin_size: int = 100,
    max_distance_bins: int = 63,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    contact_map = cooler.Cooler(str(Path(cool_path).resolve()))
    if contact_map.binsize is None or target_bin_size % int(contact_map.binsize):
        raise ValueError("target_bin_size must be an integer multiple of Cooler binsize")
    chrom_used = _resolve_chromosome(contact_map, chrom)
    chrom_size = int(contact_map.chromsizes[chrom_used])
    first_bin, last_bin = contact_map.extent(chrom_used)
    bins = contact_map.bins()[first_bin:last_bin]
    target_by_source = (
        bins["start"].to_numpy(dtype=np.int64) // int(target_bin_size)
    )
    n_target_bins = int(math.ceil(chrom_size / target_bin_size))
    pixel_selector = contact_map.pixels()
    nnz = int(contact_map.info["nnz"])

    def chunks():
        for offset in range(0, nnz, chunk_size):
            pixels = pixel_selector[offset : min(offset + chunk_size, nnz)]
            bin1 = pixels["bin1_id"].to_numpy(dtype=np.int64)
            bin2 = pixels["bin2_id"].to_numpy(dtype=np.int64)
            keep = (
                (bin1 >= first_bin)
                & (bin1 < last_bin)
                & (bin2 >= first_bin)
                & (bin2 < last_bin)
            )
            yield pd.DataFrame(
                {
                    "target1": target_by_source[bin1[keep] - first_bin],
                    "target2": target_by_source[bin2[keep] - first_bin],
                    "count": pixels.loc[keep, "count"].to_numpy(dtype=np.float64),
                }
            )

    return expected_from_target_pixels(
        chunks(), n_target_bins=n_target_bins, max_distance_bins=max_distance_bins
    )

def apply_expected(matrix: np.ndarray, expected: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(expected, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("apply_expected requires a square 2D matrix")
    distances = np.abs(np.subtract.outer(np.arange(array.shape[0]), np.arange(array.shape[1])))
    if distances.max(initial=0) >= len(vector):
        raise ValueError("expected vector is shorter than the matrix distance range")
    denominators = vector[distances]
    return np.divide(
        array,
        denominators,
        out=np.zeros_like(array, dtype=np.float64),
        where=denominators > 0,
    )

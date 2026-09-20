"""Read and aggregate fixed-size local windows from Cooler files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cooler
import numpy as np
from numpy.typing import NDArray


def _resolve_chromosome(contact_map: cooler.Cooler, requested: str) -> str:
    names = list(contact_map.chromnames)
    if requested in names:
        return requested

    casefold_matches = [name for name in names if name.casefold() == requested.casefold()]
    if len(casefold_matches) == 1:
        return casefold_matches[0]

    assembly = str(contact_map.info.get("genome-assembly", ""))
    known_mg1655_aliases = {"mg1655", "nc_000913", "nc_000913.3"}
    if (
        len(names) == 1
        and requested.casefold() in known_mg1655_aliases
        and (assembly.casefold() in known_mg1655_aliases or names[0].casefold() in known_mg1655_aliases)
    ):
        return names[0]

    raise ValueError(
        f"Chromosome {requested!r} is not present. Available chromosomes: {names}."
    )


def load_local_matrix(
    cool_path: str | Path,
    chrom: str,
    center: float,
    window_bp: int = 6400,
    target_bin_size: int = 100,
    balance: bool = False,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Fetch one genomic window and sum source contacts into target bins.

    Only the requested local interval is materialized. Source-bin midpoints are
    assigned to target genomic bins, which preserves exactly 64 output bins for
    a 6,400 bp window at 100 bp/bin even when the center is 5 bp off the source
    bin grid. Regions beyond chromosome bounds are zero padded.
    """
    path = Path(cool_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cooler file does not exist: {path}")
    if window_bp <= 0 or target_bin_size <= 0:
        raise ValueError("window_bp and target_bin_size must be positive.")
    if window_bp % target_bin_size:
        raise ValueError("window_bp must be exactly divisible by target_bin_size.")
    if not math.isfinite(float(center)):
        raise ValueError(f"center must be finite, got {center!r}.")

    contact_map = cooler.Cooler(str(path))
    if contact_map.binsize is None:
        raise ValueError("Variable-width Cooler bins are not supported.")
    source_bin_size = int(contact_map.binsize)
    if target_bin_size < source_bin_size or target_bin_size % source_bin_size:
        raise ValueError(
            "target_bin_size must be an integer multiple of the source binsize; "
            f"got target={target_bin_size}, source={source_bin_size}."
        )
    if balance and "weight" not in contact_map.bins().columns:
        raise ValueError("balance=True requested, but the Cooler has no 'weight' column.")

    chrom_used = _resolve_chromosome(contact_map, str(chrom))
    chrom_size = int(contact_map.chromsizes[chrom_used])
    center_bp = float(center)
    requested_start = center_bp - window_bp / 2
    requested_end = center_bp + window_bp / 2
    fetch_start = max(0, int(math.floor(requested_start)))
    fetch_end = min(chrom_size, int(math.ceil(requested_end)))
    if fetch_start >= fetch_end:
        raise ValueError(
            f"Window [{requested_start}, {requested_end}) does not overlap {chrom_used} "
            f"(length {chrom_size})."
        )

    region = (chrom_used, fetch_start, fetch_end)
    local_sparse = contact_map.matrix(balance=balance, sparse=True).fetch(region).tocoo()
    local_bins = contact_map.bins().fetch(region)
    if local_sparse.shape[0] != len(local_bins):
        raise RuntimeError(
            f"Fetched matrix/bin mismatch: {local_sparse.shape} versus {len(local_bins)} bins."
        )

    bin_midpoints = (
        local_bins["start"].to_numpy(dtype=np.float64)
        + local_bins["end"].to_numpy(dtype=np.float64)
    ) / 2.0
    target_indices = np.floor(
        (bin_midpoints - requested_start) / target_bin_size
    ).astype(np.int64)
    n_target_bins = window_bp // target_bin_size

    row_targets = target_indices[local_sparse.row]
    col_targets = target_indices[local_sparse.col]
    keep = (
        (row_targets >= 0)
        & (row_targets < n_target_bins)
        & (col_targets >= 0)
        & (col_targets < n_target_bins)
    )
    aggregated = np.zeros((n_target_bins, n_target_bins), dtype=np.float64)
    np.add.at(
        aggregated,
        (row_targets[keep], col_targets[keep]),
        np.asarray(local_sparse.data[keep], dtype=np.float64),
    )

    metadata: dict[str, Any] = {
        "cool_path": str(path),
        "chrom_requested": str(chrom),
        "chrom_used": chrom_used,
        "chrom_size": chrom_size,
        "center": center_bp,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "window_bp": window_bp,
        "source_bin_size": source_bin_size,
        "target_bin_size": target_bin_size,
        "source_bins_fetched": int(len(local_bins)),
        "padded": bool(requested_start < 0 or requested_end > chrom_size),
        "balance": bool(balance),
    }
    return aggregated.astype(np.float32), metadata

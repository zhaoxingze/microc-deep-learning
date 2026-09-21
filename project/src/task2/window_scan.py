"""Generate complete genomic windows and scan aligned Micro-C replicates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_microc import load_local_matrix


@dataclass(frozen=True)
class WindowCoordinate:
    window_id: str
    chrom: str
    start: int
    end: int
    center: int


def stable_window_id(chrom: str, start: int, end: int) -> str:
    key = f"{chrom.strip()}|{int(start)}|{int(end)}"
    return "WIN_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def generate_windows(
    chrom: str,
    chrom_size: int,
    *,
    window_bp: int = 6_400,
    step_bp: int = 800,
    scan_start: int = 0,
    scan_end: int | None = None,
) -> list[WindowCoordinate]:
    if window_bp <= 0 or step_bp <= 0:
        raise ValueError("window_bp and step_bp must be positive")
    end_limit = chrom_size if scan_end is None else min(int(scan_end), int(chrom_size))
    start_limit = max(0, int(scan_start))
    if start_limit >= end_limit:
        raise ValueError("scan interval is empty")
    starts = range(start_limit, end_limit - window_bp + 1, step_bp)
    return [
        WindowCoordinate(
            stable_window_id(chrom, start, start + window_bp),
            str(chrom),
            start,
            start + window_bp,
            start + window_bp // 2,
        )
        for start in starts
    ]


def _zero_axis(matrix: np.ndarray) -> tuple[bool, int, int]:
    zero_rows = int(np.all(matrix == 0, axis=1).sum())
    zero_cols = int(np.all(matrix == 0, axis=0).sum())
    return bool(zero_rows or zero_cols), zero_rows, zero_cols


def scan_replicates(
    rep1_cool: str | Path,
    rep2_cool: str | Path,
    chrom: str,
    chrom_size: int,
    output_dir: str | Path,
    *,
    window_bp: int = 6_400,
    step_bp: int = 800,
    target_bin_size: int = 100,
    scan_start: int = 0,
    scan_end: int | None = None,
    balance: bool = False,
) -> pd.DataFrame:
    if window_bp % target_bin_size:
        raise ValueError("window_bp must be divisible by target_bin_size")
    windows = generate_windows(
        chrom,
        chrom_size,
        window_bp=window_bp,
        step_bp=step_bp,
        scan_start=scan_start,
        scan_end=scan_end,
    )
    if not windows:
        raise ValueError("scan produced no complete windows")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    matrix_size = window_bp // target_bin_size
    metadata_rows: list[dict[str, object]] = []
    replicate_ids: dict[str, list[str]] = {}

    for replicate, cool_path in (("rep1", rep1_cool), ("rep2", rep2_cool)):
        output_path = root / f"genome_windows_{replicate}.npy"
        arrays = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(windows), 1, matrix_size, matrix_size),
        )
        ids: list[str] = []
        for index, window in enumerate(windows):
            matrix, loader_metadata = load_local_matrix(
                cool_path,
                chrom,
                window.center,
                window_bp=window_bp,
                target_bin_size=target_bin_size,
                balance=balance,
            )
            if matrix.shape != (matrix_size, matrix_size):
                raise ValueError(
                    f"{replicate} window {window.window_id} has shape {matrix.shape}"
                )
            if bool(loader_metadata.get("padded")):
                raise ValueError(f"complete scan window was unexpectedly padded: {window.window_id}")
            zero_axis, zero_rows, zero_cols = _zero_axis(matrix)
            arrays[index, 0] = matrix
            ids.append(window.window_id)
            metadata_rows.append(
                {
                    "window_id": window.window_id,
                    "chrom": window.chrom,
                    "start": window.start,
                    "end": window.end,
                    "center": window.center,
                    "window_bp": window_bp,
                    "target_bin_size": target_bin_size,
                    "array_index": index,
                    "replicate": replicate,
                    "zero_axis": zero_axis,
                    "zero_row_count": zero_rows,
                    "zero_col_count": zero_cols,
                    "cool_path": str(Path(cool_path).resolve()),
                }
            )
        arrays.flush()
        replicate_ids[replicate] = ids

    if replicate_ids["rep1"] != replicate_ids["rep2"]:
        raise RuntimeError("rep1/rep2 window_id order is not aligned")
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(root / "genome_windows_metadata.csv", index=False, encoding="utf-8-sig")
    return metadata

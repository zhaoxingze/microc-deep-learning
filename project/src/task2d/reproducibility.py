"""Shared genome-wide O/E and paired-replicate reproducibility measurements."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.data.load_microc import load_local_matrix


SCALE_WINDOW_BP = {'small': 3200, 'medium': 6400, 'large': 12800}
CANONICAL_WINDOW_BP = 6400
TARGET_BIN_SIZE = 100
MIN_VALID_PIXELS = 10
CORRELATION_GAP = 2


def window_for_scale(scale: str) -> int:
    try:
        return SCALE_WINDOW_BP[str(scale).lower()]
    except KeyError as exc:
        raise ValueError(f'unknown frozen representative scale: {scale}') from exc


def correlation_mask(n_bins: int, *, gap: int = CORRELATION_GAP) -> np.ndarray:
    if n_bins < 2 or gap < 0:
        raise ValueError('invalid upper-triangle mask geometry')
    rows, cols = np.indices((n_bins, n_bins))
    return (rows < cols) & ((cols - rows) > gap)


def read_pair(cool_paths: dict[str, str | Path], chrom: str, center: float,
              window_bp: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if window_bp not in set(SCALE_WINDOW_BP.values()):
        raise ValueError('window is not a frozen Task2A scale')
    matrices = {}
    details = {}
    for rep in ('rep1', 'rep2'):
        matrices[rep], details[rep] = load_local_matrix(
            cool_paths[f'{rep}_cool'], str(chrom), float(center),
            window_bp=window_bp, target_bin_size=TARGET_BIN_SIZE, balance=False,
        )
    if matrices['rep1'].shape != matrices['rep2'].shape or matrices['rep1'].shape != (
        window_bp // TARGET_BIN_SIZE, window_bp // TARGET_BIN_SIZE
    ):
        raise ValueError('replicate window geometry mismatch')
    return matrices['rep1'], matrices['rep2'], details


def normalize_pair(raw1: np.ndarray, raw2: np.ndarray,
                   expected1: np.ndarray, expected2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    one, two = np.asarray(raw1, dtype=float), np.asarray(raw2, dtype=float)
    if one.ndim != 2 or one.shape[0] != one.shape[1] or one.shape != two.shape:
        raise ValueError('paired raw matrices must have the same square shape')
    n = len(one)
    distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    output = []
    for raw, expected in ((one, expected1), (two, expected2)):
        vector = np.asarray(expected, dtype=float)
        if len(vector) < n:
            raise ValueError('genome-wide expected vector is shorter than window distances')
        denominator = vector[distance]
        with np.errstate(divide='ignore', invalid='ignore'):
            oe = raw / denominator
        oe[~np.isfinite(denominator) | (denominator <= 0)] = np.nan
        oe[~np.isfinite(oe)] = np.nan
        output.append(oe)
    return output[0], output[1]


def _is_zero_axis(raw: np.ndarray) -> bool:
    return bool(np.any(np.all(raw == 0, axis=0)) or np.any(np.all(raw == 0, axis=1)))


def _paired_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < MIN_VALID_PIXELS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def measure_pair(raw1: np.ndarray, raw2: np.ndarray,
                 expected1: np.ndarray, expected2: np.ndarray) -> dict:
    oe1, oe2 = normalize_pair(raw1, raw2, expected1, expected2)
    n_bins = oe1.shape[0]
    primary = correlation_mask(n_bins, gap=CORRELATION_GAP)
    full = correlation_mask(n_bins, gap=0)
    pair_finite = np.isfinite(oe1) & np.isfinite(oe2)
    selected = primary & pair_finite
    x, y = oe1[selected], oe2[selected]
    n_valid = int(len(x))
    if n_valid < MIN_VALID_PIXELS:
        reason = 'too_few_valid_pixels'
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        reason = 'zero_variance'
    else:
        reason = ''
    pearson, spearman = _paired_correlations(x, y)
    full_x, full_y = oe1[full & pair_finite], oe2[full & pair_finite]
    full_pearson, full_spearman = _paired_correlations(full_x, full_y)
    if not reason and (not np.isfinite(pearson) or not np.isfinite(spearman)):
        reason = 'undefined_correlation'
    if n_valid:
        pooled = np.concatenate([x, y])
        robust_scale = float(np.percentile(pooled, 75) - np.percentile(pooled, 25))
        nrmse = float(np.sqrt(np.mean((x - y) ** 2)) / (robust_scale + 1e-12))
        norms = float(np.linalg.norm(x) * np.linalg.norm(y))
        cosine = float(np.dot(x, y) / norms) if norms > 0 else np.nan
    else:
        nrmse = cosine = np.nan
    zero1, zero2 = _is_zero_axis(np.asarray(raw1)), _is_zero_axis(np.asarray(raw2))
    finite_fraction = n_valid / int(primary.sum())
    return {
        'pearson': pearson, 'spearman': spearman, 'cosine': cosine, 'nrmse': nrmse,
        'full_upper_pearson': full_pearson, 'full_upper_spearman': full_spearman,
        'n_valid_pixels': n_valid, 'finite_fraction': float(finite_fraction),
        'correlation_valid': not bool(reason), 'invalid_reason': reason,
        'rep1_zero_axis': zero1, 'rep2_zero_axis': zero2,
        'artifact_warning': bool(zero1 or zero2 or finite_fraction < 1),
        'matrix_bins': n_bins,
    }

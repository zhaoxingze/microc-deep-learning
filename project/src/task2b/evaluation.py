"""Task2B blocked reference evaluation and fixed visualization transforms."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


BLOCK_BP = 128_000
BACKGROUND_SAMPLING_BLOCK_BP = 3_200


def build_genomic_groups(rows: pd.DataFrame, *, block_bp: int = BLOCK_BP) -> np.ndarray:
    if block_bp <= 0 or not {"chrom", "center"}.issubset(rows.columns):
        raise ValueError("genomic groups need chrom/center and positive block size")
    positions = pd.to_numeric(rows.center, errors="coerce")
    if positions.isna().any() or (positions < 0).any():
        raise ValueError("invalid reference centers")
    return (rows.chrom.astype(str) + ":" + (positions.astype(int) // block_bp).astype(str)).to_numpy()


def select_background_reference(
    windows: pd.DataFrame, known: pd.DataFrame, *, n: int = 344,
    seed: int = 20_260_920,
) -> pd.DataFrame:
    """Choose deterministic strict-background windows with a feasible 4.8 kb center gap."""
    required = {"window_id", "chrom", "start", "end", "center", "background_candidate"}
    if not required.issubset(windows.columns) or not {"chrom", "start", "end"}.issubset(known.columns):
        raise ValueError("strict background sampling inputs are incomplete")
    if n <= 0:
        raise ValueError("positive background reference count required")
    frame = windows.loc[windows.background_candidate.eq(True)].drop_duplicates("window_id").copy()
    if frame.empty:
        raise ValueError("strict background pool is empty")
    eligible = np.ones(len(frame), dtype=bool)
    frame = frame.reset_index(drop=True)
    for chrom, ids in frame.groupby("chrom").groups.items():
        targets = known.loc[known.chrom.astype(str).eq(str(chrom))]
        if targets.empty:
            continue
        left = frame.loc[ids]
        overlap = (left.start.to_numpy()[:, None] < targets.end.to_numpy()[None, :]) & (
            left.end.to_numpy()[:, None] > targets.start.to_numpy()[None, :]
        )
        eligible[np.asarray(list(ids))] = ~overlap.any(axis=1)
    frame = frame.loc[eligible].copy()
    # The 12.8 kb clean pool cannot contain 344 mutually nonoverlapping windows.
    # A 4.8 kb center gap is the largest feasible fixed gap for this frozen pool.
    minimum_gap = 4_800
    ordered = frame.sort_values(["chrom", "center", "window_id"]).reset_index(drop=True)
    chosen: list[int] = []
    last_center_by_chrom: dict[str, float] = {}
    for index, row in ordered.iterrows():
        center = float(row.center)
        chrom = str(row.chrom)
        if center - last_center_by_chrom.get(chrom, float("-inf")) >= minimum_gap:
            chosen.append(int(index))
            last_center_by_chrom[chrom] = center
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} strict windows meet the {minimum_gap} bp center gap for {n} references")
    rng = np.random.default_rng(seed)
    # Remove redundant nearby picks first, reducing overlapping adjacent pairs.
    while len(chosen) > n:
        ranked = []
        for position in range(1, len(chosen) - 1):
            before = ordered.iloc[chosen[position - 1]]
            middle = ordered.iloc[chosen[position]]
            after = ordered.iloc[chosen[position + 1]]
            if not (before.chrom == middle.chrom == after.chrom):
                continue
            left_gap = float(middle.center - before.center)
            right_gap = float(after.center - middle.center)
            reduction = int(left_gap < 12_800) + int(right_gap < 12_800) - int(left_gap + right_gap < 12_800)
            ranked.append((reduction, -(left_gap + right_gap), float(rng.random()), position))
        if not ranked:
            chosen.pop(int(rng.integers(len(chosen))))
        else:
            chosen.pop(max(ranked)[-1])
    selected = ordered.iloc[chosen].copy().reset_index(drop=True)
    selected["sampling_block"] = selected.chrom.astype(str) + ":" + (
        selected.center.astype(int) // BACKGROUND_SAMPLING_BLOCK_BP
    ).astype(str)
    selected["selection_min_center_gap_bp"] = minimum_gap
    selected.insert(0, "reference_id", [f"BG_{i:04d}" for i in range(1, len(selected) + 1)])
    selected["reference_kind"] = "Background"
    selected["type"] = "Background"
    return selected


def audit_background_reference(windows: pd.DataFrame, known: pd.DataFrame,
                               selected: pd.DataFrame) -> dict[str, int | float]:
    """Quantify unavoidable background-window correlation and AE-train reuse."""
    pool = windows.loc[windows.background_candidate.eq(True)].drop_duplicates("window_id").copy()
    eligible = []
    for chrom, group in pool.groupby("chrom"):
        targets = known.loc[known.chrom.astype(str).eq(str(chrom))]
        if targets.empty:
            eligible.append(group)
            continue
        overlap = (group.start.to_numpy()[:, None] < targets.end.to_numpy()[None, :]) & (
            group.end.to_numpy()[:, None] > targets.start.to_numpy()[None, :]
        )
        eligible.append(group.iloc[np.flatnonzero(~overlap.any(axis=1))])
    clean = pd.concat(eligible, ignore_index=True)
    maximum_disjoint = 0
    for _, group in clean.groupby("chrom"):
        last_end = -1
        for row in group.sort_values(["end", "start", "window_id"]).itertuples(index=False):
            if int(row.start) >= last_end:
                maximum_disjoint += 1
                last_end = int(row.end)
    adjacent_overlaps = 0
    for _, group in selected.groupby("chrom"):
        ordered = group.sort_values(["start", "end"])
        adjacent_overlaps += int((ordered.start.to_numpy()[1:] < ordered.end.to_numpy()[:-1]).sum())
    gaps = []
    for _, group in selected.groupby("chrom"):
        gaps.extend(np.diff(np.sort(group.center.to_numpy(dtype=int))).tolist())
    return {
        "strict_pool_count": len(clean),
        "strict_pool_max_mutually_nonoverlapping_count": maximum_disjoint,
        "selected_count": len(selected),
        "selected_min_center_gap_bp": int(min(gaps)) if gaps else 0,
        "selected_adjacent_interval_overlap_pairs": adjacent_overlaps,
        "selected_reused_ae_train_count": int(selected.split.eq("train").sum()) if "split" in selected else -1,
    }


def fit_reference_scalers(
    reference_blocks: Mapping[str, np.ndarray]
) -> tuple[dict[str, StandardScaler], dict[str, np.ndarray]]:
    if not reference_blocks:
        raise ValueError("reference feature blocks are empty")
    n = len(next(iter(reference_blocks.values())))
    scalers = {}
    transformed = {}
    for name, block in reference_blocks.items():
        values = np.asarray(block, dtype=np.float64)
        if values.ndim != 2 or len(values) != n or not np.isfinite(values).all():
            raise ValueError(f"invalid reference block {name}")
        scaler = StandardScaler().fit(values)
        scalers[name] = scaler
        transformed[name] = scaler.transform(values).astype(np.float32)
    return scalers, transformed


def evaluate_blocked_probes(
    representations: Mapping[str, np.ndarray], labels: np.ndarray,
    metadata: pd.DataFrame, *, n_splits: int = 5, seed: int = 20_260_920,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(labels, dtype=int)
    if len(y) != len(metadata) or set(y) != {0, 1}:
        raise ValueError("probe references require aligned binary labels")
    if not {"start", "end"}.issubset(metadata.columns):
        raise ValueError("probe metadata needs intervals for boundary purging")
    groups = build_genomic_groups(metadata)
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y, groups):
        train_rows = metadata.iloc[train_idx]
        test_rows = metadata.iloc[test_idx]
        keep = np.ones(len(train_idx), dtype=bool)
        for chrom in test_rows.chrom.astype(str).unique():
            train_local = np.flatnonzero(train_rows.chrom.astype(str).eq(chrom).to_numpy())
            test_local = test_rows.loc[test_rows.chrom.astype(str).eq(chrom)]
            if len(train_local) and len(test_local):
                train_windows = train_rows.iloc[train_local]
                crossing = (
                    train_windows.start.to_numpy()[:, None] < test_local.end.to_numpy()[None, :]
                ) & (
                    train_windows.end.to_numpy()[:, None] > test_local.start.to_numpy()[None, :]
                )
                keep[train_local] = ~crossing.any(axis=1)
        folds.append((train_idx[keep], test_idx, int((~keep).sum())))
    per_fold = []
    summaries = []
    for name, matrix in representations.items():
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError(f"invalid probe representation: {name}")
        rows = []
        for fold_index, (train_idx, test_idx, purged_count) in enumerate(folds, start=1):
            train_groups = sorted(set(groups[train_idx]))
            test_groups = sorted(set(groups[test_idx]))
            if set(train_groups).intersection(test_groups):
                raise ValueError("genomic group leaked across probe folds")
            train_rows = metadata.iloc[train_idx]
            test_rows = metadata.iloc[test_idx]
            interval_overlap_count = 0
            for chrom in test_rows.chrom.astype(str).unique():
                left = train_rows.loc[train_rows.chrom.astype(str).eq(chrom)]
                right = test_rows.loc[test_rows.chrom.astype(str).eq(chrom)]
                if len(left) and len(right):
                    interval_overlap_count += int(((left.start.to_numpy()[:, None] < right.end.to_numpy()[None, :]) &
                                                   (left.end.to_numpy()[:, None] > right.start.to_numpy()[None, :])).sum())
            if interval_overlap_count:
                raise ValueError("probe interval leaked across train/test folds")
            if set(y[train_idx]) != {0, 1} or set(y[test_idx]) != {0, 1}:
                raise ValueError(f"GroupKFold fold {fold_index} lacks a probe class")
            probe = LogisticRegression(
                class_weight="balanced", solver="lbfgs", max_iter=2000,
                random_state=seed,
            ).fit(x[train_idx], y[train_idx])
            probabilities = probe.predict_proba(x[test_idx])[:, 1]
            predicted = (probabilities >= .5).astype(int)
            row = {
                "representation": name, "fold": fold_index,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
                "train_groups": train_groups, "test_groups": test_groups,
                "purged_train_count": purged_count,
                "cross_fold_interval_overlap_count": interval_overlap_count,
                "accuracy": float(accuracy_score(y[test_idx], predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], predicted)),
                "roc_auc": float(roc_auc_score(y[test_idx], probabilities)),
                "pr_auc": float(average_precision_score(y[test_idx], probabilities)),
            }
            rows.append(row)
            per_fold.append(row)
        summary = {"representation": name, "n_folds": len(rows)}
        for metric in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc"):
            values = np.asarray([row[metric] for row in rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(per_fold)


def fit_visualization_models(
    reference: np.ndarray, candidates: np.ndarray, *, seed: int = 20_260_920,
):
    import umap

    joined = np.vstack([reference, candidates]).astype(np.float32)
    if joined.ndim != 2 or len(joined) < 4 or not np.isfinite(joined).all():
        raise ValueError("PCA/UMAP require finite aligned reference and candidate embeddings")
    pca = PCA(n_components=2, svd_solver="full", random_state=seed)
    pca_xy = pca.fit_transform(joined)
    umap_model = umap.UMAP(
        n_components=2, n_neighbors=min(15, len(joined) - 1),
        min_dist=.1, metric="euclidean", random_state=seed, n_jobs=1,
    )
    umap_xy = umap_model.fit_transform(joined)
    return pca, pca_xy, umap_model, umap_xy

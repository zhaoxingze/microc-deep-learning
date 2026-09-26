"""Post-fit descriptive annotations, centroids, medoids, and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.task2c.clustering import PrimaryFit


KNOWN_TYPES = ("CHIN", "OPCID", "CHID")
SUMMARY_COLUMNS = (
    "cluster_id", "centroid_index", "total_members", "candidate_count", "known_count",
    "CHIN_count", "OPCID_count", "CHID_count", "known_fraction", "candidate_fraction",
    "known_containing", "dominant_known_type", "known_type_purity", "cluster_annotation",
    "candidate_known_center_overlap_count", "candidate_known_interval_overlap_count",
    "candidate_known_overlap_count", "candidate_unannotated_count",
    "mean_membership_probability", "median_membership_probability",
    "representative_sample_id", "representative_region_id", "candidate_representative_region_id",
    "novel_like_cluster_candidate", "Task2D_priority", "background_nearest_count",
    "background_min_distance", "background_median_distance",
)


@dataclass(frozen=True)
class AnalysisResult:
    membership: pd.DataFrame
    clusters: pd.DataFrame
    summary: pd.DataFrame
    centroids: np.ndarray
    representatives: pd.DataFrame
    background_proximity: pd.DataFrame


def analyze_clusters(primary: PrimaryFit, candidate_meta: pd.DataFrame,
                     known_meta: pd.DataFrame, background: np.ndarray,
                     background_meta: pd.DataFrame) -> AnalysisResult:
    """Attach labels only after the immutable vector-only primary fit exists."""
    nc, nk = primary.n_candidate, primary.n_known
    if len(candidate_meta) != nc or len(known_meta) != nk or len(primary.labels) != nc + nk:
        raise ValueError("post-fit candidate/Known alignment failed")
    if len(background) != len(background_meta) or background.ndim != 2 or background.shape[1] != primary.fused.shape[1]:
        raise ValueError("background diagnostic alignment failed")
    if candidate_meta.region_id.duplicated().any() or known_meta.reference_id.duplicated().any():
        raise ValueError("duplicate post-fit sample ID")
    required_candidate = {"region_id", "chrom", "start", "end", "center",
                          "known_center_overlap", "known_interval_overlap", "known_types"}
    required_known = {"reference_id", "type", "chrom", "start", "end", "center"}
    if not required_candidate.issubset(candidate_meta) or not required_known.issubset(known_meta):
        raise ValueError("post-fit metadata schema incomplete")
    if known_meta.type.isna().any() or not known_meta.type.isin(KNOWN_TYPES).all():
        raise ValueError("Known type labels invalid for post-fit annotation")
    cm = candidate_meta.reset_index(drop=True)
    km = known_meta.reset_index(drop=True)
    cm = cm.copy()
    for column in ("known_center_overlap", "known_interval_overlap"):
        text_values = cm[column].astype(str).str.lower()
        if not text_values.isin(["true", "false"]).all():
            raise ValueError(f"invalid candidate {column} boolean metadata")
        cm[column] = text_values.eq("true")
    member = pd.DataFrame({
        "sample_id": pd.concat([cm.region_id, km.reference_id], ignore_index=True),
        "sample_source": ["candidate"] * nc + ["known"] * nk,
        "region_id": list(cm.region_id) + [""] * nk,
        "known_type": [""] * nc + list(km.type),
        "chrom": pd.concat([cm.chrom, km.chrom], ignore_index=True),
        "start": pd.concat([cm.start, km.start], ignore_index=True),
        "end": pd.concat([cm.end, km.end], ignore_index=True),
        "center": pd.concat([cm.center, km.center], ignore_index=True),
        "cluster_id": primary.labels,
        "membership_probability": primary.probabilities,
        "outlier_score": primary.outlier_scores,
    })
    if member.sample_id.duplicated().any():
        raise ValueError("candidate/Known sample IDs collide")
    known_projected = primary.projected[nc:]
    candidate_projected = primary.projected[:nc]
    distances = cdist(candidate_projected, known_projected)
    nearest_index = distances.argmin(axis=1)
    nearest = pd.DataFrame({
        "nearest_known_id": km.reference_id.to_numpy()[nearest_index],
        "nearest_known_type": km.type.to_numpy()[nearest_index],
        "nearest_known_distance": distances[np.arange(nc), nearest_index],
    })
    for known_type in KNOWN_TYPES:
        indexes = np.flatnonzero(km.type.eq(known_type).to_numpy())
        nearest[f"nearest_{known_type}_distance"] = (
            distances[:, indexes].min(axis=1) if len(indexes) else np.full(nc, np.nan)
        )

    cluster_ids = sorted(set(map(int, primary.labels)) - {-1})
    centroids = np.empty((len(cluster_ids), primary.fused.shape[1]), dtype=np.float64)
    summaries = []
    representatives = []
    representative_distance = np.full(nc, np.nan)
    for centroid_index, cluster_id in enumerate(cluster_ids):
        member_indexes = np.flatnonzero(primary.labels == cluster_id)
        candidate_indexes = member_indexes[member_indexes < nc]
        known_indexes = member_indexes[member_indexes >= nc]
        centroid = primary.fused[member_indexes].mean(axis=0)
        centroids[centroid_index] = centroid
        distance_to_centroid = np.linalg.norm(primary.fused[member_indexes] - centroid, axis=1)
        medoid_index = int(member_indexes[np.argmin(distance_to_centroid)])
        candidate_representative_index = (
            int(candidate_indexes[np.argmin(np.linalg.norm(primary.fused[candidate_indexes] - centroid, axis=1))])
            if len(candidate_indexes) else None
        )
        representative_distance[candidate_indexes] = np.linalg.norm(primary.fused[candidate_indexes] - centroid, axis=1)
        counts = {known_type: int(km.iloc[known_indexes - nc].type.eq(known_type).sum()) for known_type in KNOWN_TYPES}
        known_count = len(known_indexes)
        dominant = max(KNOWN_TYPES, key=lambda value: counts[value]) if known_count else ""
        purity = counts[dominant] / known_count if known_count else np.nan
        annotation = f"{dominant}-like" if known_count and purity >= .5 else (
            "mixed-known" if known_count else "unannotated"
        )
        candidate_rows = cm.iloc[candidate_indexes]
        center_count = int(candidate_rows.known_center_overlap.astype(bool).sum())
        interval_count = int(candidate_rows.known_interval_overlap.astype(bool).sum())
        overlap_count = int((candidate_rows.known_center_overlap.astype(bool) |
                             candidate_rows.known_interval_overlap.astype(bool)).sum())
        novel = bool(len(candidate_indexes) >= 5 and known_count == 0 and center_count == 0 and interval_count == 0)
        priority = "High" if novel else ("Low" if known_count and annotation != "mixed-known" else "Medium")
        if not len(candidate_indexes) and known_count:
            priority = "Low"
        medoid = member.iloc[medoid_index]
        candidate_region = cm.region_id.iloc[candidate_representative_index] if candidate_representative_index is not None else ""
        representatives.append({
            "cluster_id": cluster_id, "representative_sample_id": medoid.sample_id,
            "source": medoid.sample_source, "region_id": medoid.region_id,
            "known_type": medoid.known_type, "distance_to_centroid": float(np.min(distance_to_centroid)),
            "candidate_representative_region_id": candidate_region,
        })
        summaries.append({
            "cluster_id": cluster_id, "centroid_index": centroid_index,
            "total_members": len(member_indexes), "candidate_count": len(candidate_indexes),
            "known_count": known_count, **{f"{key}_count": value for key, value in counts.items()},
            "known_fraction": known_count / len(member_indexes),
            "candidate_fraction": len(candidate_indexes) / len(member_indexes),
            "known_containing": bool(known_count), "dominant_known_type": dominant,
            "known_type_purity": purity, "cluster_annotation": annotation,
            "candidate_known_center_overlap_count": center_count,
            "candidate_known_interval_overlap_count": interval_count,
            "candidate_known_overlap_count": overlap_count,
            "candidate_unannotated_count": len(candidate_indexes) - overlap_count,
            "mean_membership_probability": float(primary.probabilities[member_indexes].mean()),
            "median_membership_probability": float(np.median(primary.probabilities[member_indexes])),
            "representative_sample_id": medoid.sample_id,
            "representative_region_id": medoid.region_id,
            "candidate_representative_region_id": candidate_region,
            "novel_like_cluster_candidate": novel,
            "Task2D_priority": priority,
            "background_nearest_count": 0, "background_min_distance": np.nan,
            "background_median_distance": np.nan,
        })

    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    if len(cluster_ids):
        background_projected = primary.pca.transform(background)
        centroid_projected = primary.pca.transform(centroids)
        bg_distances = cdist(background_projected, centroid_projected)
        bg_nearest = bg_distances.argmin(axis=1)
        bg_ids = np.asarray(cluster_ids)[bg_nearest]
        bg_distance = bg_distances[np.arange(len(background)), bg_nearest]
        for cluster_id in cluster_ids:
            values = bg_distance[bg_ids == cluster_id]
            mask = summary.cluster_id.eq(cluster_id)
            summary.loc[mask, "background_nearest_count"] = len(values)
            if len(values):
                summary.loc[mask, "background_min_distance"] = float(values.min())
                summary.loc[mask, "background_median_distance"] = float(np.median(values))
    else:
        bg_ids = np.full(len(background), -1)
        bg_distance = np.full(len(background), np.nan)
    background_proximity = pd.DataFrame({
        "reference_id": background_meta.reference_id.to_numpy(),
        "nearest_cluster_id": bg_ids,
        "distance_to_nearest_cluster_centroid": bg_distance,
    })

    clusters = pd.concat([cm, member.iloc[:nc][["cluster_id", "membership_probability", "outlier_score"]], nearest], axis=1)
    clusters["candidate_status"] = np.where(clusters.cluster_id.eq(-1), "noise", "clustered")
    annotations = summary.set_index("cluster_id") if len(summary) else None
    clusters["cluster_annotation"] = clusters.cluster_id.map(
        annotations.cluster_annotation if annotations is not None else {}
    ).fillna("noise")
    clusters["novel_like_cluster_candidate"] = clusters.cluster_id.map(
        annotations.novel_like_cluster_candidate if annotations is not None else {}
    ).eq(True)
    clusters["representative_distance"] = representative_distance
    return AnalysisResult(member, clusters, summary, centroids,
                          pd.DataFrame(representatives, columns=[
                              "cluster_id", "representative_sample_id", "source", "region_id", "known_type",
                              "distance_to_centroid", "candidate_representative_region_id",
                          ]), background_proximity)

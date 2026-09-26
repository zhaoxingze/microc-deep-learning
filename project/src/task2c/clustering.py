"""Label-blind 95%-variance PCA and pre-registered sklearn HDBSCAN."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


PRIMARY_PARAMETERS = {
    "min_cluster_size": 5,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    "cluster_selection_epsilon": 0.0,
    "allow_single_cluster": False,
    "copy": True,
}
SENSITIVITY_PARAMETERS = ((8, 5), (5, 8))


@dataclass(frozen=True)
class PrimaryFit:
    pca: PCA
    projected: np.ndarray
    fused: np.ndarray
    clusterer: HDBSCAN
    labels: np.ndarray
    probabilities: np.ndarray
    outlier_scores: np.ndarray
    n_candidate: int
    n_known: int


def fit_clustering_pca(values: np.ndarray) -> tuple[PCA, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 4 or array.shape[1] < 2 or not np.isfinite(array).all():
        raise ValueError("clustering PCA needs finite two-dimensional feature rows")
    spectrum = PCA(svd_solver="full").fit(array)
    cumulative = np.cumsum(spectrum.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, 0.95 - 1e-12, side="left") + 1)
    pca = PCA(n_components=n_components, svd_solver="full")
    projected = pca.fit_transform(array)
    if float(pca.explained_variance_ratio_.sum()) < 0.95 - 1e-10:
        raise ValueError("clustering PCA did not preserve 95% variance")
    return pca, projected


def fit_primary(candidates: np.ndarray, known: np.ndarray) -> PrimaryFit:
    candidate = np.asarray(candidates, dtype=np.float64)
    reference = np.asarray(known, dtype=np.float64)
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape[1] != reference.shape[1]:
        raise ValueError("candidate and Known embedding dimension mismatch")
    if len(candidate) == 0 or len(reference) == 0 or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("nonempty finite candidate and Known vectors required")
    fused = np.vstack([candidate, reference])
    pca, projected = fit_clustering_pca(fused)
    clusterer = HDBSCAN(**PRIMARY_PARAMETERS).fit(projected)
    labels = np.asarray(clusterer.labels_, dtype=int)
    probabilities = np.asarray(clusterer.probabilities_, dtype=float)
    outlier = np.asarray(getattr(clusterer, "outlier_scores_", np.full(len(labels), np.nan)), dtype=float)
    if len(labels) != len(fused) or len(probabilities) != len(fused) or not np.isfinite(probabilities).all():
        raise ValueError("HDBSCAN membership alignment failed")
    return PrimaryFit(pca, projected, fused, clusterer, labels, probabilities,
                      outlier, len(candidate), len(reference))


def run_sensitivity(primary: PrimaryFit) -> pd.DataFrame:
    rows = []
    for min_cluster_size, min_samples in SENSITIVITY_PARAMETERS:
        model = HDBSCAN(**{**PRIMARY_PARAMETERS, "min_cluster_size": min_cluster_size,
                           "min_samples": min_samples}).fit(primary.projected)
        labels = np.asarray(model.labels_, dtype=int)
        rows.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_count": int(len(set(labels) - {-1})),
            "noise_count": int((labels == -1).sum()),
            "noise_fraction": float((labels == -1).mean()),
            "adjusted_rand_index_to_primary": float(adjusted_rand_score(primary.labels, labels)),
        })
    return pd.DataFrame(rows)

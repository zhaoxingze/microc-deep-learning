import numpy as np
import pandas as pd
import pytest

from src.task2c.clustering import PrimaryFit, fit_clustering_pca, fit_primary, run_sensitivity
from src.task2c.analysis import analyze_clusters
from sklearn.decomposition import PCA


def test_pca_chooses_minimum_components_reaching_95_percent_without_labels():
    one_axis = np.array([[-10., 0.], [10., 0.], [0., -1.], [0., 1.]])
    pca, projected = fit_clustering_pca(one_axis)
    assert projected.shape == (4, 1)
    assert pca.explained_variance_ratio_.sum() >= .95
    two_axes = np.array([[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]])
    assert fit_clustering_pca(two_axes)[1].shape == (4, 2)


def test_pca_exact_95_percent_boundary_keeps_only_one_component():
    values = np.array([[-1., 0.], [1., 0.]] * 19 +
                      [[0., -1.], [0., 1.]])
    pca, projected = fit_clustering_pca(values)
    assert projected.shape == (40, 1)
    assert np.isclose(pca.explained_variance_ratio_.sum(), .95, atol=1e-12)


def test_primary_hdbscan_sees_only_candidate_and_known_vectors():
    rng = np.random.default_rng(21)
    candidates = np.r_[rng.normal(0, .05, size=(10, 4)), rng.normal(8, .05, size=(10, 4))]
    known = np.r_[rng.normal(0, .05, size=(5, 4)), rng.normal(8, .05, size=(5, 4))]
    fit = fit_primary(candidates, known)
    assert len(fit.labels) == 30
    assert fit.projected.shape[0] == 30
    assert fit.clusterer.min_cluster_size == fit.clusterer.min_samples == 5
    assert fit.clusterer.cluster_selection_method == "eom"
    assert fit.clusterer.metric == "euclidean"
    assert np.isfinite(fit.probabilities).all()
    assert len(set(fit.labels) - {-1}) >= 2
    audit = run_sensitivity(fit)
    assert set(zip(audit.min_cluster_size, audit.min_samples)) == {(8, 5), (5, 8)}
    assert audit.adjusted_rand_index_to_primary.notna().all()


def test_primary_rejects_bad_alignment_and_nonfinite_vectors():
    with pytest.raises(ValueError, match="dimension"):
        fit_primary(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="finite"):
        fit_primary(np.array([[np.nan, 0.], [1., 0.]]), np.ones((2, 2)))


def test_known_labels_and_background_not_in_primary_fit_signature():
    import inspect

    assert list(inspect.signature(fit_primary).parameters) == ["candidates", "known"]


def synthetic_postfit():
    candidate = np.array([[0., i * .01] for i in range(5)] + [[5., 0.], [8., 0.]])
    known = np.array([[5., .1], [5., .2], [5., .3], [12., 0.], [12., .1], [12., .2]])
    fused = np.vstack([candidate, known])
    pca = PCA(n_components=2).fit(fused)
    labels = np.array([0, 0, 0, 0, 0, 1, -1, 1, 1, 1, 2, 2, 2])
    fit = PrimaryFit(pca, pca.transform(fused), fused, None, labels,
                     np.array([.8] * 6 + [0.] + [.7] * 6), np.full(13, np.nan), 7, 6)
    candidate_meta = pd.DataFrame({
        "region_id": [f"R{i}" for i in range(7)], "chrom": ["chr"] * 7,
        "start": np.arange(7) * 1000, "end": np.arange(7) * 1000 + 500,
        "center": np.arange(7) * 1000 + 250, "length_bp": [500] * 7,
        "known_center_overlap": [False] * 7, "known_interval_overlap": [False] * 7,
        "known_types": [""] * 7,
    })
    known_meta = pd.DataFrame({
        "reference_id": [f"K{i}" for i in range(6)], "type": ["CHIN", "CHIN", "OPCID", "CHIN", "OPCID", "CHID"],
        "chrom": ["chr"] * 6, "start": np.arange(6) * 1000,
        "end": np.arange(6) * 1000 + 500, "center": np.arange(6) * 1000 + 250,
    })
    background_meta = pd.DataFrame({"reference_id": ["BG1", "BG2"]})
    background = np.array([[0., 0.], [20., 0.]])
    return fit, candidate_meta, known_meta, background, background_meta


def test_postfit_annotation_preserves_noise_and_identifies_only_clean_novel_like():
    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    result = analyze_clusters(fit, cmeta, kmeta, background, bgmeta)
    summary = result.summary.set_index("cluster_id")
    assert summary.loc[0, "candidate_count"] == 5
    assert summary.loc[0, "known_count"] == 0
    assert bool(summary.loc[0, "novel_like_cluster_candidate"])
    assert summary.loc[1, "cluster_annotation"] == "CHIN-like"
    assert summary.loc[1, "CHIN_count"] == 2
    assert summary.loc[1, "OPCID_count"] == 1
    assert summary.loc[2, "cluster_annotation"] == "mixed-known"
    assert result.clusters.loc[result.clusters.region_id.eq("R6"), "candidate_status"].iloc[0] == "noise"
    assert not bool(result.clusters.loc[result.clusters.region_id.eq("R6"), "novel_like_cluster_candidate"].iloc[0])
    assert result.centroids.shape == (3, 2)
    assert set(result.representatives.cluster_id) == {0, 1, 2}
    for row in result.summary.itertuples(index=False):
        members = fit.fused[fit.labels == row.cluster_id]
        centroid = members.mean(axis=0)
        assert np.allclose(result.centroids[row.centroid_index], centroid)
        distances = np.linalg.norm(members - centroid, axis=1)
        representative = result.representatives.set_index('cluster_id').loc[row.cluster_id]
        sample_ids = result.membership.loc[fit.labels == row.cluster_id, 'sample_id'].to_numpy()
        assert representative.representative_sample_id == sample_ids[np.argmin(distances)]
    assert np.isfinite(result.clusters.nearest_known_distance).all()
    assert len(result.background_proximity) == 2


def test_genomic_known_overlap_disqualifies_no_known_cluster():
    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    cmeta.loc[0, "known_interval_overlap"] = True
    result = analyze_clusters(fit, cmeta, kmeta, background, bgmeta)
    row = result.summary.set_index("cluster_id").loc[0]
    assert row.known_count == 0
    assert row.candidate_known_overlap_count == 1
    assert not bool(row.novel_like_cluster_candidate)


def test_all_noise_keeps_candidate_rows_and_empty_cluster_summary():
    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    noisy = PrimaryFit(fit.pca, fit.projected, fit.fused, None,
                       np.full(len(fit.labels), -1), np.zeros(len(fit.labels)),
                       fit.outlier_scores, fit.n_candidate, fit.n_known)
    result = analyze_clusters(noisy, cmeta, kmeta, background, bgmeta)
    assert result.summary.empty
    assert len(result.clusters) == 7
    assert result.clusters.candidate_status.eq("noise").all()


def test_known_type_checked_only_when_postfit_annotations_begin():
    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    kmeta.loc[0, "type"] = ""
    with pytest.raises(ValueError, match="Known type"):
        analyze_clusters(fit, cmeta, kmeta, background, bgmeta)


def test_string_boolean_overlap_metadata_is_not_truthy_by_accident():
    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    cmeta["known_center_overlap"] = "False"
    cmeta["known_interval_overlap"] = "False"
    result = analyze_clusters(fit, cmeta, kmeta, background, bgmeta)
    assert bool(result.summary.set_index("cluster_id").loc[0, "novel_like_cluster_candidate"])
    cmeta.loc[0, "known_interval_overlap"] = "True"
    changed = analyze_clusters(fit, cmeta, kmeta, background, bgmeta)
    assert not bool(changed.summary.set_index("cluster_id").loc[0, "novel_like_cluster_candidate"])

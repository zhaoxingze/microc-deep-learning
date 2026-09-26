import numpy as np
import pandas as pd
import pytest

from src.task2b.evaluation import (
    audit_background_reference,
    build_genomic_groups,
    evaluate_blocked_probes,
    fit_reference_scalers,
    fit_visualization_models,
    select_background_reference,
)
from src.task2b.pipeline import Task2BPaths, _verify_prior_output_hash, execute_task2b


def test_background_reference_is_unique_by_sampling_block_and_known_clean():
    rows = pd.DataFrame([
        dict(window_id=f"W{i}", chrom="MG1655", start=i * 3200, end=i * 3200 + 12800,
             center=i * 3200 + 6400, background_candidate=True)
        for i in range(12)
    ])
    known = pd.DataFrame([dict(chrom="MG1655", start=0, end=1000)])
    selected = select_background_reference(rows, known, n=5, seed=20260920)
    assert len(selected) == 5
    assert selected.center.astype(int).floordiv(3200).is_unique
    assert np.diff(np.sort(selected.center.to_numpy())).min() >= 4800
    assert not ((selected.start < 1000) & (selected.end > 0)).any()
    audit = audit_background_reference(rows.assign(split="train"), known, selected.assign(split="train"))
    assert audit["selected_count"] == 5
    assert audit["selected_reused_ae_train_count"] == 5
    assert audit["strict_pool_max_mutually_nonoverlapping_count"] < 5


def test_reference_scalers_fit_only_unlabeled_reference_rows():
    ref = {"cnn": np.array([[0.], [2.]]), "small": np.array([[1.], [3.]]),
           "medium": np.array([[2.], [4.]]), "large": np.array([[3.], [5.]])}
    candidate = {name: values + 100 for name, values in ref.items()}
    scalers, transformed_ref = fit_reference_scalers(ref)
    transformed_candidate = {name: scalers[name].transform(candidate[name]) for name in ref}
    assert all(np.allclose(values.mean(axis=0), 0) for values in transformed_ref.values())
    assert transformed_candidate["cnn"].min() > 90


def test_grouped_probe_has_zero_group_overlap_and_metric_schema():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(10), 4)
    y = np.tile([0, 1, 0, 1], 10)
    x = rng.normal(size=(40, 5)) + y[:, None] * .5
    centers = groups * 128000 + 1000
    centers[3] = 127000
    metadata = pd.DataFrame({"chrom": "MG1655", "center": centers,
                             "start": centers - 6400, "end": centers + 6400})
    table, folds = evaluate_blocked_probes({"cnn": x, "ae": x, "fused": x}, y, metadata, n_splits=5)
    assert set(table.representation) == {"cnn", "ae", "fused"}
    assert {"accuracy_mean", "balanced_accuracy_mean", "roc_auc_mean", "pr_auc_mean"}.issubset(table)
    assert all(not set(row.train_groups).intersection(row.test_groups) for row in folds.itertuples())
    assert folds.purged_train_count.max() > 0
    assert folds.cross_fold_interval_overlap_count.eq(0).all()


def test_pca_and_umap_use_fixed_seed_and_aligned_rows():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(20, 5))
    candidates = rng.normal(size=(4, 5))
    pca, pca_xy, umap_model, umap_xy = fit_visualization_models(reference, candidates, seed=20260920)
    assert pca_xy.shape == umap_xy.shape == (24, 2)
    assert umap_model.random_state == 20260920
    assert pca.explained_variance_ratio_.shape == (2,)
    assert np.allclose(pca_xy, pca.transform(np.vstack([reference, candidates])))


def test_genomic_group_ids_are_coordinate_based():
    data = pd.DataFrame({"chrom": ["MG1655"] * 3, "center": [0, 127999, 128000]})
    assert build_genomic_groups(data).tolist() == ["MG1655:0", "MG1655:0", "MG1655:1"]


def test_smoke_full_paths_are_isolated(tmp_path):
    full = Task2BPaths(tmp_path, "full")
    smoke = Task2BPaths(tmp_path, "smoke")
    assert full.output_root == tmp_path / "outputs" / "task2b"
    assert smoke.output_root == tmp_path / "outputs" / "pipeline_runs" / "task2b_smoke" / "outputs" / "task2b"
    assert full.output_root != smoke.output_root


def test_dry_run_with_missing_frozen_proposals_does_not_write(tmp_path):
    with pytest.raises(FileNotFoundError):
        execute_task2b(tmp_path, mode="full", dry_run=True)
    assert not (tmp_path / "outputs").exists()


def test_prior_output_hash_rejects_changed_expected_array(tmp_path):
    from src.task1.manifest import sha256_file

    path = tmp_path / "expected.npz"
    path.write_bytes(b"old")
    manifest = {"outputs": {str(path): {"sha256": sha256_file(path)}}}
    assert _verify_prior_output_hash(manifest, path) == sha256_file(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        _verify_prior_output_hash(manifest, path)

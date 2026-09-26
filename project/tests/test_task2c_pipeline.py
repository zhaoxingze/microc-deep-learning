import numpy as np
import pandas as pd
import pytest

from src.task2c.inputs import Task2CInputs, Task2CPaths, validate_alignment, freeze_protocol
from src.task2c.pipeline import _smoke_indices, build_completion, build_frozen_protocol, execute_task2c
from src.task2c.reporting import paired_oe_matrix, plot_cluster_overview, write_cluster_figures
from tests.test_task2c_clustering import synthetic_postfit
from src.task2c.analysis import analyze_clusters
from src.task1.manifest import sha256_file


def test_alignment_rejects_duplicate_ids_and_mismatched_rows():
    candidate = np.zeros((2, 256), dtype=np.float32)
    reference = np.zeros((3, 256), dtype=np.float32)
    cmeta = pd.DataFrame({"region_id": ["A", "B"]})
    rmeta = pd.DataFrame({"reference_id": ["K1", "K2", "BG1"],
                          "reference_kind": ["Known", "Known", "Background"],
                          "type": ["CHIN", "CHID", "Background"]})
    validate_alignment(candidate, reference, cmeta, rmeta)
    with pytest.raises(ValueError, match="duplicate"):
        validate_alignment(candidate, reference, cmeta.assign(region_id=["A", "A"]), rmeta)
    with pytest.raises(ValueError, match="alignment"):
        validate_alignment(candidate[:1], reference, cmeta, rmeta)
    validate_alignment(candidate, reference, cmeta, rmeta.assign(type=["CHIN", "", "Background"]))


def test_frozen_protocol_cannot_be_silently_changed(tmp_path):
    path = tmp_path / "frozen_task2c_protocol.json"
    freeze_protocol(path, {"min_cluster_size": 5})
    freeze_protocol(path, {"min_cluster_size": 5})
    with pytest.raises(ValueError, match="frozen"):
        freeze_protocol(path, {"min_cluster_size": 8})


def test_smoke_and_full_paths_are_isolated(tmp_path):
    assert Task2CPaths(tmp_path, "full").output_root == tmp_path / "outputs" / "task2c"
    assert Task2CPaths(tmp_path, "smoke").output_root == tmp_path / "outputs" / "pipeline_runs" / "task2c_smoke" / "outputs" / "task2c"


def test_dry_run_missing_inputs_does_not_write(tmp_path):
    with pytest.raises(FileNotFoundError):
        execute_task2c(tmp_path, mode="full", dry_run=True)
    assert not (tmp_path / "outputs").exists()


def test_paired_oe_heatmap_uses_same_background_expected_and_clips():
    one = np.full((2, 2), 4., dtype=float)
    two = np.full((2, 2), 8., dtype=float)
    result = paired_oe_matrix(one, two, np.array([2., 2.]), np.array([4., 4.]), 3., 3.)
    assert result.shape == (2, 2)
    assert np.allclose(result, np.log1p(2.))


def test_cluster_plots_and_novel_like_heatmap_title(tmp_path):
    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    analysis = analyze_clusters(fit, cmeta, kmeta, background, bgmeta)
    ids = list(cmeta.region_id) + list(kmeta.reference_id)
    umap = pd.DataFrame({"point_id": ids, "x": np.arange(len(ids)), "y": np.arange(len(ids)) * .1})
    plots = write_cluster_figures(fit, analysis, umap, tmp_path)
    assert {path.name for path in plots} == {
        "pca_clusters.png", "pca_known_composition.png", "umap_clusters.png", "umap_known_composition.png"
    }
    assert all(path.stat().st_size > 1000 for path in plots)
    image = plot_cluster_overview(
        [np.ones((64, 64))] * 5, ["R0"] * 5,
        np.array([[0., 0.]] * 5), np.array([[1., 1.]] * 5),
        tmp_path / "novel_like.png", title="Novel-like candidate cluster 0",
        vmax=np.log1p(3.), coordinate_text="MG1655:100-200",
    )
    assert image.stat().st_size > 1000


def test_completion_requires_aligned_primary_outputs_and_no_task2d():
    import json

    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    analysis = analyze_clusters(fit, cmeta, kmeta, background, bgmeta)
    complete = build_completion(fit, analysis, n_figures=4, n_overviews=3)
    assert complete["task2c_complete"] is True
    assert complete["ready_for_task2d"] is True
    assert complete["known_labels_used_after_clustering_only"] is True
    assert complete["umap_used_for_clustering"] is False
    assert complete["task2d_executed"] is False
    json.dumps(complete)
    broken = build_completion(fit, analysis, n_figures=0, n_overviews=3)
    assert broken["task2c_complete"] is False


def test_frozen_protocol_records_only_input_hashes_and_fixed_algorithm():
    protocol = build_frozen_protocol("full", {"candidate_embeddings": "abc"}, "codehash")
    assert protocol["clustering_pca"]["variance_fraction"] == .95
    assert protocol["primary_hdbscan"]["min_cluster_size"] == 5
    assert protocol["primary_hdbscan"]["min_samples"] == 5
    assert protocol["known_labels_used_for_fit"] is False
    assert protocol["umap_used_for_clustering"] is False
    assert "known_type" not in str(protocol)


def test_smoke_sampling_does_not_consult_known_type_labels():
    from types import SimpleNamespace

    x = SimpleNamespace(candidates=np.zeros((30, 2)), known=np.zeros((70, 2)),
                        background=np.zeros((70, 2)),
                        known_meta=pd.DataFrame({"type": ["CHIN"] * 70}))
    before = _smoke_indices(x)
    x.known_meta["type"] = ["CHID"] * 35 + ["OPCID"] * 35
    after = _smoke_indices(x)
    assert all(np.array_equal(left, right) for left, right in zip(before, after))


def test_successful_pipeline_fit_is_label_blind_and_leaves_upstream_untouched(tmp_path, monkeypatch):
    import json
    import src.task2c.pipeline as pipeline
    from src.task2.reporting import validate_task2a_outputs

    fit, cmeta, kmeta, background, bgmeta = synthetic_postfit()
    candidates, known = fit.fused[:7].copy(), fit.fused[7:].copy()
    source_keys = ('task2a_protocol', 'task2a_candidates', 'task2b_manifest',
                   'candidate_embeddings', 'reference_embeddings', 'feature_scaler')
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    paths = {key: upstream / f'{key}.txt' for key in source_keys}
    for key, path in paths.items():
        path.write_text(f'frozen {key}', encoding='utf-8')
    before = {key: sha256_file(path) for key, path in paths.items()}
    ids = list(cmeta.region_id) + list(kmeta.reference_id)
    umap = pd.DataFrame({'point_id': ids, 'x': np.arange(len(ids), dtype=float),
                         'y': np.arange(len(ids), dtype=float) / 2})
    observed = []
    real_fit = pipeline.fit_primary

    def spy_fit(candidate_vectors, known_vectors):
        observed.append((candidate_vectors.copy(), known_vectors.copy()))
        return real_fit(candidate_vectors, known_vectors)

    def stub_heatmaps(primary, analysis, umap_coordinates, source_paths, output_dir):
        result = []
        for cluster_id in analysis.summary.cluster_id:
            path = output_dir / 'cluster_overviews' / f'cluster_{cluster_id:03d}_overview.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'heatmap boundary stub')
            result.append(path)
        return result

    monkeypatch.setattr(pipeline, 'fit_primary', spy_fit)
    monkeypatch.setattr(pipeline, 'write_cluster_heatmaps', stub_heatmaps)
    for run_number in (0, 1):
        candidate_meta = cmeta.copy()
        known_meta = kmeta.copy()
        diagnostic_background = background.copy()
        rep1, rep2 = candidates + 1, candidates - 1
        if run_number:
            candidate_meta['best_p'] = 1e-200
            known_meta['type'] = 'CHID'
            diagnostic_background += 999
            rep1, rep2 = candidates + 100, candidates - 100
        inputs = Task2CInputs(
            candidates, known, diagnostic_background, rep1, rep2,
            candidate_meta, known_meta, bgmeta.copy(), umap,
            paths, before,
        )
        monkeypatch.setattr(pipeline, 'load_task2c_inputs', lambda root, value=inputs: value)
        root = tmp_path / f'run_{run_number}'
        result = execute_task2c(root, mode='full')
        assert result.manifest_path.is_file()
        payload = validate_task2a_outputs(result.manifest_path)
        assert payload['statistics']['n_candidates'] == 7
        assert payload['statistics']['n_known'] == 6
        completion = json.loads((result.output_root / 'task2c_completion.json').read_text(encoding='utf-8'))
        assert completion['task2c_complete'] is True
        assert not (root / 'outputs' / 'task2d').exists()
        assert {key: sha256_file(path) for key, path in paths.items()} == before
    assert len(observed) == 2
    for fitted_candidates, fitted_known in observed:
        assert np.array_equal(fitted_candidates, candidates)
        assert np.array_equal(fitted_known, known)

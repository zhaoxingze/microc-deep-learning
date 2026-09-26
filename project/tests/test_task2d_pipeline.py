import json

import numpy as np
import pandas as pd
import pytest

from src.task1.manifest import sha256_file
from src.task2d.inputs import (
    Task2DInputs, Task2DPaths, freeze_protocol, validate_alignment, verify_completion,
    verify_source_hashes, source_paths, validate_protocol_rules,
)
from src.task2d.pipeline import execute_task2d
from src.task2.reporting import validate_task2a_outputs


def test_task2d_alignment_rejects_reordered_and_duplicate_candidate_ids():
    task2a = pd.DataFrame({'region_id': ['R1', 'R2']})
    task2b = pd.DataFrame({'region_id': ['R1', 'R2']})
    task2c = pd.DataFrame({'region_id': ['R1', 'R2'], 'cluster_id': [1, -1]})
    membership = pd.DataFrame({'sample_source': ['candidate', 'candidate', 'known'],
                               'region_id': ['R1', 'R2', ''], 'sample_id': ['R1', 'R2', 'K1'],
                               'cluster_id': [1, -1, 1]})
    known = pd.DataFrame({'reference_kind': ['Known', 'Background'],
                          'reference_id': ['K1', 'B1']})
    validate_alignment(task2a, task2b, task2c, membership, known)
    with pytest.raises(ValueError, match='candidate.*alignment'):
        validate_alignment(task2a, task2b.iloc[::-1], task2c, membership, known)
    with pytest.raises(ValueError, match='duplicate'):
        validate_alignment(task2a, task2b, task2c.assign(region_id=['R1', 'R1']), membership, known)
    with pytest.raises(ValueError, match='membership'):
        validate_alignment(task2a, task2b, task2c, membership.assign(cluster_id=[0, -1, 1]), known)


def test_task2d_completion_gate_and_frozen_hashes(tmp_path):
    valid = {'task2a': {'protocol_frozen': True, 'ready_for_task2b': True},
             'task2b': {'task2b_complete': True, 'mode': 'full'},
             'task2c': {'task2c_complete': True, 'ready_for_task2d': True}}
    verify_completion(valid)
    with pytest.raises(ValueError, match='Task2C'):
        verify_completion({**valid, 'task2c': {'task2c_complete': True, 'ready_for_task2d': False}})
    source = tmp_path / 'candidate.csv'
    source.write_text('frozen', encoding='utf-8')
    original_hash = sha256_file(source)
    verify_source_hashes({'candidate': source}, {'candidate': original_hash})
    source.write_text('altered', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        verify_source_hashes({'candidate': source}, {'candidate': original_hash})


def test_task2d_smoke_full_paths_and_protocol_freeze(tmp_path):
    assert Task2DPaths(tmp_path, 'full').output_root == tmp_path / 'outputs' / 'task2d'
    assert Task2DPaths(tmp_path, 'smoke').output_root == tmp_path / 'outputs' / 'pipeline_runs' / 'task2d_smoke' / 'outputs' / 'task2d'
    path = tmp_path / 'protocol.json'
    freeze_protocol(path, {'known_floor_percentile': 5})
    assert json.loads(path.read_text())['known_floor_percentile'] == 5
    with pytest.raises(ValueError, match='frozen'):
        freeze_protocol(path, {'known_floor_percentile': 10})


def test_pre_registered_rules_must_match_implemented_measurement_constants():
    fixed = {
        'version': 'Task2D-Cross-Replicate-Reproducibility-v1',
        'seed': 20260920,
        'primary_window_rule': {'source': 'Task2A frozen best_scale and representative center',
                                'small': 3200, 'medium': 6400, 'large': 12800},
        'canonical_window_bp': 6400, 'target_bin_size': 100,
        'minimum_valid_pixels': 10,
        'correlation_mask': 'i < j and abs(i-j) > 2, paired finite O/E and positive expected',
        'full_upper_auxiliary_mask': 'i < j, paired finite O/E and positive expected',
        'oe_rule': "unclipped raw matrix divided by that replicate's validated genome-wide E(distance); nonpositive expected invalid",
        'pearson_implementation': 'scipy.stats.pearsonr on valid masked O/E values',
        'spearman_implementation': 'scipy.stats.spearmanr on valid masked O/E values',
        'nrmse_rule': 'masked RMSE / (pooled IQR + 1e-12)',
        'zero_axis_rule': 'any all-zero row or column in raw unbalanced replicate matrix; warning only',
        'known_control_floor': '5th percentile of valid 344 Known canonical Pearson and Spearman separately; nonfinite or negative floor stops binary filtering',
        'candidate_pass_rule': 'valid primary Pearson >= pearson_floor AND valid primary Spearman >= spearman_floor',
        'strict_novel_cluster_rule': 'Task2C novel_like flag AND >=5 candidates AND zero Known AND zero claimed-member genomic Known overlap AND >=80% of valid candidate members pass',
        'strict_cluster_valid_member_pass_fraction': .8,
        'follow_up_rule': 'zero genomic Known overlap AND reproducible unannotated noise or clustered candidate; no novel class claim',
        'ranking_rule': 'pass first, primary Spearman descending, primary Pearson descending, chrom/start/end/region_id ascending',
        'heatmap_rule': 'log1p(unclipped O/E) rep1/rep2/difference, common vmax=99.5th percentile of finite pixels of all final follow-up rep1+rep2 displays; display only',
    }
    validate_protocol_rules(fixed)
    with pytest.raises(ValueError, match='strict'):
        validate_protocol_rules({**fixed, 'strict_cluster_valid_member_pass_fraction': .7})
    with pytest.raises(ValueError, match='window'):
        validate_protocol_rules({**fixed, 'canonical_window_bp': 3200})
    with pytest.raises(ValueError, match='oe_rule'):
        validate_protocol_rules({**fixed, 'oe_rule': 'local expected'})


def test_task2d_pipeline_runs_controls_then_candidates_without_upstream_writes(tmp_path, monkeypatch):
    import src.task2d.pipeline as pipeline

    root = tmp_path / 'project'
    source_dir = tmp_path / 'frozen_inputs'
    source_dir.mkdir()
    paths = {key: source_dir / f'{key}.dat' for key in source_paths(root)}
    for key, path in paths.items():
        path.write_bytes(key.encode())
    loader_source = source_dir / 'load_microc.py'
    loader_source.write_text('frozen coordinate loader', encoding='utf-8')
    monkeypatch.setattr(pipeline, 'COORDINATE_LOADER_SOURCE', loader_source)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    protocol = {'version': 'test', 'source_sha256': hashes, 'minimum_valid_pixels': 10}
    protocol_path = root / 'outputs' / 'task2d' / 'frozen_task2d_protocol.json'
    freeze_protocol(protocol_path, protocol)
    candidates = pd.DataFrame({
        'region_id': ['R1', 'R2', 'R3'], 'chrom': ['MG1655'] * 3,
        'start': [0, 10000, 20000], 'end': [3200, 16400, 32800],
        'center': [1600, 13200, 26400], 'length_bp': [3200, 6400, 12800],
        'best_scale': ['small', 'medium', 'large'], 'best_branch': ['shape'] * 3,
        'support_channels': ['small_shape', 'medium_shape', 'large_shape'],
        'cluster_id': [-1, 1, 1], 'cluster_annotation': ['noise', 'CHIN-like', 'CHIN-like'],
        'membership_probability': [0., .8, .9],
        'known_center_overlap': [False, False, True],
        'known_interval_overlap': [False, False, False],
        'novel_like_cluster_candidate': [False] * 3,
        'nearest_known_id': ['K1'] * 3, 'nearest_known_type': ['CHIN'] * 3,
        'nearest_known_distance': [1., 2., 3.], 'known_types': ['', '', 'CHIN'],
    })
    known = pd.DataFrame({'reference_id': ['K1', 'K2', 'K3'],
                          'type': ['CHIN', 'OPCID', 'CHID'], 'chrom': ['MG1655'] * 3,
                          'start': [100, 200, 300], 'end': [500, 600, 700],
                          'center': [300, 400, 500]})
    background = known.assign(reference_id=['B1', 'B2', 'B3'], type='Background')
    summary = pd.DataFrame({'cluster_id': [1], 'cluster_annotation': ['CHIN-like'],
                            'candidate_count': [2], 'known_count': [1],
                            'novel_like_cluster_candidate': [False]})
    inputs = Task2DInputs(candidates, known, background, summary,
                           {'rep1': np.ones(128), 'rep2': np.ones(128)},
                           paths, hashes, protocol, protocol_path)
    monkeypatch.setattr(pipeline, 'load_inputs', lambda value: inputs)
    calls = []

    def fake_read_pair(source_paths, chrom, center, window_bp):
        calls.append((center, window_bp))
        n = window_bp // 100
        i, j = np.indices((n, n))
        raw = (1 + ((i * 3 + j * 5) % 11)).astype(float)
        return raw, raw.copy(), {'rep1': {'padded': False}, 'rep2': {'padded': False}}

    monkeypatch.setattr(pipeline, 'read_pair', fake_read_pair)
    dry = execute_task2d(root, mode='full', dry_run=True)
    assert dry.dry_run and calls == []
    result = execute_task2d(root, mode='full')
    assert result.manifest_path.is_file()
    manifest = validate_task2a_outputs(result.manifest_path)
    assert manifest['statistics']['n_candidates'] == 3
    assert manifest['statistics']['n_known'] == 3
    assert manifest['statistics']['n_background'] == 3
    assert str(loader_source.resolve()) in manifest['inputs']
    assert (1600, 3200) in calls and (13200, 6400) in calls and (26400, 12800) in calls
    assert all((int(center), 6400) in calls for center in [300, 400, 500])
    assert {key: sha256_file(path) for key, path in paths.items()} == hashes
    assert not (root / 'outputs' / 'task2_final').exists()
    annotated = pd.read_csv(result.output_root / 'candidate_reproducibility.csv')
    assert len(annotated) == 3
    assert annotated.loc[annotated.region_id.eq('R1'), 'candidate_pool'].iloc[0] == 'UNANNOTATED_NOISE'
    loader_source.write_text('changed coordinate loader', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        execute_task2d(root, mode='full', resume=True)

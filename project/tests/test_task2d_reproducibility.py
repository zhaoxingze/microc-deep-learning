import numpy as np
import pandas as pd
import pytest

from src.task2d.reproducibility import (
    correlation_mask, measure_pair, normalize_pair, read_pair, window_for_scale,
)
from src.task2d.analysis import calibrate_known_floor, annotate_candidates, cluster_reproducibility


def varied_matrix(n=8):
    raw = np.ones((n, n), dtype=float)
    indexes = np.argwhere(correlation_mask(n, gap=2))
    for value, (i, j) in enumerate(indexes, start=1):
        raw[i, j] = raw[j, i] = float(value)
    return raw


def test_primary_mask_excludes_diagonal_and_two_neighbor_bands():
    mask = correlation_mask(8, gap=2)
    assert int(mask.sum()) == 15
    assert mask[0, 3] and mask[2, 7]
    assert not mask[0, 2] and not mask[4, 4] and not mask[7, 0]
    assert int(correlation_mask(8, gap=0).sum()) == 28


def test_per_replicate_genome_expected_makes_scaled_raw_matrices_agree():
    raw = varied_matrix()
    expected1 = np.ones(8)
    expected2 = np.full(8, 2.)
    oe1, oe2 = normalize_pair(raw, raw * 2, expected1, expected2)
    assert np.array_equal(oe1, oe2)
    score = measure_pair(raw, raw * 2, expected1, expected2)
    assert score['correlation_valid'] is True
    assert score['n_valid_pixels'] == 15
    assert score['finite_fraction'] == 1.
    assert score['pearson'] == pytest.approx(1.)
    assert score['spearman'] == pytest.approx(1.)
    assert score['nrmse'] == pytest.approx(0.)
    assert score['cosine'] == pytest.approx(1.)
    assert score['full_upper_pearson'] == pytest.approx(1.)


def test_invalid_expected_and_nonfinite_pixels_are_excluded_not_imputed():
    raw = varied_matrix()
    raw[0, 3] = np.nan
    expected = np.ones(8)
    expected[4] = 0.
    oe1, oe2 = normalize_pair(raw, raw, expected, expected)
    assert np.isnan(oe1[0, 4]) and np.isnan(oe2[0, 4])
    score = measure_pair(raw, raw, expected, expected)
    assert score['n_valid_pixels'] == 10
    assert score['finite_fraction'] == pytest.approx(10 / 15)
    assert score['correlation_valid'] is True
    assert score['pearson'] == pytest.approx(1.)


def test_constant_pattern_stays_nan_and_zero_axis_is_warning():
    constant = np.zeros((8, 8))
    score = measure_pair(constant, constant, np.ones(8), np.ones(8))
    assert score['rep1_zero_axis'] is True
    assert score['rep2_zero_axis'] is True
    assert score['artifact_warning'] is True
    assert score['correlation_valid'] is False
    assert score['invalid_reason'] == 'zero_variance'
    assert np.isnan(score['pearson']) and np.isnan(score['spearman'])


def test_too_few_valid_pixels_have_explicit_reason():
    raw = np.ones((4, 4))
    score = measure_pair(raw, raw, np.ones(4), np.ones(4))
    assert score['n_valid_pixels'] == 1
    assert score['correlation_valid'] is False
    assert score['invalid_reason'] == 'too_few_valid_pixels'
    assert np.isnan(score['pearson'])


def test_replicates_are_read_at_exact_same_center_and_frozen_scale(monkeypatch):
    import src.task2d.reproducibility as module

    calls = []

    def fake_loader(path, chrom, center, *, window_bp, target_bin_size, balance):
        calls.append((str(path), chrom, center, window_bp, target_bin_size, balance))
        return np.ones((window_bp // 100, window_bp // 100)), {'padded': False}

    monkeypatch.setattr(module, 'load_local_matrix', fake_loader)
    assert window_for_scale('small') == 3200
    assert window_for_scale('medium') == 6400
    assert window_for_scale('large') == 12800
    raw1, raw2, details = read_pair({'rep1_cool': 'rep1', 'rep2_cool': 'rep2'},
                                     'MG1655', 12345, window_for_scale('small'))
    assert raw1.shape == raw2.shape == (32, 32)
    assert calls == [('rep1', 'MG1655', 12345, 3200, 100, False),
                     ('rep2', 'MG1655', 12345, 3200, 100, False)]
    assert details['rep1']['padded'] is False


def test_known_floor_uses_controls_only_and_negative_floor_fails():
    controls = pd.DataFrame({'correlation_valid': [True] * 3,
                             'pearson': [.2, .4, .6], 'spearman': [.1, .3, .5]})
    floors = calibrate_known_floor(controls)
    assert floors['calibration_valid'] is True
    assert floors['pearson_floor'] == pytest.approx(.22)
    assert floors['spearman_floor'] == pytest.approx(.12)
    failed = calibrate_known_floor(controls.assign(pearson=[-.2, .4, .6]))
    assert failed['calibration_valid'] is False
    assert failed['pearson_floor'] < 0


def candidate_fixture():
    names = ['O', 'N', 'K', 'U', 'S0', 'S1', 'S2', 'S3', 'S4']
    clusters = [1, -1, 1, 3, 4, 4, 4, 4, 4]
    types = ['CHIN-like', 'noise', 'CHIN-like', 'unannotated'] + ['unannotated'] * 5
    frame = pd.DataFrame({
        'region_id': names, 'cluster_id': clusters, 'cluster_annotation': types,
        'known_center_overlap': [True] + [False] * 8,
        'known_interval_overlap': [False] * 9,
        'novel_like_cluster_candidate': [False] * 4 + [True] * 5,
        'correlation_valid': [True] * 9,
        'primary_pearson': [.9, .9, .9, .9, .9, .9, .9, .9, .1],
        'primary_spearman': [.9, .9, .9, .9, .9, .9, .9, .9, .1],
    })
    summary = pd.DataFrame({'cluster_id': [1, 3, 4],
                            'cluster_annotation': ['CHIN-like', 'unannotated', 'unannotated'],
                            'candidate_count': [2, 1, 5],
                            'known_count': [2, 0, 0],
                            'novel_like_cluster_candidate': [False, False, True]})
    floor = {'calibration_valid': True, 'pearson_floor': .5, 'spearman_floor': .5}
    return frame, summary, floor


def test_noise_and_known_like_unannotated_remain_follow_up_not_novel_type():
    candidates, summary, floor = candidate_fixture()
    result = annotate_candidates(candidates, summary, floor).set_index('region_id')
    assert result.loc['O', 'final_status'] == 'KNOWN_OVERLAP'
    assert result.loc['N', 'candidate_pool'] == 'UNANNOTATED_NOISE'
    assert result.loc['N', 'final_status'] == 'UNANNOTATED_NOISE_REPRODUCIBLE'
    assert result.loc['K', 'candidate_pool'] == 'UNANNOTATED_CLUSTERED'
    assert result.loc['K', 'final_status'] == 'KNOWN_LIKE_CLUSTER_REPRODUCIBLE'
    assert result.loc['U', 'final_status'] == 'UNANNOTATED_CLUSTERED_REPRODUCIBLE'
    assert not result.loc['N', 'strict_novel_candidate']


def test_strict_cluster_requires_eighty_percent_valid_pass_and_zero_known():
    candidates, summary, floor = candidate_fixture()
    annotated = annotate_candidates(candidates, summary, floor)
    strict = annotated.loc[annotated.strict_novel_candidate]
    assert set(strict.region_id) == {'S0', 'S1', 'S2', 'S3'}
    assert annotated.loc[annotated.region_id.eq('S4'), 'final_status'].iloc[0] == 'UNANNOTATED_CLUSTERED_LOW_REPRODUCIBILITY'
    candidates.loc[candidates.region_id.eq('S3'), 'primary_spearman'] = .1
    changed = annotate_candidates(candidates, summary, floor)
    assert not changed.strict_novel_candidate.any()
    assert list(changed.loc[changed.strict_novel_candidate, 'region_id']) == []


def test_strict_cluster_uses_valid_member_denominator_and_rejects_known_control():
    candidates, summary, floor = candidate_fixture()
    candidates.loc[candidates.region_id.eq('S4'), 'correlation_valid'] = False
    valid_only = annotate_candidates(candidates, summary, floor)
    assert set(valid_only.loc[valid_only.strict_novel_candidate, 'region_id']) == {'S0', 'S1', 'S2', 'S3'}
    assert not valid_only.loc[valid_only.region_id.eq('S4'), 'strict_novel_candidate'].iloc[0]
    summary.loc[summary.cluster_id.eq(4), 'known_count'] = 1
    with_known = annotate_candidates(candidates, summary, floor)
    assert not with_known.strict_novel_candidate.any()


def test_cluster_summary_counts_only_valid_reproducible_candidates():
    candidates, summary, floor = candidate_fixture()
    annotated = annotate_candidates(candidates, summary, floor)
    audit = cluster_reproducibility(annotated, summary).set_index('cluster_id')
    assert audit.loc[4, 'candidate_count'] == 5
    assert audit.loc[4, 'candidate_valid_count'] == 5
    assert audit.loc[4, 'reproducible_candidate_count'] == 4
    assert audit.loc[4, 'reproducible_fraction'] == pytest.approx(.8)
    assert audit.loc[1, 'candidate_count'] == 2

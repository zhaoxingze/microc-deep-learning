import numpy as np
import pandas as pd
import pytest

from src.task2d.finalization import (
    build_final_completion, common_display_vmax, format_control_and_sensitivity_tables,
    select_final_tables,
    write_final_heatmaps,
)


def final_candidates():
    return pd.DataFrame({
        'candidate_id': ['C1', 'C2', 'C3', 'C4'],
        'region_id': ['C1', 'C2', 'C3', 'C4'],
        'chrom': ['MG1655'] * 4, 'start': [100, 200, 300, 400],
        'end': [200, 300, 400, 500], 'center': [150, 250, 350, 450],
        'cluster_id': [-1, 1, 1, 1],
        'cluster_annotation': ['noise', 'CHIN-like', 'CHIN-like', 'CHIN-like'],
        'known_center_overlap': [False, False, False, True],
        'known_interval_overlap': [False] * 4,
        'primary_pearson': [.8, .9, .7, .99],
        'primary_spearman': [.85, .95, .65, .99],
        'reproducible_by_known_floor': [True, True, True, True],
        'strict_novel_candidate': [False] * 4,
        'final_status': ['UNANNOTATED_NOISE_REPRODUCIBLE',
                         'KNOWN_LIKE_CLUSTER_REPRODUCIBLE',
                         'UNANNOTATED_CLUSTERED_REPRODUCIBLE', 'KNOWN_OVERLAP'],
    })


def test_empty_strict_novel_is_headered_and_followups_include_noise_and_known_like():
    all_rows, strict, followup = select_final_tables(final_candidates())
    assert list(strict.columns) == list(all_rows.columns)
    assert strict.empty
    assert list(followup.region_id) == ['C2', 'C1', 'C3']
    assert not followup.region_id.eq('C4').any()
    assert followup.known_center_overlap.eq(False).all()


def test_followup_requires_actual_known_floor_pass_not_status_text_alone():
    frame = final_candidates()
    frame.loc[frame.region_id.eq('C1'), 'reproducible_by_known_floor'] = False
    _, _, followup = select_final_tables(frame)
    assert 'C1' not in set(followup.region_id)


def test_global_visualization_scale_is_single_995th_percentile():
    one = np.array([[0., 1.], [2., 3.]])
    two = np.array([[4., 5.], [6., 7.]])
    assert common_display_vmax([one, two]) == pytest.approx(np.percentile(np.arange(8), 99.5))
    assert common_display_vmax([]) is None


def test_task2_can_complete_without_strict_novel_detection():
    completion = build_final_completion({'task2a_finalized': True, 'task2b_complete': True,
        'task2c_complete': True, 'task2d_reproducibility_complete': True,
        'all_eligible_candidates_evaluated': True, 'known_positive_control_evaluated': True,
        'background_reference_evaluated': True, 'cluster_level_reproducibility_complete': True,
        'candidate_final_status_complete': True, 'strict_novel_list_generated': True,
        'followup_unannotated_list_generated': True, 'heatmap_protocol_completed': True,
        'final_manifest_generated': True, 'final_summary_generated': True,
        'no_upstream_result_modified': True}, strict_count=0)
    assert completion['task2_complete'] is True
    assert completion['strict_novel_structure_detected'] is False
    assert all(completion['checks'].values())


def test_heatmaps_share_one_display_scale_and_same_canonical_coordinates(tmp_path, monkeypatch):
    import src.task2d.finalization as module

    calls = []

    def fake_pair(paths, chrom, center, bp):
        calls.append((chrom, center, bp))
        n = bp // 100
        raw = np.ones((n, n), dtype=float)
        raw[3:7, 12:16] = raw[12:16, 3:7] = 4.
        return raw, raw * 1.1, {'rep1': {}, 'rep2': {}}

    monkeypatch.setattr(module, 'read_pair', fake_pair)
    selected = final_candidates().iloc[:2]
    paths, vmax = write_final_heatmaps(selected, {'rep1': np.ones(128), 'rep2': np.ones(128)},
                                       {'rep1_cool': 'one', 'rep2_cool': 'two'}, tmp_path)
    assert len(paths) == 2
    assert all(path.stat().st_size > 1000 for path in paths)
    assert calls == [('MG1655', 150, 6400), ('MG1655', 250, 6400)]
    assert vmax > 0


def test_final_report_embeds_known_type_distribution_and_fixed_sensitivity():
    known = pd.DataFrame({'type': ['CHIN', 'OPCID', 'CHID'],
                          'correlation_valid': [True] * 3,
                          'pearson': [.2, .4, .6], 'spearman': [.3, .5, .7]})
    sensitivity = pd.DataFrame({'min_cluster_size': [8], 'min_samples': [5],
                                'cluster_count': [2], 'noise_count': [102]})
    rendered = format_control_and_sensitivity_tables(known, sensitivity)
    assert all(name in rendered for name in ('CHIN', 'OPCID', 'CHID'))
    assert '0.6' in rendered
    assert '102' in rendered

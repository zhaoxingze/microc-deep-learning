import numpy as np
import pandas as pd
import pytest

from src.task2 import background


def test_three_way_whole_blocks_and_nonempty():
    rows = pd.DataFrame([dict(window_id=str(i), chrom='c', start=i*100,
                             end=i*100+80, background_candidate=True) for i in range(20)])
    result = background.calibrated_background_split(rows, block_bp=200, seed=7)
    assert set(result.split) == {'train', 'validation', 'calibration'}
    assert result.groupby('block_id').split.nunique().max() == 1
    assert len(result) == 20
    background.assert_three_way_no_overlap(result)


def test_boundary_and_known_excluded():
    rows = pd.DataFrame([dict(window_id=str(i), chrom='c', start=i*100,
                             end=i*100+120, background_candidate=i != 2) for i in range(10)])
    result = background.calibrated_background_split(rows, block_bp=200)
    assert '2' not in set(result.window_id)
    assert set(result.window_id) == {'0', '4', '6', '8'}


def test_too_few_blocks_fails():
    rows = pd.DataFrame([dict(window_id=str(i), chrom='c', start=i*100,
                             end=i*100+80, background_candidate=True) for i in range(2)])
    with pytest.raises(ValueError, match='three'):
        background.calibrated_background_split(rows, block_bp=100)


def test_overlap_fails_independent_validator():
    rows = pd.DataFrame(dict(window_id=['a','b','c'], block_id=['1','2','3'],
                             chrom=['c']*3, start=[0,50,300], end=[100,150,400],
                             split=['train','validation','calibration']))
    with pytest.raises(ValueError, match='overlap'):
        background.assert_three_way_no_overlap(rows)


def test_empirical_tail_includes_ties_and_smoothing():
    from src.task2.calibration import fit_empirical_tail_calibrator, transform_to_tail_probability
    cal = fit_empirical_tail_calibrator([1,2,2,4])
    assert transform_to_tail_probability(cal, [0,1,2,3,4,5]).tolist() == [1,1,.8,.4,.4,.2]


def test_tail_monotone_positive_and_anomaly():
    from src.task2.calibration import fit_empirical_tail_calibrator, transform_to_tail_probability
    p = transform_to_tail_probability(fit_empirical_tail_calibrator([1,2,3]), np.arange(7))
    assert np.all(p > 0) and np.all(np.diff(-np.log10(p)) >= 0)
    assert -np.log10(p[-1]) == pytest.approx(np.log10(4))


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf]])
def test_invalid_calibration_fails(values):
    from src.task2.calibration import fit_empirical_tail_calibrator
    with pytest.raises(ValueError):
        fit_empirical_tail_calibrator(values)


def scores_fixture():
    return pd.DataFrame(dict(window_id=['train','val','cal1','cal2','known'],
        rep1_density_raw=[99,99,1,2,3], rep2_density_raw=[99,99,10,20,15],
        rep1_shape_raw=[99,99,100,200,50], rep2_shape_raw=[99,99,1,2,3]))


def test_calibration_only_eligible_calibration_independent_branches():
    from src.task2.calibration import calibrate_scores
    split = pd.DataFrame(dict(window_id=['train','val','cal1','cal2'],
                              split=['train','validation','calibration','calibration']))
    out, artifact = calibrate_scores(scores_fixture(), split)
    k = out.set_index('window_id').loc['known']
    assert k.rep1_density_p == pytest.approx(1/3)
    assert k.rep1_shape_p == 1
    assert k.rep2_density_p == pytest.approx(2/3)
    assert k.rep2_shape_p == pytest.approx(1/3)
    assert artifact['rep1']['density']['n'] == 2


def test_average_then_or_not_or_then_average():
    from src.task2.calibration import calibrate_scores
    split = pd.DataFrame(dict(window_id=['cal1','cal2'], split=['calibration']*2))
    out, _ = calibrate_scores(scores_fixture(), split)
    k = out.set_index('window_id').loc['known']
    assert k.density_tail_score == pytest.approx((-np.log10(1/3)-np.log10(2/3))/2)
    assert k.shape_tail_score == pytest.approx(-np.log10(1/3)/2)
    assert k.calibrated_or_score == k.density_tail_score
    assert k.density_anomaly_min == pytest.approx(-np.log10(2/3))
    assert k.rep1_p_or_bonferroni == pytest.approx(2/3)


def test_known_cannot_be_calibration():
    from src.task2.calibration import calibrate_scores
    split = pd.DataFrame(dict(window_id=['known'], split=['calibration'], background_candidate=[False]))
    with pytest.raises(ValueError, match='background'):
        calibrate_scores(scores_fixture(), split)


def test_missing_calibration_score_rejected():
    from src.task2.calibration import calibrate_scores
    split = pd.DataFrame(dict(window_id=['missing'], split=['calibration']))
    with pytest.raises(ValueError):
        calibrate_scores(scores_fixture(), split)


def test_actual_known_interval_calibration_overlap_aborts_without_resplitting():
    from src.task2.background import audit_known_interval_overlap
    split=pd.DataFrame(dict(window_id=['t','v','c'],chrom=['x']*3,start=[0,100,200],end=[90,190,290],
                            split=['train','validation','calibration']))
    known=pd.DataFrame(dict(structure_id=['s'],chrom=['x'],start=[80],end=[110]))
    audit=audit_known_interval_overlap(split,known)
    assert set(audit.window_id)=={'t','v'}
    assert split.split.tolist()==['train','validation','calibration']
    known.loc[0,['start','end']]=[280,300]
    with pytest.raises(ValueError,match='Calibration'):
        audit_known_interval_overlap(split,known)


def test_generic_paired_diagnostics_are_calibrated_and_legacy_preserved():
    from src.task2.calibration import calibrate_scores
    scores=scores_fixture()
    for c in ('score_min','score_max','score_difference','paired_candidate_score'):scores[c]=99
    out,_=calibrate_scores(scores,pd.DataFrame(dict(window_id=['cal1','cal2'],split=['calibration']*2)))
    k=out.set_index('window_id').loc['known']
    assert k.legacy_score_min==99
    assert k.score_min==pytest.approx(np.log10(3))
    assert k.score_difference==0
    assert k.paired_candidate_score==k.calibrated_or_score

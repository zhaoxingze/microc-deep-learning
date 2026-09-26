import numpy as np
import pandas as pd
import pytest


def windows():
    return pd.DataFrame(dict(window_id=list('abcde'), chrom=['c']*5,
        start=[0,800,1600,3200,8000], end=[6400,7200,8000,9600,14400],
        center=[3200,4000,4800,6400,11200], calibrated_or_score=[5,4,3,2,1],
        density_tail_score=[5,4,3,2,1], shape_tail_score=[0,0,0,0,0],
        zero_axis_rep1=[False,True,False,False,False], zero_axis_rep2=[False]*5))


def known():
    return pd.DataFrame(dict(structure_id=['s1','s2','s3'],type=['CHIN','OPCID','CHID'],
        chrom=['c']*3,start=[100,8000,14900],end=[200,8100,15000],center=[150,8050,14950]))


def test_nms_suppresses_near_but_keeps_exact_radius_and_distant():
    from src.task2.region_detection import build_candidate_regions
    regions, members = build_candidate_regions(windows())
    assert set(regions.peak_window_id) == {'a','d','e'}
    assert set(members.loc[members.peak_window_id.eq('a'),'window_id']) == {'a','b','c'}


def test_region_bounds_union_and_branch():
    from src.task2.region_detection import build_candidate_regions
    regions, _ = build_candidate_regions(windows())
    a = regions.set_index('peak_window_id').loc['a']
    assert (a.start,a.end,a.member_window_count,a.union_length_bp)==(0,8000,3,8000)
    assert a.dominant_branch == 'density'


def test_support_cannot_chain_or_jump_gap():
    from src.task2.region_detection import build_candidate_regions
    regions, members = build_candidate_regions(windows().drop(index=1))
    a = regions.set_index('peak_window_id').loc['a']
    assert a.member_window_count == 1
    assert not bool(members.set_index('window_id').loc['c','included'])


def test_ties_invariant_to_input_order():
    from src.task2.region_detection import build_candidate_regions
    w = windows(); w.calibrated_or_score = 1
    a, _ = build_candidate_regions(w)
    b, _ = build_candidate_regions(w.sample(frac=1,random_state=3))
    pd.testing.assert_frame_equal(a,b)


def test_coverage_union_budget_overshoot_and_zero():
    from src.task2.region_detection import build_candidate_regions, select_coverage_budget
    regions, _ = build_candidate_regions(windows())
    chosen, bp = select_coverage_budget(regions, .10, 20000)
    assert len(chosen)==1 and bp==8000
    chosen, bp = select_coverage_budget(regions, .45, 20000)
    assert len(chosen)==2 and bp==9600
    chosen, bp = select_coverage_budget(regions, 0, 20000)
    assert chosen.empty and bp==0


def test_halfopen_center_and_auxiliary_overlap():
    from src.task2.calibrated_evaluation import recall_metrics
    intervals = pd.DataFrame(dict(chrom=['c'], start=[0],end=[150]))
    values = recall_metrics(intervals, known())
    assert values['overall_recall']==0
    assert values['overall_overlap_recall']==pytest.approx(1/3)


def test_curve_reports_actual_coverage():
    from src.task2.region_detection import build_candidate_regions
    from src.task2.calibrated_evaluation import coverage_curve
    regions, _ = build_candidate_regions(windows())
    curve = coverage_curve(regions,known(),20000,budgets=(0,.10,.45))
    assert curve.actual_coverage.tolist()==[0,.4,.48]
    assert curve.overall_recall.tolist()==[0,1/3,2/3]


def test_region_random_exact_length_multiset_linear_and_deterministic():
    from src.task2.calibrated_evaluation import sample_region_matched
    selected = pd.DataFrame(dict(chrom=['c','c'],start=[0,100],end=[20,150]))
    a = sample_region_matched(selected,{'c':200},np.random.default_rng(5))
    b = sample_region_matched(selected,{'c':200},np.random.default_rng(5))
    pd.testing.assert_frame_equal(a,b)
    assert sorted(a.end-a.start)==[20,50]
    assert a.start.ge(0).all() and a.end.le(200).all()


def test_random_baselines_report_actual_coverage_and_repeatability():
    from src.task2.region_detection import build_candidate_regions
    from src.task2.calibrated_evaluation import random_baselines
    w=windows(); regions,_=build_candidate_regions(w)
    a,b,draws = random_baselines(w,regions,known(),{'c':20000},budgets=(.1,),repeats=5,seed=7)
    aa,bb,dd = random_baselines(w,regions,known(),{'c':20000},budgets=(.1,),repeats=5,seed=7)
    pd.testing.assert_frame_equal(a,aa); pd.testing.assert_frame_equal(b,bb)
    pd.testing.assert_frame_equal(draws,dd)
    assert len(draws)==10 and set(draws.baseline)=={'coverage','region'}
    assert a.iloc[0].target_coverage_bp==8000
    assert b.iloc[0].region_count==1


def test_known_audit_and_missed_by_budget():
    from src.task2.region_detection import build_candidate_regions, rank_windows
    from src.task2.calibrated_evaluation import known_audit, missed_summary
    w=rank_windows(windows()); regions,_=build_candidate_regions(w)
    audit=known_audit(w,regions,known(),20000)
    assert len(audit)==3
    assert audit.set_index('structure_id').loc['s1','best_overlapping_window_id']=='a'
    assert audit.set_index('structure_id').loc['s1','zero_axis_nearby']==False
    assert audit.set_index('structure_id').loc['s3','selected_at_20pct_coverage']==False
    summary=missed_summary(audit)
    assert summary.set_index('type').loc['CHID','missed_20pct']==1


def test_zero_axis_enrichment_counts_centers_in_union_once():
    from src.task2.calibrated_evaluation import zero_axis_audit
    selected = pd.DataFrame(dict(chrom=['c','c'],start=[0,500],end=[8000,9000]))
    audit=zero_axis_audit(windows(),selected)
    assert audit['selected_count']==4
    assert audit['all_rate']==.2 and audit['selected_rate']==.25
    assert audit['enrichment']==1.25


def gate_inputs():
    curve=pd.DataFrame(dict(budget=[.1,.2],budget_reached=[True,True],overall_recall=[.5,.6],
        CHIN_recall=[.4,.5],OPCID_recall=[.5,.6],CHID_recall=[.3,.4]))
    coverage=pd.DataFrame(dict(budget=[.1,.2],overall_ci_high=[.3,.4],repeats=[100,100],matched_repeats=[100,100]))
    region=pd.DataFrame(dict(budget=[.1,.2],CHIN_ci_low=[.1,.2],OPCID_ci_low=[.1,.2],CHID_ci_low=[.1,.2]))
    return curve,coverage,region


def test_readiness_all_seven_and_false_no_next_task():
    from src.task2.calibrated_evaluation import assess_readiness
    a,b,c=gate_inputs()
    r=assess_readiness(a,b,c,zero_enrichment=.5,replicate_spearman=.8,branch_spearman=-.4,no_overlap=True)
    assert r['ready_for_task2b'] and len(r['checks'])==7 and not r['task2b_executed']
    r=assess_readiness(a,b,c,zero_enrichment=1.01,replicate_spearman=.8,branch_spearman=-.4,no_overlap=True)
    assert not r['ready_for_task2b'] and not r['task2b_executed']


@pytest.mark.parametrize('failure',['random10','random20','class','replicate','branch','leak'])
def test_readiness_each_failure_blocks(failure):
    from src.task2.calibrated_evaluation import assess_readiness
    a,b,c=gate_inputs()
    if failure=='random10': a.loc[0,'overall_recall']=.3
    if failure=='random20': a.loc[1,'overall_recall']=.4
    if failure=='class': a.loc[1,'CHID_recall']=.1
    r=assess_readiness(a,b,c,zero_enrichment=.5,
        replicate_spearman=0 if failure=='replicate' else .8,
        branch_spearman=.9 if failure=='branch' else -.4,no_overlap=failure!='leak')
    assert not r['ready_for_task2b']


@pytest.mark.parametrize('invalid',['unmatched','unreached','too_few_repeats'])
def test_readiness_invalid_coverage_evidence_cannot_pass(invalid):
    from src.task2.calibrated_evaluation import assess_readiness
    a,b,c=gate_inputs()
    if invalid=='unmatched': b.loc[0,'matched_repeats']=99
    if invalid=='unreached': a.loc[0,'budget_reached']=False
    if invalid=='too_few_repeats': b[['repeats','matched_repeats']]=3
    r=assess_readiness(a,b,c,zero_enrichment=.5,replicate_spearman=.8,branch_spearman=-.4,no_overlap=True)
    assert not r['ready_for_task2b']

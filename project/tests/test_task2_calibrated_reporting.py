from types import SimpleNamespace
import numpy as np
import pandas as pd


def test_report_writes_required_figures_utf8_and_honest_failed_gate(tmp_path):
    from src.task2.calibrated_reporting import write_calibrated_report
    curve=pd.DataFrame(dict(budget=[.05,.1,.2,.3],actual_coverage=[.051,.101,.201,.301],
        region_count=[1,2,3,4],overall_recall=[.1,.2,.3,.4],CHIN_recall=[.1,.2,.3,.4],
        OPCID_recall=[.1,.2,.3,.4],CHID_recall=[.1,.2,.3,.4]))
    random=pd.DataFrame(dict(budget=[.05,.1,.2,.3],overall_mean=[.1]*4,overall_ci_low=[0]*4,overall_ci_high=[.5]*4,
                            actual_coverage_mean=[.05,.1,.2,.3]))
    audit=pd.DataFrame(dict(type=['CHIN','CHID','OPCID'],structure_id=['s1','s2','s3'],
        selected_at_20pct_coverage=[False,False,True],best_density_tail_score=[.2,.3,.4],
        best_shape_tail_score=[.4,.5,.6],best_calibrated_or_score=[.4,.5,.6],
        first_covering_region_rank=[1,2,3]))
    scores=pd.DataFrame(dict(background_candidate=[True]*3,density_tail_score=[.1,.2,.3],
        shape_tail_score=[.2,.3,.4],calibrated_or_score=[.2,.3,.4]))
    statistics=dict(n_windows=10,n_known_structures=3,n_candidate_regions=5,
        background_counts={'train':4,'validation':2,'calibration':2},block_counts={'train':2,'validation':1,'calibration':1},
        best_epoch=1,best_validation_loss=.5,correlations={'replicates':{'pearson':.8,'spearman':.7},
        'branches':{'pearson':-.3,'spearman':-.2}},zero_axis={'enrichment':.5},minimum_empirical_p=1/3,
        readiness={'ready_for_task2b':False,'checks':{'beats_random':False}},calibrated_or_unique_scores=3,calibrated_or_max_ties=2)
    tables={'coverage_recall_curve':curve,'coverage_random_baseline':random,'region_random_baseline':random,
            'known_structure_detection_audit':audit,'missed_known_summary':pd.DataFrame({'type':['CHIN'],'total':[1]})}
    paths=SimpleNamespace(output_root=tmp_path,summary_path=tmp_path/'task2a_calibrated_summary.md')
    write_calibrated_report(paths,SimpleNamespace(mode='smoke'),{},statistics,tables,scores,
                            pd.DataFrame(dict(epoch=[1],train_loss=[1],val_loss=[.5])))
    for name in ('coverage_recall_curve.png','known_score_distributions.png','task2a_calibrated_summary.md'):
        assert (tmp_path/name).stat().st_size>100
    text=paths.summary_path.read_text(encoding='utf-8-sig')
    assert 'false' in text and 'Smoke' in text and 'Task2B' in text and '23.' in text

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def test_fixed_scale_specs_and_exact_counts():
    from src.task2.multiscale import SCALES, generate_scale_windows
    assert [(s.name,s.window_bp,s.step_bp,s.matrix_bins) for s in SCALES] == [
        ('small',3200,400,32),('medium',6400,800,64),('large',12800,1600,128)]
    assert [len(generate_scale_windows(s,'MG1655',4_641_652)) for s in SCALES] == [11597,5795,2894]


def test_scale_windows_complete_stable_and_collision_free():
    from src.task2.multiscale import SCALES, generate_scale_windows
    rows=[w for s in SCALES for w in generate_scale_windows(s,'c',16000)]
    assert len({w.window_id for w in rows})==len(rows)
    assert all(w.scale in w.window_id for w in rows)
    assert min(w.start for w in rows)==0 and max(w.end for w in rows)<=16000
    assert generate_scale_windows(SCALES[0],'c',16000)[2].window_id==generate_scale_windows(SCALES[0],'c',16000)[2].window_id


def test_replicate_alignment_rejects_order_or_coordinates():
    from src.task2.multiscale import assert_replicate_alignment
    rows=pd.DataFrame(dict(window_id=['a','b','a','b'],replicate=['rep1','rep1','rep2','rep2'],
        array_index=[0,1,0,1],chrom=['c']*4,start=[0,1,0,1],end=[1,2,1,2]))
    assert_replicate_alignment(rows)
    rows.loc[3,'window_id']='x'
    with pytest.raises(ValueError,match='alignment'):assert_replicate_alignment(rows)


def test_expected_max128_and_scale_slices():
    from src.task2.multiscale import SCALES, slice_expected
    maximum={'rep1':np.arange(128.),'rep2':np.arange(128.)}
    assert [len(slice_expected(maximum,s)['rep1']) for s in SCALES]==[32,64,128]
    with pytest.raises(ValueError,match='128'):slice_expected({'rep1':np.ones(127),'rep2':np.ones(128)},SCALES[2])


def test_strict_interval_exclusion_halfopen():
    from src.task2.multiscale import strict_background_candidates
    windows=pd.DataFrame(dict(window_id=list('abcd'),chrom=['c']*4,start=[0,10,20,30],end=[10,20,30,40],center=[5,15,25,35]))
    known=pd.DataFrame(dict(structure_id=['s'],chrom=['c'],start=[10],end=[30]))
    result=strict_background_candidates(windows,known)
    assert result.set_index('window_id').background_candidate.to_dict()=={'a':True,'b':False,'c':False,'d':True}


def test_master_block_assignment_nonempty_deterministic():
    from src.task2.multiscale import make_master_block_split
    a=make_master_block_split('c',4641652,128000,seed=20260920)
    b=make_master_block_split('c',4641652,128000,seed=20260920)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==37 and a.split.value_counts().to_dict()=={'train':26,'validation':6,'calibration':5}


def test_master_split_deterministically_repairs_empty_scale_split():
    from src.task2.multiscale import make_feasible_master_block_split
    # Scale large has no eligible windows in block3, which the first seeded draw assigns validation.
    eligibility={'small':{0,1,2,3},'medium':{0,1,2,3},'large':{0,1,2}}
    a=make_feasible_master_block_split('c',100000,25600,eligibility,seed=20260920)
    b=make_feasible_master_block_split('c',100000,25600,eligibility,seed=20260920)
    pd.testing.assert_frame_equal(a,b)
    for blocks in eligibility.values():
        labels=set(a.loc[a.block_index.isin(blocks),'split'])
        assert labels=={'train','validation','calibration'}


def test_scale_background_uses_master_and_drops_crossing_windows():
    from src.task2.multiscale import assign_scale_background
    master=pd.DataFrame(dict(chrom=['c','c','c'],block_id=['c:0','c:1','c:2'],block_start=[0,100,200],block_end=[100,200,300],split=['train','validation','calibration']))
    windows=pd.DataFrame(dict(window_id=list('abcd'),chrom=['c']*4,start=[0,80,100,200],end=[40,120,140,240],background_candidate=[True]*4))
    result=assign_scale_background(windows,master,block_bp=100)
    assert result.window_id.tolist()==['a','c','d']
    assert result.split.tolist()==['train','validation','calibration']


def test_known_overlap_audit_is_zero_or_fails():
    from src.task2.multiscale import assert_background_known_clean
    rows=pd.DataFrame(dict(scale=['small'],split=['train'],window_id=['a'],chrom=['c'],start=[0],end=[10]))
    known=pd.DataFrame(dict(structure_id=['s'],chrom=['c'],start=[10],end=[20]))
    audit=assert_background_known_clean(rows,known)
    assert audit.overlap_count.tolist()==[0]
    known.loc[0,'start']=9
    with pytest.raises(ValueError,match='known interval'):assert_background_known_clean(rows,known)


def test_cross_scale_leakage_detects_overlapping_splits():
    from src.task2.multiscale import assert_cross_scale_no_leakage, split_leakage_audit
    clean=pd.DataFrame(dict(scale=['small','large','medium'],split=['train','validation','calibration'],
        window_id=['a','b','c'],chrom=['c']*3,start=[0,100,200],end=[50,150,250]))
    audit=assert_cross_scale_no_leakage(clean)
    assert len(audit)==12 and audit.overlap_count.sum()==0
    assert {'combined','small','medium','large'}==set(audit.scope)
    pd.testing.assert_frame_equal(audit,split_leakage_audit(clean))
    clean.loc[1,['start','end']]=[40,80]
    with pytest.raises(ValueError,match='cross-scale'):assert_cross_scale_no_leakage(clean)


@pytest.mark.parametrize('size',[32,64,128])
def test_same_autoencoder_supports_all_shapes(size):
    from src.task2.autoencoder import BackgroundAutoencoder
    model=BackgroundAutoencoder();x=torch.zeros(2,1,size,size)
    assert model(x).shape==x.shape


def pool_fixture():
    return pd.DataFrame([
        dict(scale='small',window_bp=3200,step_bp=400,window_id='small_a',chrom='c',start=0,end=3200,center=1600,scale_or_score=5.,density_tail_score=5.,shape_tail_score=0.,rep1_calibrated_or_score=5.,rep2_calibrated_or_score=4.,zero_axis_rep1=False,zero_axis_rep2=False),
        dict(scale='medium',window_bp=6400,step_bp=800,window_id='medium_a',chrom='c',start=0,end=6400,center=3200,scale_or_score=4.,density_tail_score=1.,shape_tail_score=4.,rep1_calibrated_or_score=4.,rep2_calibrated_or_score=3.,zero_axis_rep1=False,zero_axis_rep2=False),
        dict(scale='large',window_bp=12800,step_bp=1600,window_id='large_a',chrom='c',start=0,end=12800,center=6400,scale_or_score=3.,density_tail_score=3.,shape_tail_score=0.,rep1_calibrated_or_score=3.,rep2_calibrated_or_score=2.,zero_axis_rep1=False,zero_axis_rep2=False),
        dict(scale='small',window_bp=3200,step_bp=400,window_id='small_b',chrom='c',start=6400,end=9600,center=8000,scale_or_score=2.,density_tail_score=2.,shape_tail_score=0.,rep1_calibrated_or_score=2.,rep2_calibrated_or_score=1.,zero_axis_rep1=True,zero_axis_rep2=False),
    ])


def test_scale_or_and_independent_calibration_markers():
    from src.task2.multiscale import validate_scale_score_pool
    pool=pool_fixture();pool['calibration_id']=['small','medium','large','small']
    validate_scale_score_pool(pool)
    bad=pool.copy();bad.loc[1,'calibration_id']='small'
    with pytest.raises(ValueError,match='independent'):validate_scale_score_pool(bad)
    assert np.array_equal(pool.scale_or_score,np.maximum(pool.density_tail_score,pool.shape_tail_score))


def test_cross_scale_nms_strict_rule_and_peak_interval():
    from src.task2.multiscale import build_cross_scale_regions
    regions,members=build_cross_scale_regions(pool_fixture(),seed=7)
    # medium center differs 1600 < max(3200,6400)/2=3200: suppressed.
    # large differs 4800 < 6400: suppressed. small_b differs 6400 from small peak: retained.
    assert regions.peak_window_id.tolist()==['small_a','small_b']
    first=regions.iloc[0]
    assert (first.start,first.end,first.peak_scale)==(0,3200,'small')
    assert first.support_union_end==12800
    assert first.support_small_count==1 and first.support_medium_count==1 and first.support_large_count==1
    assert set(members.window_id)==set(pool_fixture().window_id)


def test_cross_scale_exact_radius_is_not_suppressed():
    from src.task2.multiscale import build_cross_scale_regions
    p=pool_fixture().iloc[[0,3]].copy();p.loc[p.window_id.eq('small_b'),['start','end','center']]=[3200,6400,4800]
    regions,_=build_cross_scale_regions(p,seed=7)
    assert len(regions)==2


def known_fixture():
    return pd.DataFrame(dict(structure_id=['s1','s2'],type=['CHIN','CHID'],chrom=['c','c'],start=[100,6500],end=[200,8500],center=[150,7500],length_bp=[100,2000]))


def test_known_scale_audit_best_scale_and_budget_flags():
    from src.task2.multiscale import known_scale_audit,build_cross_scale_regions
    pool=pool_fixture();regions,_=build_cross_scale_regions(pool,seed=7)
    regions_by_scale={
        'small':pd.DataFrame(dict(chrom=['c'],start=[0],end=[3200],cumulative_union_bp=[3200])),
        'medium':pd.DataFrame(dict(chrom=['c'],start=[9000],end=[15400],cumulative_union_bp=[6400])),
        'large':pd.DataFrame(dict(chrom=['c'],start=[9000],end=[20000],cumulative_union_bp=[11000])),
    }
    audit=known_scale_audit(pool,regions,known_fixture(),genome_length=20000,
                            regions_by_scale=regions_by_scale)
    assert len(audit)==2 and {'small_best_or','medium_best_or','large_best_or','best_scale','best_multiscale_score','selected_at_20pct'}<=set(audit)
    assert audit.set_index('structure_id').loc['s1','best_scale']=='small'
    s1=audit.set_index('structure_id').loc['s1']
    assert bool(s1.selected_small_at_20pct)
    assert not bool(s1.selected_medium_at_20pct) and not bool(s1.selected_large_at_20pct)
    assert s1.rescue_category_at_20pct=='small_only'


def test_scale_scan_manifest_rejects_corruption_and_stale_source(tmp_path):
    from src.task2.multiscale import SCALES, write_scale_scan_manifest, validate_scale_scan_manifest
    root=tmp_path/'small';root.mkdir()
    sources={}
    for rep in ('rep1','rep2'):
        source=tmp_path/f'{rep}.cool';source.write_bytes(f'{rep}-source'.encode());sources[rep]=source
        np.save(root/f'genome_windows_{rep}.npy',np.ones((2,1,32,32),dtype=np.float32))
    pd.DataFrame({'window_id':['a']}).to_csv(root/'metadata.csv',index=False)
    write_scale_scan_manifest(root,SCALES[0],sources)
    validate_scale_scan_manifest(root,SCALES[0],sources)
    with open(root/'genome_windows_rep1.npy','ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError,match='artifact'):
        validate_scale_scan_manifest(root,SCALES[0],sources)
    np.save(root/'genome_windows_rep1.npy',np.ones((2,1,32,32),dtype=np.float32))
    write_scale_scan_manifest(root,SCALES[0],sources)
    sources['rep1'].write_bytes(b'changed-source')
    with pytest.raises(ValueError,match='source SHA256'):
        validate_scale_scan_manifest(root,SCALES[0],sources)


def test_scale_rescue_summary_classifies_exactly_once():
    from src.task2.multiscale import scale_rescue_summary
    selections={10:{'small':{'s1'},'medium':set(),'large':set()},20:{'small':{'s1'},'medium':{'s1'},'large':{'s2'}},30:{'small':set(),'medium':set(),'large':set()}}
    out=scale_rescue_summary(known_fixture(),selections)
    chin10=out[(out.type=='CHIN')&(out.budget==.1)].iloc[0]
    assert chin10.recalled_small_only==1 and chin10.missed_all_scales==0
    chid20=out[(out.type=='CHID')&(out.budget==.2)].iloc[0]
    assert chid20.recalled_large_only==1
    assert (out[['recalled_small_only','recalled_medium_only','recalled_large_only','recalled_multiple_scales','missed_all_scales']].sum(axis=1)==out.n_total).all()


def test_multiscale_readiness_all_seven_and_never_runs_task2b():
    from src.task2.multiscale import assess_multiscale_readiness
    curve=pd.DataFrame(dict(budget=[.1,.2],budget_reached=[True,True],overall_recall=[.5,.6],CHIN_recall=[.3,.4],OPCID_recall=[.4,.5],CHID_recall=[.2,.3]))
    cov=pd.DataFrame(dict(budget=[.1,.2],repeats=[100,100],matched_repeats=[100,100],overall_ci_high=[.4,.5]))
    reg=pd.DataFrame(dict(budget=[.1,.2],CHIN_ci_low=[.1,.1],OPCID_ci_low=[.1,.1],CHID_ci_low=[.1,.1]))
    result=assess_multiscale_readiness(curve,cov,reg,zero_enrichment=.8,replicate_spearman=.7,no_leakage=True,known_overlap_zero=True)
    assert result['ready_for_task2b'] and len(result['checks'])==7 and not result['task2b_executed']
    result=assess_multiscale_readiness(curve,cov,reg,zero_enrichment=1.1,replicate_spearman=.7,no_leakage=True,known_overlap_zero=True)
    assert not result['ready_for_task2b'] and not result['task2b_executed']


def _config(tmp_path,mode='smoke',dry_run=False):
    from src.task2.pipeline import Task2AConfig
    project=tmp_path/'project';raw=tmp_path/'raw';project.mkdir(parents=True);raw.mkdir()
    for rep in ('rep1','rep2'):(raw/f'{rep}.cool').write_bytes(b'cool')
    known=project/'data'/'processed'/'structures.csv';known.parent.mkdir(parents=True)
    known.write_text('type,chrom,start,end,center\nCHIN,MG1655,1,2,1.5\n',encoding='utf-8')
    return Task2AConfig(project,raw,raw/'rep1.cool',raw/'rep2.cool',mode=mode,
                        detector_version='multiscale',dry_run=dry_run)


def test_multiscale_cli_config_and_isolated_paths(tmp_path):
    from scripts.run_task2a import parse_args
    from src.task2.multiscale_pipeline import MultiscalePaths
    assert parse_args(['--detector-version','multiscale']).detector_version=='multiscale'
    smoke=_config(tmp_path/'smoke');full=_config(tmp_path/'full',mode='full')
    sp=MultiscalePaths.from_config(smoke);fp=MultiscalePaths.from_config(full)
    assert smoke.epochs==3 and smoke.patience==2
    assert full.epochs==120 and full.patience==15
    assert sp.workspace_root.name=='task2a_multiscale_smoke'
    assert fp.data_root==full.project_root/'data'/'task2_multiscale'
    assert fp.output_root==full.project_root/'outputs'/'task2a_multiscale'


def test_multiscale_dryrun_writes_nothing_and_preserves_history(tmp_path):
    from src.task2.pipeline import execute_task2a
    config=_config(tmp_path,dry_run=True)
    for name in ('task2a','task2a_refined','task2a_calibrated'):
        path=config.project_root/'outputs'/name;path.mkdir(parents=True)
        (path/'sentinel').write_bytes(b'protected')
    result=execute_task2a(config)
    assert result.dry_run
    assert not (config.project_root/'outputs'/'pipeline_runs'/'task2a_multiscale_smoke').exists()
    for name in ('task2a','task2a_refined','task2a_calibrated'):
        assert (config.project_root/'outputs'/name/'sentinel').read_bytes()==b'protected'

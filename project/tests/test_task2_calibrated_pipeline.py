from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.task1.manifest import sha256_file
from src.task2.pipeline import Task2AConfig, Task2APaths, execute_task2a
from scripts.run_task2a import parse_args


def config_at(tmp_path, **kwargs):
    p=tmp_path/'project';p.mkdir(parents=True,exist_ok=True)
    raw=tmp_path/'raw';raw.mkdir(exist_ok=True)
    for rep in ('rep1','rep2'): (raw/f'{rep}.cool').write_bytes(b'synthetic')
    known=p/'data'/'processed'/'structures.csv';known.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(type=['CHIN'],chrom=['MG1655'],start=[10],end=[20],center=[15])).to_csv(known,index=False)
    return Task2AConfig(p,raw,raw/'rep1.cool',raw/'rep2.cool',detector_version='calibrated',**kwargs)


def test_calibrated_cli_and_output_isolation(tmp_path):
    assert parse_args(['--detector-version','calibrated']).detector_version=='calibrated'
    full=config_at(tmp_path,mode='full')
    p=Task2APaths.from_config(full);s=Task2APaths.from_config(replace(full,mode='smoke'))
    assert p.output_root==full.project_root/'outputs'/'task2a_calibrated'
    assert s.workspace_root==full.project_root/'outputs'/'pipeline_runs'/'task2a_calibrated_smoke'
    assert p.background_path.name=='background_split_calibrated.csv'
    assert full.epochs==120 and full.patience==15


def test_calibrated_dryrun_does_not_write_old_or_new_outputs(tmp_path):
    config=config_at(tmp_path,mode='full',dry_run=True)
    for name in ('task2a','task2a_refined'):
        p=config.project_root/'outputs'/name;p.mkdir(parents=True)
        (p/'sentinel.txt').write_bytes(b'protected')
    result=execute_task2a(config)
    assert result.dry_run
    assert not (config.project_root/'outputs'/'task2a_calibrated').exists()
    for name in ('task2a','task2a_refined'):
        assert (config.project_root/'outputs'/name/'sentinel.txt').read_bytes()==b'protected'


def cached_fixture(tmp_path):
    from src.task2.window_scan import generate_windows
    from src.task2.background import blocked_background_split
    config=config_at(tmp_path,mode='full')
    paths=Task2APaths.from_config(replace(config,detector_version='refined'))
    paths.data_root.mkdir(parents=True)
    wins=generate_windows('MG1655',24000)
    rows=[]
    for rep in ('rep1','rep2'):
        array=np.ones((len(wins),1,64,64),dtype=np.float32)
        np.save(paths.data_root/f'genome_windows_{rep}.npy',array)
        for i,w in enumerate(wins):
            rows.append(dict(**vars(w),replicate=rep,array_index=i,window_bp=6400,target_bin_size=100,
                             zero_axis=False,zero_row_count=0,zero_col_count=0,cool_path=str(getattr(config,f'{rep}_cool'))))
    meta=pd.DataFrame(rows);meta.to_csv(paths.metadata_path,index=False)
    np.savez(paths.expected_path,rep1=np.ones(64),rep2=np.ones(64))
    background=meta.loc[meta.replicate.eq('rep1')].copy();background['background_candidate']=True
    # Cache validator checks schema + old split integrity; tiny cache uses two blocks.
    background=blocked_background_split(background,block_bp=12000)
    background.to_csv(paths.background_path,index=False)
    files=[paths.metadata_path,paths.rep1_array_path,paths.rep2_array_path,paths.expected_path,
           paths.background_path,config.rep1_cool,config.rep2_cool,paths.structures_path]
    records={str(p.resolve()):{'sha256':sha256_file(p)} for p in files}
    return config,paths,records


def test_cache_validates_sha_shape_ids_alignment_expected(tmp_path):
    from src.task2.calibrated_pipeline import validate_scan_cache
    config,paths,records=cached_fixture(tmp_path)
    meta,expected,bg=validate_scan_cache(paths,config,24000,records)
    assert meta.window_id.nunique()==23 and len(expected['rep1'])==64 and len(bg)==23


@pytest.mark.parametrize('corruption',['sha','index','id','coordinate','missing','expected','shape','schema'])
def test_cache_rejects_corrupt_or_incomplete_artifacts(tmp_path,corruption):
    from src.task2.calibrated_pipeline import validate_scan_cache
    config,paths,records=cached_fixture(tmp_path)
    if corruption in ('sha','index','id','coordinate','missing'):
        meta=pd.read_csv(paths.metadata_path)
        if corruption in ('sha','index'): meta.loc[0,'array_index']=4
        if corruption=='id': meta.loc[0,'window_id']='invalid'
        if corruption=='coordinate': meta.loc[23,'end']=7000
        if corruption=='missing': meta=meta.drop(index=[0,23])
        meta.to_csv(paths.metadata_path,index=False)
    if corruption=='expected': np.savez(paths.expected_path,rep1=np.ones(63),rep2=np.ones(64))
    if corruption=='shape': np.save(paths.rep1_array_path,np.zeros((23,1,63,63)))
    if corruption=='schema': pd.DataFrame(dict(window_id=['x'])).to_csv(paths.background_path,index=False)
    if corruption!='sha':
        for p in records: records[p]['sha256']=sha256_file(Path(p))
    with pytest.raises(ValueError): validate_scan_cache(paths,config,24000,records)

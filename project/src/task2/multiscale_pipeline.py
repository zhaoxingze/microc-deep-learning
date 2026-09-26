"""Sequential, memory-bounded Task 2A.3 multi-scale orchestration."""
from __future__ import annotations

from dataclasses import dataclass,replace
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr,spearmanr

from src.data.known_dataset import attach_structure_ids
from src.task1.manifest import sha256_file
from src.task1.pipeline import PipelineLogger,PipelineStepError
from src.task2.autoencoder import WindowDataset,load_autoencoder,score_reconstruction,train_autoencoder,BackgroundAutoencoder
from src.task2.calibration import calibrate_scores
from src.task2.calibrated_evaluation import BUDGETS,random_baselines,zero_axis_audit
from src.task2.calibrated_pipeline import validate_scan_cache
from src.task2.expected import compute_genome_expected
from src.task2.multiscale import (SCALES,assert_background_known_clean,assert_cross_scale_no_leakage,
    assert_replicate_alignment,assign_scale_background,assess_multiscale_readiness,
    build_cross_scale_regions,detector_coverage_table,generate_scale_windows,known_scale_audit,
    make_feasible_master_block_split,scale_recall_sets,scale_rescue_summary,scan_scale_replicates,
    slice_expected,stable_scale_window_id,strict_background_candidates,validate_scale_score_pool,
    validate_scale_scan_manifest)
from src.task2.pipeline import Task2AConfig,Task2APaths,Task2AResult,_geometry,_validate_inputs
from src.task2.region_detection import select_coverage_budget
from src.task2.reporting import write_csv_utf8,write_manifest,validate_task2a_outputs
from src.task2.scoring import score_windows
from src.task2.shape_preprocessing import fit_background_oe_clips


@dataclass(frozen=True)
class MultiscalePaths:
    source_root: Path
    workspace_root: Path
    data_root: Path
    output_root: Path
    structures_path: Path
    expected_path: Path
    master_data_path: Path
    master_output_path: Path
    protocol_path: Path
    manifest_path: Path
    summary_path: Path
    log_dir: Path

    @classmethod
    def from_config(cls,config: Task2AConfig):
        source=config.project_root.resolve()
        workspace=source if config.mode=='full' else source/'outputs'/'pipeline_runs'/'task2a_multiscale_smoke'
        data=workspace/'data'/'task2_multiscale';output=workspace/'outputs'/'task2a_multiscale'
        return cls(source,workspace,data,output,source/'data'/'processed'/'structures.csv',
            data/'genome_expected_max128.npz',data/'master_block_split.csv',output/'master_block_split.csv',
            output/'frozen_multiscale_protocol.json',output/'task2a_multiscale_manifest.json',
            output/'task2a_multiscale_summary.md',source/'outputs'/'logs')


def execute_multiscale(config: Task2AConfig) -> Task2AResult:
    started=time.monotonic();paths=MultiscalePaths.from_config(config);_validate_inputs(config,paths)
    if config.dry_run:
        print(f'MULTISCALE DRY RUN: output={paths.output_root}; data={paths.data_root}',flush=True)
        print('Scales=3200/400,6400/800,12800/1600; strict interval background; shared blocks; historical outputs read-only; no Task2B.',flush=True)
        return Task2AResult(config.mode,None,None,None,time.monotonic()-started,True)
    return _execute_multiscale(config,paths,started)


def _execute_multiscale(config: Task2AConfig,paths: MultiscalePaths,started: float) -> Task2AResult:
    from src.task2.multiscale_reporting import write_multiscale_report
    paths.data_root.mkdir(parents=True,exist_ok=True);paths.output_root.mkdir(parents=True,exist_ok=True)
    log_path=paths.log_dir/f'run_task2a_multiscale_{config.mode}_{datetime.now():%Y%m%d_%H%M%S}.log'
    before=_protected_hashes(config.project_root)
    try:
        with PipelineLogger(log_path,config.project_root) as logger:
            logger.write(f'MultiScale mode={config.mode}; output={paths.output_root}')
            chrom_size,_=_geometry(config);domain=min(chrom_size,config.scan_end or chrom_size)
            known_all=attach_structure_ids(pd.read_csv(paths.structures_path,encoding='utf-8-sig'))
            if 'length_bp' not in known_all:known_all['length_bp']=known_all.end-known_all.start
            known=known_all.loc[known_all.chrom.eq(config.chrom)&known_all.center.ge(0)&known_all.center.lt(domain)].copy()
            expected_max=_load_expected(config,paths,logger)
            source_sha256={rep:sha256_file(getattr(config,f'{rep}_cool')) for rep in ('rep1','rep2')}
            metadata_by_scale={};arrays_by_scale={};input_artifacts=[];data_outputs=[paths.expected_path,paths.master_data_path]
            for spec in SCALES:
                if spec.name=='medium':
                    metadata,arrays,inputs,outputs=_reuse_medium(config,paths,spec,chrom_size,domain,logger)
                else:
                    metadata,arrays,inputs,outputs=_load_or_scan(config,paths,spec,chrom_size,source_sha256,logger)
                metadata_by_scale[spec.name]=metadata;arrays_by_scale[spec.name]=arrays
                input_artifacts.extend(inputs);data_outputs.extend(outputs)
                logger.write(f'{spec.name}: {metadata.window_id.nunique()} aligned windows')
            candidate_by_scale={};eligible_blocks={}
            for spec in SCALES:
                unique=metadata_by_scale[spec.name].loc[
                    metadata_by_scale[spec.name].replicate.eq('rep1')
                ].copy()
                candidates=strict_background_candidates(unique,known_all)
                candidate_by_scale[spec.name]=candidates
                block_end=((candidates.start//config.block_bp)+1)*config.block_bp
                fully_contained=candidates.background_candidate & candidates.end.le(block_end)
                eligible_blocks[spec.name]=set(
                    (candidates.loc[fully_contained,'start']//config.block_bp).astype(int)
                )
            master=make_feasible_master_block_split(
                config.chrom,domain,config.block_bp,eligible_blocks,seed=config.seed
            )
            write_csv_utf8(master,paths.master_data_path);write_csv_utf8(master,paths.master_output_path)
            split_by_scale={};all_background=[];clip_values={}
            for spec in SCALES:
                metadata=metadata_by_scale[spec.name]
                candidates=candidate_by_scale[spec.name]
                split=assign_scale_background(candidates,master,block_bp=config.block_bp);split['scale']=spec.name
                split_path=paths.data_root/spec.name/'background_split.csv';write_csv_utf8(split,split_path);data_outputs.append(split_path)
                split_by_scale[spec.name]=split;all_background.append(split)
                rows=metadata.merge(split[['window_id','split']],on='window_id',how='left',validate='many_to_one')
                rows['background_split']=rows.split.fillna('non_background')
                clip_values[spec.name]=fit_background_oe_clips(rows,arrays_by_scale[spec.name],slice_expected(expected_max,spec))
            background=pd.concat(all_background,ignore_index=True)
            overlap_audit=assert_background_known_clean(background,known_all)
            leakage_audit=assert_cross_scale_no_leakage(background)
            write_csv_utf8(overlap_audit,paths.output_root/'background_overlap_audit.csv')
            write_csv_utf8(overlap_audit,paths.output_root/'background_known_interval_overlap_audit.csv')
            write_csv_utf8(leakage_audit,paths.output_root/'background_split_leakage_audit.csv')
            logger.write('Strict interval background overlap=0 for all scales/splits; same/cross-scale leakage=0.')
            code_files=sorted((config.project_root/'src'/'task2').glob('*.py'))+[config.project_root/'scripts'/'run_task2a.py']
            protocol=_protocol(config,chrom_size,domain,master,metadata_by_scale,split_by_scale,clip_values,code_files)
            _write_json(paths.protocol_path,protocol);_write_json(paths.output_root/'protected_artifacts_before.json',before)
            logger.write('Multi-scale protocol, code SHA256 and master split frozen before AE training and known-recall evaluation.')
            score_tables=[];summary_rows=[]
            for spec in SCALES:
                logger.write(f'[{spec.name}] training independent background AE')
                metadata=metadata_by_scale[spec.name];split=split_by_scale[spec.name]
                rows=metadata.merge(split[['window_id','split']],on='window_id',how='left',validate='many_to_one')
                rows['background_split']=rows.split.fillna('non_background');expected=slice_expected(expected_max,spec)
                make=lambda frame:WindowDataset(frame,arrays_by_scale[spec.name],input_type='oe_log_robust',
                                                expected=expected,oe_clip_values=clip_values[spec.name])
                scale_root=paths.output_root/spec.name
                training=train_autoencoder(make(rows.loc[rows.split.eq('train')]),make(rows.loc[rows.split.eq('validation')]),
                    scale_root,epochs=config.epochs,patience=config.patience,seed=config.seed)
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model=load_autoencoder(scale_root/'best_model.pth',device)
                reconstruction=score_reconstruction(model,make(metadata),device=device)
                scores,robust=score_windows(metadata,arrays_by_scale[spec.name],expected,reconstruction,
                    set(split.loc[split.split.eq('train'),'window_id']),primary_fusion='or_max')
                scores,calibration=calibrate_scores(scores,split)
                scores['scale']=spec.name;scores['window_bp']=spec.window_bp;scores['step_bp']=spec.step_bp
                scores['scale_or_score']=scores.calibrated_or_score;scores['calibration_id']=spec.name
                scores['background_split']=scores.window_id.map(split.set_index('window_id').split).fillna('not_eligible')
                scores['background_candidate']=scores.window_id.isin(split.window_id)
                write_csv_utf8(scores,scale_root/'window_scores.csv');_write_json(scale_root/'calibration.json',calibration)
                score_tables.append(scores)
                counts=split.split.value_counts()
                summary_rows.append(dict(scale=spec.name,n_windows=len(scores),window_bp=spec.window_bp,step_bp=spec.step_bp,
                    train_background=int(counts.get('train',0)),validation_background=int(counts.get('validation',0)),
                    calibration_background=int(counts.get('calibration',0)),best_epoch=training.best_epoch,
                    best_validation_loss=training.best_validation_loss,minimum_empirical_p=1/(int(counts.get('calibration',0))+1),
                    rep1_oe_clip=clip_values[spec.name]['rep1'],rep2_oe_clip=clip_values[spec.name]['rep2'],
                    robust_scales=robust))
                logger.write(f'[{spec.name}] locked epoch={training.best_epoch}, val_loss={training.best_validation_loss:.9f}')
                del model,reconstruction,training,rows,scores;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
            pool=pd.concat(score_tables,ignore_index=True);validate_scale_score_pool(pool)
            write_csv_utf8(pool,paths.output_root/'multiscale_window_scores.csv')
            multiscale_regions,membership=build_cross_scale_regions(pool,seed=config.seed)
            write_csv_utf8(multiscale_regions,paths.output_root/'multiscale_candidate_regions.csv')
            write_csv_utf8(membership,paths.output_root/'region_membership.csv')
            detector_pools={name:pool.loc[pool.scale.eq(name)].copy() for name in ('small','medium','large')}
            detector_pools['multiscale']=pool
            coverage,regions_by_detector=detector_coverage_table(detector_pools,known,domain,seed=config.seed)
            for name in ('small','medium','large'):
                write_csv_utf8(regions_by_detector[name],paths.output_root/name/'candidate_regions.csv')
            write_csv_utf8(coverage,paths.output_root/'multiscale_coverage_recall.csv')
            write_csv_utf8(coverage,paths.output_root/'scale_detector_comparison.csv')
            audit=known_scale_audit(pool,multiscale_regions,known,domain,
                regions_by_scale={k:regions_by_detector[k] for k in ('small','medium','large')})
            write_csv_utf8(audit,paths.output_root/'known_structure_scale_audit.csv')
            recall_sets=scale_recall_sets({k:regions_by_detector[k] for k in ('small','medium','large')},known,domain)
            rescue=scale_rescue_summary(known,recall_sets);write_csv_utf8(rescue,paths.output_root/'scale_rescue_summary.csv')
            cov_random,reg_random,draws=random_baselines(pool,multiscale_regions,known,{config.chrom:domain},
                budgets=BUDGETS,repeats=config.random_repeats,seed=config.seed)
            write_csv_utf8(cov_random,paths.output_root/'coverage_random_baseline.csv')
            write_csv_utf8(reg_random,paths.output_root/'region_random_baseline.csv')
            write_csv_utf8(draws,paths.output_root/'random_baseline_draws.csv')
            replicate={'pearson':float(pearsonr(multiscale_regions.rep1_peak_score,multiscale_regions.rep2_peak_score).statistic),
                       'spearman':float(spearmanr(multiscale_regions.rep1_peak_score,multiscale_regions.rep2_peak_score).statistic),
                       'unit':'multiscale peak windows','n':len(multiscale_regions)}
            write_csv_utf8(pd.DataFrame([replicate]),paths.output_root/'replicate_consistency.csv')
            selected20,_=select_coverage_budget(multiscale_regions,.20,domain);zero=zero_axis_audit(pool,selected20)
            write_csv_utf8(pd.DataFrame([zero]),paths.output_root/'zero_axis_audit.csv')
            multi_curve=coverage.loc[coverage.detector.eq('multiscale')]
            no_leakage=bool(leakage_audit.overlap_count.sum()==0)
            readiness=assess_multiscale_readiness(multi_curve,cov_random,reg_random,zero_enrichment=zero['enrichment'],
                replicate_spearman=replicate['spearman'],no_leakage=no_leakage,known_overlap_zero=True,mode=config.mode)
            scale_summary=pd.DataFrame(summary_rows);serial_summary=scale_summary.drop(columns='robust_scales')
            write_csv_utf8(serial_summary,paths.output_root/'scale_summary.csv')
            diagnosis=_diagnosis(readiness,coverage,rescue,known)
            statistics=dict(n_windows_by_scale={r['scale']:r['n_windows'] for r in summary_rows},
                n_candidate_regions=len(multiscale_regions),scale_summary=serial_summary,
                known_overlap_zero=True,no_leakage=no_leakage,replicate_consistency=replicate,zero_axis=zero,
                readiness=readiness,ready_for_task2b=readiness['ready_for_task2b'],diagnosis=diagnosis)
            tables={'coverage':coverage,'coverage_random':cov_random,'region_random':reg_random,
                    'rescue':rescue,'known_audit':audit}
            write_multiscale_report(paths,config,protocol,statistics,tables)
            after=_protected_hashes(config.project_root)
            if before!=after:raise ValueError('historical Task2A outputs/data changed')
            _write_json(paths.output_root/'protected_artifacts_audit.json',{'unchanged':True,'file_count':len(after),'hashes':after})
            for name,digest in protocol['code_sha256'].items():
                if sha256_file(name)!=digest:raise ValueError('code changed after multi-scale protocol freeze')
            root_outputs=[p for p in paths.output_root.rglob('*') if p.is_file() and p!=paths.manifest_path]
            output_paths=sorted(set(root_outputs+data_outputs),key=lambda p:str(p))
            input_paths=sorted(set([config.rep1_cool,config.rep2_cool,paths.structures_path,*input_artifacts,*code_files]),key=lambda p:str(p))
            manifest_stats={k:v for k,v in statistics.items() if k!='scale_summary'}
            manifest_stats['scale_summary']=serial_summary.to_dict('records')
            write_manifest(paths.manifest_path,configuration=protocol,input_paths=input_paths,
                output_paths=output_paths,statistics=manifest_stats)
            validate_task2a_outputs(paths.manifest_path)
            logger.write(f'COMPLETE: regions={len(multiscale_regions)}, ready_for_task2b={readiness["ready_for_task2b"]}; Task2B not executed.')
    except Exception as exc:
        raise PipelineStepError(f'Multi-scale Task2A failed: {exc}. See {log_path}') from exc
    return Task2AResult(config.mode,paths.summary_path,paths.manifest_path,log_path,time.monotonic()-started)


def _write_json(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def _protected_hashes(project_root: Path):
    files=[]
    for name in ('task2a','task2a_refined','task2a_calibrated'):
        root=project_root/'outputs'/name
        files.extend(p for p in root.rglob('*') if p.is_file())
    root=project_root/'data'/'task2';files.extend(p for p in root.rglob('*') if p.is_file())
    return {str(p.resolve()):sha256_file(p) for p in sorted(files)}


def _load_expected(config,paths,logger):
    if paths.expected_path.is_file():
        with np.load(paths.expected_path,allow_pickle=False) as stored:
            values={r:stored[r].copy() for r in ('rep1','rep2')}
        if all(v.shape==(128,) and np.isfinite(v).all() for v in values.values()):
            logger.write('Reused validated max-distance127 expected vectors.');return values
    values={r:compute_genome_expected(getattr(config,f'{r}_cool'),config.chrom,
        target_bin_size=100,max_distance_bins=127) for r in ('rep1','rep2')}
    np.savez(paths.expected_path,**values);logger.write('Computed genome expected once per replicate through distance127.')
    return values


def _validate_scale_files(directory: Path,spec,chrom,chrom_size,scan_end,cool_paths,source_sha256):
    validate_scale_scan_manifest(directory,spec,cool_paths,source_sha256=source_sha256)
    metadata=pd.read_csv(directory/'metadata.csv',encoding='utf-8-sig');assert_replicate_alignment(metadata)
    expected=generate_scale_windows(spec,chrom,chrom_size,scan_end=scan_end)
    for rep in ('rep1','rep2'):
        rows=metadata.loc[metadata.replicate.eq(rep)].sort_values('array_index')
        if rows.window_id.tolist()!=[w.window_id for w in expected]:raise ValueError('scale cache IDs stale')
        if not np.array_equal(rows[['start','end','center']].to_numpy(),np.asarray([[w.start,w.end,w.center] for w in expected])):
            raise ValueError('scale cache coordinates stale')
        if {Path(p).resolve() for p in rows.cool_path}!={Path(cool_paths[rep]).resolve()}:raise ValueError('scale cache source stale')
        array=np.load(directory/f'genome_windows_{rep}.npy',mmap_mode='r',allow_pickle=False)
        if array.shape!=(len(expected),1,spec.matrix_bins,spec.matrix_bins) or array.dtype!=np.float32:
            raise ValueError('scale cache array invalid')
    return metadata


def _load_or_scan(config,paths,spec,chrom_size,source_sha256,logger):
    directory=paths.data_root/spec.name;arrays={r:directory/f'genome_windows_{r}.npy' for r in ('rep1','rep2')}
    try:
        metadata=_validate_scale_files(directory,spec,config.chrom,chrom_size,config.scan_end,
            {'rep1':config.rep1_cool,'rep2':config.rep2_cool},source_sha256)
        logger.write(f'{spec.name}: reused validated local memmap scan')
    except (OSError,ValueError,KeyError):
        metadata=scan_scale_replicates(config.rep1_cool,config.rep2_cool,spec,config.chrom,chrom_size,directory,
            scan_end=config.scan_end,source_sha256=source_sha256)
    outputs=[directory/'metadata.csv',directory/'scan_manifest.json',*arrays.values()]
    return metadata,arrays,[],outputs


def _reuse_medium(config,paths,spec,chrom_size,domain,logger):
    original_config=replace(config,mode='full',detector_version='refined')
    original=Task2APaths.from_config(original_config)
    calibrated=Task2APaths.from_config(replace(config,mode='full',detector_version='calibrated'))
    manifest=json.loads(calibrated.manifest_path.read_text(encoding='utf-8'))
    metadata,_,_=validate_scan_cache(original,original_config,chrom_size,{**manifest['inputs'],**manifest['outputs']})
    metadata=metadata.loc[metadata.end.le(domain)].copy()
    metadata['scale']='medium';metadata['step_bp']=spec.step_bp
    metadata['window_id']=[stable_scale_window_id('medium',r.chrom,r.start,r.end) for r in metadata.itertuples(index=False)]
    assert_replicate_alignment(metadata)
    directory=paths.data_root/'medium';directory.mkdir(parents=True,exist_ok=True)
    meta_path=directory/'metadata.csv';write_csv_utf8(metadata,meta_path)
    source_path=directory/'array_source.json';_write_json(source_path,{'reused':True,
        'rep1':str(original.rep1_array_path),'rep2':str(original.rep2_array_path),
        'reference_manifest':str(calibrated.manifest_path)})
    logger.write('medium: SHA/geometry/order validated and reused Task2A.2 float32 arrays')
    inputs=[original.metadata_path,original.rep1_array_path,original.rep2_array_path,original.expected_path,
            original.background_path,calibrated.manifest_path]
    return metadata,{'rep1':original.rep1_array_path,'rep2':original.rep2_array_path},inputs,[meta_path,source_path]


def _protocol(config,chrom_size,domain,master,metadata,split,clips,code_files):
    return dict(version='task2a.3-multiscale-v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        mode=config.mode,scales=[dict(name=s.name,window_bp=s.window_bp,step_bp=s.step_bp,matrix_bins=s.matrix_bins) for s in SCALES],
        target_bin_size=100,expected='one genome-wide vector per replicate d=0..127, sliced first32/64/128',
        chromosome_size=chrom_size,evaluation_domain_size=domain,background_rule='strict half-open known interval non-overlap; margin_bp=0',
        block_bp=config.block_bp,
        master_block_assignment='first seeded whole-block permutation giving every scale nonempty train/validation/calibration; then shared by all scales',
        master_assignment_attempt=int(master.assignment_attempt.iloc[0]) if 'assignment_attempt' in master else 1,
        master_block_sha256=sha256_file(config.project_root/'data'/'task2_multiscale'/'master_block_split.csv') if config.mode=='full' else None,
        master_block_counts=master.split.value_counts().to_dict(),window_counts={k:int(v.window_id.nunique()) for k,v in metadata.items()},
        background_counts={k:v.split.value_counts().to_dict() for k,v in split.items()},oe_clip_values=clips,
        AE=dict(architecture=str(BackgroundAutoencoder()),epochs=config.epochs,patience=config.patience,batch_size=32,
                learning_rate=.001,loss='full-pixel MSE',checkpoint='minimum background Validation MSE only'),
        calibration='per-scale/per-replicate/per-branch p=(1+count(cal>=x))/(n+1), inclusive upper tail; A=-log10(p)',
        within_scale='mean replicate anomaly within branch; scale_or=max(paired_density,paired_shape)',
        cross_scale_fusion='unweighted maximum via one ranked calibrated window pool',
        cross_scale_nms='suppress iff same chrom and abs(center difference)<max(window_bp_new,window_bp_peak)/2',
        formal_region='exact peak window [start,end); support union diagnostic only',coverage_budgets=list(BUDGETS),
        random_repetitions=config.random_repeats,random_seed=config.seed,random_domain='linear',
        readiness=['overall10>coverage random97.5','overall20>coverage random97.5',
            'all classes20 positive and >=region random2.5','zero-axis enrichment<=1','peak rep Spearman>0',
            'same/cross-scale no split leakage','all background known overlap=0'],task2b_auto_execution=False,
        torch_version=torch.__version__,torch_num_threads=torch.get_num_threads(),
        code_sha256={str(p.resolve()):sha256_file(p) for p in code_files})


def _diagnosis(readiness,coverage,rescue,known):
    if readiness['ready_for_task2b']:
        return 'Task2A multi-scale detector passed the pre-registered readiness gate. Stop; next suggested phase is Task2B Structure Representation.'
    at20=rescue.loc[np.isclose(rescue.budget,.2)]
    small_chin=int(at20.loc[at20.type.eq('CHIN'),'recalled_small_only'].sum())
    large_chid=int(at20.loc[at20.type.eq('CHID'),'recalled_large_only'].sum())
    medium=float(coverage.loc[coverage.detector.eq('medium')&np.isclose(coverage.budget,.2),'overall_recall'].iloc[0])
    multi=float(coverage.loc[coverage.detector.eq('multiscale')&np.isclose(coverage.budget,.2),'overall_recall'].iloc[0])
    missed=int(at20.missed_all_scales.sum());total=len(known)
    if small_chin+large_chid>0:
        return (f'At 20% coverage, Small uniquely rescues {small_chin} CHIN and Large uniquely rescues {large_chid} CHID; '
                f'MultiScale Overall={multi:.4f} versus Medium={medium:.4f}, while {missed}/{total} remain missed by all scales. '
                'Scale contributes distinct rescues, but failed gates indicate scale alone is insufficient; representation/anomaly strength remains a bottleneck.')
    return (f'At 20% coverage no unique Small-CHIN or Large-CHID rescue was observed; MultiScale Overall={multi:.4f} versus '
            f'Medium={medium:.4f}, and {missed}/{total} remain missed by all scales. Evidence does not support window scale as the main fix; '
            'the representation or the assumption that known structures are strong background anomalies requires re-examination.')

"""Task 2A.2 orchestration. Historical artifacts are read-only inputs."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.data.known_dataset import attach_structure_ids
from src.task1.manifest import sha256_file
from src.task1.pipeline import PipelineLogger, PipelineStepError, PipelineValidationError
from src.task2.autoencoder import BackgroundAutoencoder, WindowDataset, load_autoencoder, score_reconstruction, train_autoencoder
from src.task2.background import (assert_no_background_overlap, assert_three_way_no_overlap, audit_known_interval_overlap,
    build_exclusion_intervals, calibrated_background_split, mark_background_candidates)
from src.task2.calibration import calibrate_scores
from src.task2.calibrated_evaluation import (BUDGETS, assess_readiness, compare_detectors, coverage_curve,
    known_audit, missed_summary, random_baselines, zero_axis_audit)
from src.task2.expected import compute_genome_expected
from src.task2.pipeline import (Task2APaths, Task2AResult, _geometry, _validate_inputs,
    _validate_scan, _validate_expected)
from src.task2.region_detection import build_candidate_regions, rank_windows, select_coverage_budget
from src.task2.reporting import write_csv_utf8, write_manifest, validate_task2a_outputs
from src.task2.scoring import score_windows
from src.task2.shape_preprocessing import fit_background_oe_clips
from src.task2.window_scan import generate_windows, scan_replicates


PROTOCOL = {
    'version': 'task2a.2-calibrated-v1', 'tail_formula': '(1 + count(cal >= x))/(n+1)',
    'tail_direction': 'upper', 'ties': 'inclusive', 'anomaly_formula': '-log10(p)',
    'replicate_combination': 'mean anomaly within branch, then max between branches',
    'primary_score': 'max(paired_density_anomaly,paired_shape_anomaly)',
    'minimum_p': '1/(n_calibration+1)',
    'bonferroni': 'min(1,2*min(p_density,p_shape)), per-replicate explanatory field only',
    'paired_p_interpretation': 'averaged negative logs are ranking scores, not a proven combined p-value',
    'split_target': [.70,.15,.15], 'split_method': 'one seeded shuffle of sorted whole blocks',
    'nms_radius_bp': 3200, 'nms_condition': 'same chromosome, center distance strictly < radius',
    'region_rule': 'assign to first higher-ranked nearby peak; keep 800-bp contiguous support component containing peak; bounds=min(start),max(end); no peak merging',
    'tie_rule': 'descending score then ascending SHA256(seed|window_id)',
    'mixed_branch': 'both paired anomalies >= -log10(0.05); diagnostic only',
    'coverage_budgets': list(BUDGETS), 'curve_budgets': (np.arange(41)/100).tolist(),
    'coverage_selection': 'whole ranked regions until union >= target; overshoot retained',
    'recall': 'unique known center in half-open selected-region union',
    'auxiliary_recall': 'known interval overlaps selected-region union > 0 bp',
    'random_domain': 'linear chromosome, no circular wrap-around',
    'coverage_random': 'permuted existing scan windows, greedy union target, 2% tolerance, 32 attempts',
    'region_random': 'independent uniform integer starts within chromosome, exact count and length multiset; overlap permitted',
    'random_interval': 'empirical 2.5/97.5 percentiles over random repetitions; not confidence interval on mean',
    'readiness': ['overall at10 > coverage random97.5','overall at20 > coverage random97.5',
        'all classes at20 positive and >= region random2.5','zero-axis enrichment at20 <=1',
        'replicate OR Spearman>0','abs(paired branch Spearman)<0.9','three-way no genomic overlap'],
    'coverage_evidence_validity': 'readiness comparison requires budget reached and >=100 repeats all within 2% coverage tolerance; invalid evidence fails closed',
    'known_interval_safety': 'inherited center exclusion unchanged; audit partial Train/Val known interval overlaps; abort if any Calibration interval overlaps known interval',
    'zero_axis_definition': 'windows with centers in selected20 region union / all scan windows',
    'task_boundary': 'Task2A only; never invoke Task2B even if ready',
    'shape_preprocessing': {'input':'genome_OE', 'clip_percentile':99.5, 'clip_fit':'background_train_only',
        'transform':'log1p then per-window (x-median)/(1.4826*MAD+1e-8)',
        'valid_mask':'upper triangle distance>2 bins','MAD_le_1e-8_fallback_scale':1.0,'normalized_clip':[-10,10]},
    'AE_loss':'full-pixel MSE, unchanged architecture', 'shape_score':'upper triangle distance>2 MSE',
    'density_score':'upper triangle distance>2 mean genome O/E',
    'best_known_scores':'all best fields from highest OR-ranked interval-overlapping window; separate independent maxima also saved',
}


def _json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def protected_hashes(project_root):
    roots=[project_root/'outputs'/'task2a',project_root/'outputs'/'task2a_refined']
    files=[p for root in roots for p in root.rglob('*') if p.is_file()]
    data=project_root/'data'/'task2'
    files += [data/name for name in ('genome_windows_rep1.npy','genome_windows_rep2.npy',
        'genome_windows_metadata.csv','genome_expected.npz','background_split.csv') if (data/name).is_file()]
    return {str(p.resolve()):sha256_file(p) for p in sorted(files)}


def validate_scan_cache(paths, config, chrom_size, records):
    """Hashes are checked against a prior manifest, not a newly computed self-reference."""
    required=[paths.metadata_path,paths.rep1_array_path,paths.rep2_array_path,paths.expected_path,
              paths.background_path,config.rep1_cool,config.rep2_cool,paths.structures_path]
    for p in required:
        record=records.get(str(p.resolve()))
        if not p.is_file() or not record or sha256_file(p)!=record['sha256']:
            raise ValueError(f'cache SHA256 validation failed: {p}')
    try:
        meta=_validate_scan(paths,replace(config,mode='full'))
        expected=_validate_expected(paths,config)
    except PipelineValidationError as exc:
        raise ValueError(str(exc)) from exc
    coordinates=generate_windows(config.chrom,chrom_size,window_bp=config.window_bp,step_bp=config.step_bp)
    target=pd.DataFrame([vars(w) for w in coordinates])
    cols=['window_id','chrom','start','end','center']
    if set(meta.replicate) != {'rep1','rep2'}:
        raise ValueError('invalid replicate metadata')
    for rep,p in [('rep1',paths.rep1_array_path),('rep2',paths.rep2_array_path)]:
        rows=meta.loc[meta.replicate.eq(rep)].reset_index(drop=True)
        if len(rows)!=len(target) or rows.window_id.duplicated().any():
            raise ValueError('cache missing/duplicate complete scan windows')
        if not np.array_equal(rows.array_index.to_numpy(),np.arange(len(target))):
            raise ValueError('cache array order/index mismatch')
        for c in cols:
            if not np.array_equal(rows[c].to_numpy(),target[c].to_numpy()):
                raise ValueError(f'cache stable ID/coordinate mismatch: {rep}/{c}')
        array=np.load(p,mmap_mode='r',allow_pickle=False)
        if array.dtype!=np.float32 or not np.isfinite(array).all() or (array<0).any():
            raise ValueError('cache array has invalid dtype/values')
        for col in ('zero_axis','zero_row_count','zero_col_count'):
            if col not in rows:
                raise ValueError('cache zero-axis schema incomplete')
    bg=pd.read_csv(paths.background_path,encoding='utf-8-sig')
    required_cols={'window_id','chrom','start','end','center','background_candidate','background_block','background_split'}
    if not required_cols.issubset(bg):
        raise ValueError('cache original split schema incomplete')
    if len(bg)!=len(target) or bg.window_id.duplicated().any() or set(bg.window_id)!=set(target.window_id):
        raise ValueError('cache original split IDs incomplete')
    aligned=bg.set_index('window_id').loc[target.window_id]
    for col in ('chrom','start','end','center'):
        if not np.array_equal(aligned[col].to_numpy(),target[col].to_numpy()):
            raise ValueError('cache original split coordinates mismatch')
    if not set(bg.background_split).issubset({'train','val','boundary_excluded','non_background'}):
        raise ValueError('cache original split labels invalid')
    assert_no_background_overlap(bg)
    return meta,expected,bg


def _load_inputs(config,paths,chrom_size,logger):
    source=Task2APaths.from_config(replace(config,mode='full',detector_version='refined'))
    try:
        manifest=json.loads(source.manifest_path.read_text(encoding='utf-8'))
        records={**manifest['inputs'],**manifest['outputs']}
        metadata,expected,background=validate_scan_cache(source,config,chrom_size,records)
        logger.write('Validated prior SHA256, complete geometry/IDs/order, rep alignment, expected and original split; reusing scan.')
        arrays={'rep1':source.rep1_array_path,'rep2':source.rep2_array_path}
        used=[source.metadata_path,*arrays.values(),source.expected_path,source.background_path,source.manifest_path]
        validation={'reused':True,'reference_manifest':str(source.manifest_path),'original_background_rows':len(background)}
    except (ValueError,OSError,KeyError) as exc:
        # Never repair an old artifact in-place. An invalid cache is rebuilt separately.
        logger.write(f'Artifact validation failed; isolated scan rebuild required: {exc}')
        fresh=paths.output_root/'rebuilt_scan'
        metadata=scan_replicates(config.rep1_cool,config.rep2_cool,config.chrom,chrom_size,fresh,
            window_bp=config.window_bp,step_bp=config.step_bp,target_bin_size=config.target_bin_size,scan_end=config.scan_end)
        arrays={rep:fresh/f'genome_windows_{rep}.npy' for rep in ('rep1','rep2')}
        expected={rep:compute_genome_expected(getattr(config,f'{rep}_cool'),config.chrom,
            target_bin_size=config.target_bin_size,max_distance_bins=63) for rep in ('rep1','rep2')}
        np.savez(fresh/'genome_expected.npz',**expected)
        used=[fresh/'genome_windows_metadata.csv',*arrays.values(),fresh/'genome_expected.npz']
        validation={'reused':False,'validation_error':str(exc)}
    if config.scan_end is not None:
        metadata=metadata.loc[metadata.end.le(config.scan_end)].copy()
    return metadata,expected,arrays,used,validation


def execute_calibrated(config):
    from src.task2.calibrated_reporting import write_calibrated_report
    started=time.monotonic();paths=Task2APaths.from_config(config)
    _validate_inputs(config,paths)
    if config.dry_run:
        print(f'CALIBRATED DRY RUN: output={paths.output_root}; split={paths.background_path}',flush=True)
        print('Reuse validated original scan; fresh 3-way split / AE / calibration / regions. Historical outputs read-only. No Task2B.',flush=True)
        return Task2AResult(config.mode,None,None,None,time.monotonic()-started,True)
    paths.output_root.mkdir(parents=True,exist_ok=True);paths.data_root.mkdir(parents=True,exist_ok=True)
    log_path=paths.log_dir/f'run_task2a_calibrated_{config.mode}_{datetime.now():%Y%m%d_%H%M%S}.log'
    before=protected_hashes(config.project_root)
    try:
        with PipelineLogger(log_path,config.project_root) as logger:
            logger.write(f'Calibrated mode={config.mode}; output={paths.output_root}')
            chrom_size,_=_geometry(config);domain=min(chrom_size,config.scan_end or chrom_size)
            metadata,expected,arrays,input_artifacts,validation=_load_inputs(config,paths,chrom_size,logger)
            known_all=attach_structure_ids(pd.read_csv(paths.structures_path,encoding='utf-8-sig'))
            known=known_all.loc[known_all.chrom.eq(config.chrom)&known_all.center.ge(0)&known_all.center.lt(domain)].copy()
            exclusions=build_exclusion_intervals(known_all,window_bp=config.window_bp,margin_bp=config.exclusion_margin_bp)
            candidates=mark_background_candidates(metadata,exclusions)
            split=calibrated_background_split(candidates,block_bp=config.block_bp,seed=config.seed)
            assert_three_way_no_overlap(split)
            known_overlap_audit=audit_known_interval_overlap(split,known_all)
            write_csv_utf8(known_overlap_audit,paths.output_root/'background_known_interval_overlap_audit.csv')
            write_csv_utf8(split,paths.background_path)
            counts={s:int(split.split.eq(s).sum()) for s in ('train','validation','calibration')}
            block_counts={s:int(split.loc[split.split.eq(s),'block_id'].nunique()) for s in counts}
            logger.write(f'Three-way background unique windows={counts}; blocks={block_counts}; pairwise coordinate leakage=NONE')
            rows=metadata.merge(split[['window_id','split']],on='window_id',how='left',validate='many_to_one')
            rows['background_split']=rows.split.fillna('non_background')
            clips=fit_background_oe_clips(rows,arrays,expected)
            code_files=sorted((config.project_root/'src'/'task2').glob('*.py'))+[config.project_root/'scripts'/'run_task2a.py']
            frozen={**PROTOCOL,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                'mode':config.mode,'random_seed':config.seed,'random_repetitions':config.random_repeats,
                'window_bp':config.window_bp,'step_bp':config.step_bp,'target_bin_size':config.target_bin_size,
                'chromosome_size':chrom_size,'evaluation_domain_size':domain,'block_bp':config.block_bp,
                'background_counts':counts,'block_counts':block_counts,'oe_clip_values':clips,
                'actual_background_fractions':{s:n/sum(counts.values()) for s,n in counts.items()},
                'known_interval_overlap_counts':{s:int(known_overlap_audit.split.eq(s).sum()) for s in counts},
                'epochs':config.epochs,'patience':config.patience,'batch_size':32,'learning_rate':.001,
                'architecture':str(BackgroundAutoencoder()),'torch_version':torch.__version__,
                'torch_num_threads':torch.get_num_threads(),
                'code_sha256':{str(p):sha256_file(p) for p in code_files},
                'split_sha256':sha256_file(paths.background_path),'cache_validation':validation}
            protocol_path=paths.output_root/'frozen_protocol.json';_json(protocol_path,frozen)
            _json(paths.output_root/'protected_artifacts_before.json',before)
            logger.write('Scientific protocol and code/split hashes frozen before training and known-recall evaluation.')
            make_dataset=lambda r:WindowDataset(r,arrays,input_type='oe_log_robust',expected=expected,oe_clip_values=clips)
            training=train_autoencoder(make_dataset(rows.loc[rows.split.eq('train')]),
                make_dataset(rows.loc[rows.split.eq('validation')]),paths.autoencoder_root,
                epochs=config.epochs,patience=config.patience,seed=config.seed)
            logger.write(f'AE locked: epoch={training.best_epoch}, validation_loss={training.best_validation_loss:.9f}')
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model=load_autoencoder(paths.autoencoder_root/'best_model.pth',device)
            reconstruction=score_reconstruction(model,make_dataset(metadata),device=device)
            scores,scales=score_windows(metadata,arrays,expected,reconstruction,
                set(split.loc[split.split.eq('train'),'window_id']),primary_fusion='or_max')
            scores,calibration=calibrate_scores(scores,split)
            for dst,src in {'density_z':'paired_density_z','shape_z':'paired_shape_z','legacy_mean':'score_mean',
                            'or_max_z':'score_or_max','positive_sum_z':'score_positive_sum'}.items():
                scores[dst]=scores[src]
            scores['primary_score']=scores.calibrated_or_score;scores['primary_detector']='CALIBRATED_OR'
            scores=rank_windows(scores,seed=config.seed)
            scores['background_split_calibrated']=scores.window_id.map(split.set_index('window_id').split).fillna('not_eligible')
            scores['background_candidate']=scores.window_id.map(candidates.set_index('window_id').background_candidate)
            regions,members=build_candidate_regions(scores,seed=config.seed)
            write_csv_utf8(scores,paths.candidate_scores_path);write_csv_utf8(regions,paths.candidate_regions_path)
            write_csv_utf8(members,paths.output_root/'region_membership.csv')
            _json(paths.output_root/'calibration.json',calibration)
            logger.write(f'Frozen scoring and regions complete: {len(scores)} windows, {len(regions)} regions. Starting evaluation without tuning.')
            curve=coverage_curve(regions,known,domain)
            cov_random,reg_random,draws=random_baselines(scores,regions,known,{config.chrom:domain},
                repeats=config.random_repeats,seed=config.seed)
            logger.write('Coverage-matched and region-length-matched random baselines complete.')
            audit=known_audit(scores,regions,known,domain);missed=missed_summary(audit)
            comparison=compare_detectors(scores,known,domain,seed=config.seed)
            correlations={}
            for name,a,b in [('replicates','rep1_calibrated_or_score','rep2_calibrated_or_score'),
                             ('branches','density_tail_score','shape_tail_score')]:
                correlations[name]={'pearson':float(pearsonr(scores[a],scores[b]).statistic),
                                    'spearman':float(spearmanr(scores[a],scores[b]).statistic)}
            selected20,_=select_coverage_budget(regions,.20,domain)
            zero=zero_axis_audit(scores,selected20)
            assert_three_way_no_overlap(split)
            readiness=assess_readiness(curve,cov_random,reg_random,zero_enrichment=zero['enrichment'],
                replicate_spearman=correlations['replicates']['spearman'],branch_spearman=correlations['branches']['spearman'],
                no_overlap=True,mode=config.mode)
            tables={'coverage_recall_curve':curve,'coverage_random_baseline':cov_random,'region_random_baseline':reg_random,
                'random_baseline_draws':draws,'known_structure_detection_audit':audit,'missed_known_summary':missed,
                'detector_comparison':comparison,'replicate_consistency':pd.DataFrame([correlations['replicates']]),
                'branch_correlation':pd.DataFrame([correlations['branches']]),'zero_axis_audit':pd.DataFrame([zero])}
            for name,table in tables.items():write_csv_utf8(table,paths.output_root/f'{name}.csv')
            statistics={'n_windows':len(scores),'n_known_structures':len(known),'n_candidate_regions':len(regions),
                'background_counts':counts,'block_counts':block_counts,'best_epoch':training.best_epoch,
                'known_interval_overlap_counts':frozen['known_interval_overlap_counts'],
                'best_validation_loss':training.best_validation_loss,'correlations':correlations,'zero_axis':zero,
                'robust_scales_train_diagnostic':scales,'readiness':readiness,'ready_for_task2b':readiness['ready_for_task2b'],
                'minimum_empirical_p':1/(counts['calibration']+1),
                'calibrated_or_unique_scores':int(scores.calibrated_or_score.nunique()),
                'calibrated_or_max_ties':int(scores.calibrated_or_score.eq(scores.calibrated_or_score.max()).sum())}
            write_calibrated_report(paths,config,frozen,statistics,tables,scores,training.history)
            after=protected_hashes(config.project_root)
            if before!=after:raise ValueError('Historical output or original data hash changed')
            _json(paths.output_root/'protected_artifacts_audit.json',{'unchanged':True,'file_count':len(before),'hashes':after})
            for name,digest in frozen['code_sha256'].items():
                if sha256_file(name)!=digest:raise ValueError('Scientific code changed after protocol freeze')
            outputs=[p for p in paths.output_root.rglob('*') if p.is_file() and p!=paths.manifest_path]
            outputs.append(paths.background_path)
            write_manifest(paths.manifest_path,configuration={**frozen,'best_epoch':training.best_epoch,
                'best_validation_loss':training.best_validation_loss},input_paths=[config.rep1_cool,config.rep2_cool,
                paths.structures_path,*input_artifacts,*code_files],output_paths=outputs,statistics=statistics)
            validate_task2a_outputs(paths.manifest_path)
            logger.write(f'COMPLETE: {paths.summary_path}; ready_for_task2b={readiness["ready_for_task2b"]}; Task2B not executed.')
    except Exception as exc:
        raise PipelineStepError(f'Calibrated Task2A failed: {exc}. See {log_path}') from exc
    return Task2AResult(config.mode,paths.summary_path,paths.manifest_path,log_path,time.monotonic()-started)

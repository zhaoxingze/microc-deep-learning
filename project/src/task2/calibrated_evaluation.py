"""Frozen region/coverage validation; no fitting or parameter selection here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.task2.region_detection import build_candidate_regions, dominant_branch, rank_windows, select_coverage_budget
from src.task2.refined_evaluation import _coverage_index, _sample_coverage_with_index, union_coverage_bp

LABELS = ('CHIN','OPCID','CHID')
BUDGETS = (.05,.10,.20,.30)


def interval_hits(intervals: pd.DataFrame, targets: pd.DataFrame, *, overlap=False):
    hits = np.zeros(len(targets),dtype=bool)
    for chrom in targets.chrom.unique():
        ids = np.flatnonzero(targets.chrom.eq(chrom).to_numpy())
        selected = intervals.loc[intervals.chrom.eq(chrom)]
        if selected.empty:
            continue
        rows = targets.iloc[ids]
        starts = rows.start.to_numpy() if overlap else rows.center.to_numpy()
        ends = rows.end.to_numpy() if overlap else rows.center.to_numpy()
        if overlap:
            hit = (starts[:,None]<selected.end.to_numpy()) & (ends[:,None]>selected.start.to_numpy())
        else:
            hit = (starts[:,None]>=selected.start.to_numpy()) & (ends[:,None]<selected.end.to_numpy())
        hits[ids] = hit.any(axis=1)
    return hits


def recall_metrics(intervals: pd.DataFrame, known: pd.DataFrame) -> dict:
    if known.structure_id.duplicated().any():
        raise ValueError('known evaluation requires unique structures')
    result = {}
    for overlap in (False,True):
        hit = interval_hits(intervals,known,overlap=overlap)
        for name in ('overall',*LABELS):
            mask = np.ones(len(known),dtype=bool) if name=='overall' else known.type.eq(name).to_numpy()
            suffix = 'overlap_recall' if overlap else 'recall'
            result[f'{name}_{suffix}'] = float(hit[mask].mean()) if mask.any() else float('nan')
    return result


def coverage_curve(regions, known, genome_length, *, budgets=tuple(np.arange(41)/100)):
    rows=[]
    for budget in budgets:
        selected,bp=select_coverage_budget(regions,float(budget),genome_length)
        rows.append(dict(budget=float(budget),actual_coverage=bp/genome_length,
            union_coverage_bp=bp,overshoot_bp=max(0,bp-budget*genome_length),
            budget_reached=bp>=budget*genome_length,region_count=len(selected),
            **recall_metrics(selected,known)))
    return pd.DataFrame(rows)


def sample_region_matched(selected, chrom_sizes: dict, rng):
    rows=[]
    for row in selected.itertuples(index=False):
        length=int(row.end-row.start); size=int(chrom_sizes[row.chrom])
        if not 0 < length <= size:
            raise ValueError('random region length outside linear domain')
        start=int(rng.integers(0,size-length+1))
        rows.append(dict(chrom=row.chrom,start=start,end=start+length))
    return pd.DataFrame(rows,columns=['chrom','start','end'])


def random_baselines(scores,regions,known,chrom_sizes,*,budgets=BUDGETS,repeats=100,seed=20_260_920):
    if repeats <= 0:
        raise ValueError('positive repetitions required')
    total=sum(chrom_sizes.values())
    windows=scores[['chrom','start','end']].reset_index(drop=True)
    index=_coverage_index(windows)
    coverage_rng,region_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    rows=[]
    for budget in budgets:
        selected,target=select_coverage_budget(regions,budget,total)
        for iteration in range(repeats):
            cov,bp,matched=_sample_coverage_with_index(windows,target,coverage_rng,index,tolerance=.02)
            reg=sample_region_matched(selected,chrom_sizes,region_rng)
            for name,draw,actual,match in [('coverage',cov,bp,matched),('region',reg,union_coverage_bp(reg),True)]:
                rows.append(dict(baseline=name,budget=budget,iteration=iteration,
                    target_coverage_bp=target,actual_coverage_bp=actual,actual_coverage=actual/total,
                    matched=bool(match),region_count=len(selected),sample_interval_count=len(draw),
                    **recall_metrics(draw,known)))
    draws=pd.DataFrame(rows)
    summaries=[]
    for name in ('coverage','region'):
        table=[]
        for budget,group in draws.loc[draws.baseline.eq(name)].groupby('budget',sort=True):
            entry=dict(budget=budget,repeats=len(group),target_coverage_bp=int(group.target_coverage_bp.iloc[0]),
                region_count=int(group.region_count.iloc[0]),matched_repeats=int(group.matched.sum()),
                actual_coverage_mean=float(group.actual_coverage.mean()),
                actual_coverage_min=float(group.actual_coverage.min()),actual_coverage_max=float(group.actual_coverage.max()),
                tolerance=.02 if name=='coverage' else np.nan)
            for label in ('overall',*LABELS):
                values=group[f'{label}_recall'].to_numpy()
                for stat,value in [('mean',np.mean(values)),('std',np.std(values)),
                                   ('ci_low',np.quantile(values,.025)),('ci_high',np.quantile(values,.975))]:
                    entry[f'{label}_{stat}']=float(value)
            table.append(entry)
        summaries.append(pd.DataFrame(table))
    return *summaries,draws


def known_audit(scores,regions,known,genome_length):
    selected={int(round(b*100)):select_coverage_budget(regions,b,genome_length)[0] for b in BUDGETS}
    selected_hits={b:interval_hits(r,known) for b,r in selected.items()}
    rows=[]
    for i,k in enumerate(known.itertuples(index=False)):
        windows=scores.loc[scores.chrom.eq(k.chrom)]
        overlap=windows.loc[windows.start.lt(k.end)&windows.end.gt(k.start)].sort_values('rank')
        row={key:getattr(k,key) for key in ('structure_id','type','chrom','start','end','center')}
        row.update(nearest_scan_center_distance_bp=float((windows.center-k.center).abs().min()),
                   zero_axis_nearby=bool(overlap[['zero_axis_rep1','zero_axis_rep2']].any(axis=None)))
        row.update({f'selected_at_{b}pct_coverage':bool(hit[i]) for b,hit in selected_hits.items()})
        if len(overlap):
            best=overlap.iloc[0]
            row.update(best_overlapping_window_id=best.window_id,best_density_tail_score=best.density_tail_score,
                best_shape_tail_score=best.shape_tail_score,best_calibrated_or_score=best.calibrated_or_score,
                best_rank=int(best['rank']),best_percentile=best.percentile,
                dominant_branch=dominant_branch(best.density_tail_score,best.shape_tail_score),
                max_overlapping_density_tail_score=float(overlap.density_tail_score.max()),
                max_overlapping_shape_tail_score=float(overlap.shape_tail_score.max()))
        else:
            row.update(best_overlapping_window_id='',best_density_tail_score=np.nan,best_shape_tail_score=np.nan,
                       best_calibrated_or_score=np.nan,best_rank=np.nan,best_percentile=np.nan,dominant_branch='unscanned')
        covering=regions.loc[regions.chrom.eq(k.chrom)&regions.start.le(k.center)&regions.end.gt(k.center)]
        row['first_covering_region_rank']=float(covering.region_rank.min())
        row['first_covering_region_peak_score']=float(covering.iloc[0].peak_score) if len(covering) else np.nan
        row['first_covering_region_coverage']=float(covering.iloc[0].cumulative_union_bp/genome_length) if len(covering) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def missed_summary(audit):
    rows=[]
    for label in LABELS:
        group=audit.loc[audit.type.eq(label)]
        row=dict(type=label,total=len(group))
        for budget in (5,10,20,30):
            missed=group.loc[~group[f'selected_at_{budget}pct_coverage']]
            row[f'missed_{budget}pct']=len(missed)
            for branch,column in [('density','best_density_tail_score'),('shape','best_shape_tail_score'),('or','best_calibrated_or_score')]:
                finite = missed[column].dropna()
                row[f'missed_{budget}pct_median_{branch}']=float(finite.median()) if len(finite) else np.nan
            row[f'missed_{budget}pct_both_below_p05']=int(((missed.best_density_tail_score < -np.log10(.05)) &
                (missed.best_shape_tail_score < -np.log10(.05))).sum())
            row[f'missed_{budget}pct_no_covering_region']=int(missed.first_covering_region_rank.isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def zero_axis_audit(scores,selected):
    mask=interval_hits(selected,scores)
    zero=scores[['zero_axis_rep1','zero_axis_rep2']].any(axis=1).to_numpy()
    all_rate=float(zero.mean())
    selected_rate=float(zero[mask].mean()) if mask.any() else float('nan')
    enrichment=selected_rate/all_rate if all_rate else (0.0 if selected_rate==0 else float('inf'))
    return dict(budget=.20,definition='scan window centers within selected region union',
        all_count=len(scores),selected_count=int(mask.sum()),all_zero_count=int(zero.sum()),
        selected_zero_count=int(zero[mask].sum()),all_rate=all_rate,selected_rate=selected_rate,enrichment=enrichment)


def assess_readiness(curve,coverage_random,region_random,*,zero_enrichment,
                     replicate_spearman,branch_spearman,no_overlap,mode='full'):
    at=lambda frame,b:frame.loc[np.isclose(frame.budget,b)].iloc[0]
    c20=at(curve,.2);r20=at(region_random,.2)
    valid_coverage = lambda b: bool(at(curve,b).budget_reached and
        at(coverage_random,b).repeats >= 100 and
        at(coverage_random,b).matched_repeats == at(coverage_random,b).repeats)
    checks={
        'beats_coverage_random_at_10pct':bool(valid_coverage(.1) and at(curve,.1).overall_recall>at(coverage_random,.1).overall_ci_high),
        'beats_coverage_random_at_20pct':bool(valid_coverage(.2) and c20.overall_recall>at(coverage_random,.2).overall_ci_high),
        'all_classes_positive_not_below_region_random_low_at_20pct':bool(all(
            c20[f'{c}_recall']>0 and c20[f'{c}_recall']>=r20[f'{c}_ci_low'] for c in LABELS)),
        'zero_axis_enrichment_le_one':bool(np.isfinite(zero_enrichment) and zero_enrichment<=1),
        'replicate_spearman_positive':bool(np.isfinite(replicate_spearman) and replicate_spearman>0),
        'branch_abs_spearman_lt_point9':bool(np.isfinite(branch_spearman) and abs(branch_spearman)<.9),
        'train_validation_calibration_no_overlap':bool(no_overlap)}
    return dict(checks=checks,ready_for_task2b=bool(mode=='full' and all(checks.values())),
                task2b_executed=False,smoke_not_scientific=mode!='full',
                coverage_evidence_valid_10pct=valid_coverage(.1),coverage_evidence_valid_20pct=valid_coverage(.2))


def compare_detectors(scores,known,genome_length,*,seed=20_260_920):
    methods={'Density-only Z':'paired_density_z','Shape-only Z':'paired_shape_z',
        'Legacy Mean Z':'score_mean','Refined OR-Max Z':'score_or_max',
        'Calibrated Density':'density_tail_score','Calibrated Shape':'shape_tail_score',
        'Calibrated OR':'calibrated_or_score'}
    rows=[]
    for name,column in methods.items():
        regions,_=build_candidate_regions(scores,score_column=column,seed=seed)
        table=coverage_curve(regions,known,genome_length,budgets=BUDGETS)
        table['method']=name;table['evaluation']='region_coverage';table['score_source']='current_calibrated_run_AE'
        rows.extend(table.to_dict('records'))
        ranked=rank_windows(scores,score_column=column,seed=seed)
        for fraction in (.01,.02,.05,.10,.20):
            selected=ranked.head(int(np.ceil(fraction*len(scores))))
            rows.append(dict(method=name,evaluation='auxiliary_window_fraction',budget=fraction,
                actual_coverage=union_coverage_bp(selected)/genome_length,n_windows=len(selected),
                score_source='current_calibrated_run_AE',**recall_metrics(selected,known)))
    return pd.DataFrame(rows)

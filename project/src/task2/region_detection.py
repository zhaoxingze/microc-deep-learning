"""Frozen label-free NMS and bounded contiguous support regions."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from src.task2.refined_evaluation import _coverage_index


def rank_windows(scores: pd.DataFrame, *, score_column='calibrated_or_score', seed=20_260_920):
    result = scores.copy()
    if result.window_id.duplicated().any() or not np.isfinite(result[score_column]).all():
        raise ValueError('ranking requires unique IDs and finite scores')
    result['tie_key'] = [hashlib.sha256(f'{seed}|{wid}'.encode()).hexdigest() for wid in result.window_id]
    result = result.sort_values([score_column,'tie_key'], ascending=[False,True]).reset_index(drop=True)
    result['rank'] = np.arange(1,len(result)+1)
    result['percentile'] = 1-(result['rank']-1)/len(result) if len(result) else np.nan
    return result


def dominant_branch(density: float, shape: float) -> str:
    if density >= -np.log10(.05) and shape >= -np.log10(.05):
        return 'mixed'
    return 'density' if density >= shape else 'shape'


def build_candidate_regions(scores: pd.DataFrame, *, score_column='calibrated_or_score',
                            nms_radius_bp=3200, step_bp=800, seed=20_260_920):
    if scores.empty or nms_radius_bp <= 0 or step_bp <= 0:
        raise ValueError('nonempty windows and positive geometry required')
    ranked = rank_windows(scores, score_column=score_column, seed=seed)
    peaks, owners = [], []
    peak_centers, peak_chroms = [], []
    for row in ranked.itertuples(index=False):
        nearby = [i for i,(c,x) in enumerate(zip(peak_chroms,peak_centers))
                  if c==row.chrom and abs(x-row.center)<nms_radius_bp]
        if nearby:
            owner = nearby[0]
        else:
            owner = len(peaks)
            peaks.append(row.window_id); peak_centers.append(row.center); peak_chroms.append(row.chrom)
        owners.append(owner)
    ranked['owner'] = owners
    regions, membership = [], []
    indexed = ranked.set_index('window_id')
    for i, wid in enumerate(peaks):
        peak = indexed.loc[wid]
        support = ranked.loc[ranked.owner.eq(i)].sort_values('center').copy()
        support['component'] = support.center.diff().gt(step_bp).cumsum()
        component = support.loc[support.window_id.eq(wid),'component'].iloc[0]
        included = support.component.eq(component)
        members = support.loc[included]
        region_id = 'REG_' + str(wid).removeprefix('WIN_')
        start,end = int(members.start.min()),int(members.end.max())
        regions.append(dict(region_id=region_id,chrom=peak.chrom,start=start,end=end,
            center=(start+end)/2,peak_center=peak.center,peak_window_id=wid,
            peak_score=float(peak[score_column]),tie_key=peak.tie_key,
            member_window_count=len(members),discarded_noncontiguous_count=int((~included).sum()),
            mean_score=float(members[score_column].mean()),max_score=float(members[score_column].max()),
            union_length_bp=end-start,density_peak_score=peak.density_tail_score,
            shape_peak_score=peak.shape_tail_score,
            dominant_branch=dominant_branch(peak.density_tail_score,peak.shape_tail_score)))
        for row, keep in zip(support.itertuples(index=False),included,strict=True):
            membership.append(dict(window_id=row.window_id,region_id=region_id,
                                   peak_window_id=wid,included=bool(keep)))
    result = pd.DataFrame(regions)
    # Result already follows rank; union is computed once for all later budgets.
    index = _coverage_index(result)
    occupied = np.zeros(len(index.segment_lengths),dtype=bool)
    coverage, cumulative = 0, []
    for left,right in index.slices:
        coverage += int(index.segment_lengths[left:right][~occupied[left:right]].sum())
        occupied[left:right] = True
        cumulative.append(coverage)
    result['cumulative_union_bp'] = cumulative
    result['region_rank'] = np.arange(1,len(result)+1)
    return result, pd.DataFrame(membership)


def select_coverage_budget(regions: pd.DataFrame, budget: float, genome_length: int):
    if not 0 <= budget <= 1 or genome_length <= 0:
        raise ValueError('invalid coverage budget/domain')
    if budget == 0 or regions.empty:
        return regions.iloc[:0].copy(),0
    cumulative = regions.cumulative_union_bp.to_numpy()
    count = min(len(regions),int(np.searchsorted(cumulative,budget*genome_length,side='left'))+1)
    return regions.iloc[:count].copy(),int(cumulative[count-1])

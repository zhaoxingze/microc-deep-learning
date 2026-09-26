"""Pre-registered Task 2A.3 scale geometry, shared splits, fusion and audits."""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_microc import load_local_matrix
from src.task1.manifest import sha256_file
from src.task2.calibrated_evaluation import BUDGETS, interval_hits, recall_metrics
from src.task2.refined_evaluation import _coverage_index
from src.task2.region_detection import dominant_branch, select_coverage_budget
from src.task2.window_scan import _zero_axis


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    window_bp: int
    step_bp: int
    target_bin_size: int = 100

    @property
    def matrix_bins(self) -> int:
        return self.window_bp // self.target_bin_size


@dataclass(frozen=True)
class ScaleWindow:
    scale: str
    window_id: str
    chrom: str
    start: int
    end: int
    center: int


SCALES = (
    ScaleSpec('small', 3_200, 400),
    ScaleSpec('medium', 6_400, 800),
    ScaleSpec('large', 12_800, 1_600),
)


def stable_scale_window_id(scale: str, chrom: str, start: int, end: int) -> str:
    key=f'{scale}|{chrom.strip()}|{int(start)}|{int(end)}'
    return f'WIN_{scale}_'+hashlib.sha256(key.encode()).hexdigest()[:16]


def generate_scale_windows(spec: ScaleSpec, chrom: str, chrom_size: int,
                           *, scan_end: int | None = None) -> list[ScaleWindow]:
    limit=min(chrom_size,scan_end) if scan_end is not None else chrom_size
    if limit < spec.window_bp:
        raise ValueError(f'{spec.name} scan interval has no complete window')
    return [ScaleWindow(spec.name,stable_scale_window_id(spec.name,chrom,start,start+spec.window_bp),
        chrom,start,start+spec.window_bp,start+spec.window_bp//2)
        for start in range(0,limit-spec.window_bp+1,spec.step_bp)]


def scan_scale_replicates(rep1_cool: str | Path, rep2_cool: str | Path, spec: ScaleSpec,
                          chrom: str, chrom_size: int, output_dir: str | Path,
                          *, scan_end: int | None = None,
                          source_sha256: dict[str,str] | None = None) -> pd.DataFrame:
    windows=generate_scale_windows(spec,chrom,chrom_size,scan_end=scan_end)
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    cool_paths={'rep1':Path(rep1_cool).resolve(),'rep2':Path(rep2_cool).resolve()}
    temporary=[root/f'genome_windows_{rep}.tmp.npy' for rep in ('rep1','rep2')]+[root/'metadata.tmp.csv']
    for path in temporary:path.unlink(missing_ok=True)
    records=[]
    try:
        for replicate,cool_path in cool_paths.items():
            arrays=np.lib.format.open_memmap(root/f'genome_windows_{replicate}.tmp.npy',mode='w+',
                dtype=np.float32,shape=(len(windows),1,spec.matrix_bins,spec.matrix_bins))
            for index,window in enumerate(windows):
                matrix,meta=load_local_matrix(cool_path,chrom,window.center,window_bp=spec.window_bp,
                    target_bin_size=spec.target_bin_size,balance=False)
                if matrix.shape!=(spec.matrix_bins,spec.matrix_bins) or bool(meta.get('padded')):
                    raise ValueError(f'invalid complete {spec.name} window {window.window_id}')
                if not np.isfinite(matrix).all() or (matrix<0).any():
                    raise ValueError(f'invalid numeric values in {spec.name} window {window.window_id}')
                zero,zrows,zcols=_zero_axis(matrix);arrays[index,0]=matrix
                records.append(dict(**vars(window),window_bp=spec.window_bp,step_bp=spec.step_bp,
                    target_bin_size=spec.target_bin_size,array_index=index,replicate=replicate,
                    zero_axis=zero,zero_row_count=zrows,zero_col_count=zcols,
                    cool_path=str(cool_path)))
            arrays.flush();del arrays
        frame=pd.DataFrame(records);assert_replicate_alignment(frame)
        frame.to_csv(root/'metadata.tmp.csv',index=False,encoding='utf-8-sig')
        for rep in ('rep1','rep2'):
            (root/f'genome_windows_{rep}.tmp.npy').replace(root/f'genome_windows_{rep}.npy')
        (root/'metadata.tmp.csv').replace(root/'metadata.csv')
        write_scale_scan_manifest(root,spec,cool_paths,source_sha256=source_sha256)
        return frame
    finally:
        for path in temporary:path.unlink(missing_ok=True)


def write_scale_scan_manifest(directory: str | Path,spec: ScaleSpec,
                              cool_paths: dict[str,str | Path],*,
                              source_sha256: dict[str,str] | None = None) -> dict:
    """Write the completion marker last, after checksumming finalized scan artifacts."""
    root=Path(directory);resolved={rep:Path(path).resolve() for rep,path in cool_paths.items()}
    source_hashes=source_sha256 or {rep:sha256_file(path) for rep,path in resolved.items()}
    artifacts={}
    for name in ('metadata.csv','genome_windows_rep1.npy','genome_windows_rep2.npy'):
        path=root/name
        if not path.is_file():raise ValueError(f'missing scale scan artifact: {name}')
        artifacts[name]={'size_bytes':path.stat().st_size,'sha256':sha256_file(path)}
    payload={'complete':True,'scale':spec.name,'window_bp':spec.window_bp,'step_bp':spec.step_bp,
        'target_bin_size':spec.target_bin_size,'matrix_bins':spec.matrix_bins,
        'sources':{rep:{'path':str(path),'sha256':source_hashes[rep]} for rep,path in resolved.items()},
        'artifacts':artifacts}
    temporary=root/'scan_manifest.json.tmp'
    temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    temporary.replace(root/'scan_manifest.json')
    return payload


def validate_scale_scan_manifest(directory: str | Path,spec: ScaleSpec,
                                 cool_paths: dict[str,str | Path],*,
                                 source_sha256: dict[str,str] | None = None) -> dict:
    root=Path(directory);path=root/'scan_manifest.json'
    try:payload=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError) as exc:raise ValueError('scale scan completion manifest missing or invalid') from exc
    expected_geometry=(spec.name,spec.window_bp,spec.step_bp,spec.target_bin_size,spec.matrix_bins)
    actual_geometry=tuple(payload.get(k) for k in ('scale','window_bp','step_bp','target_bin_size','matrix_bins'))
    if payload.get('complete') is not True or actual_geometry!=expected_geometry:
        raise ValueError('scale scan completion/geometry mismatch')
    resolved={rep:Path(value).resolve() for rep,value in cool_paths.items()}
    expected_hashes=source_sha256 or {rep:sha256_file(value) for rep,value in resolved.items()}
    for rep,source in resolved.items():
        record=payload.get('sources',{}).get(rep,{})
        if record.get('path')!=str(source):raise ValueError('scale scan source path mismatch')
        if record.get('sha256')!=expected_hashes[rep]:raise ValueError('scale scan source SHA256 mismatch')
    for name,record in payload.get('artifacts',{}).items():
        artifact=root/name
        if not artifact.is_file() or artifact.stat().st_size!=record.get('size_bytes'):
            raise ValueError('scale scan artifact size mismatch')
        if sha256_file(artifact)!=record.get('sha256'):raise ValueError('scale scan artifact SHA256 mismatch')
    if set(payload.get('artifacts',{}))!={'metadata.csv','genome_windows_rep1.npy','genome_windows_rep2.npy'}:
        raise ValueError('scale scan artifact manifest incomplete')
    return payload


def assert_replicate_alignment(metadata: pd.DataFrame) -> None:
    required={'window_id','replicate','array_index','chrom','start','end'}
    if not required.issubset(metadata) or set(metadata.replicate)!={'rep1','rep2'}:
        raise ValueError('replicate alignment schema invalid')
    views=[]
    for rep in ('rep1','rep2'):
        rows=metadata.loc[metadata.replicate.eq(rep)].sort_values('array_index')
        if rows.window_id.duplicated().any() or not np.array_equal(rows.array_index,np.arange(len(rows))):
            raise ValueError('replicate alignment index invalid')
        views.append(rows[['window_id','chrom','start','end']].reset_index(drop=True))
    if not views[0].equals(views[1]):
        raise ValueError('replicate alignment IDs/coordinates differ')


def slice_expected(maximum: dict[str,np.ndarray], spec: ScaleSpec) -> dict[str,np.ndarray]:
    if set(maximum)!={'rep1','rep2'} or any(np.asarray(v).shape!=(128,) for v in maximum.values()):
        raise ValueError('maximum expected must contain two length-128 vectors')
    return {rep:np.asarray(values[:spec.matrix_bins]).copy() for rep,values in maximum.items()}


def strict_background_candidates(windows: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    required={'window_id','chrom','start','end'}
    if not required.issubset(windows) or not {'chrom','start','end'}.issubset(known):
        raise ValueError('strict background schema incomplete')
    out=windows.drop_duplicates('window_id').copy().reset_index(drop=True)
    eligible=np.ones(len(out),dtype=bool)
    for chrom,indices in out.groupby('chrom').groups.items():
        structures=known.loc[known.chrom.astype(str).eq(str(chrom))]
        if structures.empty:continue
        overlap=(out.loc[indices,'start'].to_numpy()[:,None]<structures.end.to_numpy()[None,:]) & (
            out.loc[indices,'end'].to_numpy()[:,None]>structures.start.to_numpy()[None,:])
        eligible[np.asarray(list(indices))]=~overlap.any(axis=1)
    out['background_candidate']=eligible
    return out


def make_master_block_split(chrom: str, domain_size: int, block_bp: int,
                            *, seed: int=20_260_920) -> pd.DataFrame:
    if domain_size<=0 or block_bp<=0:raise ValueError('invalid master block domain')
    n=math.ceil(domain_size/block_bp)
    if n<3:raise ValueError('master split needs at least three blocks')
    blocks=np.arange(n);order=np.random.default_rng(seed).permutation(n)
    n_train=min(max(1,round(.70*n)),n-2)
    n_val=min(max(1,round(.15*n)),n-n_train-1)
    labels={int(block):('train' if i<n_train else 'validation' if i<n_train+n_val else 'calibration')
            for i,block in enumerate(order)}
    result=pd.DataFrame(dict(chrom=chrom,block_index=blocks,
        block_id=[f'{chrom}:{i}' for i in blocks],block_start=blocks*block_bp,
        block_end=np.minimum((blocks+1)*block_bp,domain_size),split=[labels[int(i)] for i in blocks]))
    if set(result.split)!={'train','validation','calibration'}:raise ValueError('master split nonempty invariant failed')
    return result


def make_feasible_master_block_split(chrom: str,domain_size: int,block_bp: int,
                                     eligible_blocks_by_scale: dict[str,set[int]],*,
                                     seed: int=20_260_920,max_attempts: int=10_000) -> pd.DataFrame:
    """First seeded whole-block allocation giving every scale all three background splits."""
    if not eligible_blocks_by_scale or any(len(blocks)<3 for blocks in eligible_blocks_by_scale.values()):
        raise ValueError('each scale needs at least three eligible background blocks')
    n=math.ceil(domain_size/block_bp);rng=np.random.default_rng(seed)
    n_train=min(max(1,round(.70*n)),n-2);n_val=min(max(1,round(.15*n)),n-n_train-1)
    for attempt in range(1,max_attempts+1):
        order=rng.permutation(n)
        labels={int(block):('train' if i<n_train else 'validation' if i<n_train+n_val else 'calibration')
                for i,block in enumerate(order)}
        if all({labels[b] for b in blocks if b in labels}=={'train','validation','calibration'}
               for blocks in eligible_blocks_by_scale.values()):
            blocks=np.arange(n)
            return pd.DataFrame(dict(chrom=chrom,block_index=blocks,block_id=[f'{chrom}:{i}' for i in blocks],
                block_start=blocks*block_bp,block_end=np.minimum((blocks+1)*block_bp,domain_size),
                split=[labels[int(i)] for i in blocks],assignment_attempt=attempt))
    raise ValueError(f'no feasible shared master split after {max_attempts} seeded attempts')


def assign_scale_background(windows: pd.DataFrame, master: pd.DataFrame, *, block_bp: int) -> pd.DataFrame:
    required={'window_id','chrom','start','end','background_candidate'}
    if not required.issubset(windows):raise ValueError('scale background schema incomplete')
    out=windows.copy();index=out.start.astype(int)//block_bp
    out['block_id']=out.chrom.astype(str)+':'+index.astype(str)
    assigned=out.merge(master[['block_id','block_end','split']],on='block_id',how='left',validate='many_to_one')
    assigned=assigned.loc[assigned.background_candidate.eq(True)&assigned.end.le(assigned.block_end)].copy()
    if assigned.empty or set(assigned.split)!={'train','validation','calibration'}:
        raise ValueError('scale lacks a nonempty background split')
    return assigned.drop(columns='block_end').sort_values(['chrom','start','window_id']).reset_index(drop=True)


def assert_background_known_clean(rows: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    records=[]
    for (scale,split),group in rows.groupby(['scale','split'],dropna=False):
        count=0
        for chrom,subset in group.groupby('chrom'):
            structures=known.loc[known.chrom.astype(str).eq(str(chrom))]
            if len(structures):
                count+=int(((subset.start.to_numpy()[:,None]<structures.end.to_numpy()[None,:]) &
                    (subset.end.to_numpy()[:,None]>structures.start.to_numpy()[None,:])).sum())
        records.append(dict(scale=scale,split=split,background_window_count=len(group),overlap_count=count))
    audit=pd.DataFrame(records)
    if audit.overlap_count.sum():raise ValueError('background known interval overlap detected')
    return audit


def _any_interval_overlap(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    for chrom,a in left.groupby('chrom'):
        b=right.loc[right.chrom.astype(str).eq(str(chrom))]
        if b.empty:continue
        aa=a.sort_values('start');bb=b.sort_values('start')
        j=0;starts=bb.start.to_numpy();ends=bb.end.to_numpy()
        for row in aa.itertuples(index=False):
            while j<len(bb) and ends[j]<=row.start:j+=1
            if j<len(bb) and starts[j]<row.end:return True
    return False


def _interval_overlap_count(left: pd.DataFrame,right: pd.DataFrame) -> int:
    count=0
    for chrom,a in left.groupby('chrom'):
        b=right.loc[right.chrom.astype(str).eq(str(chrom))]
        if b.empty:continue
        starts=np.sort(b.start.to_numpy(dtype=np.int64));ends=np.sort(b.end.to_numpy(dtype=np.int64))
        for row in a.itertuples(index=False):
            count+=int(np.searchsorted(starts,int(row.end),side='left')-
                       np.searchsorted(ends,int(row.start),side='right'))
    return count


def split_leakage_audit(rows: pd.DataFrame) -> pd.DataFrame:
    required={'scale','split','chrom','start','end'}
    if not required.issubset(rows):raise ValueError('cross-scale leakage schema incomplete')
    from itertools import combinations
    records=[]
    for scope in ('combined','small','medium','large'):
        frame=rows if scope=='combined' else rows.loc[rows.scale.eq(scope)]
        for a,b in combinations(('train','validation','calibration'),2):
            records.append(dict(scope=scope,split_a=a,split_b=b,
                overlap_count=_interval_overlap_count(frame.loc[frame.split.eq(a)],frame.loc[frame.split.eq(b)])))
    return pd.DataFrame(records)


def assert_cross_scale_no_leakage(rows: pd.DataFrame) -> pd.DataFrame:
    audit=split_leakage_audit(rows)
    if audit.overlap_count.sum():raise ValueError('same/cross-scale coordinate leakage detected')
    return audit


def validate_scale_score_pool(pool: pd.DataFrame) -> None:
    required={'scale','calibration_id','scale_or_score','density_tail_score','shape_tail_score'}
    if not required.issubset(pool) or pool.window_id.duplicated().any():raise ValueError('invalid multiscale score pool')
    if not (pool.scale.astype(str)==pool.calibration_id.astype(str)).all():
        raise ValueError('scale calibration must be independent')
    if not np.allclose(pool.scale_or_score,np.maximum(pool.density_tail_score,pool.shape_tail_score)):
        raise ValueError('scale OR is not max calibrated branches')


def _rank_pool(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    out=pool.copy();out['tie_key']=[hashlib.sha256(f'{seed}|{x}'.encode()).hexdigest() for x in out.window_id]
    return out.sort_values(['scale_or_score','tie_key'],ascending=[False,True]).reset_index(drop=True)


def build_cross_scale_regions(pool: pd.DataFrame, *, seed: int=20_260_920):
    if pool.empty:raise ValueError('empty cross-scale pool')
    ranked=_rank_pool(pool,seed);selected=[];centers=[];owners=[]
    # centers stores (coordinate, peak_index), enabling bounded bisect lookup.
    for row in ranked.itertuples(index=False):
        left=bisect_left(centers,(int(row.center)-6_400,-1));right=bisect_right(centers,(int(row.center)+6_400,10**12))
        candidates=[]
        for _,idx in centers[left:right]:
            peak=selected[idx]
            if peak.chrom==row.chrom and abs(int(row.center)-int(peak.center))<max(int(row.window_bp),int(peak.window_bp))/2:
                candidates.append(idx)
        if candidates:owner=min(candidates)
        else:
            owner=len(selected);selected.append(row);insort(centers,(int(row.center),owner))
        owners.append(owner)
    ranked['owner']=owners;regions=[];members=[]
    for index,peak in enumerate(selected):
        support=ranked.loc[ranked.owner.eq(index)]
        region_id='MSREG_'+str(peak.window_id).removeprefix('WIN_')
        counts=support.scale.value_counts()
        regions.append(dict(region_id=region_id,chrom=peak.chrom,start=int(peak.start),end=int(peak.end),
            center=float(peak.center),peak_scale=peak.scale,peak_window_id=peak.window_id,
            peak_score=float(peak.scale_or_score),peak_window_bp=int(peak.window_bp),
            density_anomaly=float(peak.density_tail_score),shape_anomaly=float(peak.shape_tail_score),
            dominant_branch=dominant_branch(peak.density_tail_score,peak.shape_tail_score),
            rep1_peak_score=float(peak.rep1_calibrated_or_score),rep2_peak_score=float(peak.rep2_calibrated_or_score),
            support_small_count=int(counts.get('small',0)),support_medium_count=int(counts.get('medium',0)),
            support_large_count=int(counts.get('large',0)),support_union_start=int(support.start.min()),
            support_union_end=int(support.end.max()),member_window_count=len(support),
            tie_key=peak.tie_key,union_length_bp=int(peak.end-peak.start)))
        for row in support.itertuples(index=False):members.append(dict(region_id=region_id,
            peak_window_id=peak.window_id,window_id=row.window_id,scale=row.scale,window_score=row.scale_or_score))
    frame=pd.DataFrame(regions);coverage_index=_coverage_index(frame)
    occupied=np.zeros(len(coverage_index.segment_lengths),dtype=bool);total=0;values=[]
    for left,right in coverage_index.slices:
        total+=int(coverage_index.segment_lengths[left:right][~occupied[left:right]].sum());occupied[left:right]=True;values.append(total)
    frame['cumulative_union_bp']=values;frame['region_rank']=np.arange(1,len(frame)+1)
    return frame,pd.DataFrame(members)


def detector_coverage_table(pools: dict[str,pd.DataFrame], known: pd.DataFrame, genome_length: int,
                            *, seed: int=20_260_920):
    rows=[];regions={}
    for detector,pool in pools.items():
        reg,_=build_cross_scale_regions(pool,seed=seed);regions[detector]=reg
        for budget in BUDGETS:
            selected,bp=select_coverage_budget(reg,budget,genome_length)
            rows.append(dict(detector=detector,budget=budget,actual_coverage=bp/genome_length,
                union_coverage_bp=bp,region_count=len(selected),budget_reached=bp>=budget*genome_length,
                **recall_metrics(selected,known)))
    return pd.DataFrame(rows),regions


def known_scale_audit(pool: pd.DataFrame, regions: pd.DataFrame, known: pd.DataFrame, genome_length: int,
                      *, regions_by_scale: dict[str,pd.DataFrame] | None = None):
    selected={int(b*100):select_coverage_budget(regions,b,genome_length)[0] for b in BUDGETS}
    flags={b:interval_hits(value,known) for b,value in selected.items()};rows=[]
    scale_flags={}
    if regions_by_scale is not None:
        for scale,scale_regions in regions_by_scale.items():
            scale_flags[scale]={int(b*100):interval_hits(select_coverage_budget(scale_regions,b,genome_length)[0],known)
                                for b in BUDGETS}
    ranked={scale:_rank_pool(pool.loc[pool.scale.eq(scale)],20_260_920).assign(rank=lambda x:np.arange(1,len(x)+1))
            for scale in ('small','medium','large')}
    for i,item in enumerate(known.itertuples(index=False)):
        row={k:getattr(item,k) for k in ('structure_id','type','chrom','start','end','center')}
        row['length_bp']=getattr(item,'length_bp',item.end-item.start);best=[]
        for scale,table in ranked.items():
            overlapping=table.loc[table.chrom.eq(item.chrom)&table.start.lt(item.end)&table.end.gt(item.start)].sort_values('rank')
            if len(overlapping):
                value=overlapping.iloc[0];best.append((value.scale_or_score,scale))
                row.update({f'{scale}_best_density':value.density_tail_score,f'{scale}_best_shape':value.shape_tail_score,
                            f'{scale}_best_or':value.scale_or_score,f'{scale}_best_rank':int(value['rank'])})
            else:row.update({f'{scale}_best_density':np.nan,f'{scale}_best_shape':np.nan,f'{scale}_best_or':np.nan,f'{scale}_best_rank':np.nan})
        row['best_multiscale_score'],row['best_scale']=max(best) if best else (np.nan,'unscanned')
        row.update({f'selected_at_{b}pct':bool(hit[i]) for b,hit in flags.items()})
        if scale_flags:
            for b in flags:
                selected_scales=[]
                for scale in ('small','medium','large'):
                    value=bool(scale_flags[scale][b][i]);row[f'selected_{scale}_at_{b}pct']=value
                    if value:selected_scales.append(scale)
                row[f'rescue_category_at_{b}pct']=(f'{selected_scales[0]}_only' if len(selected_scales)==1
                    else 'multiple_scales' if selected_scales else 'missed_all_scales')
        rows.append(row)
    return pd.DataFrame(rows)


def scale_recall_sets(regions_by_scale: dict[str,pd.DataFrame], known: pd.DataFrame, genome_length: int):
    result={}
    for budget in (.10,.20,.30):
        result[int(budget*100)]={}
        for scale,regions in regions_by_scale.items():
            selected,_=select_coverage_budget(regions,budget,genome_length)
            result[int(budget*100)][scale]=set(known.loc[interval_hits(selected,known),'structure_id'])
    return result


def scale_rescue_summary(known: pd.DataFrame, selections: dict[int,dict[str,set[str]]]):
    rows=[]
    for budget,by_scale in selections.items():
        for label,group in known.groupby('type'):
            counts={'small':by_scale['small'],'medium':by_scale['medium'],'large':by_scale['large']};ids=set(group.structure_id)
            patterns={sid:tuple(name for name,values in counts.items() if sid in values) for sid in ids}
            rows.append(dict(type=label,budget=budget/100,n_total=len(ids),
                recalled_small_only=sum(v==('small',) for v in patterns.values()),
                recalled_medium_only=sum(v==('medium',) for v in patterns.values()),
                recalled_large_only=sum(v==('large',) for v in patterns.values()),
                recalled_multiple_scales=sum(len(v)>1 for v in patterns.values()),
                missed_all_scales=sum(len(v)==0 for v in patterns.values())))
    return pd.DataFrame(rows)


def assess_multiscale_readiness(curve,coverage_random,region_random,*,zero_enrichment,
                                replicate_spearman,no_leakage,known_overlap_zero,mode='full'):
    at=lambda frame,b:frame.loc[np.isclose(frame.budget,b)].iloc[0]
    valid=lambda b:bool(at(curve,b).budget_reached and at(coverage_random,b).repeats>=100 and
                        at(coverage_random,b).matched_repeats==at(coverage_random,b).repeats)
    c20=at(curve,.2);r20=at(region_random,.2)
    checks={'beats_coverage_random_at_10pct':bool(valid(.1) and at(curve,.1).overall_recall>at(coverage_random,.1).overall_ci_high),
        'beats_coverage_random_at_20pct':bool(valid(.2) and c20.overall_recall>at(coverage_random,.2).overall_ci_high),
        'all_classes_positive_not_below_region_random_low_at_20pct':bool(all(c20[f'{c}_recall']>0 and c20[f'{c}_recall']>=r20[f'{c}_ci_low'] for c in ('CHIN','OPCID','CHID'))),
        'zero_axis_enrichment_le_one':bool(np.isfinite(zero_enrichment) and zero_enrichment<=1),
        'replicate_spearman_positive':bool(np.isfinite(replicate_spearman) and replicate_spearman>0),
        'same_and_cross_scale_no_leakage':bool(no_leakage),'all_background_known_overlap_zero':bool(known_overlap_zero)}
    return dict(checks=checks,ready_for_task2b=bool(mode=='full' and all(checks.values())),
                task2b_executed=False,coverage_evidence_valid_10pct=valid(.1),coverage_evidence_valid_20pct=valid(.2))

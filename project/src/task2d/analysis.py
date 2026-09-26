"""Post-measurement Known-floor calibration, candidate pools and final statuses."""
from __future__ import annotations

import numpy as np
import pandas as pd

KNOWN_FLOOR_PERCENTILE = 5
STRICT_VALID_MEMBER_PASS_FRACTION = .8


def calibrate_known_floor(known: pd.DataFrame) -> dict:
    required = {'correlation_valid', 'pearson', 'spearman'}
    if not required.issubset(known):
        raise ValueError(f'Known control fields missing: {sorted(required - set(known))}')
    good = known.loc[known.correlation_valid.astype(bool)].copy()
    good = good.loc[np.isfinite(good.pearson.to_numpy(float)) & np.isfinite(good.spearman.to_numpy(float))]
    if good.empty:
        return {'calibration_valid': False, 'failure_reason': 'no_valid_known_controls',
                'pearson_floor': None, 'spearman_floor': None, 'n_valid_known': 0,
                'rule': 'valid Known canonical Pearson and Spearman 5th percentile'}
    pearson = float(np.percentile(good.pearson, KNOWN_FLOOR_PERCENTILE))
    spearman = float(np.percentile(good.spearman, KNOWN_FLOOR_PERCENTILE))
    valid = bool(np.isfinite([pearson, spearman]).all() and pearson >= 0 and spearman >= 0)
    return {'calibration_valid': valid,
            'failure_reason': '' if valid else 'negative_or_nonfinite_known_floor',
            'pearson_floor': pearson, 'spearman_floor': spearman,
            'n_valid_known': int(len(good)),
            'rule': 'valid Known canonical Pearson and Spearman 5th percentile'}


def _strict_boolean(values: pd.Series, name: str) -> pd.Series:
    text = values.astype(str).str.lower()
    if not text.isin(['true', 'false']).all():
        raise ValueError(f'invalid {name} boolean metadata')
    return text.eq('true')


def annotate_candidates(candidates: pd.DataFrame, cluster_summary: pd.DataFrame,
                        floors: dict) -> pd.DataFrame:
    required = {'region_id', 'cluster_id', 'cluster_annotation',
                'known_center_overlap', 'known_interval_overlap', 'novel_like_cluster_candidate',
                'correlation_valid', 'primary_pearson', 'primary_spearman'}
    if not required.issubset(candidates):
        raise ValueError(f'candidate annotation fields missing: {sorted(required - set(candidates))}')
    if candidates.region_id.isna().any() or candidates.region_id.duplicated().any():
        raise ValueError('candidate IDs missing or duplicated')
    output = candidates.copy().reset_index(drop=True)
    for key in ('known_center_overlap', 'known_interval_overlap', 'novel_like_cluster_candidate'):
        output[key] = _strict_boolean(output[key], key)
    output['known_overlap'] = output.known_center_overlap | output.known_interval_overlap
    output['candidate_pool'] = np.select(
        [output.known_overlap, output.novel_like_cluster_candidate,
         output.cluster_id.eq(-1), output.cluster_id.ge(0)],
        ['KNOWN_OVERLAP', 'STRICT_NOVEL_CLUSTER', 'UNANNOTATED_NOISE', 'UNANNOTATED_CLUSTERED'],
        default='UNCLASSIFIED',
    )
    if output.candidate_pool.eq('UNCLASSIFIED').any():
        raise ValueError('candidate cluster ID is neither noise nor clustered')
    valid = (output.correlation_valid.astype(bool) &
             np.isfinite(output.primary_pearson.to_numpy(float)) &
             np.isfinite(output.primary_spearman.to_numpy(float)))
    calibration_valid = bool(floors.get('calibration_valid', False))
    passed = pd.Series(pd.NA, index=output.index, dtype='boolean')
    if calibration_valid:
        passed.loc[valid] = (output.loc[valid, 'primary_pearson'].ge(floors['pearson_floor']) &
                             output.loc[valid, 'primary_spearman'].ge(floors['spearman_floor'])).to_numpy(bool)
    output['pearson_floor'] = floors.get('pearson_floor')
    output['spearman_floor'] = floors.get('spearman_floor')
    output['reproducible_by_known_floor'] = passed
    output['strict_novel_candidate'] = False
    summary = cluster_summary.set_index('cluster_id') if len(cluster_summary) else cluster_summary
    if calibration_valid:
        for cluster_id, members in output.loc[output.cluster_id.ge(0)].groupby('cluster_id'):
            if cluster_id not in summary.index:
                raise ValueError(f'cluster {cluster_id} absent from frozen Task2C summary')
            row = summary.loc[cluster_id]
            valid_members = members.loc[valid.loc[members.index]]
            fraction = (float(valid_members.reproducible_by_known_floor.fillna(False).astype(bool).mean())
                        if len(valid_members) else 0.)
            qualifies = bool(row.novel_like_cluster_candidate) and bool(members.novel_like_cluster_candidate.all()) and (
                int(row.candidate_count) >= 5 and int(row.candidate_count) == len(members) and
                int(row.known_count) == 0 and not bool(members.known_overlap.any()) and
                len(valid_members) > 0 and fraction >= STRICT_VALID_MEMBER_PASS_FRACTION
            )
            if qualifies:
                chosen = members.index[valid.loc[members.index] & members.reproducible_by_known_floor.fillna(False).astype(bool)]
                output.loc[chosen, 'strict_novel_candidate'] = True
    statuses = []
    notes = []
    for row in output.itertuples(index=False):
        is_pass = bool(row.reproducible_by_known_floor) if pd.notna(row.reproducible_by_known_floor) else False
        if not row.correlation_valid:
            status, note = 'INSUFFICIENT_QUALITY', 'Primary correlation undefined; retain continuous diagnostics.'
        elif not calibration_valid:
            status, note = 'CONTROL_CALIBRATION_FAILURE', 'Known-control floor invalid; binary candidate filtering stopped.'
        elif row.known_overlap:
            status, note = 'KNOWN_OVERLAP', 'Genomic center or interval overlaps a Known annotation.'
        elif row.strict_novel_candidate:
            status, note = 'STRICT_NOVEL_CLUSTER_MEMBER', 'Pre-registered Task2C cluster and 80% valid-member reproducibility rule passed.'
        elif row.cluster_id == -1:
            status = 'UNANNOTATED_NOISE_REPRODUCIBLE' if is_pass else 'UNANNOTATED_NOISE_LOW_REPRODUCIBILITY'
            note = 'Reproducible unclustered candidate, not a new structure class.' if is_pass else 'HDBSCAN noise; no novel-class inference.'
        elif str(row.cluster_annotation).endswith('-like'):
            status = 'KNOWN_LIKE_CLUSTER_REPRODUCIBLE' if is_pass else 'KNOWN_LIKE_CLUSTER_LOW_REPRODUCIBILITY'
            note = 'Genomically unannotated, but in a Known-containing representation cluster; not a type assignment.'
        else:
            status = 'UNANNOTATED_CLUSTERED_REPRODUCIBLE' if is_pass else 'UNANNOTATED_CLUSTERED_LOW_REPRODUCIBILITY'
            note = 'Unannotated clustered candidate; not confirmed novel biology.'
        statuses.append(status)
        notes.append(note)
    output['final_status'] = statuses
    output['evidence_note'] = notes
    return output


def cluster_reproducibility(candidates: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in cluster_summary.itertuples(index=False):
        members = candidates.loc[candidates.cluster_id.eq(source.cluster_id)]
        valid = members.loc[members.correlation_valid.astype(bool)]
        passed = int(valid.reproducible_by_known_floor.fillna(False).astype(bool).sum())
        def spread(column: str) -> tuple[float, float]:
            values = valid[column].dropna()
            return ((float(values.median()), float(values.quantile(.75) - values.quantile(.25)))
                    if len(values) else (np.nan, np.nan))
        pearson, pearson_iqr = spread('primary_pearson')
        spearman, spearman_iqr = spread('primary_spearman')
        rows.append({
            'cluster_id': int(source.cluster_id), 'cluster_annotation': source.cluster_annotation,
            'candidate_count': len(members), 'candidate_valid_count': len(valid),
            'median_primary_pearson': pearson, 'primary_pearson_iqr': pearson_iqr,
            'median_primary_spearman': spearman, 'primary_spearman_iqr': spearman_iqr,
            'reproducible_candidate_count': passed,
            'reproducible_fraction': passed / len(valid) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows, columns=[
        'cluster_id', 'cluster_annotation', 'candidate_count', 'candidate_valid_count',
        'median_primary_pearson', 'primary_pearson_iqr', 'median_primary_spearman',
        'primary_spearman_iqr', 'reproducible_candidate_count', 'reproducible_fraction',
    ])

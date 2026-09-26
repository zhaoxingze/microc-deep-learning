"""Independent background-calibration upper tails; no known-label fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_empirical_tail_calibrator(values) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('calibration requires nonempty finite one-dimensional scores')
    return {'sorted_values': np.sort(values).tolist(), 'n': len(values),
            'minimum_p': 1 / (len(values) + 1),
            'formula': '(1 + count(calibration >= x)) / (n + 1)'}


def transform_to_tail_probability(calibrator: dict, values) -> np.ndarray:
    reference = np.asarray(calibrator['sorted_values'], dtype=float)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(reference):
        raise ValueError('tail transform requires finite values and fitted reference')
    return (1 + len(reference) - np.searchsorted(reference, values, side='left')) / (len(reference)+1)


def calibrate_scores(scores: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if scores.window_id.duplicated().any() or split.window_id.duplicated().any():
        raise ValueError('calibration requires unique window IDs')
    reference = split.loc[split.split.eq('calibration')]
    if 'background_candidate' in reference and not reference.background_candidate.eq(True).all():
        raise ValueError('calibration must contain only eligible background')
    if reference.empty or not set(reference.window_id).issubset(set(scores.window_id)):
        raise ValueError('missing calibration scores')
    output = scores.copy()
    output = output.rename(columns={c:f'legacy_{c}' for c in
        ('score_min','score_max','score_difference','paired_candidate_score') if c in output})
    mask = output.window_id.isin(reference.window_id)
    artifact = {'n_calibration_windows': int(mask.sum()), 'calibration_window_ids': reference.window_id.tolist(),
                'tail': 'upper, inclusive ties, add-one smoothing'}
    for rep in ('rep1','rep2'):
        artifact[rep] = {}
        for branch in ('density','shape'):
            raw = f'{rep}_{branch}_raw'
            cal = fit_empirical_tail_calibrator(output.loc[mask, raw].to_numpy())
            artifact[rep][branch] = cal
            p = transform_to_tail_probability(cal, output[raw].to_numpy())
            output[f'{rep}_{branch}_p'] = p
            output[f'{rep}_{branch}_anomaly'] = -np.log10(p)
        output[f'{rep}_calibrated_or_score'] = output[[f'{rep}_density_anomaly', f'{rep}_shape_anomaly']].max(axis=1)
        # These are per-replicate explanatory bounds, never used for ranking.
        output[f'{rep}_p_or_bonferroni'] = np.minimum(1, 2*output[[f'{rep}_density_p',f'{rep}_shape_p']].min(axis=1))
    for branch in ('density','shape'):
        cols = [f'rep1_{branch}_anomaly', f'rep2_{branch}_anomaly']
        output[f'paired_{branch}_anomaly'] = output[cols].mean(axis=1)
        output[f'{branch}_tail_score'] = output[f'paired_{branch}_anomaly']
        output[f'{branch}_anomaly_min'] = output[cols].min(axis=1)
    output['calibrated_or_score'] = output[['density_tail_score','shape_tail_score']].max(axis=1)
    output['paired_candidate_score'] = output.calibrated_or_score
    rep_cols = ['rep1_calibrated_or_score','rep2_calibrated_or_score']
    output['score_min'] = output[rep_cols].min(axis=1)
    output['score_max'] = output[rep_cols].max(axis=1)
    output['score_difference'] = (output[rep_cols[0]]-output[rep_cols[1]]).abs()
    return output, artifact

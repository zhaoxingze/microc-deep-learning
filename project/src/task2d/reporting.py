"""Descriptive Task2D control/candidate summaries without novelty overclaim."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def known_distribution_table(known: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for structure_type in ('CHIN', 'OPCID', 'CHID', 'ALL'):
        subset = known if structure_type == 'ALL' else known.loc[known.type.eq(structure_type)]
        valid = subset.loc[subset.correlation_valid.astype(bool)]
        row = {'type': structure_type, 'n_total': len(subset), 'n_valid': len(valid)}
        for metric in ('pearson', 'spearman'):
            values = valid[metric].dropna()
            for label, quantile in (('p05', .05), ('p25', .25), ('median', .5), ('p75', .75), ('p95', .95)):
                row[f'{metric}_{label}'] = float(values.quantile(quantile)) if len(values) else np.nan
            row[f'{metric}_iqr'] = row[f'{metric}_p75'] - row[f'{metric}_p25']
        rows.append(row)
    return pd.DataFrame(rows)


def write_task2d_summary(path: Path, *, mode: str, candidates: pd.DataFrame,
                         known: pd.DataFrame, background: pd.DataFrame,
                         distributions: pd.DataFrame, floors: dict,
                         clusters: pd.DataFrame, completion: dict) -> Path:
    def median(frame: pd.DataFrame, metric: str) -> float:
        return float(frame[metric].median()) if len(frame) else np.nan

    unannotated = candidates.loc[~candidates.known_overlap]
    noise = candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_NOISE')]
    clustered = candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_CLUSTERED')]
    passed = int(candidates.reproducible_by_known_floor.fillna(False).astype(bool).sum())
    lines = [
        '# Task2D Cross-replicate Reproducibility Validation', '',
        f"> {'SMOKE engineering check only; no scientific conclusions.' if mode == 'smoke' else 'FULL: frozen validation; reproducibility is stability, not structural identity.'}", '',
        '## Frozen measurement', '',
        '- Candidate primary window uses Task2A representative best_scale/center: 3.2, 6.4 or 12.8 kb; canonical 6.4 kb is supplemental. All windows use 100 bp bins and identical coordinates for rep1/rep2.',
        '- Each raw matrix is divided by its own replicate genome-wide E(d), without clipping for correlation. Formal mask: upper triangle i<j with |i-j|>2; full upper is auxiliary.',
        '- Constant vectors or fewer than ten finite paired pixels yield NaN, never an artificial zero. Pearson/Spearman are the primary measurements; cosine and pooled-IQR NRMSE are auxiliary.', '',
        '## Positive and background controls', '',
        f'- Known: {len(known)} rows, {int(known.correlation_valid.sum())} valid; canonical Pearson median {median(known, "pearson"):.6f}; Spearman median {median(known, "spearman"):.6f}.',
        f'- Known floor rule: separate 5th percentiles of valid canonical Pearson and Spearman. Pearson floor `{floors.get("pearson_floor")}`; Spearman floor `{floors.get("spearman_floor")}`; calibration_valid `{str(floors["calibration_valid"]).lower()}`.',
        '- Known type distributions (median, IQR, 5th/25th/75th/95th):', '',
        distributions.to_markdown(index=False), '',
        f'- Background: {len(background)} rows, {int(background.correlation_valid.sum())} valid; Pearson median {median(background, "pearson"):.6f}; Spearman median {median(background, "spearman"):.6f}. Background may also be highly reproducible, so correlation is not a structure classifier.', '',
        '## Candidates and frozen clusters', '',
        f'- Evaluated candidates: {len(candidates)}; valid primary correlations: {int(candidates.correlation_valid.sum())}; zero-overlap eligible: {len(unannotated)}; unannotated noise: {len(noise)}; unannotated clustered (excluding strict-pool rows): {len(clustered)}.',
        f'- Candidate primary Pearson/Spearman medians: {median(candidates, "primary_pearson"):.6f} / {median(candidates, "primary_spearman"):.6f}; pass Known floor: {passed}.',
        f'- Noise valid {int(noise.correlation_valid.sum())} / {len(noise)}, median Pearson/Spearman {median(noise, "primary_pearson"):.6f} / {median(noise, "primary_spearman"):.6f}, pass {int(noise.reproducible_by_known_floor.fillna(False).astype(bool).sum())}. Noise remains unclustered, not a novel class.',
        f'- Unannotated clustered valid {int(clustered.correlation_valid.sum())} / {len(clustered)}, median Pearson/Spearman {median(clustered, "primary_pearson"):.6f} / {median(clustered, "primary_spearman"):.6f}. Known-like cluster membership is not a CHIN/OPCID/CHID prediction.',
        '- Per-cluster candidate reproducibility:', '',
        clusters.to_markdown(index=False) if len(clusters) else 'No non-noise clusters.', '',
        f'- Zero-axis or nonfinite-pixel artifact warnings: {int(candidates.artifact_warning.sum())} candidates. They remain in all tables.',
        '- Strict novel-cluster membership can only come from Task2C novel-like clusters with at least five candidates, zero Known and zero genomic Known overlap, plus at least 80% of valid members passing both floors. No Task2C labels were changed.',
        '- If a Known floor is negative/nonfinite, binary filtering is stopped and continuous candidate metrics remain available; no substitute threshold is chosen.', '',
        '## Interpretation', '',
        '- The same Known structures participated in earlier classifier/method development; these controls are operational reference levels, not independent discovery validation. Background windows are partially correlated/reused. Two replicates and one chromosome cannot establish a new structure type.',
        f'- `task2d_complete={str(completion["task2d_complete"]).lower()}`; Final Task2 status is determined separately after verified finalization.', '',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8-sig')
    return path

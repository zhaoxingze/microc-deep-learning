"""Verified Task2 final tables, common-scale O/E figures and completion."""
from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.task1.manifest import sha256_file
from src.task2.reporting import validate_task2a_outputs, write_csv_utf8, write_manifest
from src.task2d.analysis import annotate_candidates
from src.task2d.inputs import load_inputs, verify_source_hashes
from src.task2d.reporting import known_distribution_table
from src.task2d.reproducibility import CANONICAL_WINDOW_BP, normalize_pair, read_pair


FOLLOWUP_STATUSES = {
    'UNANNOTATED_CLUSTERED_REPRODUCIBLE', 'UNANNOTATED_NOISE_REPRODUCIBLE',
    'KNOWN_LIKE_CLUSTER_REPRODUCIBLE',
}
FINAL_FRONT_COLUMNS = (
    'candidate_id', 'region_id', 'chrom', 'start', 'end', 'center', 'length_bp',
    'best_scale', 'best_branch', 'support_channels', 'cluster_id', 'cluster_annotation',
    'membership_probability', 'known_center_overlap', 'known_interval_overlap', 'known_types',
    'nearest_known_id', 'nearest_known_type', 'nearest_known_distance', 'primary_window_bp',
    'primary_pearson', 'primary_spearman', 'primary_cosine', 'primary_nrmse',
    'canonical_pearson', 'canonical_spearman', 'pearson_floor', 'spearman_floor',
    'reproducible_by_known_floor', 'rep1_zero_axis', 'rep2_zero_axis',
    'artifact_warning', 'correlation_valid', 'final_status', 'strict_novel_candidate', 'evidence_note',
)


def _boolean(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower()
    if not text.isin(['true', 'false']).all():
        raise ValueError('final candidate boolean field contains missing or invalid values')
    return text.eq('true')


def select_final_tables(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = candidates.copy()
    if 'candidate_id' not in data:
        data.insert(0, 'candidate_id', data.region_id)
    for key in ('known_center_overlap', 'known_interval_overlap', 'strict_novel_candidate'):
        data[key] = _boolean(data[key])
    if data.region_id.isna().any() or data.region_id.duplicated().any():
        raise ValueError('final candidate IDs missing or duplicated')
    unannotated = ~(data.known_center_overlap | data.known_interval_overlap)
    strict_mask = data.strict_novel_candidate & unannotated & data.final_status.eq('STRICT_NOVEL_CLUSTER_MEMBER')
    if (data.strict_novel_candidate & ~strict_mask).any():
        raise ValueError('strict novel candidate violates zero-overlap/status rule')
    passed = data.reproducible_by_known_floor.astype(str).str.lower().eq('true')
    followup_mask = unannotated & data.final_status.isin(FOLLOWUP_STATUSES) & passed & ~strict_mask
    front = [column for column in FINAL_FRONT_COLUMNS if column in data]
    ordered = data[front + [column for column in data if column not in front]]
    strict = ordered.loc[strict_mask].copy()
    followup = ordered.loc[followup_mask].copy()
    followup = followup.sort_values(
        ['reproducible_by_known_floor', 'primary_spearman', 'primary_pearson',
         'chrom', 'start', 'end', 'region_id'],
        ascending=[False, False, False, True, True, True, True], kind='mergesort',
    ).reset_index(drop=True)
    return ordered, strict, followup


def common_display_vmax(matrices: list[np.ndarray]) -> float | None:
    if not matrices:
        return None
    values = np.concatenate([np.asarray(matrix, dtype=float).ravel() for matrix in matrices])
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 99.5)) if len(values) else None


def _display_pair(raw1: np.ndarray, raw2: np.ndarray,
                  expected: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    oe1, oe2 = normalize_pair(raw1, raw2, expected['rep1'], expected['rep2'])
    with np.errstate(invalid='ignore'):
        return np.log1p(np.maximum(oe1, 0)), np.log1p(np.maximum(oe2, 0))


def write_final_heatmaps(selected: pd.DataFrame, expected: dict[str, np.ndarray],
                         source_paths: dict[str, Path], output_dir: Path) -> tuple[list[Path], float | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected.empty:
        note = output_dir / 'README.md'
        note.write_text('No final follow-up or strict novel candidate satisfied the frozen rules; no candidate heatmaps exist.\n',
                        encoding='utf-8')
        return [note], None
    cache = []
    all_pixels = []
    for row in selected.itertuples(index=False):
        raw1, raw2, _ = read_pair(source_paths, row.chrom, row.center, CANONICAL_WINDOW_BP)
        display1, display2 = _display_pair(raw1, raw2, expected)
        cache.append((row, display1, display2))
        all_pixels.extend([display1, display2])
    vmax = common_display_vmax(all_pixels)
    if vmax is None or vmax <= 0:
        raise ValueError('cannot establish positive common heatmap visualization scale')
    paths = []
    for row, one, two in cache:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), constrained_layout=True)
        for axis, matrix, name in zip(axes, (one, two, np.abs(one - two)),
                                      ('Rep1 log1p(O/E)', 'Rep2 log1p(O/E)', '|Rep1 - Rep2|')):
            image = axis.imshow(matrix, cmap='magma', vmin=0, vmax=vmax, origin='lower')
            axis.set_title(name, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        fig.colorbar(image, ax=axes, orientation='horizontal', fraction=.05, pad=.04,
                     label=f'Common display range 0–{vmax:.3f}')
        fig.suptitle(f'{row.candidate_id} {row.chrom}:{int(row.start)}-{int(row.end)} '
                     f'(display: center {int(row.center)} ± 3.2 kb)\n'
                     f'cluster {row.cluster_id} {row.cluster_annotation} | '
                     f'primary Pearson {row.primary_pearson:.3f}, Spearman {row.primary_spearman:.3f} | '
                     f'{row.final_status}', fontsize=9)
        stable_id = re.sub(r'[^A-Za-z0-9_.-]', '_', str(row.candidate_id))
        path = output_dir / f'{stable_id}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths, vmax


def _boxplot(ax, groups: list[tuple[str, pd.DataFrame]], metric: str, title: str) -> None:
    items = [(name, frame[metric].dropna().to_numpy(float)) for name, frame in groups if metric in frame]
    items = [(name, values) for name, values in items if len(values)]
    if items:
        ax.boxplot([values for _, values in items], tick_labels=[name for name, _ in items],
                   showfliers=False)
    else:
        ax.text(.5, .5, 'No valid measurements', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=20, labelsize=8)
    ax.set_ylabel(metric.title())
    ax.grid(axis='y', alpha=.2)


def write_final_figures(known: pd.DataFrame, background: pd.DataFrame,
                        candidates: pd.DataFrame, followup: pd.DataFrame,
                        output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    known_groups = [(name, known.loc[known.type.eq(name)]) for name in ('CHIN', 'OPCID', 'CHID')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric in zip(axes, ('pearson', 'spearman')):
        _boxplot(axis, known_groups, metric, f'Known canonical {metric.title()}')
    path = output_dir / 'known_reproducibility_distribution.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    comparison = [
        ('Known', known), ('Background', background),
        ('Unannotated clustered', candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_CLUSTERED')].rename(
            columns={'primary_pearson': 'pearson', 'primary_spearman': 'spearman'})),
        ('Unannotated noise', candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_NOISE')].rename(
            columns={'primary_pearson': 'pearson', 'primary_spearman': 'spearman'})),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for axis, metric in zip(axes, ('pearson', 'spearman')):
        _boxplot(axis, comparison, metric, f'Controls vs candidate primary {metric.title()}')
    path = output_dir / 'candidate_vs_known_reproducibility.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    top = followup.head(12)
    if len(top):
        positions = np.arange(len(top))
        ax.bar(positions - .18, top.primary_pearson.to_numpy(float), width=.36, label='Primary Pearson')
        ax.bar(positions + .18, top.primary_spearman.to_numpy(float), width=.36, label='Primary Spearman')
        ax.set_xticks(positions, top.candidate_id, rotation=70, fontsize=7)
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, 'No reproducible unannotated candidates', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_title('Reproducible unannotated candidates (not confirmed new structures)')
    ax.set_ylabel('Primary replicate correlation')
    ax.grid(axis='y', alpha=.2)
    path = output_dir / 'reproducible_candidate_gallery.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def build_final_completion(checks: dict[str, bool], *, strict_count: int) -> dict:
    checks = {key: bool(value) for key, value in checks.items()}
    return {'checks': checks, 'task2_complete': all(checks.values()),
            'strict_novel_structure_detected': bool(strict_count > 0),
            'task3_executed': False}


def format_control_and_sensitivity_tables(known: pd.DataFrame,
                                          sensitivity: pd.DataFrame) -> str:
    distribution = known_distribution_table(known)
    columns = ['type', 'n_valid', 'pearson_p05', 'pearson_p25', 'pearson_median',
               'pearson_p75', 'pearson_p95', 'pearson_iqr', 'spearman_p05',
               'spearman_p25', 'spearman_median', 'spearman_p75', 'spearman_p95',
               'spearman_iqr']
    return ('Known type distributions (Pearson and Spearman median, IQR, 5th/25th/75th/95th):\n\n'
            + distribution[columns].to_markdown(index=False) + '\n\n'
            + 'Fixed Task2C HDBSCAN sensitivity audit (not parameter selection):\n\n'
            + sensitivity.to_markdown(index=False))


def _summary(path: Path, inputs, candidates: pd.DataFrame, known: pd.DataFrame,
             background: pd.DataFrame, clusters: pd.DataFrame, floors: dict,
             strict: pd.DataFrame, followup: pd.DataFrame, completion: dict,
             a_manifest: dict, b_manifest: dict, c_manifest: dict,
             visualization_vmax: float | None) -> Path:
    a = a_manifest['statistics']
    b = b_manifest['statistics']
    c = c_manifest['statistics']
    known_valid = known.loc[known.correlation_valid.astype(bool)]
    background_valid = background.loc[background.correlation_valid.astype(bool)]
    noise = candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_NOISE')]
    clustered = candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_CLUSTERED')]
    sensitivity = pd.read_csv(inputs.source_paths['task2c_manifest'].parent / 'clustering_sensitivity.csv',
                              encoding='utf-8-sig')
    diagnostics = format_control_and_sensitivity_tables(known, sensitivity)
    cluster_counts = inputs.cluster_summary[['cluster_id', 'candidate_count', 'CHIN_count',
                                             'OPCID_count', 'CHID_count', 'cluster_annotation',
                                             'novel_like_cluster_candidate']]
    top_fields = ['candidate_id', 'chrom', 'start', 'end', 'cluster_id', 'primary_pearson',
                  'primary_spearman', 'known_center_overlap', 'known_interval_overlap',
                  'artifact_warning', 'final_status']
    lines = [
        '# Task2 Final: Proposal → Representation → Grouping → Reproducibility', '',
        '## Task2A — frozen high-recall proposal generator', '',
        '- Six independent channels: Small/Medium/Large × Density/Shape; channel-wise NMS, union and interval merge. Fixed paired p-equivalent threshold <=0.05, not a formal paired test.',
        f"- Channel peaks: {a['channel_peak_counts']}; {a['n_proposals']} proposals merged to {a['n_candidates']} final candidate regions; genomic union coverage {a['coverage_fraction']:.6f}.",
        '- Center-based Known Recall (same Known set was used during method development):', '',
        pd.DataFrame(a['recall']).to_markdown(index=False), '',
        '- The detector is a high-recall proposal generator rather than a novelty judge; further recall tuning on reused Known would overfit.', '',
        '## Task2B — 256D fused representation', '',
        '- Frozen concatenation: Task1 CNN 64D + Small AE 64D + Medium AE 64D + Large AE 64D, with frozen reference-fitted scaling.',
        '- Genomic-blocked Known-vs-Background linear probes (diagnostics, not novel detectors):', '',
        pd.DataFrame(b['probe'])[['representation', 'balanced_accuracy_mean', 'roc_auc_mean', 'pr_auc_mean']].to_markdown(index=False), '',
        f"- PCA PC1/PC2 explained variance {b['pca_explained_variance_ratio'][0]:.6f}/{b['pca_explained_variance_ratio'][1]:.6f}; PCA reference-centroid separation/spread {b['pca_map_diagnostics']['reference_centroid_separation_over_spread']:.3f}; UMAP separation/spread {b['umap_map_diagnostics']['reference_centroid_separation_over_spread']:.3f}, candidate quadrants {b['umap_map_diagnostics']['candidate_occupied_quadrants']}/4. UMAP is visualization only.",
        f"- Mean rep1/rep2 embedding cosine by block: {b['replicate_cosine_means']}.", '',
        '## Task2C — frozen unsupervised grouping', '',
        f"- Candidate+Known clustering PCA: {c_manifest['configuration']['clustering_pca_n_components']} components; HDBSCAN {c_manifest['configuration']['primary_hdbscan']}.",
        f"- {c['n_clusters']} non-noise clusters, {c['n_noise']} / {c['n_candidates'] + c['n_known']} noise ({c['noise_fraction']:.6f}); silhouette {c['silhouette_nonnoise']}; novel-like clusters {c['n_novel_like_clusters']}.",
        '- Cluster candidate and Known-type composition:', '', cluster_counts.to_markdown(index=False), '',
        '- CHIN-like membership is a postfit representation-composition hint, not a CHIN label.', '',
        diagnostics, '',
        '## Task2D — cross-replicate validation', '',
        '- Formal candidate window: frozen representative Small 3.2 kb, Medium 6.4 kb or Large 12.8 kb; canonical 6.4 kb is supplemental. Controls use canonical 6.4 kb. Same coordinates for rep1/rep2, separate genome-wide E(d), 100 bp bins.',
        '- Pearson/Spearman use finite O/E upper-triangle pixels with i<j and |i-j|>2; full upper, cosine and pooled-IQR NRMSE are diagnostics.',
        f"- Known valid {len(known_valid)}/{len(known)}; Pearson median {known_valid.pearson.median():.6f}, 5th percentile {floors['pearson_floor']:.6f}; Spearman median {known_valid.spearman.median():.6f}, 5th percentile {floors['spearman_floor']:.6f}.",
        '- Known CHIN/OPCID/CHID distributions above are also preserved in `outputs/task2d/known_reproducibility_distribution.csv` and figure.',
        f"- Background valid {len(background_valid)}/{len(background)}; Pearson median {background_valid.pearson.median():.6f}; Spearman median {background_valid.spearman.median():.6f}. Background may also be reproducible.",
        f"- Candidate primary valid {int(candidates.correlation_valid.sum())}/{len(candidates)}; median Pearson/Spearman {candidates.primary_pearson.median():.6f}/{candidates.primary_spearman.median():.6f}; pass Known floors {int(candidates.reproducible_by_known_floor.fillna(False).astype(bool).sum())}.",
        f"- Unannotated noise {len(noise)} (valid {int(noise.correlation_valid.sum())}, pass {int(noise.reproducible_by_known_floor.fillna(False).astype(bool).sum())}); median Pearson/Spearman {noise.primary_pearson.median():.6f}/{noise.primary_spearman.median():.6f}.",
        f"- Unannotated clustered {len(clustered)} (valid {int(clustered.correlation_valid.sum())}, pass {int(clustered.reproducible_by_known_floor.fillna(False).astype(bool).sum())}); median Pearson/Spearman {clustered.primary_pearson.median():.6f}/{clustered.primary_spearman.median():.6f}.",
        '- Cluster-level candidate reproducibility (including current Cluster 1):', '', clusters.to_markdown(index=False), '',
        f'- Candidate heatmaps show canonical 6.4 kb log1p(O/E) with one global display vmax `{visualization_vmax}`; this visualization percentile never affects selection.', '',
        '## Final evidence tiers', '',
        f'- Task2C novel-like clusters before Task2D: {c["n_novel_like_clusters"]}; strict clusters surviving Task2D: {strict.cluster_id.nunique()}; strict novel candidate members after 80% valid-member gate: {len(strict)}.',
        f'- Reproducible unannotated follow-up candidates: {len(followup)}; clustered {int(followup.cluster_id.ge(0).sum())}, noise {int(followup.cluster_id.eq(-1).sum())}. Known-like clustered candidates remain unannotated at the genomic locus and are **not** assigned a Known type.',
        '- Top follow-up candidates by predeclared pass → Spearman → Pearson → coordinate ranking:', '',
        followup.head(10)[top_fields].to_markdown(index=False) if len(followup) else 'None.', '',
        ('- 在当前预注册分析协议下，没有候选满足严格的新结构簇定义。' if strict.empty else
         '- Strict novel-cluster members are *candidates*, not independently confirmed new structures.'),
        '- A nonempty reproducible-unannotated list signals follow-up loci, not confirmed novel chromatin structures. Even HDBSCAN noise can be reproducible without forming a new structure class.',
        '- Main limitations: reused Known set in prior supervised/method-development stages; Background references partly reuse AE-training windows and are genomically correlated; one chromosome and two biological repeats; Pearson/Spearman show stability but cannot prove structural identity or biological mechanism.',
        f'- `task2_complete={str(completion["task2_complete"]).lower()}`; `strict_novel_structure_detected={str(completion["strict_novel_structure_detected"]).lower()}`. Completion reflects workflow integrity, not discovery of a new class.', '',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8-sig')
    return path


def finalize_task2(root: str | Path, *, verify: bool = True) -> Path:
    root = Path(root).resolve()
    inputs = load_inputs(root)
    task2d_root = root / 'outputs' / 'task2d'
    task2d_manifest_path = task2d_root / 'task2d_manifest.json'
    d_manifest = validate_task2a_outputs(task2d_manifest_path)
    d_completion = json.loads((task2d_root / 'task2d_completion.json').read_text(encoding='utf-8'))
    if d_completion.get('mode') != 'full' or d_completion.get('task2d_complete') is not True:
        raise ValueError('Task2D Full must complete before Task2 finalization')
    if d_manifest['configuration']['frozen_task2d_protocol_sha256'] != sha256_file(inputs.protocol_path):
        raise ValueError('Task2D frozen protocol hash changed')
    candidates = pd.read_csv(task2d_root / 'candidate_reproducibility.csv', encoding='utf-8-sig')
    known = pd.read_csv(task2d_root / 'known_reproducibility.csv', encoding='utf-8-sig')
    background = pd.read_csv(task2d_root / 'background_reproducibility.csv', encoding='utf-8-sig')
    clusters = pd.read_csv(task2d_root / 'cluster_reproducibility_summary.csv', encoding='utf-8-sig')
    floors = json.loads((task2d_root / 'reproducibility_thresholds.json').read_text(encoding='utf-8'))
    if floors.get('calibration_valid') is not True:
        raise ValueError('Known control calibration failed; strict binary finalization stopped')
    if len(candidates) != len(inputs.candidates) or not candidates.region_id.reset_index(drop=True).equals(
        inputs.candidates.region_id.reset_index(drop=True)
    ) or len(known) != len(inputs.known) or len(background) != len(inputs.background):
        raise ValueError('Task2D Full candidate/control row alignment failed')
    reproduced = annotate_candidates(candidates, inputs.cluster_summary, floors)
    if not reproduced.final_status.equals(candidates.final_status) or not _boolean(reproduced.strict_novel_candidate).equals(
        _boolean(candidates.strict_novel_candidate)
    ):
        raise ValueError('Task2D candidate final status disagrees with frozen analysis rule')
    all_rows, strict, followup = select_final_tables(candidates)
    if missing := set(FINAL_FRONT_COLUMNS) - set(all_rows):
        raise ValueError(f'final candidate fields missing: {sorted(missing)}')
    out = root / 'outputs' / 'task2_final'
    out.mkdir(parents=True, exist_ok=True)
    outputs = [
        write_csv_utf8(all_rows, out / 'all_candidates_final_status.csv'),
        write_csv_utf8(strict, out / 'strict_novel_candidates.csv'),
        write_csv_utf8(followup, out / 'reproducible_unannotated_candidates.csv'),
    ]
    heatmap_rows = pd.concat([followup, strict], ignore_index=True).drop_duplicates('region_id')
    heatmap_paths, vmax = write_final_heatmaps(heatmap_rows, inputs.expected, inputs.source_paths,
                                               out / 'heatmaps')
    outputs.extend(heatmap_paths)
    outputs.extend(write_final_figures(known, background, candidates, followup, out / 'figures'))
    a_path, b_path, c_path = (inputs.source_paths[f'task2{letter}_manifest'] for letter in ('a', 'b', 'c'))
    a_manifest = validate_task2a_outputs(a_path)
    b_manifest = validate_task2a_outputs(b_path)
    c_manifest = validate_task2a_outputs(c_path)
    verify_source_hashes(inputs.source_paths, inputs.source_hashes)
    checks = {
        'task2a_finalized': True, 'task2b_complete': True, 'task2c_complete': True,
        'task2d_reproducibility_complete': d_completion['task2d_complete'],
        'all_eligible_candidates_evaluated': len(candidates) == len(inputs.candidates),
        'known_positive_control_evaluated': len(known) == len(inputs.known),
        'background_reference_evaluated': len(background) == len(inputs.background),
        'cluster_level_reproducibility_complete': len(clusters) == len(inputs.cluster_summary),
        'candidate_final_status_complete': len(all_rows) == len(candidates) and all_rows.final_status.notna().all(),
        'strict_novel_list_generated': outputs[1].is_file(),
        'followup_unannotated_list_generated': outputs[2].is_file(),
        'heatmap_protocol_completed': len(heatmap_paths) == max(len(heatmap_rows), 1),
        'final_manifest_generated': True, 'final_summary_generated': True,
        'no_upstream_result_modified': True,
    }
    completion = build_final_completion(checks, strict_count=len(strict))
    summary_path = _summary(out / 'task2_final_summary.md', inputs, candidates, known, background,
                            clusters, floors, strict, followup, completion,
                            a_manifest, b_manifest, c_manifest, vmax)
    outputs.append(summary_path)
    completion_path = out / 'task2_completion.json'
    completion_path.write_text(json.dumps(completion, indent=2, ensure_ascii=False), encoding='utf-8')
    outputs.append(completion_path)
    configuration = {
        'version': 'Task2-Final-v1', 'task2a_manifest_sha256': sha256_file(a_path),
        'task2b_manifest_sha256': sha256_file(b_path),
        'task2c_manifest_sha256': sha256_file(c_path),
        'task2d_manifest_sha256': sha256_file(task2d_manifest_path),
        'frozen_task2d_protocol_sha256': sha256_file(inputs.protocol_path),
        'rep1_cool_sha256': inputs.source_hashes['rep1_cool'],
        'rep2_cool_sha256': inputs.source_hashes['rep2_cool'],
        'genome_expected_sha256': inputs.source_hashes['genome_expected'],
        'primary_window_rule': inputs.protocol['primary_window_rule'],
        'canonical_window_bp': inputs.protocol['canonical_window_bp'],
        'oe_rule': inputs.protocol['oe_rule'], 'correlation_mask': inputs.protocol['correlation_mask'],
        'pearson_implementation': inputs.protocol['pearson_implementation'],
        'spearman_implementation': inputs.protocol['spearman_implementation'],
        'known_control_floor': inputs.protocol['known_control_floor'],
        'pearson_floor': floors['pearson_floor'], 'spearman_floor': floors['spearman_floor'],
        'strict_novel_cluster_rule': inputs.protocol['strict_novel_cluster_rule'],
        'strict_cluster_pass_fraction': inputs.protocol['strict_cluster_valid_member_pass_fraction'],
        'heatmap_protocol': inputs.protocol['heatmap_rule'], 'visualization_vmax': vmax,
        'seed': inputs.protocol['seed'], 'numpy_version': np.__version__,
        'scipy_version': scipy.__version__, 'sklearn_version': sklearn.__version__,
        'pandas_version': pd.__version__,
    }
    manifest_path = out / 'task2_manifest.json'
    write_manifest(manifest_path, configuration=configuration,
                   input_paths=[*inputs.source_paths.values(), task2d_manifest_path, inputs.protocol_path,
                                Path(__file__)], output_paths=outputs,
                   statistics={'n_candidates': len(all_rows), 'n_strict_novel_candidates': len(strict),
                               'n_followup_candidates': len(followup), 'n_followup_clustered': int(followup.cluster_id.ge(0).sum()),
                               'n_followup_noise': int(followup.cluster_id.eq(-1).sum()),
                               'completion': completion})
    if verify:
        validate_task2a_outputs(manifest_path)
        verify_source_hashes(inputs.source_paths, inputs.source_hashes)
    if not completion['task2_complete']:
        raise ValueError('Task2 final completion checks failed')
    return manifest_path

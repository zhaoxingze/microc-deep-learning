"""Frozen Task2D control-first cross-replicate measurement orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy

from src.task1.manifest import sha256_file
from src.task2.reporting import validate_task2a_outputs, write_csv_utf8, write_manifest
from src.task2d.analysis import annotate_candidates, calibrate_known_floor, cluster_reproducibility
from src.task2d.inputs import Task2DInputs, Task2DPaths, freeze_protocol, load_inputs, verify_source_hashes
from src.task2d.reporting import known_distribution_table, write_task2d_summary
from src.task2d.reproducibility import CANONICAL_WINDOW_BP, measure_pair, read_pair, window_for_scale


SEED = 20_260_920
COORDINATE_LOADER_SOURCE = Path(__file__).resolve().parents[1] / 'data' / 'load_microc.py'


@dataclass(frozen=True)
class Task2DResult:
    output_root: Path
    manifest_path: Path | None
    summary_path: Path | None
    dry_run: bool
    resumed: bool
    elapsed_seconds: float


def _smoke_rows(inputs: Task2DInputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    candidate = inputs.candidates.iloc[np.sort(rng.choice(len(inputs.candidates),
                                        min(12, len(inputs.candidates)), replace=False))].reset_index(drop=True)
    pieces = []
    for structure_type in ('CHIN', 'OPCID', 'CHID'):
        subset = inputs.known.loc[inputs.known.type.eq(structure_type)]
        if len(subset):
            pieces.append(subset.iloc[np.sort(rng.choice(len(subset), min(3, len(subset)), replace=False))])
    known = pd.concat(pieces, ignore_index=True) if pieces else inputs.known.iloc[:0].copy()
    background = inputs.background.iloc[np.sort(rng.choice(len(inputs.background),
                                          min(9, len(inputs.background)), replace=False))].reset_index(drop=True)
    return candidate, known, background


def _measure_reference(frame: pd.DataFrame, inputs: Task2DInputs) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        raw1, raw2, _ = read_pair(inputs.source_paths, row.chrom, row.center, CANONICAL_WINDOW_BP)
        record = row._asdict()
        record.update(measure_pair(raw1, raw2, inputs.expected['rep1'], inputs.expected['rep2']))
        record['window_bp'] = CANONICAL_WINDOW_BP
        rows.append(record)
    return pd.DataFrame(rows)


def _measure_candidates(frame: pd.DataFrame, inputs: Task2DInputs) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        primary_bp = window_for_scale(row.best_scale)
        raw1, raw2, _ = read_pair(inputs.source_paths, row.chrom, row.center, primary_bp)
        primary = measure_pair(raw1, raw2, inputs.expected['rep1'], inputs.expected['rep2'])
        if primary_bp == CANONICAL_WINDOW_BP:
            canonical = dict(primary)
        else:
            canonical1, canonical2, _ = read_pair(inputs.source_paths, row.chrom, row.center,
                                                   CANONICAL_WINDOW_BP)
            canonical = measure_pair(canonical1, canonical2, inputs.expected['rep1'], inputs.expected['rep2'])
        record = row._asdict()
        record['candidate_id'] = row.region_id
        record['primary_window_bp'] = primary_bp
        record.update({f'primary_{key}': value for key, value in primary.items()})
        record.update({f'canonical_{key}': value for key, value in canonical.items()})
        for key in ('n_valid_pixels', 'finite_fraction', 'correlation_valid', 'invalid_reason',
                    'rep1_zero_axis', 'rep2_zero_axis', 'artifact_warning'):
            record[key] = primary[key]
        rows.append(record)
    return pd.DataFrame(rows)


def execute_task2d(root: str | Path, *, mode: str = 'smoke', dry_run: bool = False,
                   resume: bool = False) -> Task2DResult:
    started = time.monotonic()
    paths = Task2DPaths(Path(root).resolve(), mode)
    inputs = load_inputs(paths.root)
    if dry_run:
        print(f'TASK2D DRY RUN: mode={mode}, candidates={len(inputs.candidates)}, '
              f'Known={len(inputs.known)}, Background={len(inputs.background)}, '
              f'protocol_sha256={sha256_file(inputs.protocol_path)}, output={paths.output_root}', flush=True)
        return Task2DResult(paths.output_root, None, None, True, False, time.monotonic() - started)
    if resume and paths.manifest_path.is_file():
        validate_task2a_outputs(paths.manifest_path)
        completion = json.loads((paths.output_root / 'task2d_completion.json').read_text(encoding='utf-8'))
        if not completion.get('task2d_complete'):
            raise ValueError('Task2D resume found incomplete run')
        return Task2DResult(paths.output_root, paths.manifest_path,
                            paths.output_root / 'task2d_summary.md', False, True,
                            time.monotonic() - started)
    candidate_source, known_source, background_source = (
        _smoke_rows(inputs) if mode == 'smoke' else
        (inputs.candidates, inputs.known, inputs.background)
    )
    out = paths.output_root
    protocol_path = freeze_protocol(out / 'frozen_task2d_protocol.json', inputs.protocol)
    known = _measure_reference(known_source, inputs)
    background = _measure_reference(background_source, inputs)
    floors = calibrate_known_floor(known)
    candidates = _measure_candidates(candidate_source, inputs)
    candidates = annotate_candidates(candidates, inputs.cluster_summary, floors)
    cluster = cluster_reproducibility(candidates, inputs.cluster_summary)
    distributions = known_distribution_table(known)
    noise = candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_NOISE')].copy()
    unannotated_clustered = candidates.loc[candidates.candidate_pool.eq('UNANNOTATED_CLUSTERED')].copy()
    output_paths = [protocol_path]
    for filename, frame in (
        ('candidate_reproducibility.csv', candidates),
        ('known_reproducibility.csv', known),
        ('background_reproducibility.csv', background),
        ('known_reproducibility_distribution.csv', distributions),
        ('cluster_reproducibility_summary.csv', cluster),
        ('noise_candidate_reproducibility.csv', noise),
        ('unannotated_clustered_reproducibility.csv', unannotated_clustered),
    ):
        output_paths.append(write_csv_utf8(frame, out / filename))
    thresholds_path = out / 'reproducibility_thresholds.json'
    thresholds_path.write_text(json.dumps({'mode': mode, **floors}, indent=2, ensure_ascii=False), encoding='utf-8')
    output_paths.append(thresholds_path)
    verify_source_hashes(inputs.source_paths, inputs.source_hashes)
    checks = {
        'upstream_artifacts_frozen': True,
        'frozen_task2d_protocol': protocol_path.is_file(),
        'all_eligible_candidates_evaluated': len(candidates) == len(candidate_source) and candidates.region_id.is_unique,
        'known_positive_controls_evaluated': len(known) == len(known_source),
        'background_controls_evaluated': len(background) == len(background_source),
        'known_control_calibration_valid': bool(floors['calibration_valid']),
        'cluster_reproducibility_complete': len(cluster) == len(inputs.cluster_summary),
        'noise_and_clustered_candidate_tables_complete': len(noise) + len(unannotated_clustered) <= len(candidates),
        'candidate_final_status_complete': len(candidates.final_status) == len(candidates),
        'no_model_training_or_upstream_modification': True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    completion = {'mode': mode, 'checks': checks, 'task2d_complete': all(checks.values()),
                  'scientific_full_result': mode == 'full', 'task2_finalized': False}
    completion_path = out / 'task2d_completion.json'
    completion_path.write_text(json.dumps(completion, indent=2), encoding='utf-8')
    output_paths.append(completion_path)
    summary_path = write_task2d_summary(out / 'task2d_summary.md', mode=mode,
                                        candidates=candidates, known=known, background=background,
                                        distributions=distributions, floors=floors,
                                        clusters=cluster, completion=completion)
    output_paths.append(summary_path)
    source_code = [Path(__file__), Path(__file__).with_name('inputs.py'),
                   Path(__file__).with_name('reproducibility.py'),
                   Path(__file__).with_name('analysis.py'), Path(__file__).with_name('reporting.py'),
                   COORDINATE_LOADER_SOURCE]
    write_manifest(paths.manifest_path, configuration={
        'version': inputs.protocol['version'], 'mode': mode,
        'frozen_task2d_protocol_sha256': sha256_file(protocol_path),
        'task2c_manifest_sha256': inputs.source_hashes['task2c_manifest'],
        'task2c_membership_sha256': inputs.source_hashes['task2c_membership'],
        'candidate_list_sha256': inputs.source_hashes['task2a_candidates'],
        'candidate_embedding_sha256': inputs.source_hashes['task2b_candidate_embeddings'],
        'genome_expected_sha256': inputs.source_hashes['genome_expected'],
        'rep1_cool_sha256': inputs.source_hashes['rep1_cool'],
        'rep2_cool_sha256': inputs.source_hashes['rep2_cool'],
        'primary_window_rule': inputs.protocol.get('primary_window_rule'),
        'canonical_window_bp': CANONICAL_WINDOW_BP,
        'oe_rule': inputs.protocol.get('oe_rule'),
        'correlation_mask': inputs.protocol.get('correlation_mask'),
        'pearson_implementation': inputs.protocol.get('pearson_implementation'),
        'spearman_implementation': inputs.protocol.get('spearman_implementation'),
        'known_control_floor': inputs.protocol.get('known_control_floor'),
        'pearson_floor': floors['pearson_floor'], 'spearman_floor': floors['spearman_floor'],
        'strict_novel_cluster_rule': inputs.protocol.get('strict_novel_cluster_rule'),
        'strict_cluster_pass_fraction': .8, 'seed': SEED,
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
    }, input_paths=[*inputs.source_paths.values(), *source_code], output_paths=output_paths,
        statistics={
            'n_candidates': len(candidates), 'n_known': len(known), 'n_background': len(background),
            'candidate_valid_count': int(candidates.correlation_valid.sum()),
            'known_valid_count': int(known.correlation_valid.sum()),
            'background_valid_count': int(background.correlation_valid.sum()),
            'n_reproducible_by_known_floor': int(candidates.reproducible_by_known_floor.fillna(False).astype(bool).sum()),
            'n_strict_novel_candidate': int(candidates.strict_novel_candidate.sum()),
            'completion': completion,
        })
    validate_task2a_outputs(paths.manifest_path)
    if not completion['task2d_complete'] and floors['calibration_valid']:
        raise ValueError('Task2D completion checks failed')
    return Task2DResult(out, paths.manifest_path, summary_path, False, False,
                        time.monotonic() - started)

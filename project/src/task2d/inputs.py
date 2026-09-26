"""Read-only Task2A/B/C input and pre-registered protocol boundary."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.task1.manifest import sha256_file
from src.task2.reporting import validate_task2a_outputs
from src.task2d.analysis import KNOWN_FLOOR_PERCENTILE, STRICT_VALID_MEMBER_PASS_FRACTION
from src.task2d.reproducibility import (
    SCALE_WINDOW_BP, CANONICAL_WINDOW_BP, TARGET_BIN_SIZE, MIN_VALID_PIXELS, CORRELATION_GAP,
)


@dataclass(frozen=True)
class Task2DPaths:
    root: Path
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {'smoke', 'full'}:
            raise ValueError('Task2D mode must be smoke or full')

    @property
    def output_root(self) -> Path:
        if self.mode == 'smoke':
            return self.root / 'outputs' / 'pipeline_runs' / 'task2d_smoke' / 'outputs' / 'task2d'
        return self.root / 'outputs' / 'task2d'

    @property
    def manifest_path(self) -> Path:
        return self.output_root / 'task2d_manifest.json'


@dataclass(frozen=True)
class Task2DInputs:
    candidates: pd.DataFrame
    known: pd.DataFrame
    background: pd.DataFrame
    cluster_summary: pd.DataFrame
    expected: dict[str, np.ndarray]
    source_paths: dict[str, Path]
    source_hashes: dict[str, str]
    protocol: dict
    protocol_path: Path


def source_paths(root: Path) -> dict[str, Path]:
    a = root / 'outputs' / 'task2a_final'
    b = root / 'outputs' / 'task2b'
    c = root / 'outputs' / 'task2c'
    return {
        'task2a_manifest': a / 'task2a_final_manifest.json',
        'task2b_manifest': b / 'task2b_manifest.json',
        'task2c_manifest': c / 'task2c_manifest.json',
        'task2a_finalization': a / 'task2a_finalization.json',
        'task2b_completion': b / 'task2b_completion.json',
        'task2c_completion': c / 'task2c_completion.json',
        'task2a_candidates': a / 'task2a_final_candidates.csv',
        'task2b_candidate_embeddings': b / 'candidate_embeddings.npy',
        'task2b_reference_metadata': b / 'reference_embedding_metadata.csv',
        'task2c_membership': c / 'cluster_membership.csv',
        'task2c_clusters': c / 'clusters.csv',
        'task2c_cluster_summary': c / 'cluster_summary.csv',
        'multiscale_protocol': root / 'outputs' / 'task2a_multiscale' / 'frozen_multiscale_protocol.json',
        'genome_expected': root / 'data' / 'task2_multiscale' / 'genome_expected_max128.npz',
        'rep1_cool': root.parent / 'micro-c数据' / 'GSE272159_37C_rep1.mapq_30.10.cool',
        'rep2_cool': root.parent / 'micro-c数据' / 'GSE272159_37C_rep2.mapq_30.10.cool',
    }


def verify_source_hashes(paths: dict[str, Path], expected_hashes: dict[str, str]) -> None:
    if set(paths) != set(expected_hashes):
        raise ValueError('frozen source hash keys differ from actual sources')
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[key]:
            raise ValueError(f'frozen source hash mismatch: {key}: {path}')


def verify_completion(values: dict[str, dict]) -> None:
    a, b, c = values['task2a'], values['task2b'], values['task2c']
    if a.get('protocol_frozen') is not True or a.get('ready_for_task2b') is not True:
        raise ValueError('Task2A is not finalized')
    if b.get('task2b_complete') is not True or b.get('mode') != 'full':
        raise ValueError('Task2B Full is incomplete')
    if c.get('task2c_complete') is not True or c.get('ready_for_task2d') is not True:
        raise ValueError('Task2C is incomplete or not ready for Task2D')


def validate_alignment(task2a: pd.DataFrame, task2b: pd.DataFrame,
                       clusters: pd.DataFrame, membership: pd.DataFrame,
                       references: pd.DataFrame) -> None:
    for frame in (task2a, task2b, clusters):
        if 'region_id' not in frame or frame.region_id.isna().any() or frame.region_id.duplicated().any():
            raise ValueError('duplicate or missing candidate region ID')
    ids = task2a.region_id.reset_index(drop=True)
    if not ids.equals(task2b.region_id.reset_index(drop=True)) or not ids.equals(clusters.region_id.reset_index(drop=True)):
        raise ValueError('candidate row alignment differs across Task2A/B/C')
    candidate_members = membership.loc[membership.sample_source.eq('candidate')].reset_index(drop=True)
    if len(candidate_members) != len(ids) or not ids.equals(candidate_members.region_id) or (
        not ids.equals(candidate_members.sample_id)
    ) or not np.array_equal(clusters.cluster_id.to_numpy(), candidate_members.cluster_id.to_numpy()):
        raise ValueError('Task2C candidate membership alignment mismatch')
    if references.reference_id.isna().any() or references.reference_id.duplicated().any():
        raise ValueError('duplicate or missing reference ID')
    known = references.loc[references.reference_kind.eq('Known')].reset_index(drop=True)
    background = references.loc[references.reference_kind.eq('Background')].reset_index(drop=True)
    known_members = membership.loc[membership.sample_source.eq('known')].reset_index(drop=True)
    if len(known) == 0 or len(background) == 0 or not known.reference_id.equals(known_members.sample_id):
        raise ValueError('Task2C Known membership alignment mismatch')


def freeze_protocol(path: Path, payload: dict) -> Path:
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError('frozen Task2D protocol differs from current inputs or rules')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def validate_protocol_rules(protocol: dict) -> None:
    """Fail closed if frozen scientific rules differ from the current implementation."""
    expected = {
        'version': 'Task2D-Cross-Replicate-Reproducibility-v1',
        'seed': 20260920,
        'primary_window_rule': {
            'source': 'Task2A frozen best_scale and representative center', **SCALE_WINDOW_BP,
        },
        'canonical_window_bp': CANONICAL_WINDOW_BP,
        'target_bin_size': TARGET_BIN_SIZE,
        'minimum_valid_pixels': MIN_VALID_PIXELS,
        'oe_rule': "unclipped raw matrix divided by that replicate's validated genome-wide E(distance); nonpositive expected invalid",
        'correlation_mask': f'i < j and abs(i-j) > {CORRELATION_GAP}, paired finite O/E and positive expected',
        'full_upper_auxiliary_mask': 'i < j, paired finite O/E and positive expected',
        'pearson_implementation': 'scipy.stats.pearsonr on valid masked O/E values',
        'spearman_implementation': 'scipy.stats.spearmanr on valid masked O/E values',
        'nrmse_rule': 'masked RMSE / (pooled IQR + 1e-12)',
        'zero_axis_rule': 'any all-zero row or column in raw unbalanced replicate matrix; warning only',
        'known_control_floor': (
            f'{KNOWN_FLOOR_PERCENTILE}th percentile of valid 344 Known canonical Pearson and Spearman '
            'separately; nonfinite or negative floor stops binary filtering'
        ),
        'candidate_pass_rule': 'valid primary Pearson >= pearson_floor AND valid primary Spearman >= spearman_floor',
        'strict_novel_cluster_rule': 'Task2C novel_like flag AND >=5 candidates AND zero Known AND zero claimed-member genomic Known overlap AND >=80% of valid candidate members pass',
        'strict_cluster_valid_member_pass_fraction': STRICT_VALID_MEMBER_PASS_FRACTION,
        'follow_up_rule': 'zero genomic Known overlap AND reproducible unannotated noise or clustered candidate; no novel class claim',
        'ranking_rule': 'pass first, primary Spearman descending, primary Pearson descending, chrom/start/end/region_id ascending',
        'heatmap_rule': 'log1p(unclipped O/E) rep1/rep2/difference, common vmax=99.5th percentile of finite pixels of all final follow-up rep1+rep2 displays; display only',
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f'frozen Task2D protocol {key} differs from implemented rule')


def load_inputs(root: str | Path) -> Task2DInputs:
    root = Path(root).resolve()
    paths = source_paths(root)
    missing = [f'{key}: {path}' for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Task2D frozen input missing: {missing}')
    manifest_a = validate_task2a_outputs(paths['task2a_manifest'])
    manifest_b = validate_task2a_outputs(paths['task2b_manifest'])
    validate_task2a_outputs(paths['task2c_manifest'])
    completions = {
        name: json.loads(paths[f'{name}_finalization' if name == 'task2a' else f'{name}_completion'].read_text(encoding='utf-8'))
        for name in ('task2a', 'task2b', 'task2c')
    }
    verify_completion(completions)
    if manifest_b['configuration']['candidate_list_sha256'] != sha256_file(paths['task2a_candidates']):
        raise ValueError('Task2A candidate hash disagrees with Task2B')
    manifest_c = json.loads(paths['task2c_manifest'].read_text(encoding='utf-8'))
    if manifest_c['configuration']['candidate_embedding_sha256'] != sha256_file(paths['task2b_candidate_embeddings']):
        raise ValueError('Task2B embedding hash disagrees with Task2C')
    protocol_path = root / 'outputs' / 'task2d' / 'frozen_task2d_protocol.json'
    if not protocol_path.is_file():
        raise FileNotFoundError(f'Task2D pre-registered protocol missing: {protocol_path}')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    validate_protocol_rules(protocol)
    verify_source_hashes(paths, protocol['source_sha256'])
    a = pd.read_csv(paths['task2a_candidates'], encoding='utf-8-sig')
    b = pd.read_csv(root / 'outputs' / 'task2b' / 'candidate_embedding_metadata.csv', encoding='utf-8-sig')
    c = pd.read_csv(paths['task2c_clusters'], encoding='utf-8-sig')
    membership = pd.read_csv(paths['task2c_membership'], encoding='utf-8-sig')
    refs = pd.read_csv(paths['task2b_reference_metadata'], encoding='utf-8-sig')
    validate_alignment(a, b, c, membership, refs)
    for column in ('best_scale', 'best_branch', 'support_channels', 'known_center_overlap',
                   'known_interval_overlap', 'cluster_annotation', 'membership_probability'):
        if column not in c:
            raise ValueError(f'Task2C candidate metadata missing: {column}')
    if not c.best_scale.isin({'small', 'medium', 'large'}).all():
        raise ValueError('invalid frozen representative scale')
    with np.load(paths['genome_expected'], allow_pickle=False) as stored:
        expected = {rep: stored[rep].copy() for rep in ('rep1', 'rep2')}
    if any(len(vector) < 128 or not np.isfinite(vector[:128]).all() for vector in expected.values()):
        raise ValueError('genome-wide expected artifact invalid')
    return Task2DInputs(
        c, refs.loc[refs.reference_kind.eq('Known')].reset_index(drop=True),
        refs.loc[refs.reference_kind.eq('Background')].reset_index(drop=True),
        pd.read_csv(paths['task2c_cluster_summary'], encoding='utf-8-sig'), expected,
        paths, protocol['source_sha256'], protocol, protocol_path,
    )

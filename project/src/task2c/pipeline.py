"""Run Task2C from read-only frozen Task2A/Task2B artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from src.task1.manifest import sha256_file
from src.task2.reporting import validate_task2a_outputs, write_csv_utf8, write_manifest
from src.task2c.analysis import AnalysisResult, analyze_clusters
from src.task2c.clustering import PRIMARY_PARAMETERS, PrimaryFit, fit_primary, run_sensitivity
from src.task2c.inputs import Task2CInputs, Task2CPaths, freeze_protocol, load_task2c_inputs
from src.task2c.reporting import (
    write_cluster_figures, write_cluster_heatmaps, write_task2c_summary,
)


SEED = 20_260_920


@dataclass(frozen=True)
class Task2CResult:
    output_root: Path
    manifest_path: Path | None
    summary_path: Path | None
    dry_run: bool
    resumed: bool
    elapsed_seconds: float


def build_frozen_protocol(mode: str, source_hashes: dict[str, str],
                          clustering_code_sha256: str) -> dict:
    if mode not in {"smoke", "full"}:
        raise ValueError("Task2C mode must be smoke or full")
    return {
        "version": "Task2C-Structure-Grouping-v1", "mode": mode, "seed": SEED,
        "primary_population": "Task2B paired-average fused candidate + unique Known reference embeddings",
        "background_in_pca_or_hdbscan_fit": False,
        "replicates_as_independent_fit_rows": False,
        "proposal_scores_used_for_fit": False,
        "known_labels_used_for_fit": False,
        "clustering_pca": {"variance_fraction": .95, "svd_solver": "full",
                           "fit_population": "candidate+Known without labels"},
        "primary_hdbscan": dict(PRIMARY_PARAMETERS),
        "sensitivity_only": [{"min_cluster_size": 8, "min_samples": 5},
                             {"min_cluster_size": 5, "min_samples": 8}],
        "known_dominant_purity": .5,
        "novel_like": "cluster>=0 and >=5 candidates and 0 Known and 0 candidate known-center/interval overlap",
        "nearest_known_distance_as_filter": False,
        "umap_used_for_clustering": False,
        "task2d_executed": False,
        "source_sha256": dict(source_hashes),
        "clustering_code_sha256": clustering_code_sha256,
    }


def build_completion(primary: PrimaryFit, analysis: AnalysisResult,
                     *, n_figures: int, n_overviews: int) -> dict:
    clusters = sorted(set(map(int, primary.labels)) - {-1})
    checks = {
        "candidate_embeddings_metadata_aligned": len(analysis.clusters) == primary.n_candidate,
        "known_reference_metadata_aligned": int(analysis.membership.sample_source.eq("known").sum()) == primary.n_known,
        "clustering_pca_complete": primary.projected.shape[0] == len(primary.labels)
                                   and primary.pca.explained_variance_ratio_.sum() >= .95 - 1e-10,
        "primary_hdbscan_complete": len(primary.labels) == primary.n_candidate + primary.n_known,
        "cluster_membership_complete": len(analysis.membership) == len(primary.labels)
                                       and np.isfinite(primary.probabilities).all(),
        "cluster_summary_complete": sorted(analysis.summary.cluster_id.tolist()) == clusters,
        "known_like_annotation_complete": len(analysis.clusters.cluster_annotation) == primary.n_candidate,
        "novel_like_identification_complete": analysis.clusters.novel_like_cluster_candidate.notna().all(),
        "centroids_and_medoids_complete": len(analysis.centroids) == len(clusters)
                                        and len(analysis.representatives) == len(clusters),
        "pca_umap_visualization_complete": n_figures == 4 and n_overviews >= len(clusters),
        "known_labels_used_after_clustering_only": True,
        "umap_not_used_for_clustering": True,
        "task2d_not_executed": True,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    complete = all(checks.values())
    return {
        "checks": checks, "task2c_complete": complete, "ready_for_task2d": complete,
        "known_labels_used_after_clustering_only": True,
        "umap_used_for_clustering": False, "task2d_executed": False,
    }


def _smoke_indices(inputs: Task2CInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    candidate_idx = np.sort(rng.choice(len(inputs.candidates), size=min(24, len(inputs.candidates)), replace=False))
    known_idx = np.sort(rng.choice(len(inputs.known), size=min(48, len(inputs.known)), replace=False))
    background_idx = np.sort(rng.choice(len(inputs.background), size=min(48, len(inputs.background)), replace=False))
    return candidate_idx, known_idx, background_idx


def _replicate_centroid_diagnostic(primary: PrimaryFit, analysis: AnalysisResult,
                                   rep1: np.ndarray, rep2: np.ndarray) -> pd.DataFrame:
    region_ids = analysis.clusters.region_id.to_numpy()
    if not len(analysis.centroids):
        return pd.DataFrame({"region_id": region_ids, "rep1_nearest_cluster": -1,
                             "rep2_nearest_cluster": -1, "nearest_cluster_agreement": False})
    cluster_ids = analysis.summary.cluster_id.to_numpy(dtype=int)
    centers = primary.pca.transform(analysis.centroids)
    one = cluster_ids[cdist(primary.pca.transform(rep1), centers).argmin(axis=1)]
    two = cluster_ids[cdist(primary.pca.transform(rep2), centers).argmin(axis=1)]
    return pd.DataFrame({"region_id": region_ids, "rep1_nearest_cluster": one,
                         "rep2_nearest_cluster": two, "nearest_cluster_agreement": one == two})


def execute_task2c(root: str | Path, *, mode: str = "smoke", dry_run: bool = False,
                   resume: bool = False) -> Task2CResult:
    started = time.monotonic()
    paths = Task2CPaths(Path(root).resolve(), mode)
    inputs = load_task2c_inputs(paths.root)
    if dry_run:
        print(f"TASK2C DRY RUN: mode={mode}, candidates={len(inputs.candidates)}, "
              f"Known={len(inputs.known)}, Background_diagnostic={len(inputs.background)}, "
              f"dimension={inputs.candidates.shape[1]}, output={paths.output_root}", flush=True)
        return Task2CResult(paths.output_root, None, None, True, False, time.monotonic() - started)
    if resume and paths.manifest_path.is_file():
        validate_task2a_outputs(paths.manifest_path)
        completion = json.loads((paths.output_root / "task2c_completion.json").read_text(encoding="utf-8"))
        if completion.get("task2c_complete") is not True:
            raise ValueError("Task2C resume found incomplete manifest")
        return Task2CResult(paths.output_root, paths.manifest_path,
                            paths.output_root / "task2c_summary.md", False, True,
                            time.monotonic() - started)

    if mode == "smoke":
        ci, ki, bi = _smoke_indices(inputs)
    else:
        ci, ki, bi = np.arange(len(inputs.candidates)), np.arange(len(inputs.known)), np.arange(len(inputs.background))
    candidates = inputs.candidates[ci]
    known = inputs.known[ki]
    background = inputs.background[bi]
    candidate_meta = inputs.candidate_meta.iloc[ci].reset_index(drop=True)
    known_meta = inputs.known_meta.iloc[ki].reset_index(drop=True)
    background_meta = inputs.background_meta.iloc[bi].reset_index(drop=True)
    rep1, rep2 = inputs.candidate_rep1[ci], inputs.candidate_rep2[ci]

    out = paths.output_root
    protocol = build_frozen_protocol(mode, inputs.source_hashes,
                                     sha256_file(Path(__file__).with_name("clustering.py")))
    protocol_path = freeze_protocol(out / "frozen_task2c_protocol.json", protocol)
    primary = fit_primary(candidates, known)  # No metadata or labels enter PCA/HDBSCAN.
    analysis = analyze_clusters(primary, candidate_meta, known_meta, background, background_meta)
    sensitivity = run_sensitivity(primary)
    assigned = primary.labels != -1
    distinct = set(primary.labels[assigned])
    silhouette = (float(silhouette_score(primary.projected[assigned], primary.labels[assigned]))
                  if len(distinct) >= 2 and int(assigned.sum()) > len(distinct) else None)
    rep_diagnostic = _replicate_centroid_diagnostic(primary, analysis, rep1, rep2)

    output_paths: list[Path] = [protocol_path]
    pca_path = out / "clustering_pca_model.pkl"
    joblib.dump(primary.pca, pca_path)
    output_paths.append(pca_path)
    variance = pd.DataFrame({
        "component": np.arange(1, len(primary.pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": primary.pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(primary.pca.explained_variance_ratio_),
    })
    output_paths.extend([
        write_csv_utf8(variance, out / "clustering_pca_variance.csv"),
        write_csv_utf8(analysis.membership, out / "cluster_membership.csv"),
        write_csv_utf8(analysis.clusters, out / "clusters.csv"),
        write_csv_utf8(analysis.summary, out / "cluster_summary.csv"),
        write_csv_utf8(analysis.representatives, out / "cluster_representatives.csv"),
        write_csv_utf8(analysis.background_proximity, out / "background_proximity.csv"),
        write_csv_utf8(sensitivity, out / "clustering_sensitivity.csv"),
        write_csv_utf8(rep_diagnostic, out / "replicate_cluster_stability.csv"),
    ])
    centroid_path = out / "cluster_centroids.npy"
    np.save(centroid_path, analysis.centroids, allow_pickle=False)
    output_paths.append(centroid_path)
    figure_paths = write_cluster_figures(primary, analysis, inputs.umap_coordinates, out)
    output_paths.extend(figure_paths)
    heatmap_paths = write_cluster_heatmaps(primary, analysis, inputs.umap_coordinates,
                                           inputs.source_paths, out)
    output_paths.extend(heatmap_paths)
    completion = build_completion(primary, analysis, n_figures=len(figure_paths),
                                  n_overviews=len([p for p in heatmap_paths if p.parent.name == "cluster_overviews"]))
    completion_path = out / "task2c_completion.json"
    completion_path.write_text(json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths.append(completion_path)
    summary_path = write_task2c_summary(out / "task2c_summary.md", mode=mode,
                                        primary=primary, analysis=analysis,
                                        sensitivity=sensitivity, silhouette=silhouette,
                                        completion=completion)
    output_paths.append(summary_path)
    current_hashes = {key: sha256_file(path) for key, path in inputs.source_paths.items()
                      if key not in {"rep1_cool", "rep2_cool"}}
    if current_hashes != inputs.source_hashes:
        raise ValueError("Task2A/Task2B input artifact changed during Task2C")
    known_sha = hashlib.sha256(np.ascontiguousarray(known).tobytes()).hexdigest()
    configuration = {
        "version": protocol["version"], "mode": mode, "seed": SEED,
        "task2a_final_protocol_sha256": inputs.source_hashes["task2a_protocol"],
        "task2a_candidate_csv_sha256": inputs.source_hashes["task2a_candidates"],
        "task2b_manifest_sha256": inputs.source_hashes["task2b_manifest"],
        "candidate_embedding_sha256": inputs.source_hashes["candidate_embeddings"],
        "known_reference_embedding_subset_sha256": known_sha,
        "reference_embedding_file_sha256": inputs.source_hashes["reference_embeddings"],
        "feature_scaler_sha256": inputs.source_hashes["feature_scaler"],
        "frozen_protocol_sha256": sha256_file(protocol_path),
        "clustering_pca": protocol["clustering_pca"],
        "clustering_pca_n_components": int(primary.projected.shape[1]),
        "clustering_pca_cumulative_explained_variance": float(primary.pca.explained_variance_ratio_.sum()),
        "hdbscan_implementation": "sklearn.cluster.HDBSCAN",
        "sklearn_version": sklearn.__version__, "primary_hdbscan": protocol["primary_hdbscan"],
        "cluster_summary_sha256": sha256_file(out / "cluster_summary.csv"),
        "known_labels_used_after_clustering_only": True,
        "umap_used_for_clustering": False, "task2d_executed": False,
    }
    source_code = [Path(__file__), Path(__file__).with_name("inputs.py"),
                   Path(__file__).with_name("clustering.py"), Path(__file__).with_name("analysis.py"),
                   Path(__file__).with_name("reporting.py")]
    write_manifest(paths.manifest_path, configuration=configuration,
                   input_paths=[*inputs.source_paths.values(), *source_code],
                   output_paths=output_paths,
                   statistics={
                       "n_candidates": len(candidates), "n_known": len(known),
                       "n_background_diagnostic": len(background),
                       "n_clusters": len(analysis.summary),
                       "n_noise": int((primary.labels == -1).sum()),
                       "noise_fraction": float((primary.labels == -1).mean()),
                       "n_novel_like_clusters": int(analysis.summary.novel_like_cluster_candidate.sum()) if len(analysis.summary) else 0,
                       "silhouette_nonnoise": silhouette,
                       "replicate_nearest_centroid_agreement": float(rep_diagnostic.nearest_cluster_agreement.mean()),
                       "completion": completion,
                   })
    validate_task2a_outputs(paths.manifest_path)
    if not completion["task2c_complete"]:
        raise ValueError("Task2C completion checks failed")
    return Task2CResult(out, paths.manifest_path, summary_path, False, False,
                        time.monotonic() - started)

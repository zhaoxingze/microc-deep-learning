"""Orchestrate frozen Task2B embedding extraction, evaluation and reporting."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from src.data.known_dataset import attach_structure_ids
from src.task1.manifest import sha256_file
from src.task1.pipeline import read_validation_selected_input
from src.task2.final_pipeline import execute_task2a_final
from src.task2.reporting import validate_task2a_outputs, write_csv_utf8, write_manifest
from src.task2b.evaluation import (
    audit_background_reference, build_genomic_groups, evaluate_blocked_probes, fit_reference_scalers,
    fit_visualization_models, select_background_reference,
)
from src.task2b.features import (
    BLOCK_ORDER, average_replicate_blocks, block_cosine_similarity,
    extract_embedding_blocks, fuse_blocks, load_model_bundle,
)
from src.task2b.reporting import (
    map_diagnostics, plot_embedding_map, plot_pca_variance, write_task2b_summary,
)


SEED = 20_260_920


@dataclass(frozen=True)
class Task2BPaths:
    project_root: Path
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("Task2B mode must be smoke or full")

    @property
    def output_root(self) -> Path:
        if self.mode == "smoke":
            return self.project_root / "outputs" / "pipeline_runs" / "task2b_smoke" / "outputs" / "task2b"
        return self.project_root / "outputs" / "task2b"

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "task2b_manifest.json"


@dataclass(frozen=True)
class Task2BResult:
    output_root: Path
    manifest_path: Path | None
    summary_path: Path | None
    dry_run: bool
    resumed: bool
    elapsed_seconds: float


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _verify_prior_output_hash(prior_manifest: dict, path: Path) -> str:
    records = prior_manifest.get("outputs", {})
    record = next((value for source, value in records.items()
                   if str(Path(source).resolve()).casefold() == str(path.resolve()).casefold()), None)
    if not isinstance(record, dict) or not path.is_file():
        raise ValueError(f"source artifact not covered by frozen Task2A.3 manifest: {path}")
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise ValueError(f"source artifact SHA256 differs from Task2A.3: {path}")
    return actual


def _reference_rows(project_root: Path) -> tuple[pd.DataFrame, int, int, dict]:
    known = attach_structure_ids(pd.read_csv(
        project_root / "data" / "processed" / "structures.csv", encoding="utf-8-sig"
    ))
    if len(known) != 344:
        raise ValueError(f"expected 344 unique known structures, found {len(known)}")
    known = known.copy()
    known["reference_id"] = known.structure_id
    known["reference_kind"] = "Known"
    background_pool = pd.read_csv(
        project_root / "data" / "task2_multiscale" / "large" / "background_split.csv",
        encoding="utf-8-sig",
    )
    background = select_background_reference(background_pool, known, n=len(known), seed=SEED)
    background_audit = audit_background_reference(background_pool, known, background)
    columns = ["reference_id", "reference_kind", "type", "chrom", "start", "end", "center"]
    combined = pd.concat([known[columns], background[columns]], ignore_index=True)
    if combined.reference_id.duplicated().any():
        raise ValueError("reference IDs are not unique")
    return combined, len(known), len(background), background_audit


def _smoke_subset(frame: pd.DataFrame, n: int, *, seed_offset: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    selected = np.random.default_rng(SEED + seed_offset).choice(len(frame), size=n, replace=False)
    return frame.iloc[np.sort(selected)].reset_index(drop=True)


def _coordinate_table(reference: pd.DataFrame, candidates: pd.DataFrame,
                      values: np.ndarray) -> pd.DataFrame:
    if len(values) != len(reference) + len(candidates):
        raise ValueError("visualization coordinate alignment failed")
    rows = pd.concat([
        pd.DataFrame({"point_id": reference.reference_id, "point_kind": "Reference",
                      "type": reference.type, "chrom": reference.chrom, "center": reference.center}),
        pd.DataFrame({"point_id": candidates.region_id, "point_kind": "Candidate",
                      "type": "Candidate", "chrom": candidates.chrom, "center": candidates.center}),
    ], ignore_index=True)
    rows["x"] = values[:, 0]
    rows["y"] = values[:, 1]
    return rows


def _valid_source_inputs(root: Path) -> dict[str, Path]:
    final_root = root / "outputs" / "task2a_final"
    paths = {
        "task2a_final_manifest": final_root / "task2a_final_manifest.json",
        "task2a_final_protocol": final_root / "frozen_proposal_protocol.json",
        "task2a_final_candidates": final_root / "task2a_final_candidates.csv",
        "task2a_finalization": final_root / "task2a_finalization.json",
        "task1_manifest": root / "outputs" / "task1" / "task1_manifest.json",
        "task1_input_comparison": root / "outputs" / "task1" / "input_comparison.csv",
        "task2a_multiscale_manifest": root / "outputs" / "task2a_multiscale" / "task2a_multiscale_manifest.json",
        "task2a_multiscale_protocol": root / "outputs" / "task2a_multiscale" / "frozen_multiscale_protocol.json",
        "expected": root / "data" / "task2_multiscale" / "genome_expected_max128.npz",
        "large_background": root / "data" / "task2_multiscale" / "large" / "background_split.csv",
        "known": root / "data" / "processed" / "structures.csv",
        "rep1_cool": root.parent / "micro-c数据" / "GSE272159_37C_rep1.mapq_30.10.cool",
        "rep2_cool": root.parent / "micro-c数据" / "GSE272159_37C_rep2.mapq_30.10.cool",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Task2B source artifacts missing: {missing}")
    return paths


def execute_task2b(
    project_root: str | Path, *, mode: str = "smoke", dry_run: bool = False,
    resume: bool = False,
) -> Task2BResult:
    started = time.monotonic()
    paths = Task2BPaths(Path(project_root).resolve(), mode)
    final_manifest_path = paths.project_root / "outputs" / "task2a_final" / "task2a_final_manifest.json"
    if not final_manifest_path.is_file():
        if dry_run:
            raise FileNotFoundError("Task2A Final proposals are required for Task2B dry-run")
        execute_task2a_final(paths.project_root)
    source = _valid_source_inputs(paths.project_root)
    validate_task2a_outputs(source["task2a_final_manifest"])
    finalization = _load_json(source["task2a_finalization"])
    if finalization.get("ready_for_task2b") is not True:
        raise ValueError("Task2A Final engineering integrity gate is not satisfied")
    task1 = _load_json(source["task1_manifest"])
    selected_input = read_validation_selected_input(source["task1_input_comparison"])
    if selected_input != task1.get("selected_input_type"):
        raise ValueError("Task1 manifest and validation-selected input type disagree")
    cnn_checkpoint = paths.project_root / "outputs" / "task1" / selected_input / "best_model.pth"
    if sha256_file(cnn_checkpoint) != task1["hashes"]["selected_checkpoint_sha256"]:
        raise ValueError("Task1 selected checkpoint hash mismatch")
    ae_paths = {scale: paths.project_root / "outputs" / "task2a_multiscale" / scale / "best_model.pth"
                for scale in ("small", "medium", "large")}
    prior_manifest = _load_json(source["task2a_multiscale_manifest"])
    ae_hashes = {scale: _verify_prior_output_hash(prior_manifest, checkpoint)
                 for scale, checkpoint in ae_paths.items()}
    source_data_hashes = {name: _verify_prior_output_hash(prior_manifest, source[name])
                          for name in ("expected", "large_background")}
    task2a_manifest = _load_json(source["task2a_final_manifest"])
    task2a_stats = task2a_manifest["statistics"]
    if dry_run:
        print(f"TASK2B DRY RUN: mode={mode}, selected_task1_input={selected_input}, "
              f"frozen_candidates={task2a_stats['n_candidates']}, output={paths.output_root}", flush=True)
        return Task2BResult(paths.output_root, None, None, True, False, time.monotonic() - started)
    if resume and paths.manifest_path.is_file():
        validate_task2a_outputs(paths.manifest_path)
        completion = _load_json(paths.output_root / "task2b_completion.json")
        if completion.get("task2b_complete") is not True:
            raise ValueError("Task2B resume manifest exists but completion is false")
        return Task2BResult(paths.output_root, paths.manifest_path,
                            paths.output_root / "task2b_summary.md", False, True,
                            time.monotonic() - started)

    candidate_all = pd.read_csv(source["task2a_final_candidates"], encoding="utf-8-sig")
    reference_all, full_known, full_background, background_audit = _reference_rows(paths.project_root)
    if mode == "smoke":
        candidates = _smoke_subset(candidate_all, 24, seed_offset=1)
        known = _smoke_subset(reference_all.loc[reference_all.reference_kind.eq("Known")], 48, seed_offset=2)
        background = _smoke_subset(reference_all.loc[reference_all.reference_kind.eq("Background")], 48, seed_offset=3)
        reference = pd.concat([known, background], ignore_index=True)
        n_splits = 3
    else:
        candidates, reference, n_splits = candidate_all, reference_all, 5
    n_known = int(reference.reference_kind.eq("Known").sum())
    n_background = int(reference.reference_kind.eq("Background").sum())
    if len(candidates) == 0 or n_known == 0 or n_background == 0:
        raise ValueError("Task2B requires nonempty candidate and two reference groups")
    combined = pd.concat([
        candidates[["chrom", "center"]], reference[["chrom", "center"]]
    ], ignore_index=True)
    bundle = load_model_bundle(cnn_checkpoint, ae_paths, selected_input)
    expected_file = np.load(source["expected"], allow_pickle=False)
    expected = {rep: expected_file[rep].copy() for rep in ("rep1", "rep2")}
    expected_file.close()
    protocol = _load_json(source["task2a_multiscale_protocol"])
    clips = protocol["oe_clip_values"]
    cool_paths = {rep: source[f"{rep}_cool"] for rep in ("rep1", "rep2")}
    extracted = extract_embedding_blocks(combined, bundle, cool_paths, expected, clips)
    n_candidate = len(candidates)
    averaged = average_replicate_blocks(extracted.rep1, extracted.rep2)
    reference_raw = {name: values[n_candidate:] for name, values in averaged.items()}
    candidate_raw = {name: values[:n_candidate] for name, values in averaged.items()}
    scalers, reference_blocks = fit_reference_scalers(reference_raw)
    candidate_blocks = {name: scalers[name].transform(candidate_raw[name]).astype(np.float32)
                        for name in BLOCK_ORDER}
    rep1_blocks = {name: scalers[name].transform(extracted.rep1[name][:n_candidate]).astype(np.float32)
                   for name in BLOCK_ORDER}
    rep2_blocks = {name: scalers[name].transform(extracted.rep2[name][:n_candidate]).astype(np.float32)
                   for name in BLOCK_ORDER}
    candidate_fused = fuse_blocks(candidate_blocks).astype(np.float32)
    reference_fused = fuse_blocks(reference_blocks).astype(np.float32)
    candidate_rep1 = fuse_blocks(rep1_blocks).astype(np.float32)
    candidate_rep2 = fuse_blocks(rep2_blocks).astype(np.float32)
    if not np.allclose(candidate_fused, (candidate_rep1 + candidate_rep2) / 2, atol=1e-5):
        raise ValueError("replicate and mean candidate embeddings disagree")

    y = reference.reference_kind.eq("Known").astype(int).to_numpy()
    representations = {
        "CNN-only": reference_blocks["cnn"],
        "AE-only": np.concatenate([reference_blocks[name] for name in ("small", "medium", "large")], axis=1),
        "Fused": reference_fused,
    }
    probes, fold_rows = evaluate_blocked_probes(representations, y, reference, n_splits=n_splits)
    for row in fold_rows.itertuples(index=False):
        if set(row.train_groups).intersection(row.test_groups):
            raise ValueError("probe genomic group leakage")
    if not fold_rows.cross_fold_interval_overlap_count.eq(0).all():
        raise ValueError("probe interval leakage")
    pca, pca_xy, umap_model, umap_xy = fit_visualization_models(reference_fused, candidate_fused)
    pca_coordinates = _coordinate_table(reference, candidates, pca_xy)
    umap_coordinates = _coordinate_table(reference, candidates, umap_xy)
    pca_diagnostics = map_diagnostics(pca_coordinates)
    umap_diagnostics = map_diagnostics(umap_coordinates)
    consistency = pd.DataFrame({"region_id": candidates.region_id})
    for name in BLOCK_ORDER:
        consistency[f"{name}_cosine"] = block_cosine_similarity(rep1_blocks[name], rep2_blocks[name])
        consistency[f"{name}_euclidean"] = np.linalg.norm(rep1_blocks[name] - rep2_blocks[name], axis=1)
    consistency["fused_cosine"] = block_cosine_similarity(candidate_rep1, candidate_rep2)
    consistency["fused_euclidean"] = np.linalg.norm(candidate_rep1 - candidate_rep2, axis=1)
    if consistency.drop(columns="region_id").isna().any(axis=None):
        raise ValueError("undefined candidate replicate similarity")
    metadata = candidates.copy().reset_index(drop=True)
    metadata["support_channels"] = metadata.apply(lambda row: ";".join(
        f"{scale}_{branch}" for scale, branch in
        ((s, b) for s in ("small", "medium", "large") for b in ("density", "shape"))
        if bool(row[f"from_{scale}_{branch}"])), axis=1)
    metadata["known_overlap"] = metadata.known_center_overlap
    for name in BLOCK_ORDER:
        metadata[f"{name}_replicate_cosine"] = consistency[f"{name}_cosine"].to_numpy()
    metadata["fused_replicate_cosine"] = consistency.fused_cosine.to_numpy()
    metadata["padded_any_replicate"] = (
        extracted.padded.loc[extracted.padded.row_index.lt(n_candidate)]
        .groupby("row_index").padded_any_scale.any().reindex(range(n_candidate), fill_value=False).to_numpy()
    )
    reference_metadata = reference.copy().reset_index(drop=True)
    reference_metadata["genomic_group"] = build_genomic_groups(reference_metadata)
    reference_metadata["probe_label"] = y

    out = paths.output_root
    out.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for name, array in (
        ("candidate_embeddings.npy", candidate_fused),
        ("candidate_embeddings_rep1.npy", candidate_rep1),
        ("candidate_embeddings_rep2.npy", candidate_rep2),
        ("reference_embeddings.npy", reference_fused),
    ):
        path = out / name
        np.save(path, array, allow_pickle=False)
        output_paths.append(path)
    output_paths.extend([
        write_csv_utf8(metadata, out / "candidate_embedding_metadata.csv"),
        write_csv_utf8(reference_metadata, out / "reference_embedding_metadata.csv"),
        write_csv_utf8(probes, out / "known_vs_background_linear_probe.csv"),
        write_csv_utf8(probes, out / "representation_ablation.csv"),
        write_csv_utf8(fold_rows.assign(
            train_groups=fold_rows.train_groups.map(lambda x: ";".join(x)),
            test_groups=fold_rows.test_groups.map(lambda x: ";".join(x)),
        ), out / "linear_probe_folds.csv"),
        write_csv_utf8(consistency, out / "candidate_embedding_replicate_consistency.csv"),
        write_csv_utf8(pca_coordinates, out / "pca_coordinates.csv"),
        write_csv_utf8(umap_coordinates, out / "umap_coordinates.csv"),
        write_csv_utf8(pd.DataFrame([
            {"component": i + 1, "explained_variance_ratio": float(value),
             "cumulative_explained_variance": float(pca.explained_variance_ratio_[:i+1].sum())}
            for i, value in enumerate(pca.explained_variance_ratio_)
        ]), out / "pca_explained_variance.csv"),
    ])
    scaler_path, pca_path, umap_path = (out / name for name in
                                        ("feature_scaler.pkl", "pca_model.pkl", "umap_model.pkl"))
    joblib.dump(scalers, scaler_path)
    joblib.dump(pca, pca_path)
    joblib.dump(umap_model, umap_path)
    output_paths.extend([scaler_path, pca_path, umap_path])
    output_paths.extend([
        plot_embedding_map(pca_coordinates, out / "pca_reference_candidates.png", method="PCA"),
        plot_pca_variance(pca.explained_variance_ratio_, out / "pca_explained_variance.png"),
        plot_embedding_map(umap_coordinates, out / "umap_reference_candidates.png", method="UMAP"),
    ])
    complete = bool(
        candidate_fused.shape == (len(candidates), sum(bundle.dimensions.values()))
        and reference_fused.shape == (len(reference), sum(bundle.dimensions.values()))
        and np.isfinite(candidate_fused).all() and np.isfinite(reference_fused).all()
        and len(metadata) == len(candidates) and len(reference_metadata) == len(reference)
        and len(probes) == 3 and len(pca_coordinates) == len(umap_coordinates) == len(combined)
        and all(not set(row.train_groups).intersection(row.test_groups) for row in fold_rows.itertuples())
        and fold_rows.cross_fold_interval_overlap_count.eq(0).all()
    )
    completion = {
        "mode": mode, "task2b_complete": complete,
        "candidate_embeddings_aligned": len(metadata) == len(candidate_fused),
        "reference_embeddings_aligned": len(reference_metadata) == len(reference_fused),
        "genomic_probe_group_leakage_zero": True,
        "genomic_probe_interval_leakage_zero": bool(fold_rows.cross_fold_interval_overlap_count.eq(0).all()),
        "known_labels_used_for_embedding_or_scaler_selection": False,
        "prior_task1_cnn_supervision_overlaps_known_reference": True,
        "independent_cnn_probe_generalization_claim": False,
        "task2c_executed": False,
    }
    output_paths.append(_write_json(out / "task2b_completion.json", completion))
    padded_count = int(extracted.padded.padded_any_scale.sum())
    summary_path = write_task2b_summary(
        out / "task2b_summary.md", mode=mode, task2a_stats=task2a_stats,
        dims=bundle.dimensions, n_candidates=len(candidates), n_known=n_known,
        n_background=n_background, probes=probes, pca_ratios=pca.explained_variance_ratio_,
        pca_diagnostics=pca_diagnostics, umap_diagnostics=umap_diagnostics,
        consistency=consistency, padded_count=padded_count, completion=completion,
        n_splits=n_splits, background_audit=background_audit,
    )
    output_paths.append(summary_path)
    import umap
    configuration = {
        "version": "Task2B-Hybrid-Shape-Representation-v1", "mode": mode,
        "seed": SEED, "task2a_final_protocol_sha256": sha256_file(source["task2a_final_protocol"]),
        "candidate_list_sha256": sha256_file(source["task2a_final_candidates"]),
        "task1_checkpoint_sha256": sha256_file(cnn_checkpoint),
        "task1_selected_input_type": selected_input,
        "ae_checkpoint_sha256": ae_hashes,
        "source_data_sha256": source_data_hashes,
        "cnn_feature_layer": "classifier[0:3]: Flatten -> Linear(128,hidden) -> ReLU; before Dropout and final logits",
        "ae_pooling": "AdaptiveAvgPool2d(1) on encoder output, then flatten",
        "feature_dimensions": bundle.dimensions,
        "fused_dimension": int(sum(bundle.dimensions.values())),
        "scaler_fitting": "unlabeled Known+strict Background reference only; separate StandardScaler per feature block",
        "background_reference": "344 strict known-disjoint large windows; deterministic >=4800 bp within-chrom center gap; seed 20260920; overlapping windows and AE-train reuse audited",
        "pca": {"n_components": 2, "svd_solver": "full", "fit": "reference+candidate standardized fused"},
        "umap": {"n_components": 2, "n_neighbors": min(15, len(combined)-1),
                 "min_dist": .1, "metric": "euclidean", "random_state": SEED,
                 "visualization_only": True},
        "linear_probe": {"model": "LogisticRegression", "class_weight": "balanced",
                         "max_iter": 2000, "n_splits": n_splits, "groups": "chrom:floor(center/128000)",
                         "splitter": "GroupKFold with training intervals crossing test boundaries purged", "threshold": .5},
        "software_versions": {"torch": torch.__version__, "sklearn": sklearn.__version__,
                              "umap_learn": umap.__version__},
        "prior_cnn_supervision_overlaps_known_reference": True,
        "task2c_executed": False,
    }
    input_paths = [*source.values(), cnn_checkpoint, *ae_paths.values(),
                   Path(__file__), Path(__file__).with_name("features.py"),
                   Path(__file__).with_name("evaluation.py"),
                   Path(__file__).with_name("reporting.py"),
                   paths.project_root / "src" / "task1" / "model.py"]
    write_manifest(paths.manifest_path, configuration=configuration,
                   input_paths=input_paths, output_paths=output_paths,
                   statistics={"n_candidates": len(candidates), "n_reference_known": n_known,
                               "n_reference_background": n_background,
                               "probe": probes.to_dict("records"),
                               "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                               "pca_map_diagnostics": pca_diagnostics,
                               "umap_map_diagnostics": umap_diagnostics,
                               "replicate_cosine_means": {
                                   name: float(consistency[f"{name}_cosine"].mean())
                                   for name in (*BLOCK_ORDER, "fused")},
                               "padded_extraction_records": padded_count,
                               "background_reference_audit": background_audit,
                               "probe_train_rows_purged_per_fold": fold_rows.loc[fold_rows.representation.eq("Fused"), "purged_train_count"].tolist(),
                               "completion": completion})
    validate_task2a_outputs(paths.manifest_path)
    if not complete:
        raise ValueError("Task2B completion checks failed")
    return Task2BResult(out, paths.manifest_path, summary_path, False, False,
                        time.monotonic() - started)

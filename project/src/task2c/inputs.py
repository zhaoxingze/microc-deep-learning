"""Read-only validated Task2A/Task2B artifact boundary for Task2C."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.task1.manifest import sha256_file
from src.task2.reporting import validate_task2a_outputs


@dataclass(frozen=True)
class Task2CPaths:
    root: Path
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("Task2C mode must be smoke or full")

    @property
    def output_root(self) -> Path:
        if self.mode == "smoke":
            return self.root / "outputs" / "pipeline_runs" / "task2c_smoke" / "outputs" / "task2c"
        return self.root / "outputs" / "task2c"

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "task2c_manifest.json"


@dataclass(frozen=True)
class Task2CInputs:
    candidates: np.ndarray
    known: np.ndarray
    background: np.ndarray
    candidate_rep1: np.ndarray
    candidate_rep2: np.ndarray
    candidate_meta: pd.DataFrame
    known_meta: pd.DataFrame
    background_meta: pd.DataFrame
    umap_coordinates: pd.DataFrame
    source_paths: dict[str, Path]
    source_hashes: dict[str, str]


def validate_alignment(candidate: np.ndarray, reference: np.ndarray,
                       candidate_meta: pd.DataFrame, reference_meta: pd.DataFrame) -> None:
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape[1] != reference.shape[1] or candidate.shape[1] == 0:
        raise ValueError("candidate/reference embedding dimension alignment failed")
    if len(candidate) != len(candidate_meta) or len(reference) != len(reference_meta):
        raise ValueError("embedding and metadata row alignment failed")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite Task2B embedding")
    if not {"region_id"}.issubset(candidate_meta) or not {"reference_id", "reference_kind"}.issubset(reference_meta):
        raise ValueError("Task2B metadata schema incomplete")
    if candidate_meta.region_id.isna().any() or candidate_meta.region_id.duplicated().any() or (
        reference_meta.reference_id.isna().any() or reference_meta.reference_id.duplicated().any()
    ):
        raise ValueError("duplicate or missing embedding sample ID")
    if not set(reference_meta.reference_kind).issubset({"Known", "Background"}):
        raise ValueError("unexpected reference source")
    if not reference_meta.reference_kind.eq("Known").any():
        raise ValueError("Known reference rows missing")


def freeze_protocol(path: Path, payload: dict) -> Path:
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != payload:
            raise ValueError("frozen Task2C protocol differs from current inputs or code")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _source_paths(root: Path) -> dict[str, Path]:
    task2a = root / "outputs" / "task2a_final"
    task2b = root / "outputs" / "task2b"
    return {
        "task2a_manifest": task2a / "task2a_final_manifest.json",
        "task2a_protocol": task2a / "frozen_proposal_protocol.json",
        "task2a_candidates": task2a / "task2a_final_candidates.csv",
        "task2b_manifest": task2b / "task2b_manifest.json",
        "task2b_completion": task2b / "task2b_completion.json",
        "candidate_embeddings": task2b / "candidate_embeddings.npy",
        "candidate_rep1": task2b / "candidate_embeddings_rep1.npy",
        "candidate_rep2": task2b / "candidate_embeddings_rep2.npy",
        "candidate_metadata": task2b / "candidate_embedding_metadata.csv",
        "reference_embeddings": task2b / "reference_embeddings.npy",
        "reference_metadata": task2b / "reference_embedding_metadata.csv",
        "feature_scaler": task2b / "feature_scaler.pkl",
        "umap_coordinates": task2b / "umap_coordinates.csv",
        "expected": root / "data" / "task2_multiscale" / "genome_expected_max128.npz",
        "multiscale_protocol": root / "outputs" / "task2a_multiscale" / "frozen_multiscale_protocol.json",
        "rep1_cool": root.parent / "micro-c数据" / "GSE272159_37C_rep1.mapq_30.10.cool",
        "rep2_cool": root.parent / "micro-c数据" / "GSE272159_37C_rep2.mapq_30.10.cool",
    }


def load_task2c_inputs(root: str | Path) -> Task2CInputs:
    root = Path(root).resolve()
    paths = _source_paths(root)
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Task2C source artifact missing: {missing}")
    validate_task2a_outputs(paths["task2a_manifest"])
    validate_task2a_outputs(paths["task2b_manifest"])
    task2b_complete = json.loads(paths["task2b_completion"].read_text(encoding="utf-8"))
    if task2b_complete.get("task2b_complete") is not True or task2b_complete.get("mode") != "full":
        raise ValueError("Task2C requires completed Full Task2B")
    manifest = json.loads(paths["task2b_manifest"].read_text(encoding="utf-8"))
    config = manifest["configuration"]
    if config["task2a_final_protocol_sha256"] != sha256_file(paths["task2a_protocol"]) or (
        config["candidate_list_sha256"] != sha256_file(paths["task2a_candidates"])
    ):
        raise ValueError("Task2A freeze hashes disagree with Task2B")
    candidates = np.load(paths["candidate_embeddings"], allow_pickle=False)
    rep1 = np.load(paths["candidate_rep1"], allow_pickle=False)
    rep2 = np.load(paths["candidate_rep2"], allow_pickle=False)
    reference = np.load(paths["reference_embeddings"], allow_pickle=False)
    candidate_meta = pd.read_csv(paths["candidate_metadata"], encoding="utf-8-sig")
    reference_meta = pd.read_csv(paths["reference_metadata"], encoding="utf-8-sig")
    validate_alignment(candidates, reference, candidate_meta, reference_meta)
    task2a_ids = pd.read_csv(paths["task2a_candidates"], usecols=["region_id"], encoding="utf-8-sig").region_id
    if not candidate_meta.region_id.reset_index(drop=True).equals(task2a_ids.reset_index(drop=True)):
        raise ValueError("Task2A candidate and Task2B metadata ID alignment failed")
    if rep1.shape != candidates.shape or rep2.shape != candidates.shape or not np.isfinite(rep1).all() or not np.isfinite(rep2).all() or (
        not np.allclose(candidates, (rep1 + rep2) / 2, atol=1e-5)
    ):
        raise ValueError("Task2B paired-average replicate alignment failed")
    if candidates.shape[1] != int(config["fused_dimension"]):
        raise ValueError("Task2B manifest fused dimension mismatch")
    known_mask = reference_meta.reference_kind.eq("Known").to_numpy()
    background_mask = reference_meta.reference_kind.eq("Background").to_numpy()
    if known_mask.sum() != 344 or background_mask.sum() != 344:
        raise ValueError("Full Task2B must contain 344 Known and 344 Background references")
    source_hashes = {key: sha256_file(path) for key, path in paths.items() if key not in {"rep1_cool", "rep2_cool"}}
    umap_coordinates = pd.read_csv(paths["umap_coordinates"], encoding="utf-8-sig")
    return Task2CInputs(
        candidates, reference[known_mask], reference[background_mask], rep1, rep2,
        candidate_meta.reset_index(drop=True), reference_meta.loc[known_mask].reset_index(drop=True),
        reference_meta.loc[background_mask].reset_index(drop=True), umap_coordinates,
        paths, source_hashes,
    )

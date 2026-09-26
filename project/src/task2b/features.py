"""Extract Task1 CNN and three frozen AE encoder representations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.load_microc import load_local_matrix
from src.data.normalize import compute_oe, log1p_normalize
from src.task1.model import SmallMicroCCNN
from src.task2.autoencoder import BackgroundAutoencoder
from src.task2.shape_preprocessing import preprocess_shape_window


BLOCK_ORDER = ("cnn", "small", "medium", "large")
AE_WINDOW_BP = {"small": 3200, "medium": 6400, "large": 12800}


@dataclass(frozen=True)
class ModelBundle:
    cnn: SmallMicroCCNN
    aes: dict[str, BackgroundAutoencoder]
    input_type: str
    dimensions: dict[str, int]
    device: torch.device


@dataclass(frozen=True)
class ExtractedBlocks:
    rep1: dict[str, np.ndarray]
    rep2: dict[str, np.ndarray]
    padded: pd.DataFrame


def cnn_preprocess(raw: np.ndarray, input_type: str) -> np.ndarray:
    if input_type == "raw":
        transformed = np.asarray(raw, dtype=np.float32).copy()
    elif input_type == "log1p":
        transformed = log1p_normalize(raw).astype(np.float32)
    elif input_type == "oe":
        transformed = compute_oe(raw).astype(np.float32)
    else:
        raise ValueError(f"unknown Task1 selected input type: {input_type}")
    if transformed.shape != (64, 64) or not np.isfinite(transformed).all():
        raise ValueError("Task1 CNN preprocessing requires finite 64x64 input")
    return transformed


def ae_encoder_embedding(model: BackgroundAutoencoder, inputs: torch.Tensor) -> torch.Tensor:
    encoded = model.encoder(inputs)
    return nn.functional.adaptive_avg_pool2d(encoded, 1).flatten(1)


def average_replicate_blocks(
    rep1: Mapping[str, np.ndarray], rep2: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if set(rep1) != set(BLOCK_ORDER) or set(rep2) != set(BLOCK_ORDER):
        raise ValueError("four aligned blocks required for each replicate")
    result = {}
    for name in BLOCK_ORDER:
        left, right = np.asarray(rep1[name]), np.asarray(rep2[name])
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError(f"replicate block shape mismatch: {name}")
        result[name] = (left + right) / 2
    return result


def fuse_blocks(blocks: Mapping[str, np.ndarray]) -> np.ndarray:
    if set(blocks) != set(BLOCK_ORDER):
        raise ValueError("fused representation needs exactly CNN and three AE blocks")
    arrays = [np.asarray(blocks[name]) for name in BLOCK_ORDER]
    if any(array.ndim != 2 or len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("embedding blocks must have aligned 2D rows")
    return np.concatenate(arrays, axis=1)


def block_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    one, two = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if one.shape != two.shape or one.ndim != 2:
        raise ValueError("cosine blocks must be aligned 2D arrays")
    norms = np.linalg.norm(one, axis=1) * np.linalg.norm(two, axis=1)
    dots = (one * two).sum(axis=1)
    result = np.full(len(one), np.nan)
    both_zero = (np.linalg.norm(one, axis=1) == 0) & (np.linalg.norm(two, axis=1) == 0)
    result[both_zero] = 1.0
    np.divide(dots, norms, out=result, where=norms > 0)
    return np.clip(result, -1.0, 1.0)


def load_model_bundle(
    cnn_checkpoint: str | Path,
    ae_checkpoints: Mapping[str, str | Path],
    input_type: str,
    *,
    device: torch.device | None = None,
) -> ModelBundle:
    if set(ae_checkpoints) != set(AE_WINDOW_BP):
        raise ValueError("three scale-specific AE checkpoints required")
    device = device or torch.device("cpu")
    cnn = SmallMicroCCNN().to(device)
    cnn_state = torch.load(Path(cnn_checkpoint), map_location=device, weights_only=True)
    cnn.load_state_dict(cnn_state["model_state_dict"])
    cnn.eval()
    aes = {}
    for scale in AE_WINDOW_BP:
        model = BackgroundAutoencoder().to(device)
        state = torch.load(Path(ae_checkpoints[scale]), map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        aes[scale] = model
    with torch.inference_mode():
        dimensions = {"cnn": int(cnn.forward_features(torch.zeros(1, 1, 64, 64, device=device)).shape[1])}
        for scale, bp in AE_WINDOW_BP.items():
            dimensions[scale] = int(ae_encoder_embedding(
                aes[scale], torch.zeros(1, 1, bp // 100, bp // 100, device=device)
            ).shape[1])
    return ModelBundle(cnn, aes, input_type, dimensions, device)


def extract_embedding_blocks(
    rows: pd.DataFrame,
    bundle: ModelBundle,
    cool_paths: Mapping[str, str | Path],
    expected: Mapping[str, np.ndarray],
    clips: Mapping[str, Mapping[str, float]],
    *,
    batch_size: int = 32,
) -> ExtractedBlocks:
    """Apply one identical matrix and feature path to candidate and reference rows."""
    if batch_size <= 0 or not {"chrom", "center"}.issubset(rows.columns):
        raise ValueError("rows need chrom/center and positive batch size")
    if set(cool_paths) != {"rep1", "rep2"} or set(expected) != {"rep1", "rep2"}:
        raise ValueError("both biological replicates are required")
    output: dict[str, dict[str, list[np.ndarray]]] = {
        rep: {block: [] for block in BLOCK_ORDER} for rep in ("rep1", "rep2")
    }
    padding = []
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    for offset in range(0, len(rows), batch_size):
        group = rows.iloc[offset:offset + batch_size]
        for rep in ("rep1", "rep2"):
            inputs: dict[str, list[np.ndarray]] = {block: [] for block in BLOCK_ORDER}
            for index, row in group.iterrows():
                padded_any = False
                for scale, bp in AE_WINDOW_BP.items():
                    raw, metadata = load_local_matrix(
                        cool_paths[rep], str(row.chrom), float(row.center),
                        window_bp=bp, target_bin_size=100, balance=False,
                    )
                    padded_any |= bool(metadata["padded"])
                    if scale == "medium":
                        inputs["cnn"].append(cnn_preprocess(raw, bundle.input_type))
                    transformed = preprocess_shape_window(
                        raw, np.asarray(expected[rep])[:bp // 100], float(clips[scale][rep])
                    )
                    inputs[scale].append(transformed)
                padding.append({"row_index": int(index), "replicate": rep, "padded_any_scale": padded_any})
            with torch.inference_mode():
                for block in BLOCK_ORDER:
                    tensor = torch.from_numpy(np.stack(inputs[block])[:, None]).to(
                        device=bundle.device, dtype=torch.float32
                    )
                    feature = (
                        bundle.cnn.forward_features(tensor) if block == "cnn"
                        else ae_encoder_embedding(bundle.aes[block], tensor)
                    )
                    output[rep][block].append(feature.cpu().numpy().astype(np.float32))
    return ExtractedBlocks(
        rep1={block: np.concatenate(output["rep1"][block], axis=0) for block in BLOCK_ORDER},
        rep2={block: np.concatenate(output["rep2"][block], axis=0) for block in BLOCK_ORDER},
        padded=pd.DataFrame(padding),
    )

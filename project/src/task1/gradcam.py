"""Grad-CAM and quantitative attention summaries for Task 1 CNNs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def compute(self, inputs: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        logits = self.model(inputs)
        class_id = int(logits.argmax(dim=1)[0]) if target_class is None else int(target_class)
        logits[0, class_id].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy().astype(np.float64)
        minimum, maximum = float(cam.min()), float(cam.max())
        if maximum > minimum:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = np.zeros_like(cam)
        return cam

    def close(self) -> None:
        self._handle.remove()


def diagonal_attention_ratio(cam: np.ndarray, band: int = 2) -> float:
    heatmap = np.asarray(cam, dtype=np.float64)
    if heatmap.ndim != 2:
        raise ValueError("CAM must be two-dimensional")
    rows, columns = np.indices(heatmap.shape)
    mask = np.abs(rows - columns) <= int(band)
    total = float(heatmap.sum())
    return float(heatmap[mask].sum() / total) if total > 0 else 0.0


def zero_axis_attention_ratio(cam: np.ndarray, raw_matrix: np.ndarray) -> float:
    heatmap = np.asarray(cam, dtype=np.float64)
    raw = np.asarray(raw_matrix, dtype=np.float64)
    if heatmap.shape != raw.shape or heatmap.ndim != 2:
        raise ValueError("CAM and raw matrix must be aligned 2D arrays")
    zero_rows = np.all(raw == 0, axis=1)
    zero_columns = np.all(raw == 0, axis=0)
    mask = zero_rows[:, None] | zero_columns[None, :]
    total = float(heatmap.sum())
    return float(heatmap[mask].sum() / total) if total > 0 else 0.0


def attention_intensity_correlations(cam: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    x = np.asarray(cam, dtype=np.float64).reshape(-1)
    y = np.asarray(matrix, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)

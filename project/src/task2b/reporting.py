"""Task2B plots and evidence-bounded summary."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_embedding_map(coordinates: pd.DataFrame, path: Path, *, method: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    candidate = coordinates.loc[coordinates.point_kind.eq("Candidate")]
    ax.scatter(candidate.x, candidate.y, s=9, alpha=.18, color="#555555", label="Candidate", rasterized=True)
    colors = {"Background": "#777777", "CHIN": "#2166AC", "OPCID": "#B2182B", "CHID": "#4D9221"}
    markers = {"Background": "x", "CHIN": "o", "OPCID": "^", "CHID": "s"}
    for label in ("Background", "CHIN", "OPCID", "CHID"):
        subset = coordinates.loc[coordinates.type.eq(label) & coordinates.point_kind.eq("Reference")]
        if not subset.empty:
            ax.scatter(subset.x, subset.y, s=22, alpha=.78, marker=markers[label],
                       color=colors[label], label=f"{label} (n={len(subset)})", rasterized=True)
    ax.set(xlabel=f"{method} 1", ylabel=f"{method} 2", title=f"Task2B {method}: frozen fused embeddings")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pca_variance(ratios: np.ndarray, path: Path) -> Path:
    values = np.asarray(ratios, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.bar(["PC1", "PC2"], values, color=["#2166AC", "#B2182B"])
    ax.set(ylabel="Explained variance ratio", ylim=(0, max(.05, float(values.max()) * 1.15)),
           title=f"First two PCs explain {values.sum():.1%}")
    for x, value in enumerate(values):
        ax.text(x, value, f"{value:.1%}", ha="center", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def map_diagnostics(coordinates: pd.DataFrame) -> dict[str, float | int]:
    known = coordinates.loc[coordinates.point_kind.eq("Reference") & ~coordinates.type.eq("Background")]
    background = coordinates.loc[coordinates.type.eq("Background")]
    candidates = coordinates.loc[coordinates.point_kind.eq("Candidate")]
    center_known = known[["x", "y"]].to_numpy().mean(axis=0)
    center_background = background[["x", "y"]].to_numpy().mean(axis=0)
    known_spread = np.linalg.norm(known[["x", "y"]].to_numpy() - center_known, axis=1).mean()
    background_spread = np.linalg.norm(background[["x", "y"]].to_numpy() - center_background, axis=1).mean()
    separation = np.linalg.norm(center_known - center_background) / max((known_spread + background_spread) / 2, 1e-12)
    origin = coordinates[["x", "y"]].to_numpy().mean(axis=0)
    relative = candidates[["x", "y"]].to_numpy() - origin
    quadrants = set(zip(relative[:, 0] >= 0, relative[:, 1] >= 0))
    return {"reference_centroid_separation_over_spread": float(separation),
            "candidate_occupied_quadrants": len(quadrants),
            "candidate_count": len(candidates)}


def write_task2b_summary(
    path: Path, *, mode: str, task2a_stats: dict, dims: dict[str, int],
    n_candidates: int, n_known: int, n_background: int,
    probes: pd.DataFrame, pca_ratios: np.ndarray,
    pca_diagnostics: dict, umap_diagnostics: dict,
    consistency: pd.DataFrame, padded_count: int, completion: dict,
    n_splits: int, background_audit: dict,
) -> Path:
    lookup = probes.set_index("representation")
    metric_lines = []
    for name in ("CNN-only", "AE-only", "Fused"):
        row = lookup.loc[name]
        metric_lines.append(
            f"- {name}: Accuracy {row.accuracy_mean:.4f} ± {row.accuracy_std:.4f}; "
            f"Balanced Accuracy {row.balanced_accuracy_mean:.4f} ± {row.balanced_accuracy_std:.4f}; "
            f"ROC-AUC {row.roc_auc_mean:.4f} ± {row.roc_auc_std:.4f}; "
            f"PR-AUC {row.pr_auc_mean:.4f} ± {row.pr_auc_std:.4f}"
        )
    cosines = {name: float(consistency[f"{name}_cosine"].mean())
               for name in ("cnn", "small", "medium", "large", "fused")}
    recall = {row["type"]: row["center_recall"] for row in task2a_stats["recall"]}
    caveat = ("SMOKE: reduced sample sizes validate engineering only; all metrics and figures are non-scientific."
              if mode == "smoke" else "FULL: descriptive representation analysis; no novelty claim.")
    lines = [
        "# Task2B Structure Representation", "", f"> {caveat}", "",
        "## Task2A final proposal context", "",
        "1. Task2A is frozen as an independent six-channel high-recall proposal generator because repeated tuning on the same 344 known structures risks validation overfitting. It proposes regions; it does not determine novelty.",
        f"2. Final candidate regions: `{task2a_stats['n_candidates']}` (this {mode} run embeds `{n_candidates}`).",
        f"3. Genomic union coverage: `{task2a_stats['coverage_bp']}` bp, fraction `{task2a_stats['coverage_fraction']:.6f}`.",
        f"4. Final center-based Known Recall: Overall `{recall['overall']:.6f}`, CHIN `{recall['CHIN']:.6f}`, OPCID `{recall['OPCID']:.6f}`, CHID `{recall['CHID']:.6f}`.",
        "5. Recall is measured on known structures already used repeatedly during development; it does not establish novel-structure precision or full-genome discovery sensitivity.",
        "6. The threshold and six-channel merge were frozen before this last evaluation. Known labels will no longer tune Task2A.", "",
        "## Representation and references", "",
        "7. Fused = standardized Task1 CNN hidden feature + standardized Small/Medium/Large AE encoder pooled features.",
        f"8–12. Dimensions: CNN `{dims['cnn']}`, Small AE `{dims['small']}`, Medium AE `{dims['medium']}`, Large AE `{dims['large']}`, Fused `{sum(dims.values())}`.",
        f"Reference Known `{n_known}`; strict blocked Background `{n_background}`. Candidate/reference matrices use one identical extraction and preprocessing function.",
        f"Background selection audit (full reference): strict eligible pool `{background_audit['strict_pool_count']}`; maximum mutually nonoverlapping 12.8 kb windows `{background_audit['strict_pool_max_mutually_nonoverlapping_count']}`; selected minimum center gap `{background_audit['selected_min_center_gap_bp']}` bp; adjacent overlapping pairs `{background_audit['selected_adjacent_interval_overlap_pairs']}`; previously used in AE training `{background_audit['selected_reused_ae_train_count']}`. These correlations are limitations, not independent background observations.",
        "Scalers were fit on the unlabeled Known+Background reference features only; candidates were transformed afterward.", "",
        "## Genomic GroupKFold known-vs-background probe", "",
        f"13–19. {n_splits}-fold mean ± std (random Balanced Accuracy reference ≈ 0.5):", *metric_lines,
        "GroupKFold uses 128 kb genomic center blocks, then purges training intervals crossing each test fold boundary; both group and interval overlap audits are zero. The probe is diagnostic, not a detector or a feature-selection loop. The frozen Task1 CNN was trained earlier using these same known structures, so CNN-only and Fused probe results are not independent generalization estimates. AE-only is less affected by this prior supervision but its background references partly reuse AE training windows.", "",
        "## PCA and UMAP", "",
        f"20. PCA explained variance: PC1 `{pca_ratios[0]:.6f}`, PC2 `{pca_ratios[1]:.6f}`, cumulative `{pca_ratios.sum():.6f}`.",
        f"21. PCA map: known/background centroid separation relative to within-group spread `{pca_diagnostics['reference_centroid_separation_over_spread']:.3f}`; candidates occupy `{pca_diagnostics['candidate_occupied_quadrants']}` of four quadrants.",
        f"22. UMAP map: known/background centroid separation relative to spread `{umap_diagnostics['reference_centroid_separation_over_spread']:.3f}`; candidates occupy `{umap_diagnostics['candidate_occupied_quadrants']}` of four quadrants. UMAP is visualization only.",
        f"23. Candidate embeddings span `{umap_diagnostics['candidate_occupied_quadrants']}` UMAP quadrants; this is descriptive, not a cluster or novelty decision.", "",
        "## Replicate stability and limitations", "",
        f"24. Mean rep1/rep2 cosine: CNN `{cosines['cnn']:.6f}`, AE Small `{cosines['small']:.6f}`, Medium `{cosines['medium']:.6f}`, Large `{cosines['large']:.6f}`, Fused `{cosines['fused']:.6f}`.",
        f"Padded matrix extraction records: `{padded_count}` across candidate and reference replicate windows.",
        "25. Main limits: prior supervised CNN exposure to known references, partial AE-training reuse among background references, correlated overlapping background windows, class imbalance/limited CHID count, coordinate-block CV on one chromosome, padded edge windows, and two-replicate data. Neither PCA nor UMAP validates novel structures.",
        f"26. Technical conditions for Task2C: `{'met' if completion['task2b_complete'] else 'not met'}`; `task2b_complete={str(completion['task2b_complete']).lower()}`. This is not a scientific claim of new structure discovery.",
        "", "No HDBSCAN, KMeans, DBSCAN, cluster labeling, or Task2C operation was executed.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path

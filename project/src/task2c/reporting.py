"""Task2C fixed-space plots, uniform O/E overviews, and scientific summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load_microc import load_local_matrix
from src.task2.expected import apply_expected
from src.task2c.analysis import AnalysisResult
from src.task2c.clustering import PrimaryFit


def paired_oe_matrix(raw1: np.ndarray, raw2: np.ndarray,
                     expected1: np.ndarray, expected2: np.ndarray,
                     clip1: float, clip2: float) -> np.ndarray:
    """Mean replicate O/E with frozen per-replicate clips, then log1p."""
    if not np.isfinite([clip1, clip2]).all() or min(clip1, clip2) <= 0:
        raise ValueError("positive finite O/E clips required")
    oe1 = np.clip(np.nan_to_num(apply_expected(raw1, expected1), nan=0., posinf=clip1), 0., clip1)
    oe2 = np.clip(np.nan_to_num(apply_expected(raw2, expected2), nan=0., posinf=clip2), 0., clip2)
    if oe1.shape != oe2.shape:
        raise ValueError("replicate heatmap shape mismatch")
    return np.log1p((oe1 + oe2) / 2).astype(np.float32)


def _two_dimensions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not 1 <= array.shape[1]:
        raise ValueError("plot coordinates need at least one dimension")
    return array[:, :2] if array.shape[1] >= 2 else np.column_stack([array[:, 0], np.zeros(len(array))])


def _cluster_color(cluster_id: int):
    return "#777777" if cluster_id == -1 else plt.get_cmap("tab20")(cluster_id % 20)


def _plot_labels(coordinates: np.ndarray, members: pd.DataFrame, path: Path,
                 *, space: str, by_type: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    coords = _two_dimensions(coordinates)
    if by_type:
        palette = {"Candidate": "#888888", "CHIN": "#2166AC", "OPCID": "#B2182B", "CHID": "#4D9221"}
        labels = np.where(members.sample_source.eq("candidate"), "Candidate", members.known_type)
        for label in ("Candidate", "CHIN", "OPCID", "CHID"):
            indices = np.flatnonzero(labels == label)
            if len(indices):
                ax.scatter(coords[indices, 0], coords[indices, 1], s=15 if label == "Candidate" else 25,
                           alpha=.35 if label == "Candidate" else .8,
                           marker="o" if label == "Candidate" else "^", color=palette[label],
                           label=f"{label} (n={len(indices)})", rasterized=True)
    else:
        for cluster_id in sorted(members.cluster_id.unique()):
            for source, marker in (("candidate", "o"), ("known", "^")):
                indices = np.flatnonzero(members.cluster_id.eq(cluster_id) & members.sample_source.eq(source))
                if len(indices):
                    ax.scatter(coords[indices, 0], coords[indices, 1], s=20, alpha=.7,
                               marker=marker, color=_cluster_color(int(cluster_id)),
                               label=f"{cluster_id}: {source} ({len(indices)})", rasterized=True)
    ax.set(xlabel=f"{space} 1", ylabel=f"{space} 2",
           title=f"Task2C {space}: {'Known composition' if by_type else 'Primary HDBSCAN labels'}")
    ax.grid(alpha=.2)
    if len(ax.get_legend_handles_labels()[0]) <= 24:
        ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_cluster_figures(primary: PrimaryFit, analysis: AnalysisResult,
                          umap_coordinates: pd.DataFrame, output_dir: Path) -> list[Path]:
    members = analysis.membership
    if not {"point_id", "x", "y"}.issubset(umap_coordinates) or umap_coordinates.point_id.duplicated().any():
        raise ValueError("Task2B UMAP coordinates are missing or duplicated")
    aligned = umap_coordinates.set_index("point_id").reindex(members.sample_id)
    if aligned[["x", "y"]].isna().any(axis=None):
        raise ValueError("Task2B UMAP coordinates do not align with Task2C members")
    umap_xy = aligned[["x", "y"]].to_numpy(dtype=float)
    return [
        _plot_labels(primary.projected, members, output_dir / "pca_clusters.png", space="Clustering PCA", by_type=False),
        _plot_labels(primary.projected, members, output_dir / "pca_known_composition.png", space="Clustering PCA", by_type=True),
        _plot_labels(umap_xy, members, output_dir / "umap_clusters.png", space="Task2B UMAP", by_type=False),
        _plot_labels(umap_xy, members, output_dir / "umap_known_composition.png", space="Task2B UMAP", by_type=True),
    ]


def plot_cluster_overview(images: list[np.ndarray], labels: list[str],
                          pca_positions: np.ndarray, umap_positions: np.ndarray,
                          path: Path, *, title: str, vmax: float,
                          coordinate_text: str) -> Path:
    if not images or len(images) != len(labels) or len(images) > 9:
        raise ValueError("overview needs one to nine aligned heatmaps")
    if len(pca_positions) != len(images) or len(umap_positions) != len(images):
        raise ValueError("overview coordinate alignment failed")
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)
    for index, axis in enumerate(axes.flat[:9]):
        if index < len(images):
            image = axis.imshow(images[index], cmap="magma", vmin=0, vmax=vmax, origin="lower")
            axis.set_title(labels[index], fontsize=8)
            axis.set_xticks([]); axis.set_yticks([])
        else:
            axis.axis("off")
    for axis, values, name in ((axes[3, 0], pca_positions, "PCA local positions"),
                               (axes[3, 1], umap_positions, "UMAP local positions")):
        points = _two_dimensions(values)
        axis.scatter(points[:, 0], points[:, 1], color="#2166AC", s=24)
        axis.scatter(points[0, 0], points[0, 1], color="#B2182B", marker="*", s=130,
                     label="medoid")
        axis.set_title(name, fontsize=9)
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=7)
    axes[3, 2].axis("off")
    axes[3, 2].text(0, .9, coordinate_text, ha="left", va="top", fontsize=7, wrap=True)
    fig.suptitle(title, fontsize=14)
    fig.colorbar(image, ax=axes[:3, :].ravel().tolist(), orientation="horizontal",
                 fraction=.02, pad=.01, label="log1p(paired O/E)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_cluster_heatmaps(primary: PrimaryFit, analysis: AnalysisResult,
                           umap_coordinates: pd.DataFrame,
                           source_paths: dict[str, Path], output_dir: Path) -> list[Path]:
    """Use only local canonical 6.4 kb reads for display, never for fitting."""
    protocol = json.loads(source_paths["multiscale_protocol"].read_text(encoding="utf-8"))
    clips = protocol["oe_clip_values"]["medium"]
    with np.load(source_paths["expected"], allow_pickle=False) as expected_file:
        expected = {rep: expected_file[rep][:64].copy() for rep in ("rep1", "rep2")}
    aligned_umap = umap_coordinates.set_index("point_id").reindex(analysis.membership.sample_id)
    if aligned_umap[["x", "y"]].isna().any(axis=None):
        raise ValueError("UMAP heatmap context is misaligned")
    umap_xy = aligned_umap[["x", "y"]].to_numpy(dtype=float)
    paths = []
    for row in analysis.summary.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        indexes = np.flatnonzero(primary.labels == cluster_id)
        centroid = analysis.centroids[int(row.centroid_index)]
        ordered = indexes[np.argsort(np.linalg.norm(primary.fused[indexes] - centroid, axis=1), kind="stable")]
        medoid = int(np.flatnonzero(analysis.membership.sample_id.eq(row.representative_sample_id))[0])
        chosen = [medoid, *[int(index) for index in ordered if int(index) != medoid][:8]]
        images = []
        labels = []
        coordinates = []
        for index in chosen:
            member = analysis.membership.iloc[index]
            matrices = {}
            display_meta = None
            for rep in ("rep1", "rep2"):
                matrices[rep], local_meta = load_local_matrix(
                    source_paths[f"{rep}_cool"], str(member.chrom), float(member.center),
                    window_bp=6400, target_bin_size=100, balance=False,
                )
                if display_meta is None:
                    display_meta = local_meta
            images.append(paired_oe_matrix(matrices["rep1"], matrices["rep2"],
                                           expected["rep1"], expected["rep2"],
                                           float(clips["rep1"]), float(clips["rep2"])))
            labels.append(f"{'Medoid ' if index == medoid else ''}{member.sample_source}: {member.sample_id}\n"
                          f"annotation {member.chrom}:{int(member.start)}-{int(member.end)}")
            coordinates.append(f"{member.sample_id}: {member.chrom}:"
                               f"{display_meta['requested_start']:.0f}-{display_meta['requested_end']:.0f}")
        pca_positions = _two_dimensions(primary.projected[chosen])
        umap_positions = umap_xy[chosen]
        vmax = float(np.log1p(max(clips.values())))
        paths.append(plot_cluster_overview(
            images, labels, pca_positions, umap_positions,
            output_dir / "cluster_overviews" / f"cluster_{cluster_id:03d}_overview.png",
            title=f"Cluster {cluster_id} overview (paired O/E)", vmax=vmax,
            coordinate_text="Displayed 6.4 kb windows (100 bp/bin);\n"
                            "panel titles = annotation intervals.\n"
                            "Requested display bounds:\n" + "\n".join(coordinates),
        ))
        if bool(row.novel_like_cluster_candidate):
            paths.append(plot_cluster_overview(
                images, labels, pca_positions, umap_positions,
                output_dir / "novel_like_clusters" / f"cluster_{cluster_id:03d}_overview.png",
                title=f"Novel-like candidate cluster {cluster_id} (unvalidated)", vmax=vmax,
                coordinate_text="Displayed 6.4 kb windows (100 bp/bin);\n"
                                "panel titles = annotation intervals.\n"
                                "Requested display bounds:\n" + "\n".join(coordinates),
            ))
    return paths


def write_task2c_summary(path: Path, *, mode: str, primary: PrimaryFit,
                         analysis: AnalysisResult, sensitivity: pd.DataFrame,
                         silhouette: float | None, completion: dict) -> Path:
    """Write a descriptive report without treating clusters as validated novelty."""
    summary = analysis.summary
    noise = int((primary.labels == -1).sum())
    known_like = {name: summary.loc[summary.cluster_annotation.eq(f"{name}-like"), "cluster_id"].tolist()
                  for name in ("CHIN", "OPCID", "CHID")}
    mixed = summary.loc[summary.cluster_annotation.eq("mixed-known"), "cluster_id"].tolist()
    novel = summary.loc[summary.novel_like_cluster_candidate] if len(summary) else summary
    nearest = analysis.clusters.nearest_known_distance
    prob = analysis.membership.membership_probability
    cluster_counts = summary[["cluster_id", "candidate_count", "CHIN_count", "OPCID_count", "CHID_count",
                              "known_count", "cluster_annotation", "novel_like_cluster_candidate",
                              "Task2D_priority"]] if len(summary) else summary
    novel_details = novel[["cluster_id", "candidate_count", "candidate_representative_region_id",
                           "mean_membership_probability", "candidate_known_overlap_count"]] if len(novel) else novel
    overlap_count = int((analysis.clusters.known_center_overlap.astype(bool) |
                         analysis.clusters.known_interval_overlap.astype(bool)).sum())
    candidate_noise = analysis.clusters.loc[analysis.clusters.cluster_id.eq(-1), "region_id"].tolist()
    full_observations = []
    if mode == "full" and len(summary):
        dominant = summary.sort_values("total_members").iloc[-1]
        full_observations.append(
            f"- The largest cluster is {int(dominant.cluster_id)} with {int(dominant.candidate_count)} "
            f"candidates and {int(dominant.known_count)} Known members; "
            f"{int((summary.candidate_count.eq(0) & summary.known_count.gt(0)).sum())} "
            "clusters are Known-only. PCA/UMAP 2D views are descriptive and do not establish candidate novelty."
        )
        inventory = summary.set_index("cluster_id")
        if (primary.n_candidate == 231 and primary.n_known == 344 and
            {0, 1, 2}.issubset(inventory.index) and
            int(inventory.loc[0, "known_count"]) == 5 and
            int(inventory.loc[1, "candidate_count"]) == 189 and
            int(inventory.loc[2, "known_count"]) == 10):
            full_observations.append(
                "- Visual review of this frozen run: cluster 2's overview has pronounced dark row/column bands "
                "in most displayed Known windows; cluster 0's five Known members have heavily overlapping "
                "annotation intervals within MG1655 2.101–2.111 Mb. These are descriptive cautions only; "
                "the cause of the bands is unknown and the members are not five independent loci."
            )
    lines = [
        "# Task2C Unsupervised Grouping / Candidate Clustering", "",
        f"> {'SMOKE: engineering only; no scientific cluster conclusions.' if mode == 'smoke' else 'FULL: frozen descriptive clustering; no confirmed novel structures.'}", "",
        "## Frozen primary protocol", "",
        f"- Input: `{primary.n_candidate}` Task2B paired-average candidate fused vectors + `{primary.n_known}` unique paired-average Known reference vectors, each `{primary.fused.shape[1]}`D. Background is excluded from PCA/HDBSCAN fit.",
        "- Existing Task2B feature scaling is retained; proposal scores, Known type labels, genomic-overlap fields, and individual rep1/rep2 rows are excluded from fit.",
        "- Fit full-SVD PCA without labels and keep the smallest component count reaching at least 95% cumulative variance. UMAP 2D is not used for clustering because its nonlinear projection can distort distance and density.",
        f"- PCA components `{primary.projected.shape[1]}`, cumulative explained variance `{primary.pca.explained_variance_ratio_.sum():.6f}`.",
        "- Primary sklearn HDBSCAN: min_cluster_size=5, min_samples=5, Euclidean, eom, epsilon=0.0, allow_single_cluster=False. No parameter was chosen from Known composition, silhouette, or plot appearance.", "",
        "## Primary cluster inventory", "",
        f"- Non-noise clusters: `{len(summary)}`. Noise: `{noise}` / `{len(primary.labels)}` = `{noise/len(primary.labels):.6f}`.",
        f"- Membership probability: mean `{prob.mean():.6f}`, median `{prob.median():.6f}`, 10th/90th percentiles `{prob.quantile(.1):.6f}` / `{prob.quantile(.9):.6f}`.",
        f"- Silhouette in clustering PCA space on non-noise samples: `{silhouette if silhouette is not None else 'not computable (<2 clusters or insufficient members)'}`. Descriptive only; not optimized.",
        "- Per-cluster candidate and Known-type counts:", "", cluster_counts.to_markdown(index=False) if len(cluster_counts) else "No non-noise clusters.", "",
        f"- CHIN-like clusters `{known_like['CHIN']}`; OPCID-like `{known_like['OPCID']}`; CHID-like `{known_like['CHID']}`; mixed-known `{mixed}`. Type names are post-fit composition annotations, not supervised predictions.",
        f"- Clusters with zero Known members: `{int(summary.known_count.eq(0).sum()) if len(summary) else 0}`; strictly novel-like candidate clusters: `{len(novel)}`.",
        "- Novel-like entries (all have >=5 candidates, no Known member, and zero candidate center/interval Known overlap):", "",
        novel_details.to_markdown(index=False) if len(novel_details) else "None.", "",
        f"- Candidate genomic Known-overlap audit: `{overlap_count}` of `{len(analysis.clusters)}` candidates overlap a Known center and/or interval. Every novel-like cluster has zero by construction.",
        f"- Candidate HDBSCAN noise IDs (`{len(candidate_noise)}`): `{'; '.join(candidate_noise) if candidate_noise else 'none'}`. Noise remains in `clusters.csv` and is not declared false.", "",
        "## Diagnostics, maps, and stability", "",
        f"- Nearest Known distance in clustering PCA space: median `{nearest.median():.6f}`, 10th/90th percentiles `{nearest.quantile(.1):.6f}` / `{nearest.quantile(.9):.6f}`. Distances have no calibrated novelty cutoff.",
        f"- Background reference proximity is projected only after fit; `{len(analysis.background_proximity)}` Background points are assigned a nearest cluster centroid for diagnostics and never filtered by distance.",
        "- `pca_clusters.png` uses the formal clustering PCA's first two components; `umap_clusters.png` overlays the same primary labels on Task2B's existing visualization-only UMAP. The 2D displays may overlap or separate visually but never determine membership.",
        "- Each non-noise cluster has a 256D centroid and true medoid, with uniform 6.4 kb paired O/E heatmaps. Novel-like figures are explicitly labeled unvalidated.",
        *full_observations,
        "- Fixed sensitivity audit (8/5 and 5/8), not a model selection search:", "",
        sensitivity.to_markdown(index=False), "",
        "## Task2D handoff and limits", "",
        f"- High-priority Task2D clusters: `{novel.cluster_id.tolist() if len(novel) else []}`. Medium/Low are descriptive ambiguity/known-composition hints, not a final ranking score.",
        "- Main limits: the Task1 CNN was previously supervised on this Known set; the Known set was reused during Task2A method development; 234/344 Task2B Background references reuse AE-training windows and many overlap; one chromosome, two replicates, representation/parameter dependence, and no independent biological validation. Cluster membership and 2D geometry alone cannot establish a new chromatin structure.",
        "- Task2D must independently validate candidate structure patterns in rep1/rep2 O/E matrices with a predeclared reproducibility and background comparison protocol, including zero-axis and known-overlap review. No Task2D correlation gate or final novel-candidate decision was run here.",
        f"- `task2c_complete={str(completion['task2c_complete']).lower()}`; `ready_for_task2d={str(completion['ready_for_task2d']).lower()}` means technically ready for validation, not confirmed novelty.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path

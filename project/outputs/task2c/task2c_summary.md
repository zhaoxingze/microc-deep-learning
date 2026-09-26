# Task2C Unsupervised Grouping / Candidate Clustering

> FULL: frozen descriptive clustering; no confirmed novel structures.

## Frozen primary protocol

- Input: `231` Task2B paired-average candidate fused vectors + `344` unique paired-average Known reference vectors, each `256`D. Background is excluded from PCA/HDBSCAN fit.
- Existing Task2B feature scaling is retained; proposal scores, Known type labels, genomic-overlap fields, and individual rep1/rep2 rows are excluded from fit.
- Fit full-SVD PCA without labels and keep the smallest component count reaching at least 95% cumulative variance. UMAP 2D is not used for clustering because its nonlinear projection can distort distance and density.
- PCA components `13`, cumulative explained variance `0.954072`.
- Primary sklearn HDBSCAN: min_cluster_size=5, min_samples=5, Euclidean, eom, epsilon=0.0, allow_single_cluster=False. No parameter was chosen from Known composition, silhouette, or plot appearance.

## Primary cluster inventory

- Non-noise clusters: `3`. Noise: `97` / `575` = `0.168696`.
- Membership probability: mean `0.807563`, median `1.000000`, 10th/90th percentiles `0.000000` / `1.000000`.
- Silhouette in clustering PCA space on non-noise samples: `0.2560571721516928`. Descriptive only; not optimized.
- Per-cluster candidate and Known-type counts:

|   cluster_id |   candidate_count |   CHIN_count |   OPCID_count |   CHID_count |   known_count | cluster_annotation   | novel_like_cluster_candidate   | Task2D_priority   |
|-------------:|------------------:|-------------:|--------------:|-------------:|--------------:|:---------------------|:-------------------------------|:------------------|
|            0 |                 0 |            4 |             0 |            1 |             5 | CHIN-like            | False                          | Low               |
|            1 |               189 |          203 |            54 |           17 |           274 | CHIN-like            | False                          | Low               |
|            2 |                 0 |            7 |             0 |            3 |            10 | CHIN-like            | False                          | Low               |

- CHIN-like clusters `[0, 1, 2]`; OPCID-like `[]`; CHID-like `[]`; mixed-known `[]`. Type names are post-fit composition annotations, not supervised predictions.
- Clusters with zero Known members: `0`; strictly novel-like candidate clusters: `0`.
- Novel-like entries (all have >=5 candidates, no Known member, and zero candidate center/interval Known overlap):

None.

- Candidate genomic Known-overlap audit: `119` of `231` candidates overlap a Known center and/or interval. Every novel-like cluster has zero by construction.
- Candidate HDBSCAN noise IDs (`42`): `FREG_a30994c0bfee6cb5; FREG_647ab946884b4026; FREG_8814232e60629894; FREG_4b056190f9431017; FREG_3d4a95743b89a9c1; FREG_b281c30546d2f2dd; FREG_f5ef0631b11a8bec; FREG_a4bcec0e05a4ae7e; FREG_927d40c86ef7b1d4; FREG_bf274765f64dcf2e; FREG_42d3e2f15815267c; FREG_c8486e55b019f350; FREG_f58b946f4319d3b7; FREG_82a811abb2714642; FREG_7e16c0c4325e5d1d; FREG_d2d392e1e3af249a; FREG_7ea68c9c73985ae9; FREG_cf3d3827bf6dd780; FREG_9f6595c9a76c4a7b; FREG_017c9e4604cf65bb; FREG_0af1ebaaab86f1c8; FREG_7089a736aaf0a1d4; FREG_2f3af1b9c3bbb05e; FREG_6e09d3bd094c8b1d; FREG_8e33a6d1e6f10e59; FREG_182d4c2fa8a9af54; FREG_be825ad8b6a2fe4e; FREG_6df32034ec393913; FREG_5f65ee0fcb5cc0d3; FREG_e015e1cbb82f3fda; FREG_3fb3b4c0cc922b63; FREG_03698d335d449109; FREG_b78622a8d089b9a6; FREG_b75c71fb585afd01; FREG_05a0cf3fde05e2f7; FREG_8a7aa8e9ef61cfa1; FREG_be1e3265dcb6bee1; FREG_4bb24303d6e4e3ac; FREG_21b7d5ac1419bc39; FREG_e2b6e60709b341ec; FREG_2a458a4eb550141c; FREG_d0437ba01c05d49f`. Noise remains in `clusters.csv` and is not declared false.

## Diagnostics, maps, and stability

- Nearest Known distance in clustering PCA space: median `3.965908`, 10th/90th percentiles `2.346645` / `9.509639`. Distances have no calibrated novelty cutoff.
- Background reference proximity is projected only after fit; `344` Background points are assigned a nearest cluster centroid for diagnostics and never filtered by distance.
- `pca_clusters.png` uses the formal clustering PCA's first two components; `umap_clusters.png` overlays the same primary labels on Task2B's existing visualization-only UMAP. The 2D displays may overlap or separate visually but never determine membership.
- Each non-noise cluster has a 256D centroid and true medoid, with uniform 6.4 kb paired O/E heatmaps. Novel-like figures are explicitly labeled unvalidated.
- The largest cluster is 1 with 189 candidates and 274 Known members; 2 clusters are Known-only. PCA/UMAP 2D views are descriptive and do not establish candidate novelty.
- Visual review of this frozen run: cluster 2's overview has pronounced dark row/column bands in most displayed Known windows; cluster 0's five Known members have heavily overlapping annotation intervals within MG1655 2.101–2.111 Mb. These are descriptive cautions only; the cause of the bands is unknown and the members are not five independent loci.
- Fixed sensitivity audit (8/5 and 5/8), not a model selection search:

|   min_cluster_size |   min_samples |   cluster_count |   noise_count |   noise_fraction |   adjusted_rand_index_to_primary |
|-------------------:|--------------:|----------------:|--------------:|-----------------:|---------------------------------:|
|                  8 |             5 |               2 |           102 |         0.177391 |                         0.993268 |
|                  5 |             8 |               2 |           105 |         0.182609 |                         0.930171 |

## Task2D handoff and limits

- High-priority Task2D clusters: `[]`. Medium/Low are descriptive ambiguity/known-composition hints, not a final ranking score.
- Main limits: the Task1 CNN was previously supervised on this Known set; the Known set was reused during Task2A method development; 234/344 Task2B Background references reuse AE-training windows and many overlap; one chromosome, two replicates, representation/parameter dependence, and no independent biological validation. Cluster membership and 2D geometry alone cannot establish a new chromatin structure.
- Task2D must independently validate candidate structure patterns in rep1/rep2 O/E matrices with a predeclared reproducibility and background comparison protocol, including zero-axis and known-overlap review. No Task2D correlation gate or final novel-candidate decision was run here.
- `task2c_complete=true`; `ready_for_task2d=true` means technically ready for validation, not confirmed novelty.

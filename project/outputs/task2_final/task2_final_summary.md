# Task2 Final: Proposal → Representation → Grouping → Reproducibility

## Task2A — frozen high-recall proposal generator

- Six independent channels: Small/Medium/Large × Density/Shape; channel-wise NMS, union and interval merge. Fixed paired p-equivalent threshold <=0.05, not a formal paired test.
- Channel peaks: {'small_density': 244, 'small_shape': 226, 'medium_density': 114, 'medium_shape': 72, 'large_density': 51, 'large_shape': 69}; 776 proposals merged to 231 final candidate regions; genomic union coverage 0.461301.
- Center-based Known Recall (same Known set was used during method development):

| type    |   n_known |   center_recall |   interval_overlap_recall |
|:--------|----------:|----------------:|--------------------------:|
| overall |       344 |        0.476744 |                  0.625    |
| CHIN    |       250 |        0.42     |                  0.556    |
| OPCID   |        68 |        0.705882 |                  0.852941 |
| CHID    |        26 |        0.423077 |                  0.692308 |

- The detector is a high-recall proposal generator rather than a novelty judge; further recall tuning on reused Known would overfit.

## Task2B — 256D fused representation

- Frozen concatenation: Task1 CNN 64D + Small AE 64D + Medium AE 64D + Large AE 64D, with frozen reference-fitted scaling.
- Genomic-blocked Known-vs-Background linear probes (diagnostics, not novel detectors):

| representation   |   balanced_accuracy_mean |   roc_auc_mean |   pr_auc_mean |
|:-----------------|-------------------------:|---------------:|--------------:|
| CNN-only         |                 0.795726 |       0.89978  |      0.910947 |
| AE-only          |                 0.82003  |       0.905201 |      0.906395 |
| Fused            |                 0.840147 |       0.925422 |      0.933851 |

- PCA PC1/PC2 explained variance 0.364488/0.192070; PCA reference-centroid separation/spread 0.221; UMAP separation/spread 0.380, candidate quadrants 4/4. UMAP is visualization only.
- Mean rep1/rep2 embedding cosine by block: {'cnn': 0.678990756199335, 'small': 0.8826925742190627, 'medium': 0.9400901639173359, 'large': 0.953277837012565, 'fused': 0.892722914443657}.

## Task2C — frozen unsupervised grouping

- Candidate+Known clustering PCA: 13 components; HDBSCAN {'min_cluster_size': 5, 'min_samples': 5, 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.0, 'allow_single_cluster': False, 'copy': True}.
- 3 non-noise clusters, 97 / 575 noise (0.168696); silhouette 0.2560571721516928; novel-like clusters 0.
- Cluster candidate and Known-type composition:

|   cluster_id |   candidate_count |   CHIN_count |   OPCID_count |   CHID_count | cluster_annotation   | novel_like_cluster_candidate   |
|-------------:|------------------:|-------------:|--------------:|-------------:|:---------------------|:-------------------------------|
|            0 |                 0 |            4 |             0 |            1 | CHIN-like            | False                          |
|            1 |               189 |          203 |            54 |           17 | CHIN-like            | False                          |
|            2 |                 0 |            7 |             0 |            3 | CHIN-like            | False                          |

- CHIN-like membership is a postfit representation-composition hint, not a CHIN label.

Known type distributions (Pearson and Spearman median, IQR, 5th/25th/75th/95th):

| type   |   n_valid |   pearson_p05 |   pearson_p25 |   pearson_median |   pearson_p75 |   pearson_p95 |   pearson_iqr |   spearman_p05 |   spearman_p25 |   spearman_median |   spearman_p75 |   spearman_p95 |   spearman_iqr |
|:-------|----------:|--------------:|--------------:|-----------------:|--------------:|--------------:|--------------:|---------------:|---------------:|------------------:|---------------:|---------------:|---------------:|
| CHIN   |       250 |      0.307625 |      0.468736 |         0.661416 |      0.787247 |      0.886709 |      0.318511 |       0.27242  |       0.38033  |          0.532179 |       0.689919 |       0.852927 |       0.309588 |
| OPCID  |        68 |      0.346656 |      0.576431 |         0.699142 |      0.833567 |      0.961651 |      0.257136 |       0.361797 |       0.506892 |          0.637208 |       0.767997 |       0.902255 |       0.261105 |
| CHID   |        26 |      0.547544 |      0.662013 |         0.791889 |      0.835998 |      0.89077  |      0.173986 |       0.480692 |       0.573353 |          0.651547 |       0.788516 |       0.893276 |       0.215163 |
| ALL    |       344 |      0.311862 |      0.492398 |         0.683971 |      0.800851 |      0.907502 |      0.308453 |       0.275738 |       0.411996 |          0.575414 |       0.713247 |       0.875519 |       0.301251 |

Fixed Task2C HDBSCAN sensitivity audit (not parameter selection):

|   min_cluster_size |   min_samples |   cluster_count |   noise_count |   noise_fraction |   adjusted_rand_index_to_primary |
|-------------------:|--------------:|----------------:|--------------:|-----------------:|---------------------------------:|
|                  8 |             5 |               2 |           102 |         0.177391 |                         0.993268 |
|                  5 |             8 |               2 |           105 |         0.182609 |                         0.930171 |

## Task2D — cross-replicate validation

- Formal candidate window: frozen representative Small 3.2 kb, Medium 6.4 kb or Large 12.8 kb; canonical 6.4 kb is supplemental. Controls use canonical 6.4 kb. Same coordinates for rep1/rep2, separate genome-wide E(d), 100 bp bins.
- Pearson/Spearman use finite O/E upper-triangle pixels with i<j and |i-j|>2; full upper, cosine and pooled-IQR NRMSE are diagnostics.
- Known valid 344/344; Pearson median 0.683971, 5th percentile 0.311862; Spearman median 0.575414, 5th percentile 0.275738.
- Known CHIN/OPCID/CHID distributions above are also preserved in `outputs/task2d/known_reproducibility_distribution.csv` and figure.
- Background valid 344/344; Pearson median 0.438881; Spearman median 0.378832. Background may also be reproducible.
- Candidate primary valid 231/231; median Pearson/Spearman 0.365431/0.364952; pass Known floors 144.
- Unannotated noise 16 (valid 16, pass 15); median Pearson/Spearman 0.718571/0.695559.
- Unannotated clustered 96 (valid 96, pass 56); median Pearson/Spearman 0.345762/0.343171.
- Cluster-level candidate reproducibility (including current Cluster 1):

|   cluster_id | cluster_annotation   |   candidate_count |   candidate_valid_count |   median_primary_pearson |   primary_pearson_iqr |   median_primary_spearman |   primary_spearman_iqr |   reproducible_candidate_count |   reproducible_fraction |
|-------------:|:---------------------|------------------:|------------------------:|-------------------------:|----------------------:|--------------------------:|-----------------------:|-------------------------------:|------------------------:|
|            0 | CHIN-like            |                 0 |                       0 |               nan        |            nan        |                nan        |             nan        |                              0 |              nan        |
|            1 | CHIN-like            |               189 |                     189 |                 0.337016 |              0.457112 |                  0.341003 |               0.478534 |                            103 |                0.544974 |
|            2 | CHIN-like            |                 0 |                       0 |               nan        |            nan        |                nan        |             nan        |                              0 |              nan        |

- Candidate heatmaps show canonical 6.4 kb log1p(O/E) with one global display vmax `2.2075789698562627`; this visualization percentile never affects selection.

## Final evidence tiers

- Task2C novel-like clusters before Task2D: 0; strict clusters surviving Task2D: 0; strict novel candidate members after 80% valid-member gate: 0.
- Reproducible unannotated follow-up candidates: 71; clustered 56, noise 15. Known-like clustered candidates remain unannotated at the genomic locus and are **not** assigned a Known type.
- Top follow-up candidates by predeclared pass → Spearman → Pearson → coordinate ranking:

| candidate_id          | chrom   |   start |     end |   cluster_id |   primary_pearson |   primary_spearman | known_center_overlap   | known_interval_overlap   | artifact_warning   | final_status                    |
|:----------------------|:--------|--------:|--------:|-------------:|------------------:|-------------------:|:-----------------------|:-------------------------|:-------------------|:--------------------------------|
| FREG_d2d392e1e3af249a | MG1655  | 1896000 | 1910000 |           -1 |          0.947721 |           0.947143 | False                  | False                    | False              | UNANNOTATED_NOISE_REPRODUCIBLE  |
| FREG_f5ef0631b11a8bec | MG1655  |  685600 |  688800 |           -1 |          0.8926   |           0.930467 | False                  | False                    | True               | UNANNOTATED_NOISE_REPRODUCIBLE  |
| FREG_ab8c4d5f9e40d1d7 | MG1655  | 1256800 | 1264800 |            1 |          0.922217 |           0.927952 | False                  | False                    | False              | KNOWN_LIKE_CLUSTER_REPRODUCIBLE |
| FREG_fee885b91f90be59 | MG1655  | 1988800 | 1995200 |            1 |          0.933989 |           0.915929 | False                  | False                    | False              | KNOWN_LIKE_CLUSTER_REPRODUCIBLE |
| FREG_11a5f6c7b97c1f4a | MG1655  | 2735200 | 2744000 |            1 |          0.946159 |           0.915088 | False                  | False                    | False              | KNOWN_LIKE_CLUSTER_REPRODUCIBLE |
| FREG_0af1ebaaab86f1c8 | MG1655  | 2512000 | 2515200 |           -1 |          0.86614  |           0.914971 | False                  | False                    | True               | UNANNOTATED_NOISE_REPRODUCIBLE  |
| FREG_4bb24303d6e4e3ac | MG1655  | 4172800 | 4190400 |           -1 |          0.895088 |           0.88199  | False                  | False                    | False              | UNANNOTATED_NOISE_REPRODUCIBLE  |
| FREG_a356c920b79fd72a | MG1655  | 4414800 | 4418000 |            1 |          0.92739  |           0.878462 | False                  | False                    | False              | KNOWN_LIKE_CLUSTER_REPRODUCIBLE |
| FREG_a635ba46a50b3a49 | MG1655  | 2269600 | 2272800 |            1 |          0.86176  |           0.878379 | False                  | False                    | False              | KNOWN_LIKE_CLUSTER_REPRODUCIBLE |
| FREG_df9791c4a4cf8e2c | MG1655  | 2042400 | 2045600 |            1 |          0.896956 |           0.876002 | False                  | False                    | False              | KNOWN_LIKE_CLUSTER_REPRODUCIBLE |

- 在当前预注册分析协议下，没有候选满足严格的新结构簇定义。
- A nonempty reproducible-unannotated list signals follow-up loci, not confirmed novel chromatin structures. Even HDBSCAN noise can be reproducible without forming a new structure class.
- Main limitations: reused Known set in prior supervised/method-development stages; Background references partly reuse AE-training windows and are genomically correlated; one chromosome and two biological repeats; Pearson/Spearman show stability but cannot prove structural identity or biological mechanism.
- `task2_complete=true`; `strict_novel_structure_detected=false`. Completion reflects workflow integrity, not discovery of a new class.

# Task2D Cross-replicate Reproducibility Validation

> FULL: frozen validation; reproducibility is stability, not structural identity.

## Frozen measurement

- Candidate primary window uses Task2A representative best_scale/center: 3.2, 6.4 or 12.8 kb; canonical 6.4 kb is supplemental. All windows use 100 bp bins and identical coordinates for rep1/rep2.
- Each raw matrix is divided by its own replicate genome-wide E(d), without clipping for correlation. Formal mask: upper triangle i<j with |i-j|>2; full upper is auxiliary.
- Constant vectors or fewer than ten finite paired pixels yield NaN, never an artificial zero. Pearson/Spearman are the primary measurements; cosine and pooled-IQR NRMSE are auxiliary.

## Positive and background controls

- Known: 344 rows, 344 valid; canonical Pearson median 0.683971; Spearman median 0.575414.
- Known floor rule: separate 5th percentiles of valid canonical Pearson and Spearman. Pearson floor `0.31186150698910403`; Spearman floor `0.27573844201209236`; calibration_valid `true`.
- Known type distributions (median, IQR, 5th/25th/75th/95th):

| type   |   n_total |   n_valid |   pearson_p05 |   pearson_p25 |   pearson_median |   pearson_p75 |   pearson_p95 |   pearson_iqr |   spearman_p05 |   spearman_p25 |   spearman_median |   spearman_p75 |   spearman_p95 |   spearman_iqr |
|:-------|----------:|----------:|--------------:|--------------:|-----------------:|--------------:|--------------:|--------------:|---------------:|---------------:|------------------:|---------------:|---------------:|---------------:|
| CHIN   |       250 |       250 |      0.307625 |      0.468736 |         0.661416 |      0.787247 |      0.886709 |      0.318511 |       0.27242  |       0.38033  |          0.532179 |       0.689919 |       0.852927 |       0.309588 |
| OPCID  |        68 |        68 |      0.346656 |      0.576431 |         0.699142 |      0.833567 |      0.961651 |      0.257136 |       0.361797 |       0.506892 |          0.637208 |       0.767997 |       0.902255 |       0.261105 |
| CHID   |        26 |        26 |      0.547544 |      0.662013 |         0.791889 |      0.835998 |      0.89077  |      0.173986 |       0.480692 |       0.573353 |          0.651547 |       0.788516 |       0.893276 |       0.215163 |
| ALL    |       344 |       344 |      0.311862 |      0.492398 |         0.683971 |      0.800851 |      0.907502 |      0.308453 |       0.275738 |       0.411996 |          0.575414 |       0.713247 |       0.875519 |       0.301251 |

- Background: 344 rows, 344 valid; Pearson median 0.438881; Spearman median 0.378832. Background may also be highly reproducible, so correlation is not a structure classifier.

## Candidates and frozen clusters

- Evaluated candidates: 231; valid primary correlations: 231; zero-overlap eligible: 112; unannotated noise: 16; unannotated clustered (excluding strict-pool rows): 96.
- Candidate primary Pearson/Spearman medians: 0.365431 / 0.364952; pass Known floor: 144.
- Noise valid 16 / 16, median Pearson/Spearman 0.718571 / 0.695559, pass 15. Noise remains unclustered, not a novel class.
- Unannotated clustered valid 96 / 96, median Pearson/Spearman 0.345762 / 0.343171. Known-like cluster membership is not a CHIN/OPCID/CHID prediction.
- Per-cluster candidate reproducibility:

|   cluster_id | cluster_annotation   |   candidate_count |   candidate_valid_count |   median_primary_pearson |   primary_pearson_iqr |   median_primary_spearman |   primary_spearman_iqr |   reproducible_candidate_count |   reproducible_fraction |
|-------------:|:---------------------|------------------:|------------------------:|-------------------------:|----------------------:|--------------------------:|-----------------------:|-------------------------------:|------------------------:|
|            0 | CHIN-like            |                 0 |                       0 |               nan        |            nan        |                nan        |             nan        |                              0 |              nan        |
|            1 | CHIN-like            |               189 |                     189 |                 0.337016 |              0.457112 |                  0.341003 |               0.478534 |                            103 |                0.544974 |
|            2 | CHIN-like            |                 0 |                       0 |               nan        |            nan        |                nan        |             nan        |                              0 |              nan        |

- Zero-axis or nonfinite-pixel artifact warnings: 13 candidates. They remain in all tables.
- Strict novel-cluster membership can only come from Task2C novel-like clusters with at least five candidates, zero Known and zero genomic Known overlap, plus at least 80% of valid members passing both floors. No Task2C labels were changed.
- If a Known floor is negative/nonfinite, binary filtering is stopped and continuous candidate metrics remain available; no substitute threshold is chosen.

## Interpretation

- The same Known structures participated in earlier classifier/method development; these controls are operational reference levels, not independent discovery validation. Background windows are partially correlated/reused. Two replicates and one chromosome cannot establish a new structure type.
- `task2d_complete=true`; Final Task2 status is determined separately after verified finalization.

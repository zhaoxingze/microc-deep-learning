# Task 2A.1 refined candidate detector summary

- primary_detector = OR_MAX
- shape_input_mode = oe_log_robust
- Fusion rules were fixed before known-recall evaluation; no alpha/beta search was performed.

## Requested questions

1. **Why change the old score?** DensityZ and ShapeZ were strongly anticorrelated, so their mean suppressed one-branch anomalies.
2. **Old Full issue:** the legacy paired branches had Pearson -0.671440 and Spearman -0.799045.
3. **Why mean is unsuitable:** the scientific target is Density anomaly OR Shape anomaly; a negative branch must not cancel a positive branch.
4. **Shape normalization:** genome O/E → background-train clip → log1p → per-window valid-pixel median/MAD normalization → fixed [-10, 10] winsorization.
5. **Does AE focus more on shape?** This is diagnosed, not assumed, by the branch-correlation change and the separate Shape-only recall.
6. **Correlation change:** refined Pearson -0.264303; refined Spearman -0.427552.
   Original legacy Full Top-20% recall was Overall=0.500000, CHIN=0.416000, OPCID=0.750000, CHID=0.653846.
7. **OR-Max Top-20% recall:** 0.412791.
8. **Legacy Mean Top-20% recall with refined shape input:** 0.334302.
9. **Positive-Sum Top-20% recall:** 0.398256.
10. **Density-only / Shape-only Top-20% recall:** 0.389535 / 0.308140.
11. **Window-count random Top-20% mean:** 0.829477.
12. **Coverage-matched random Top-20% mean / 95% high:** 0.349855 / 0.420276.
13. **CHIN OR-Max Top-20% recall:** 0.316000.
14. **OPCID OR-Max Top-20% recall:** 0.808824.
15. **CHID OR-Max Top-20% recall:** 0.307692.
16. **Zero-axis audit:** all=0.093011, top=0.034483, enrichment=0.370738.
17. **Replicate consistency:** Pearson=0.969765, Spearman=0.926920.
18. **Task2B readiness:** Task2A detector still needs refinement.

## AutoEncoder checkpoint

- best_epoch: 85
- best_validation_loss: 0.35617421872049404
- Checkpoint selection used background validation loss only.

## Readiness checks

- beats_coverage_random_95pct_at_top10_and_top20: False
- all_classes_recalled_at_top20pct: True
- retains_legacy_top20_overall: False
- zero_axis_not_dominant: True
- replicate_scores_positively_correlated: True
- shape_branch_not_near_duplicate: True

## Detector comparison

| method       |   candidate_fraction |   n_windows |   union_coverage_bp |   overall_recall |   CHIN_recall |   OPCID_recall |   CHID_recall |
|:-------------|---------------------:|------------:|--------------------:|-----------------:|--------------:|---------------:|--------------:|
| density_only |                 0.01 |          58 |              107200 |        0.0261628 |         0.004 |      0.117647  |     0         |
| density_only |                 0.02 |         116 |              192800 |        0.0552326 |         0.016 |      0.220588  |     0         |
| density_only |                 0.05 |         290 |              470400 |        0.127907  |         0.068 |      0.382353  |     0.0384615 |
| density_only |                 0.1  |         580 |              828000 |        0.223837  |         0.148 |      0.544118  |     0.115385  |
| density_only |                 0.2  |        1159 |             1520800 |        0.389535  |         0.292 |      0.794118  |     0.269231  |
| shape_only   |                 0.01 |          58 |              220000 |        0.0406977 |         0.032 |      0.0735294 |     0.0384615 |
| shape_only   |                 0.02 |         116 |              409600 |        0.0494186 |         0.044 |      0.0735294 |     0.0384615 |
| shape_only   |                 0.05 |         290 |              838400 |        0.113372  |         0.128 |      0.0882353 |     0.0384615 |
| shape_only   |                 0.1  |         580 |             1432000 |        0.186047  |         0.224 |      0.102941  |     0.0384615 |
| shape_only   |                 0.2  |        1159 |             2296800 |        0.30814   |         0.364 |      0.176471  |     0.115385  |
| legacy_mean  |                 0.01 |          58 |              103200 |        0.0232558 |         0.004 |      0.102941  |     0         |
| legacy_mean  |                 0.02 |         116 |              199200 |        0.0552326 |         0.016 |      0.220588  |     0         |
| legacy_mean  |                 0.05 |         290 |              483200 |        0.133721  |         0.068 |      0.411765  |     0.0384615 |
| legacy_mean  |                 0.1  |         580 |              924800 |        0.238372  |         0.148 |      0.617647  |     0.115385  |
| legacy_mean  |                 0.2  |        1159 |             1740800 |        0.334302  |         0.248 |      0.735294  |     0.115385  |
| or_max       |                 0.01 |          58 |              101600 |        0.0232558 |         0.004 |      0.102941  |     0         |
| or_max       |                 0.02 |         116 |              192800 |        0.0552326 |         0.016 |      0.220588  |     0         |
| or_max       |                 0.05 |         290 |              470400 |        0.127907  |         0.068 |      0.382353  |     0.0384615 |
| or_max       |                 0.1  |         580 |              839200 |        0.229651  |         0.156 |      0.544118  |     0.115385  |
| or_max       |                 0.2  |        1159 |             1627200 |        0.412791  |         0.316 |      0.808824  |     0.307692  |
| positive_sum |                 0.01 |          58 |              102400 |        0.0232558 |         0.004 |      0.102941  |     0         |
| positive_sum |                 0.02 |         116 |              192800 |        0.0552326 |         0.016 |      0.220588  |     0         |
| positive_sum |                 0.05 |         290 |              470400 |        0.127907  |         0.068 |      0.382353  |     0.0384615 |
| positive_sum |                 0.1  |         580 |              840000 |        0.223837  |         0.148 |      0.544118  |     0.115385  |
| positive_sum |                 0.2  |        1159 |             1648000 |        0.398256  |         0.304 |      0.794118  |     0.269231  |

## Branch correlations

| subset         |   n_windows |    pearson |   spearman |
|:---------------|------------:|-----------:|-----------:|
| all            |        5795 | -0.264303  |  -0.427552 |
| background     |        1932 | -0.166415  |  -0.298534 |
| known_overlap  |        1859 | -0.226871  |  -0.509414 |
| top_candidates |         580 |  0.0765139 |  -0.458123 |

## Artifacts

- `F:\Micro-C\project\outputs\task2a_refined\background_autoencoder\best_model.pth`
- `F:\Micro-C\project\outputs\task2a_refined\background_autoencoder\training_history.csv`
- `F:\Micro-C\project\outputs\task2a_refined\candidate_scores.csv`
- `F:\Micro-C\project\outputs\task2a_refined\candidate_regions_or_max.csv`
- `F:\Micro-C\project\outputs\task2a_refined\replicate_consistency.csv`
- `F:\Micro-C\project\outputs\task2a_refined\zero_axis_audit.csv`
- `F:\Micro-C\project\outputs\task2a_refined\figures\training_curve.png`
- `F:\Micro-C\project\outputs\task2a_refined\branch_scatter.png`
- `F:\Micro-C\project\outputs\task2a_refined\detector_recall_curve.png`
- `F:\Micro-C\project\outputs\task2a_refined\or_max_vs_random.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\reconstruction_examples\01_CHIN_WIN_fe1c169ce29c009b.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\reconstruction_examples\02_OPCID_WIN_9f5b327682402a48.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\reconstruction_examples\03_CHID_WIN_ce5eb97de406aa43.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\reconstruction_examples\04_background_WIN_1fc62eb1213185ea.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0001_WIN_9f5b327682402a48.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0002_WIN_2214b3c0c0d86f57.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0003_WIN_91dcf33c3970744c.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0004_WIN_82f2371819f88fe7.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0005_WIN_bef12e4567ecea75.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0006_WIN_a3cb3466cbe597e1.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0007_WIN_800654f3b2e5c71f.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0008_WIN_3a40a15a184c85d2.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0009_WIN_3ce8ce92b481ad1a.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0010_WIN_732474f1d27ead2d.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0011_WIN_26e0519bf8f18e42.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0012_WIN_ea1df0fc54d42b34.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0013_WIN_170ba20bfffcb12c.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0014_WIN_413ad38ab2c6e6ad.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0015_WIN_d3db49576796cf26.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0016_WIN_3230d3b53d31e102.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0017_WIN_43c1190bc97ab8c3.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0018_WIN_79d080ff7faf66ba.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0019_WIN_25491494c23a8b0b.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0020_WIN_fe1c169ce29c009b.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0021_WIN_09f660e6a0188b86.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0022_WIN_c696500d6f097b88.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0023_WIN_6136fab8c3725fa8.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0024_WIN_6d475c96b9abad0b.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0025_WIN_18dfbf56f068f216.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0026_WIN_dad8dc9208ff99ab.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0027_WIN_79e6f8258c86633d.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0028_WIN_5f882094e010a0d9.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0029_WIN_647c406011cf06d7.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0030_WIN_d7ad951b8768b8de.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0031_WIN_758c26b5add4d8f2.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0032_WIN_9f8a445e5b32c211.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0033_WIN_6aa0715331f59580.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0034_WIN_1b12d355fdaa5a23.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0035_WIN_48b2f8bc45724609.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0036_WIN_16e2864f7e1bcaab.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0037_WIN_ccf1ed4ea07bcfc6.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0038_WIN_76d0dc9e98742175.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0039_WIN_b32ec38f24ef6726.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0040_WIN_6f4d44f740726e3e.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0041_WIN_40f78fa5bc3233e9.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0042_WIN_be0c77abdf014970.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0043_WIN_891b8fa65c46cfb1.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0044_WIN_edc2b5aba269fae7.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0045_WIN_4a8c76b6a9a59938.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0046_WIN_2b9594fa868b0efb.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0047_WIN_7af38791001ac21f.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0048_WIN_f50b815f9d79a677.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0049_WIN_565a9c9c997ce276.png`
- `F:\Micro-C\project\outputs\task2a_refined\figures\top_candidates\rank_0050_WIN_4ca48f7fb63fc1c1.png`
- `F:\Micro-C\project\outputs\task2a_refined\detector_comparison.csv`
- `F:\Micro-C\project\outputs\task2a_refined\random_baseline_comparison.csv`
- `F:\Micro-C\project\outputs\task2a_refined\branch_correlation.csv`

## Scope boundary

No PCA, UMAP, HDBSCAN, clustering, latent clustering, novel-class claim, biological candidate interpretation, or Task2B–2D analysis was performed.

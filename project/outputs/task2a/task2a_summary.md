# Task 2A candidate detector summary

## Run scope

- Mode: `full`
- windows: 5795
- background_train: 1570
- background_val: 362
- known_structures: 344
- candidate_regions: 68

## Replicate consistency

- pearson: 0.9887654119085845
- spearman: 0.8187457679850091

## Zero-axis audit

- all_rate: 0.09301121656600518
- top_rate: 0.04310344827586207
- enrichment: 0.4634220459343612

zero-axis 富集只作为数据质量警告；本流程不自动宣称任何新型结构或生物学发现。

## Known-structure recall

|   candidate_fraction |   candidate_windows |   overall_recall |   CHIN_recall |   OPCID_recall |   CHID_recall |
|---------------------:|--------------------:|-----------------:|--------------:|---------------:|--------------:|
|                 0.01 |                  58 |        0.0232558 |         0.004 |       0.102941 |     0         |
|                 0.02 |                 116 |        0.0494186 |         0.016 |       0.191176 |     0         |
|                 0.05 |                 290 |        0.127907  |         0.08  |       0.338235 |     0.0384615 |
|                 0.1  |                 580 |        0.238372  |         0.164 |       0.544118 |     0.153846  |
|                 0.2  |                1159 |        0.5       |         0.416 |       0.75     |     0.653846  |

## Artifacts

- `F:\Micro-C\project\data\task2\genome_windows_metadata.csv`
- `F:\Micro-C\project\data\task2\genome_windows_rep1.npy`
- `F:\Micro-C\project\data\task2\genome_windows_rep2.npy`
- `F:\Micro-C\project\data\task2\genome_expected.npz`
- `F:\Micro-C\project\data\task2\background_split.csv`
- `F:\Micro-C\project\outputs\task2a\background_autoencoder\best_model.pth`
- `F:\Micro-C\project\outputs\task2a\background_autoencoder\training_history.csv`
- `F:\Micro-C\project\outputs\task2a\candidate_scores.csv`
- `F:\Micro-C\project\outputs\task2a\top_candidate_regions.csv`
- `F:\Micro-C\project\outputs\task2a\known_structure_recall.csv`
- `F:\Micro-C\project\outputs\task2a\random_recall_baseline.csv`
- `F:\Micro-C\project\outputs\task2a\score_ablation.csv`
- `F:\Micro-C\project\outputs\task2a\replicate_consistency.csv`
- `F:\Micro-C\project\outputs\task2a\zero_axis_audit.csv`
- `F:\Micro-C\project\outputs\task2a\figures\training_curve.png`
- `F:\Micro-C\project\outputs\task2a\figures\recall_curve.png`
- `F:\Micro-C\project\outputs\task2a\figures\reconstruction_examples\01_CHIN_WIN_fe1c169ce29c009b.png`
- `F:\Micro-C\project\outputs\task2a\figures\reconstruction_examples\02_OPCID_WIN_26e0519bf8f18e42.png`
- `F:\Micro-C\project\outputs\task2a\figures\reconstruction_examples\03_CHID_WIN_ce5eb97de406aa43.png`
- `F:\Micro-C\project\outputs\task2a\figures\reconstruction_examples\04_background_WIN_1fc62eb1213185ea.png`
- `F:\Micro-C\project\outputs\task2a\figures\reconstruction_examples\05_top_candidate_WIN_82f2371819f88fe7.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0001_WIN_82f2371819f88fe7.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0002_WIN_bef12e4567ecea75.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0003_WIN_a3cb3466cbe597e1.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0004_WIN_800654f3b2e5c71f.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0005_WIN_3a40a15a184c85d2.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0006_WIN_3ce8ce92b481ad1a.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0007_WIN_91dcf33c3970744c.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0008_WIN_732474f1d27ead2d.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0009_WIN_ea1df0fc54d42b34.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0010_WIN_26e0519bf8f18e42.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0011_WIN_170ba20bfffcb12c.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0012_WIN_413ad38ab2c6e6ad.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0013_WIN_d3db49576796cf26.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0014_WIN_3230d3b53d31e102.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0015_WIN_43c1190bc97ab8c3.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0016_WIN_25491494c23a8b0b.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0017_WIN_79d080ff7faf66ba.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0018_WIN_fe1c169ce29c009b.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0019_WIN_c696500d6f097b88.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0020_WIN_09f660e6a0188b86.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0021_WIN_6136fab8c3725fa8.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0022_WIN_6d475c96b9abad0b.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0023_WIN_18dfbf56f068f216.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0024_WIN_d7ad951b8768b8de.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0025_WIN_dad8dc9208ff99ab.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0026_WIN_79e6f8258c86633d.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0027_WIN_5f882094e010a0d9.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0028_WIN_647c406011cf06d7.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0029_WIN_9f8a445e5b32c211.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0030_WIN_6aa0715331f59580.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0031_WIN_ccf1ed4ea07bcfc6.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0032_WIN_48b2f8bc45724609.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0033_WIN_1b12d355fdaa5a23.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0034_WIN_16e2864f7e1bcaab.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0035_WIN_76d0dc9e98742175.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0036_WIN_b32ec38f24ef6726.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0037_WIN_be0c77abdf014970.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0038_WIN_6f4d44f740726e3e.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0039_WIN_40f78fa5bc3233e9.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0040_WIN_891b8fa65c46cfb1.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0041_WIN_edc2b5aba269fae7.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0042_WIN_565a9c9c997ce276.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0043_WIN_4a8c76b6a9a59938.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0044_WIN_2b9594fa868b0efb.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0045_WIN_e0fe915948c420d4.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0046_WIN_f50b815f9d79a677.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0047_WIN_7af38791001ac21f.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0048_WIN_4ca48f7fb63fc1c1.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0049_WIN_62068c38aaed90c3.png`
- `F:\Micro-C\project\outputs\task2a\figures\top_candidates\rank_0050_WIN_c24de159ec86a7b7.png`

## Readiness boundary

输出仅支持候选排序和 Task 2A 内部验证；未进行聚类、UMAP、HDBSCAN、novel class 或 Task 2B/2C 分析。

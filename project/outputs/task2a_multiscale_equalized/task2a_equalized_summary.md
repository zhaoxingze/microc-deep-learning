# Task2A.4 cross-scale calibration equalization summary

> 本轮只复用现有 empirical p-values 做共同下限校准、排序、NMS 和冻结评估；未扫描 Micro-C、未计算 E(d)、未训练或加载 AE、未执行 Task2B。

## Frozen calibration

- Calibration counts: `{'small': 1054, 'medium': 431, 'large': 147}`
- common_p_floor: `0.006756756756757` (`1/148`)
- equalized maximum anomaly: `2.170261715394957`

## Scale contribution before and after

| phase   | scale   |   peak_count |   peak_fraction |   top20_count |   total_regions |
|:--------|:--------|-------------:|----------------:|--------------:|----------------:|
| before  | small   |         1589 |       0.858919  |            20 |            1850 |
| before  | medium  |          185 |       0.1       |             0 |            1850 |
| before  | large   |           76 |       0.0410811 |             0 |            1850 |
| after   | small   |         1587 |       0.860629  |            20 |            1844 |
| after   | medium  |          178 |       0.0965293 |             0 |            1844 |
| after   | large   |           79 |       0.0428416 |             0 |            1844 |

## Coverage results

- 10% overall recall: `0.119186047`
- 10% coverage-random 97.5%: `0.156976744`
- 10% region-random 97.5%: `0.157122093`
- 20% overall recall: `0.212209302`
- 20% coverage-random 97.5%: `0.264825581`
- 20% region-random 97.5%: `0.266206395`
- 20% CHIN recall: `0.148000000`
- 20% OPCID recall: `0.455882353`
- 20% CHID recall: `0.192307692`

## Stability and data quality

- Rep1/Rep2 Spearman: `0.798465466`
- Zero-axis enrichment: `0.854096643`

## Frozen readiness gate

| check                             | passed   |
|:----------------------------------|:---------|
| beats_coverage_random_at_10pct    | False    |
| beats_coverage_random_at_20pct    | False    |
| all_classes_positive_at_20pct     | True     |
| zero_axis_enrichment_le_one       | True     |
| replicate_spearman_positive       | True     |
| same_and_cross_scale_no_leakage   | True     |
| all_background_known_overlap_zero | True     |

- ready_for_task2b: `false`
- task2b_executed: `false`

TASK2A NOT READY FOR TASK2B

在修复 cross-scale calibration-size bias 后，pure anomaly detector 仍不能稳定富集 known structures。
下一阶段应改变 proposal-generation paradigm；不再增加 scale、替换 AE 或调整 threshold。

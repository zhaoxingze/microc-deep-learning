# Task 2A.3 multi-scale candidate detector summary

> FULL：所有尺度、融合、NMS、预算、随机基线与 readiness 均在 known recall 前冻结。

## 1. 为什么做 multi-scale？

单一 6.4 kb 窗口可能稀释较短 CHIN，又无法完整表达较长 CHID；本轮只改变物理窗口尺度，100 bp bin resolution 保持不变。

## 2. 三个尺度是什么？

`small=3200/400`、`medium=6400/800`、`large=12800/1600`（window/step bp），矩阵为32/64/128。

## 3. 各尺度窗口数

| scale | n_windows |
| --- | --- |
| small | 11597 |
| medium | 5795 |
| large | 2894 |

## 4. 各尺度 Background 数量

| scale | train_background | validation_background | calibration_background |
| --- | --- | --- | --- |
| small | 5337 | 1251 | 1054 |
| medium | 2144 | 531 | 431 |
| large | 708 | 204 | 147 |

## 5. 所有 background known overlap 是否为0？

`True`。严格使用完整 `[structure_start,structure_end)`，margin=0；详见 background_overlap_audit.csv。

## 6. 是否存在跨尺度 split leakage？

`False`；审计结论 no_leakage=`True`。

## 7. Small AE

| best_epoch | best_validation_loss |
| --- | --- |
| 115 | 0.309315 |

## 8. Medium AE

| best_epoch | best_validation_loss |
| --- | --- |
| 118 | 0.394363 |

## 9. Large AE

| best_epoch | best_validation_loss |
| --- | --- |
| 120 | 0.471916 |

## 10. 每尺度 Calibration sample size

| scale | calibration_background | minimum_empirical_p |
| --- | --- | --- |
| small | 1054 | 0.000948 |
| medium | 431 | 0.002315 |
| large | 147 | 0.006757 |

经验上尾固定为 `(1+count(cal>=x))/(n+1)`，每个 scale/replicate/branch 独立拟合后才跨尺度比较。

## 11. Small-only 5/10/20/30% Recall

| budget | actual_coverage | overall_recall | CHIN_recall | OPCID_recall | CHID_recall | region_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 0.050585 | 0.069767 | 0.024000 | 0.250000 | 0.038462 | 87 |
| 0.100000 | 0.100654 | 0.133721 | 0.064000 | 0.397059 | 0.115385 | 169 |
| 0.200000 | 0.200187 | 0.267442 | 0.180000 | 0.588235 | 0.269231 | 357 |
| 0.300000 | 0.300324 | 0.401163 | 0.292000 | 0.808824 | 0.384615 | 546 |

## 12. Medium-only 5/10/20/30% Recall

| budget | actual_coverage | overall_recall | CHIN_recall | OPCID_recall | CHID_recall | region_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 0.050327 | 0.069767 | 0.032000 | 0.235294 | 0.000000 | 44 |
| 0.100000 | 0.100654 | 0.104651 | 0.052000 | 0.323529 | 0.038462 | 84 |
| 0.200000 | 0.200618 | 0.229651 | 0.172000 | 0.485294 | 0.115385 | 170 |
| 0.300000 | 0.300066 | 0.279070 | 0.208000 | 0.588235 | 0.153846 | 265 |

## 13. Large-only 5/10/20/30% Recall

| budget | actual_coverage | overall_recall | CHIN_recall | OPCID_recall | CHID_recall | region_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 0.050327 | 0.040698 | 0.028000 | 0.102941 | 0.000000 | 19 |
| 0.100000 | 0.100309 | 0.084302 | 0.064000 | 0.191176 | 0.000000 | 40 |
| 0.200000 | 0.200618 | 0.171512 | 0.152000 | 0.279412 | 0.076923 | 81 |
| 0.300000 | 0.300583 | 0.290698 | 0.276000 | 0.397059 | 0.153846 | 125 |

## 14. MultiScale 5/10/20/30% Recall

| budget | actual_coverage | overall_recall | CHIN_recall | OPCID_recall | CHID_recall | region_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 0.050413 | 0.075581 | 0.024000 | 0.279412 | 0.038462 | 79 |
| 0.100000 | 0.100912 | 0.113372 | 0.076000 | 0.279412 | 0.038462 | 109 |
| 0.200000 | 0.200101 | 0.223837 | 0.168000 | 0.441176 | 0.192308 | 205 |
| 0.300000 | 0.301186 | 0.319767 | 0.260000 | 0.544118 | 0.307692 | 337 |

## 15. CHIN Recall

四个 detector 的 CHIN 结果见上述统一 coverage 表及 `multiscale_coverage_recall.csv`。

## 16. OPCID Recall

四个 detector 的 OPCID 结果使用相同实际 genomic union coverage，不使用 Top-X windows。

## 17. CHID Recall

四个 detector 的 CHID 结果同样为 known center 落入 peak-window region union。

## 18. Coverage-random 95%范围

| budget | repeats | target_coverage_bp | region_count | matched_repeats | actual_coverage_mean | actual_coverage_min | actual_coverage_max | tolerance | overall_mean | overall_std | overall_ci_low | overall_ci_high | CHIN_mean | CHIN_std | CHIN_ci_low | CHIN_ci_high | OPCID_mean | OPCID_std | OPCID_ci_low | OPCID_ci_high | CHID_mean | CHID_std | CHID_ci_low | CHID_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 100 | 234000 | 79 | 100 | 0.050413 | 0.050413 | 0.050413 | 0.020000 | 0.051773 | 0.013824 | 0.030451 | 0.081395 | 0.052520 | 0.017136 | 0.028000 | 0.094100 | 0.048235 | 0.025732 | 0.006985 | 0.102941 | 0.053846 | 0.040339 | 0.000000 | 0.153846 |
| 0.100000 | 100 | 468400 | 109 | 100 | 0.100912 | 0.100912 | 0.100912 | 0.020000 | 0.097209 | 0.020466 | 0.063953 | 0.139535 | 0.097600 | 0.024311 | 0.056000 | 0.148000 | 0.097059 | 0.032818 | 0.044118 | 0.161765 | 0.093846 | 0.051979 | 0.000000 | 0.192308 |
| 0.200000 | 100 | 928800 | 205 | 100 | 0.200101 | 0.200101 | 0.200101 | 0.020000 | 0.202965 | 0.028284 | 0.155451 | 0.257340 | 0.205240 | 0.033935 | 0.149700 | 0.270100 | 0.196324 | 0.046054 | 0.109926 | 0.294118 | 0.198462 | 0.083510 | 0.038462 | 0.366346 |
| 0.300000 | 100 | 1398000 | 337 | 100 | 0.301186 | 0.301186 | 0.301186 | 0.020000 | 0.303721 | 0.030192 | 0.247093 | 0.362137 | 0.301800 | 0.035019 | 0.241900 | 0.372000 | 0.309853 | 0.056162 | 0.198162 | 0.419485 | 0.306154 | 0.095136 | 0.115385 | 0.481731 |

## 19. Region-random 95%范围

| budget | repeats | target_coverage_bp | region_count | matched_repeats | actual_coverage_mean | actual_coverage_min | actual_coverage_max | tolerance | overall_mean | overall_std | overall_ci_low | overall_ci_high | CHIN_mean | CHIN_std | CHIN_ci_low | CHIN_ci_high | OPCID_mean | OPCID_std | OPCID_ci_low | OPCID_ci_high | CHID_mean | CHID_std | CHID_ci_low | CHID_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 100 | 234000 | 79 | 100 | 0.056918 | 0.054809 | 0.058600 | NA | 0.058343 | 0.014836 | 0.029070 | 0.087209 | 0.059000 | 0.018316 | 0.025900 | 0.100000 | 0.057500 | 0.028900 | 0.014706 | 0.125368 | 0.054231 | 0.044885 | 0.000000 | 0.153846 |
| 0.100000 | 100 | 468400 | 109 | 100 | 0.107049 | 0.100010 | 0.111384 | NA | 0.105930 | 0.023639 | 0.066860 | 0.152834 | 0.106320 | 0.026237 | 0.064000 | 0.160000 | 0.104412 | 0.033502 | 0.044118 | 0.176471 | 0.106154 | 0.061327 | 0.000000 | 0.230769 |
| 0.200000 | 100 | 928800 | 205 | 100 | 0.198412 | 0.184737 | 0.210236 | NA | 0.195843 | 0.028273 | 0.143677 | 0.251526 | 0.193960 | 0.032331 | 0.132000 | 0.260200 | 0.202353 | 0.045618 | 0.102941 | 0.279412 | 0.196923 | 0.074833 | 0.076923 | 0.346154 |
| 0.300000 | 100 | 1398000 | 337 | 100 | 0.292676 | 0.279971 | 0.304388 | NA | 0.297645 | 0.031963 | 0.231032 | 0.356613 | 0.300600 | 0.039181 | 0.223800 | 0.376000 | 0.283235 | 0.061786 | 0.168750 | 0.404779 | 0.306923 | 0.083733 | 0.153846 | 0.461538 |

两种 baseline 均为 100 次；区间是随机重复的2.5/97.5分位，不是均值置信区间。Region random 保留正式 MultiScale region 的数量和长度多重集。

## 20. MultiScale 在10%是否稳定超过random？

`False`。

## 21. MultiScale 在20%是否稳定超过random？

`False`。

## 22. 哪些 CHIN 被 Small rescue？

以下为20% coverage下 Small-only 的 CHIN；其他预算和逐尺度 membership 见 `known_structure_scale_audit.csv`。

| structure_id | start | end | small_best_or | small_best_rank |
| --- | --- | --- | --- | --- |
| CHIN_8bc99ad127e36986 | 153540 | 155700 | 2.134177 | 322 |
| CHIN_80c12c16922655e6 | 526710 | 529470 | 2.872737 | 166 |
| CHIN_12300449726e7c4d | 676260 | 680280 | 1.714752 | 563 |
| CHIN_39d26f4e365a3b6a | 1219500 | 1220780 | 1.685822 | 600 |
| CHIN_76e05e2f594dedf6 | 1581250 | 1584230 | 1.956483 | 401 |
| CHIN_413c7aed4a1f2e45 | 1869840 | 1873960 | 1.621183 | 718 |
| CHIN_9336b6dd24567a55 | 2102940 | 2104420 | 1.531665 | 918 |
| CHIN_4c5b118b76314438 | 2102940 | 2105920 | 1.531665 | 918 |
| CHIN_27fcb8c59cdd19ed | 2102940 | 2106440 | 1.531665 | 918 |
| CHIN_dc7847370035ae4b | 2488280 | 2490700 | 1.468463 | 1047 |
| CHIN_776dbab456673217 | 2784290 | 2786430 | 2.061113 | 357 |
| CHIN_fa05997ad11be666 | 2784290 | 2789230 | 2.061113 | 357 |
| CHIN_58d99e489f853d61 | 2787400 | 2788660 | 1.580007 | 805 |
| CHIN_7547f8c089f6bd2d | 2986330 | 2990830 | 1.718455 | 559 |
| CHIN_5d54f5e6e2eeaaf5 | 3631380 | 3633920 | 1.446955 | 1109 |
| CHIN_276dc2f80d3fd17b | 3631380 | 3634760 | 1.446955 | 1109 |
| CHIN_597d783bf467e297 | 3631380 | 3636320 | 1.446955 | 1109 |

数量汇总：

| type | budget | n_total | recalled_small_only | recalled_medium_only | recalled_large_only | recalled_multiple_scales | missed_all_scales |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CHIN | 0.100000 | 250 | 8 | 5 | 11 | 9 | 217 |
| CHIN | 0.200000 | 250 | 17 | 13 | 24 | 32 | 164 |
| CHIN | 0.300000 | 250 | 33 | 10 | 37 | 47 | 123 |

## 23. 哪些 CHID 被 Large rescue？

以下为20% coverage下 Large-only 的 CHID；逐结构三个 scale 的分数、排名和 membership 见 audit。

| structure_id | start | end | large_best_or | large_best_rank |
| --- | --- | --- | --- | --- |
| CHID_c16193a866007239 | 872940 | 876560 | 1.931701 | 166 |

数量汇总：

| type | budget | n_total | recalled_small_only | recalled_medium_only | recalled_large_only | recalled_multiple_scales | missed_all_scales |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CHID | 0.100000 | 26 | 3 | 1 | 0 | 0 | 22 |
| CHID | 0.200000 | 26 | 4 | 1 | 1 | 3 | 17 |
| CHID | 0.300000 | 26 | 6 | 1 | 2 | 4 | 13 |

## 24. 多少结构仍 missed by all scales？

| type | budget | n_total | missed_all_scales |
| --- | --- | --- | --- |
| CHID | 0.100000 | 26 | 22 |
| CHIN | 0.100000 | 250 | 217 |
| OPCID | 0.100000 | 68 | 37 |
| CHID | 0.200000 | 26 | 17 |
| CHIN | 0.200000 | 250 | 164 |
| OPCID | 0.200000 | 68 | 25 |
| CHID | 0.300000 | 26 | 13 |
| CHIN | 0.300000 | 250 | 123 |
| OPCID | 0.300000 | 68 | 12 |

## 25. MultiScale candidate region 数量

1850 个排序 peak-window regions。

## 26. Rep1/Rep2 consistency

{"pearson": 0.7868386803971267, "spearman": 0.791902745125259, "unit": "multiscale peak windows", "n": 1850}

## 27. Zero-axis enrichment

{"budget": 0.2, "definition": "scan window centers within selected region union", "all_count": 20286, "selected_count": 4061, "all_zero_count": 1664, "selected_zero_count": 288, "all_rate": 0.08202701370403234, "selected_rate": 0.07091849298202413, "enrichment": 0.8645748489383062}

## 28. Readiness 每一项

| check | passed |
| --- | --- |
| beats_coverage_random_at_10pct | False |
| beats_coverage_random_at_20pct | False |
| all_classes_positive_not_below_region_random_low_at_20pct | True |
| zero_axis_enrichment_le_one | True |
| replicate_spearman_positive | True |
| same_and_cross_scale_no_leakage | True |
| all_background_known_overlap_zero | True |

## 29. ready_for_task2b

`false`。Task2B executed=`False`。

## 30. 根因诊断（仅解释，不调参）

At 20% coverage, Small uniquely rescues 17 CHIN and Large uniquely rescues 1 CHID; MultiScale Overall=0.2238 versus Medium=0.2297, while 206/344 remain missed by all scales. Scale contributes distinct rescues, but failed gates indicate scale alone is insufficient; representation/anomaly strength remains a bottleneck.

## 科学和实现限制

- Expected 在每个 replicate 上仅计算一次到 distance127，各尺度严格切片。
- Formal coverage 使用 peak window；support union 只作诊断，Large support 不会扩张正式预算。
- Genome scan 与 random 均为线性坐标，未引入 circular wrap-around。
- `best_scale`/rescue 是冻结后的解释，不参与尺度选择或权重学习。
- 即使 readiness=true，本流水线也不会自动进入 Task2B。

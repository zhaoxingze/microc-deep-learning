# Task 2A.2 calibrated detector summary

> Full：一次固定协议运行；已知标注仅用于背景排除和最终评价，未用于参数选择。

## 1. 为什么 robust z-score 不足？

相同 robust z 不代表相同背景尾部罕见程度，两个分支的尾部形状可能不同。

## 2. 为什么 empirical tail？

用独立背景经验上尾转换到同一罕见度量纲；这是背景相对 anomaly，不是已证明的生物学显著性。重叠窗口具有空间依赖，不能当成独立检验。

## 3. Calibration set 如何构建？

背景窗口数：{'train': 1263, 'validation': 411, 'calibration': 258}；完整 block 数：{'train': 26, 'validation': 6, 'calibration': 5}。
预先固定 seed 的整块三分，目标 70/15/15，实际窗口比例允许偏离；跨块窗口和已知结构排除区域不进入三分。
实际背景窗口比例：{'train': 0.6537267080745341, 'validation': 0.2127329192546584, 'calibration': 0.13354037267080746}。
Train 拟合 clip 并训练 AE；Validation 只选 checkpoint；Calibration 只拟合分支尾分布。
AE best epoch=119；best validation loss=0.345369810。

## 4. 是否无 overlap？

执行独立 Train/Validation/Calibration 两两半开区间重叠检查；任意泄漏立即失败。最终七项检查见下表。

边界限制：继承的已知排除是 center±6400，而不是扩展完整标注区间；较长标注仍可能与部分 Train/Val 窗口相交。实际计数：{'train': 3, 'validation': 7, 'calibration': 0}。详见 background_known_interval_overlap_audit.csv。Calibration 与真实已知区间另行检查，任何相交直接中止，不重新抽划分。

## 5. Density tail 如何计算？

每个 replicate 的 d 为 genome O/E 上三角、距离主对角线 >2 bins 的均值。p_D=(1+count(D_cal>=d))/(n+1)，A_D=-log10(p_D)。

## 6. Shape tail 如何计算？

每个 replicate 的 s 为 shape-normalized AE 重构在同一 off-diagonal mask 上的 MSE。p_S=(1+count(S_cal>=s))/(n+1)，A_S=-log10(p_S)。
最小可达 p=0.003861004；更极端值会饱和并产生同分。

## 7. Calibrated OR 如何计算？

每个分支先平均 rep1/rep2 anomaly，再取两个 paired branch 的最大值。min replicate 只作诊断。每重复 Bonferroni 字段不参与排名；平均 -log10(p) 不宣称是有效 combined p-value。
主分数有 2605 个不同值，最高分有 109 个窗口并列；同分用 SHA256(seed|window_id) 固定排序，不读取标注。

## 8. 为什么不再以 Top-X% windows 为主？

6400 bp 窗口每 800 bp 滑动，高度重叠。窗口比例不等于实际覆盖预算。主比较固定为 region union coverage；旧窗口比例仅为辅助对照。

## 9. NMS / region 如何工作？

降序遍历，小于 3200 bp 的同染色体窗口归属最高排名邻近峰；支持窗口只保留包含峰的连续 800 bp center 分量。区域取成员区间边界，不跨峰链式合并。区域可重叠，覆盖只计并集。
整块加入排序区域至达到预算，保留并报告 overshoot；mixed 标签固定为两分支 anomaly 均 >= -log10(.05)，不影响排序。

## 10. 真实候选 regions 数量？

全排序区域共 1224 个，来源于 5795 个配对窗口。这不是全部需要后续接受的候选：正式候选集合由固定覆盖预算截取。

## 11. 5/10/20/30% coverage Overall Recall

| budget | actual_coverage | region_count | overall_recall | CHIN_recall | OPCID_recall | CHID_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 0.051706 | 32 | 0.061047 | 0.028000 | 0.205882 | 0.000000 |
| 0.100000 | 0.100309 | 53 | 0.116279 | 0.088000 | 0.250000 | 0.038462 |
| 0.200000 | 0.201825 | 103 | 0.206395 | 0.192000 | 0.308824 | 0.076923 |
| 0.300000 | 0.300755 | 160 | 0.273256 | 0.256000 | 0.397059 | 0.115385 |

## 12. CHIN Recall / 13. OPCID Recall / 14. CHID Recall

逐类主召回见上表，均以 unique structure center 落入候选并集定义；区间 overlap 辅助召回见 coverage_recall_curve.csv。

## 15. Coverage-matched random

| budget | repeats | target_coverage_bp | region_count | matched_repeats | actual_coverage_mean | actual_coverage_min | actual_coverage_max | tolerance | overall_mean | overall_std | overall_ci_low | overall_ci_high | CHIN_mean | CHIN_std | CHIN_ci_low | CHIN_ci_high | OPCID_mean | OPCID_std | OPCID_ci_low | OPCID_ci_high | CHID_mean | CHID_std | CHID_ci_low | CHID_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 100 | 240000 | 32 | 100 | 0.051706 | 0.051706 | 0.051706 | 0.020000 | 0.049855 | 0.014762 | 0.026163 | 0.074201 | 0.050840 | 0.017365 | 0.020000 | 0.084000 | 0.047647 | 0.024793 | 0.000000 | 0.102941 | 0.046154 | 0.038844 | 0.000000 | 0.115385 |
| 0.100000 | 100 | 465600 | 53 | 100 | 0.100309 | 0.100309 | 0.100309 | 0.020000 | 0.100291 | 0.020725 | 0.062282 | 0.142442 | 0.099600 | 0.023957 | 0.057700 | 0.152200 | 0.102647 | 0.035780 | 0.029412 | 0.184191 | 0.100769 | 0.059031 | 0.000000 | 0.212500 |
| 0.200000 | 100 | 936800 | 103 | 100 | 0.201825 | 0.201825 | 0.201825 | 0.020000 | 0.199564 | 0.030298 | 0.151163 | 0.267442 | 0.199160 | 0.036355 | 0.133900 | 0.280200 | 0.200735 | 0.046428 | 0.124632 | 0.287132 | 0.200385 | 0.076594 | 0.056731 | 0.346154 |
| 0.300000 | 100 | 1396000 | 160 | 100 | 0.300755 | 0.300755 | 0.300755 | 0.020000 | 0.295581 | 0.030621 | 0.236701 | 0.357558 | 0.293040 | 0.035967 | 0.215800 | 0.366100 | 0.304412 | 0.054013 | 0.205882 | 0.397059 | 0.296923 | 0.091999 | 0.113462 | 0.461538 |

## 16. Region-matched random

| budget | repeats | target_coverage_bp | region_count | matched_repeats | actual_coverage_mean | actual_coverage_min | actual_coverage_max | tolerance | overall_mean | overall_std | overall_ci_low | overall_ci_high | CHIN_mean | CHIN_std | CHIN_ci_low | CHIN_ci_high | OPCID_mean | OPCID_std | OPCID_ci_low | OPCID_ci_high | CHID_mean | CHID_std | CHID_ci_low | CHID_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 100 | 240000 | 32 | 100 | 0.068351 | 0.063836 | 0.070664 | NA | 0.069128 | 0.019808 | 0.040698 | 0.107849 | 0.069240 | 0.024434 | 0.029900 | 0.118100 | 0.068088 | 0.030753 | 0.014706 | 0.125368 | 0.070769 | 0.050065 | 0.000000 | 0.192308 |
| 0.100000 | 100 | 465600 | 53 | 100 | 0.114225 | 0.107568 | 0.119241 | NA | 0.117703 | 0.026093 | 0.068241 | 0.168895 | 0.117840 | 0.032592 | 0.060000 | 0.182100 | 0.116618 | 0.037102 | 0.051103 | 0.176471 | 0.119231 | 0.064931 | 0.000000 | 0.250962 |
| 0.200000 | 100 | 936800 | 103 | 100 | 0.212883 | 0.199546 | 0.228601 | NA | 0.210843 | 0.031256 | 0.149491 | 0.266061 | 0.210280 | 0.037251 | 0.141900 | 0.276000 | 0.212647 | 0.046384 | 0.117647 | 0.301838 | 0.211538 | 0.081317 | 0.076923 | 0.346154 |
| 0.300000 | 100 | 1396000 | 160 | 100 | 0.309320 | 0.287877 | 0.324091 | NA | 0.313227 | 0.033480 | 0.244186 | 0.376672 | 0.311480 | 0.039556 | 0.231800 | 0.386100 | 0.321765 | 0.052543 | 0.227574 | 0.411765 | 0.307692 | 0.088712 | 0.153846 | 0.481731 |

以上区间为 100 次随机重复的 2.5/97.5 分位范围，不是均值的置信区间。每次抽样覆盖、是否匹配和召回见 random_baseline_draws.csv。Region-random 保持数量及长度多重集，但随机重叠会改变实际并集覆盖。全部使用与 scan 一致的线性域。

## 17. 是否稳定优于 random？

10% coverage 判定：False；20% coverage 判定：False。两者均超过 coverage-random 上分位才满足稳定富集门槛。

## 18. Replicate consistency 与 branch correlation

Rep1/Rep2 calibrated OR：{'pearson': 0.8550841687186909, 'spearman': 0.8695980344501323}。Density/Shape paired anomaly：{'pearson': -0.3374839134893235, 'spearman': -0.4311313886166414}。

## 19. Zero-axis enrichment

{"budget": 0.2, "definition": "scan window centers within selected region union", "all_count": 5795, "selected_count": 1171, "all_zero_count": 539, "selected_zero_count": 65, "all_rate": 0.09301121656600518, "selected_rate": 0.05550811272416738, "enrichment": 0.59678944941846}
固定统计 20% 预算区域并集包含中心的扫描窗口，与全部扫描窗口比较；两个 replicate 任一 zero-axis 即记为 zero-axis。

## 20. 哪些结构仍漏检？

逐条 ID、最佳分数、排名、最近扫描中心距离、覆盖判定见 known_structure_detection_audit.csv；下表分预算统计。

| type | total | missed_5pct | missed_5pct_median_density | missed_5pct_median_shape | missed_5pct_median_or | missed_5pct_both_below_p05 | missed_5pct_no_covering_region | missed_10pct | missed_10pct_median_density | missed_10pct_median_shape | missed_10pct_median_or | missed_10pct_both_below_p05 | missed_10pct_no_covering_region | missed_20pct | missed_20pct_median_density | missed_20pct_median_shape | missed_20pct_median_or | missed_20pct_both_below_p05 | missed_20pct_no_covering_region | missed_30pct | missed_30pct_median_density | missed_30pct_median_shape | missed_30pct_median_or | missed_30pct_both_below_p05 | missed_30pct_no_covering_region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHIN | 250 | 243 | 0.223294 | 0.927662 | 1.109572 | 167 | 0 | 228 | 0.236630 | 0.874439 | 1.062084 | 167 | 0 | 202 | 0.250485 | 0.659855 | 1.022422 | 165 | 0 | 186 | 0.274479 | 0.579352 | 0.984393 | 164 | 0 |
| OPCID | 68 | 54 | 0.637088 | 0.697535 | 1.162125 | 39 | 0 | 51 | 0.717832 | 0.650001 | 1.158027 | 38 | 0 | 47 | 0.718894 | 0.610647 | 1.146287 | 37 | 0 | 41 | 0.774839 | 0.391285 | 1.091573 | 35 | 0 |
| CHID | 26 | 26 | 0.229962 | 1.014967 | 1.217096 | 14 | 0 | 25 | 0.236630 | 1.007512 | 1.209180 | 14 | 0 | 24 | 0.260858 | 0.830778 | 1.197529 | 14 | 0 | 23 | 0.285085 | 0.654043 | 1.185877 | 14 | 0 |

## 21. CHIN / CHID 漏检的分数模式

- CHIN：20% coverage 漏检 202 个；最佳 OR 窗口 density/shape/OR 中位数分别为 0.2505/0.6599/1.0224；两分支均低于固定 p=.05 标尺的有 165 个；全区域集合无中心覆盖的有 0 个。
- CHID：20% coverage 漏检 24 个；最佳 OR 窗口 density/shape/OR 中位数分别为 0.2609/0.8308/1.1975；两分支均低于固定 p=.05 标尺的有 14 个；全区域集合无中心覆盖的有 0 个。

高分但未在预算内召回，需结合 first_covering_region_rank/coverage 检查区域优先级或窗口边界；低分提示当前窗口/表征没有把该结构转为显著背景偏离。仅凭这些诊断不能断言具体生物机制或唯一原因。本轮不据此修改 detector。

## 22. ready_for_task2b

`false`

## 23. 为什么？全部固定检查

| check | passed |
| --- | --- |
| beats_coverage_random_at_10pct | False |
| beats_coverage_random_at_20pct | False |
| all_classes_positive_not_below_region_random_low_at_20pct | True |
| zero_axis_enrichment_le_one | True |
| replicate_spearman_positive | True |
| branch_abs_spearman_lt_point9 | True |
| train_validation_calibration_no_overlap | True |

全部门槛同时满足才可冻结该版本为后续候选输入；任何失败均保留 false，并停止 Task2A。无论结果如何，本次没有运行 Task2B、聚类、embedding 或生物学解释。

## 对照、限制与可追溯性

- detector_comparison.csv 的七个方法共用本次 AE 与三分背景；它们是本轮配对消融，不冒充旧模型原始结果。旧输出目录保留了各自真实历史结果。
- known_score_distributions.png：每个 known structure 使用最佳 OR 重叠窗口，背景使用 eligible windows；粒度不同，图用于诊断，不能当成无偏检验。
- Calibration 有限且窗口空间相关，极端分数分辨率受限。SHA tie-break 仅保证确定性，不消除统计分辨率限制。
- MG1655 的 circular boundary handling 尚未启用；scan 与 random 都采用线性坐标，未来可单独改进。
- frozen_protocol.json 保存预先规则、代码哈希与 split 哈希；calibration.json 保存每个重复/分支完整校准分布。Manifest 验证所有输入和输出 SHA256。
- protected_artifacts_audit.json 验证 legacy/refined 结果与原始五个 Task2 数据工件未被覆盖。

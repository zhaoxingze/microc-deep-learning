# Micro-C 任务一总结

## 结论口径

主结论全部采用 **structure-level** 指标：同一 structure 的可用 rep1/rep2 logits 先求均值，再产生一次预测。Best checkpoint 与 input type 都只由 Validation structure-level Macro-F1 决定；Test 在 checkpoint 锁定后独立运行。replicate-level 结果仅作为辅助稳定性分析。

## 24 个问题的逐项回答

1. **最终有多少 structures？** 344（CHIN 250、OPCID 68、CHID 26）。
2. **有多少 genomic groups？** 175；其中 mixed-class groups=39，最大 group=12 structures。
3. **MILP 最终如何划分？** Train 115 groups / 241 structures（70.06%）；Validation 31 / 52（15.12%）；Test 29 / 51（14.83%）。
4. **是否存在 genomic leakage？** 否。train∩val、train∩test、val∩test 均为空。
5. **各 split 类别数？** Train CHIN/OPCID/CHID=175/48/18；Validation=38/10/4；Test=37/10/4。
6. **Rep1/Rep2 一致性如何？** 344 对全部可计算。上三角 Pearson mean=0.9957、Spearman mean=0.8338；排除 |i-j|≤2 后 Pearson=0.9941、Spearman=0.7789。Pearson 极高，秩一致性良好但更有限。
7. **Raw 的 Validation/Test 结果？** Validation Accuracy=0.7115, Macro-F1=0.6730; Test Accuracy=0.7255, Macro-F1=0.6715, Weighted-F1=0.7694。
8. **log1p 的 Validation/Test 结果？** Validation Accuracy=0.7885, Macro-F1=0.6252; Test Accuracy=0.8039, Macro-F1=0.5561, Weighted-F1=0.7869。
9. **O/E 的 Validation/Test 结果？** Validation Accuracy=0.5769, Macro-F1=0.5726; Test Accuracy=0.6078, Macro-F1=0.6017, Weighted-F1=0.6714。
10. **哪种 input 根据 Validation 最优？** RAW，Validation structure-level Macro-F1=0.6730；该选择未查看 Test。
11. **最佳 CNN Test Accuracy？** 0.7255。
12. **最佳 CNN Test Macro-F1？** 0.6715。
13. **CHIN Recall/F1？** 0.6757/0.7812。
14. **OPCID Recall/F1？** 0.9000/0.9000。
15. **CHID Recall/F1？** 0.7500/0.3333。CHID 仅 4 个 Test structures，估计不稳定。
16. **哪两个类别最容易混淆？** 最大单向误差为 CHIN→CHID（11 个），因此主要问题是 CHIN 与 CHID 的区分。
17. **Brightness baseline 表现？** Validation structure Macro-F1=0.4643；Test Accuracy=0.6471、Macro-F1=0.6053。
18. **CNN 是否可能依赖整体 contact density？** 可能。brightness baseline 的 Test Macro-F1 已达 0.6053，接近 RAW CNN 的 0.6715；说明类别存在明显强度/密度差异。O/E CNN 仍有 0.6017，提示形态信息可能也有贡献，但不能仅凭本实验断言模型学到了特定生物学机制。
19. **Grad-CAM 主要关注哪里？** 所选样本 CAM 对输入强度的 mean Spearman=0.4937，且对角带 attention ratio=0.1486；关注既与高强度区域相关，也偏向主对角邻域。
20. **是否过度集中在主对角线？** |i-j|≤2 的均匀面积基线为 0.0767，实际均值为 0.1486，约为 1.94 倍。存在明显富集，但平均权重并非大部分都落在对角带内，因此描述为“偏向”而非“完全支配”。
21. **是否关注 zero-axis artifact？** 所选 CAM 的总体 zero-axis attention ratio 均值=0.0423；在含 zero-axis 的选中 replicate 中均值=0.1524。存在一定关注，必须作为潜在 artifact 保留警示。
22. **CAM 与输入强度相关程度？** Pearson mean=0.3886，Spearman mean=0.4937，为中等正相关，支持模型部分依赖强度分布。
23. **zero-axis sensitivity 是否显著改变结果？** Validation Macro-F1 从 0.6730 到 0.6735，基本不变；Test Macro-F1 从 0.6715 降至 0.6124（Δ=-0.0592），Accuracy 从 0.7255 到 0.7442。但评价 structures 从 51 变为 43，样本集合不同，未做配对显著性检验，故不能宣称统计学显著；结果提示少数类表现对 zero-axis 排除较敏感。
24. **任务一是否完全满足要求？** 是。数据、MILP split、三组 CNN、公平选择/独立 Test、主辅指标、重复一致性、亮度基线、Grad-CAM 定量分析、zero-axis 敏感性、manifest 与核心测试均已完成；未进入任务二。

## Zero-axis 敏感性参与量

| Split | Structures | Replicates |
|---|---:|---:|
| Train | 208 | 416 |
| Validation | 42 | 84 |
| Test | 43 | 84 |

## 边界

本报告仅完成“任务一：已知结构的有监督识别与可解释分析”。没有运行 AutoEncoder、全基因组候选检测、UMAP、HDBSCAN 或未知结构发现。

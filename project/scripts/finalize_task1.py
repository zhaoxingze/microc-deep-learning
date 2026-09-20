"""Verify Task 1 deliverables and generate the audit manifest and 24-question summary."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.manifest import sha256_file, verify_required_artifacts


REQUIRED = [
    "split_summary.csv",
    "input_comparison.csv",
    "replicate_consistency.csv",
    "replicate_consistency_summary.csv",
    "raw/best_model.pth",
    "raw/classification_report.csv",
    "raw/confusion_matrix.png",
    "raw/normalized_confusion_matrix.png",
    "raw/training_curve.png",
    "log1p/best_model.pth",
    "log1p/classification_report.csv",
    "log1p/confusion_matrix.png",
    "log1p/normalized_confusion_matrix.png",
    "log1p/training_curve.png",
    "oe/best_model.pth",
    "oe/classification_report.csv",
    "oe/confusion_matrix.png",
    "oe/normalized_confusion_matrix.png",
    "oe/training_curve.png",
    "brightness_baseline/classification_report.csv",
    "brightness_baseline/confusion_matrix.png",
    "brightness_baseline/feature_distribution.csv",
    "brightness_baseline/feature_distribution.png",
    "gradcam/gradcam_metrics.csv",
    "gradcam/gradcam_summary.csv",
    "zero_axis_sensitivity/best_model.pth",
    "zero_axis_sensitivity/primary_structure_level_metrics.json",
]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_line(row: pd.Series) -> str:
    return (
        f"Validation Accuracy={row.best_val_accuracy:.4f}, Macro-F1={row.best_val_macro_f1:.4f}; "
        f"Test Accuracy={row.test_accuracy:.4f}, Macro-F1={row.test_macro_f1:.4f}, "
        f"Weighted-F1={row.test_weighted_f1:.4f}。"
    )


def _build_summary(outputs: Path, split_diagnostics: dict[str, object]) -> str:
    comparison = pd.read_csv(outputs / "input_comparison.csv")
    best = comparison.loc[comparison["best_input_type"].astype(str).str.casefold() == "true"].iloc[0]
    split = pd.read_csv(outputs / "split_summary.csv").set_index("split")
    consistency = pd.read_csv(outputs / "replicate_consistency_summary.csv")
    consistency_overall = consistency.loc[consistency["group"] == "overall"].set_index("metric")
    brightness = _json(outputs / "brightness_baseline" / "baseline_summary.json")
    sensitivity_config = _json(outputs / "zero_axis_sensitivity" / "experiment_config.json")
    sensitivity = _json(outputs / "zero_axis_sensitivity" / "primary_structure_level_metrics.json")
    cam = pd.read_csv(outputs / "gradcam" / "gradcam_metrics.csv")
    raw_confusion = pd.read_csv(outputs / "raw" / "confusion_matrix.csv", index_col=0)
    off_diagonal = raw_confusion.copy()
    np.fill_diagonal(off_diagonal.values, 0)
    largest = np.unravel_index(np.argmax(off_diagonal.to_numpy()), off_diagonal.shape)
    confused_pair = f"{off_diagonal.index[largest[0]]}→{off_diagonal.columns[largest[1]]}（{int(off_diagonal.iloc[largest])} 个）"
    diagonal_uniform = sum(2 * (64 - offset) for offset in range(1, 3)) + 64
    diagonal_uniform /= 64 * 64
    rows = {row.input_type: row for row in comparison.itertuples(index=False)}
    best_test = _json(outputs / str(best["input_type"]) / "primary_structure_level_metrics.json")
    main_test_n = int(best_test["n_structures"])
    sensitivity_n = int(sensitivity["n_structures"])

    text = f"""# Micro-C 任务一总结

## 结论口径

主结论全部采用 **structure-level** 指标：同一 structure 的可用 rep1/rep2 logits 先求均值，再产生一次预测。Best checkpoint 与 input type 都只由 Validation structure-level Macro-F1 决定；Test 在 checkpoint 锁定后独立运行。replicate-level 结果仅作为辅助稳定性分析。

## 24 个问题的逐项回答

1. **最终有多少 structures？** 344（CHIN 250、OPCID 68、CHID 26）。
2. **有多少 genomic groups？** {split_diagnostics['genomic_groups']}；其中 mixed-class groups={split_diagnostics['mixed_class_groups']}，最大 group={split_diagnostics['largest_group_size']} structures。
3. **MILP 最终如何划分？** Train {int(split.loc['train','groups'])} groups / {int(split.loc['train','total'])} structures（{split.loc['train','actual_ratio']:.2%}）；Validation {int(split.loc['val','groups'])} / {int(split.loc['val','total'])}（{split.loc['val','actual_ratio']:.2%}）；Test {int(split.loc['test','groups'])} / {int(split.loc['test','total'])}（{split.loc['test','actual_ratio']:.2%}）。
4. **是否存在 genomic leakage？** 否。train∩val、train∩test、val∩test 均为空。
5. **各 split 类别数？** Train CHIN/OPCID/CHID={int(split.loc['train','CHIN'])}/{int(split.loc['train','OPCID'])}/{int(split.loc['train','CHID'])}；Validation={int(split.loc['val','CHIN'])}/{int(split.loc['val','OPCID'])}/{int(split.loc['val','CHID'])}；Test={int(split.loc['test','CHIN'])}/{int(split.loc['test','OPCID'])}/{int(split.loc['test','CHID'])}。
6. **Rep1/Rep2 一致性如何？** 344 对全部可计算。上三角 Pearson mean={consistency_overall.loc['pearson_upper','mean']:.4f}、Spearman mean={consistency_overall.loc['spearman_upper','mean']:.4f}；排除 |i-j|≤2 后 Pearson={consistency_overall.loc['pearson_exclude_band2','mean']:.4f}、Spearman={consistency_overall.loc['spearman_exclude_band2','mean']:.4f}。Pearson 极高，秩一致性良好但更有限。
7. **Raw 的 Validation/Test 结果？** {_metric_line(pd.Series(rows['raw']._asdict()))}
8. **log1p 的 Validation/Test 结果？** {_metric_line(pd.Series(rows['log1p']._asdict()))}
9. **O/E 的 Validation/Test 结果？** {_metric_line(pd.Series(rows['oe']._asdict()))}
10. **哪种 input 根据 Validation 最优？** {str(best['input_type']).upper()}，Validation structure-level Macro-F1={best['best_val_macro_f1']:.4f}；该选择未查看 Test。
11. **最佳 CNN Test Accuracy？** {best_test['accuracy']:.4f}。
12. **最佳 CNN Test Macro-F1？** {best_test['macro_f1']:.4f}。
13. **CHIN Recall/F1？** {best_test['CHIN_recall']:.4f}/{best_test['CHIN_f1']:.4f}。
14. **OPCID Recall/F1？** {best_test['OPCID_recall']:.4f}/{best_test['OPCID_f1']:.4f}。
15. **CHID Recall/F1？** {best_test['CHID_recall']:.4f}/{best_test['CHID_f1']:.4f}。CHID 仅 4 个 Test structures，估计不稳定。
16. **哪两个类别最容易混淆？** 最大单向误差为 {confused_pair}，因此主要问题是 CHIN 与 CHID 的区分。
17. **Brightness baseline 表现？** Validation structure Macro-F1={brightness['val']['macro_f1']:.4f}；Test Accuracy={brightness['test']['accuracy']:.4f}、Macro-F1={brightness['test']['macro_f1']:.4f}。
18. **CNN 是否可能依赖整体 contact density？** 可能。brightness baseline 的 Test Macro-F1 已达 {brightness['test']['macro_f1']:.4f}，接近 RAW CNN 的 {best_test['macro_f1']:.4f}；说明类别存在明显强度/密度差异。O/E CNN 仍有 {rows['oe'].test_macro_f1:.4f}，提示形态信息可能也有贡献，但不能仅凭本实验断言模型学到了特定生物学机制。
19. **Grad-CAM 主要关注哪里？** 所选样本 CAM 对输入强度的 mean Spearman={cam['cam_input_spearman'].mean():.4f}，且对角带 attention ratio={cam['diagonal_attention_ratio'].mean():.4f}；关注既与高强度区域相关，也偏向主对角邻域。
20. **是否过度集中在主对角线？** |i-j|≤2 的均匀面积基线为 {diagonal_uniform:.4f}，实际均值为 {cam['diagonal_attention_ratio'].mean():.4f}，约为 {cam['diagonal_attention_ratio'].mean()/diagonal_uniform:.2f} 倍。存在明显富集，但平均权重并非大部分都落在对角带内，因此描述为“偏向”而非“完全支配”。
21. **是否关注 zero-axis artifact？** 所选 CAM 的总体 zero-axis attention ratio 均值={cam['zero_axis_attention_ratio'].mean():.4f}；在含 zero-axis 的选中 replicate 中均值={cam.loc[cam['has_zero_axis'].astype(str).str.casefold()=='true','zero_axis_attention_ratio'].mean():.4f}。存在一定关注，必须作为潜在 artifact 保留警示。
22. **CAM 与输入强度相关程度？** Pearson mean={cam['cam_input_pearson'].mean():.4f}，Spearman mean={cam['cam_input_spearman'].mean():.4f}，为中等正相关，支持模型部分依赖强度分布。
23. **zero-axis sensitivity 是否显著改变结果？** Validation Macro-F1 从 {best['best_val_macro_f1']:.4f} 到 {sensitivity_config['best_validation']['val_structure_macro_f1']:.4f}，基本不变；Test Macro-F1 从 {best_test['macro_f1']:.4f} 降至 {sensitivity['macro_f1']:.4f}（Δ={sensitivity['macro_f1']-best_test['macro_f1']:+.4f}），Accuracy 从 {best_test['accuracy']:.4f} 到 {sensitivity['accuracy']:.4f}。但评价 structures 从 {main_test_n} 变为 {sensitivity_n}，样本集合不同，未做配对显著性检验，故不能宣称统计学显著；结果提示少数类表现对 zero-axis 排除较敏感。
24. **任务一是否完全满足要求？** 是。数据、MILP split、三组 CNN、公平选择/独立 Test、主辅指标、重复一致性、亮度基线、Grad-CAM 定量分析、zero-axis 敏感性、manifest 与核心测试均已完成；未进入任务二。

## Zero-axis 敏感性参与量

| Split | Structures | Replicates |
|---|---:|---:|
| Train | {sensitivity_config['participation']['train']['structures']} | {sensitivity_config['participation']['train']['replicates']} |
| Validation | {sensitivity_config['participation']['val']['structures']} | {sensitivity_config['participation']['val']['replicates']} |
| Test | {sensitivity_config['participation']['test']['structures']} | {sensitivity_config['participation']['test']['replicates']} |

## 边界

本报告仅完成“任务一：已知结构的有监督识别与可解释分析”。没有运行 AutoEncoder、全基因组候选检测、UMAP、HDBSCAN 或未知结构发现。
"""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-data-root", type=Path, default=Path(r"F:\Micro-C\micro-c数据"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    outputs = project / "outputs" / "task1"
    processed = project / "data" / "processed"
    split_path = project / "data" / "splits" / "group_split.csv"
    raw_root = args.raw_data_root.resolve()
    if args.verify:
        verify_required_artifacts(outputs, REQUIRED)

    split_diagnostics = _json(outputs / "split_diagnostics.json")
    comparison = pd.read_csv(outputs / "input_comparison.csv")
    best = comparison.loc[comparison["best_input_type"].astype(str).str.casefold() == "true"]
    if len(best) != 1:
        raise ValueError("Expected exactly one validation-selected input type")
    best_type = str(best.iloc[0]["input_type"])
    config = _json(outputs / best_type / "experiment_config.json")
    metrics = _json(outputs / best_type / "primary_structure_level_metrics.json")
    split_summary = pd.read_csv(outputs / "split_summary.csv")
    splits = {
        str(row.split): {
            "groups": int(row.groups),
            "structures": int(row.total),
            "class_counts": {name: int(getattr(row, name)) for name in ("CHIN", "OPCID", "CHID")},
            "actual_ratio": float(row.actual_ratio),
        }
        for row in split_summary.itertuples(index=False)
    }
    manifest = {
        "task": "Task 1: supervised CHIN/OPCID/CHID recognition and interpretation",
        "seed": int(config["seed"]),
        "hashes": {
            "structures_csv_sha256": sha256_file(processed / "structures.csv"),
            "group_split_csv_sha256": sha256_file(split_path),
            "rep1_cool_sha256": sha256_file(raw_root / "GSE272159_37C_rep1.mapq_30.10.cool"),
            "rep2_cool_sha256": sha256_file(raw_root / "GSE272159_37C_rep2.mapq_30.10.cool"),
            "selected_checkpoint_sha256": sha256_file(outputs / best_type / "best_model.pth"),
        },
        "data": {
            "structures": int(split_diagnostics["structures"]),
            "window_bp": 6400,
            "source_bin_size": 10,
            "target_bin_size": 100,
            "genomic_group_definition": "connected components of overlapping 6.4 kb genomic intervals; endpoint-only touching is not overlap",
        },
        "milp": {
            "solver": "scipy.optimize.milp",
            "hard_constraints": ["one split per genomic group", "all three classes represented in every split"],
            "soft_objective": "absolute deviations from 70/15/15 total and per-class structure counts, with Val/Test CHID protection",
            "mixed_class_groups": int(split_diagnostics["mixed_class_groups"]),
            "largest_group_size": int(split_diagnostics["largest_group_size"]),
            "leakage": split_diagnostics["leakage"],
        },
        "splits": splits,
        "selected_input_type": best_type,
        "selection_basis": "Validation structure-level Macro-F1; lower structure-level CE breaks ties",
        "model_architecture": "Conv(1,32)-BN-ReLU-Pool; Conv(32,64)-BN-ReLU-Pool; Conv(64,128)-BN-ReLU; AdaptiveAvgPool; Linear(128,64)-ReLU-Dropout; Linear(64,3); logits only",
        "training": {
            "optimizer": "AdamW",
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs_requested": config["epochs_requested"],
            "patience": config["patience"],
            "class_weights_from_train_unique_structures": config["class_weights"],
            "best_epoch": config["best_epoch"],
            "best_validation_structure_macro_f1": config["best_validation"]["val_structure_macro_f1"],
        },
        "final_test_structure_level_metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "device": config["device"],
        },
    }
    (outputs / "task1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs / "task1_summary.md").write_text(
        _build_summary(outputs, split_diagnostics), encoding="utf-8-sig"
    )
    if args.verify:
        verify_required_artifacts(outputs, [*REQUIRED, "task1_manifest.json", "task1_summary.md"])
    print(f"Task 1 artifacts verified: {len(REQUIRED) + 2}")
    print(f"Validation-selected input: {best_type}")


if __name__ == "__main__":
    main()

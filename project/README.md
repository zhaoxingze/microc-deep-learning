# Micro-C 任务一：已知结构有监督识别与可解释分析

本工程完成真实 Micro-C 数据与 CHIN / OPCID / CHID 已知标注的数据工程、group-aware 划分、小型 CNN 公平对照及可解释分析。任务范围止于任务一；不包含 AutoEncoder、UMAP、HDBSCAN、全基因组候选检测或新结构发现。

## 数据结论

- 原始目录：`F:\Micro-C\micro-c数据`
- 可直接读取的候选基线：`GSE272159_37C_rep1.mapq_30.10.cool`、`GSE272159_37C_rep2.mapq_30.10.cool`
- 两个文件均为 10 bp/bin，染色体为 `NC_000913.3`，长度 4,641,652 bp，无 balance weight。
- 文件名没有明确写出 `WT`；因此构建脚本不会静默认定身份，必须用 `--cool-path` 明确指定。当前交付数据集使用 rep1，选择会记录在 metadata 和 manifest 中。
- 标注来自 `标注数据.xlsx`：Supplementary Table 4 = OPCID（68），Table 5 = CHIN（250），Table 6 = CHID（26），合计 344。
- 标注染色体写作 `MG1655`，Cooler 染色体写作 `NC_000913.3`；加载器只在文件明确声明 MG1655 且为单染色体时应用该别名映射，并在 metadata 中同时保留请求名和实际名。
- rep1 与 rep2 均已生成 raw/log1p/O/E 三组 `(344, 1, 64, 64)` 矩阵，并用稳定 `structure_id` 一一对齐。
- 主实验保留 zero-axis 样本并在 metadata 标记；另对 Validation 选出的最佳 input 单独运行 `--exclude-zero-axis` 敏感性分析，不改变主实验。

`GSE272161_RAW.tar` 内还有 21 个 `.cool.gz` 条件文件。盘点会列出归档成员，但不会擅自解包约 5 GB 的原始归档；`check_cool_files.py` 只验证磁盘上可直接寻址的 Cooler 文件。

## 安装与运行

```powershell
cd F:\Micro-C\project
python -m pip install -r requirements.txt
python scripts\inspect_data.py
python scripts\check_cool_files.py
python scripts\find_structure_annotations.py
python scripts\build_known_structure_dataset.py --cool-path "F:\Micro-C\micro-c数据\GSE272159_37C_rep1.mapq_30.10.cool"
python scripts\qc_known_structures.py
python scripts\build_paired_known_dataset.py
python scripts\build_group_split.py
python scripts\run_task1_experiments.py
python scripts\analyze_replicate_consistency.py
python scripts\run_brightness_baseline.py
python scripts\explain_task1.py
python scripts\train_task1.py --input-type raw --exclude-zero-axis --output-dir outputs\task1\zero_axis_sensitivity --mode both
python scripts\finalize_task1.py --verify
python -m pytest -q
pytest -q
```

不带 `--cool-path` 运行构建脚本时，只有文件名中明确包含 `WT` 且唯一匹配 rep1 才会自动选择；否则脚本会列出候选并停止。

## 实现约定

- `load_local_matrix` 只 fetch 单个局部区间，绝不将整条染色体转为 dense。
- 6,400 bp 窗口按源基因组 bin 的中点归属，二维接触计数通过求和聚合到 100 bp/bin，得到 64×64；这不是图片缩放。
- 越过基因组边界的部分以 0 padding，并写入 `padded` metadata。
- 输出同时保留 raw、log1p 和 O/E；O/E 按 `|i-j|` 距离的全体像素均值计算。
- CSV 使用 UTF-8 with BOM，便于 Windows Excel/VS Code 直接打开；JSON 和代码为 UTF-8。
- `dataset_manifest.json` 记录参数、依赖版本、输入与输出 SHA-256，便于复现和完整性核验。
- 不会删除局部覆盖缺口：含完整零行/零列的窗口会以 `has_zero_axis`、`zero_row_count`、`zero_col_count` 标记，供训练前明确决定保留、掩码或分层分析。

## 任务一评价口径

- `genomic_group_id` 是 Train/Validation/Test 的不可拆分单位；固定划分保存在 `data/splits/group_split.csv`。
- Train 将可用 rep1/rep2 作为训练观测；Validation/Test 先对同一 structure 的可用 replicate logits 求均值，再计算一次结构级预测与 loss。
- checkpoint 只按 Validation structure-level Macro-F1 选择，同分时使用更低的 structure-level validation loss。
- RAW/log1p/O/E 使用相同 split、seed、网络与训练参数；best input 只依据 Validation，Test 在 checkpoint 锁定后独立运行。
- 主结果在 `outputs/task1/task1_summary.md`，完整可复现信息及 SHA-256 在 `outputs/task1/task1_manifest.json`。

## 主要输出

- `outputs/data_inventory.csv`
- `outputs/cool_file_summary.csv`
- `outputs/structure_annotation_summary.csv`
- `data/processed/structures.csv`
- `data/processed/known_windows_{raw,log1p,oe}.npy`
- `data/processed/known_labels.npy`
- `data/processed/known_metadata.csv`
- `data/processed/dataset_manifest.json`
- `outputs/qc/*_examples.png`、`*_mean.png`、`class_distribution.png`
- `outputs/qc/qc_summary.csv`、`qc_sample_metrics.csv`
- `data/processed/paired_known_metadata.csv`
- `data/splits/group_split.csv`
- `outputs/task1/input_comparison.csv`
- `outputs/task1/{raw,log1p,oe}/`
- `outputs/task1/brightness_baseline/`
- `outputs/task1/gradcam/`
- `outputs/task1/zero_axis_sensitivity/`
- `outputs/task1/task1_summary.md`
- `outputs/task1/task1_manifest.json`

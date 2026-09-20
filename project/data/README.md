# Local data layout

Large Micro-C inputs and generated matrices are intentionally excluded from Git and must be supplied locally.

The current scripts expect the original inputs at:

```text
F:\Micro-C\micro-c数据\
├── 标注数据.xlsx
├── GSE272159_37C_rep1.mapq_30.10.cool
├── GSE272159_37C_rep2.mapq_30.10.cool
└── GSE272161_RAW.tar
```

Generated arrays and metadata are written to `project/data/processed/` and remain local. The fixed, compact genomic-group split in `project/data/splits/group_split.csv` is versioned for reproducibility.

Do not commit `.cool`, `.mcool`, `.hic`, `.npy`, `.npz`, archives, or model checkpoints. The root `.gitignore` enforces these rules without deleting local files.

# Micro-C Deep Learning

This repository contains the Python implementation for supervised recognition and interpretation of known CHIN, OPCID, and CHID structures (Task 1), followed by candidate generation, representation, unsupervised grouping and cross-replicate reproducibility validation (Task 2).

The executable project is in [`project/`](project/). See [`project/README.md`](project/README.md) for installation, data preparation, training, evaluation, and verification commands.

Large experimental inputs, generated NumPy matrices, model checkpoints, serialized models, dependency folders and temporary files are intentionally excluded from Git. They remain local. Data placement is documented in [`project/data/README.md`](project/data/README.md).

Formal experiment CSV tables, JSON provenance records, Markdown reports and PNG heatmaps are published under `project/outputs/task*/`; Smoke runs and logs are excluded. The current final Task 2 report is [`project/outputs/task2_final/task2_final_summary.md`](project/outputs/task2_final/task2_final_summary.md). The workflow is complete, with 71 reproducible unannotated follow-up loci and no strict novel-structure cluster; these loci are not confirmed new structure types.

Published manifests record the original local paths and hashes, including inputs and model artifacts intentionally omitted from GitHub. They are provenance records, not a claim that a fresh clone contains every referenced file. Re-running the full experiment requires placing the source data locally and generating the omitted artifacts with the documented scripts. `.gitattributes` preserves published result bytes without Git newline conversion.

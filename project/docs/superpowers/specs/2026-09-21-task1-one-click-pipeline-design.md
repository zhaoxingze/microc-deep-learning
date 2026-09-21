# Task 1 One-Click Pipeline Design

## Purpose

Provide one-command, reproducible orchestration for the existing Micro-C Task 1 workflow without changing any scientific algorithm, model, label, split, metric, or existing formal result. Windows users run `run_all.ps1`; all platforms may run `scripts/run_all.py`. Both entry points consume one Python pipeline definition.

## Scope and boundaries

The pipeline covers environment validation, tests, raw-data inventory, Cooler validation, annotation extraction, rep1 dataset construction, QC, paired rep1/rep2 construction, genomic-group MILP split, raw/log1p/O/E CNN experiments, replicate consistency, brightness baseline, Grad-CAM, validation-selected input resolution, zero-axis sensitivity, finalization, and final tests.

It must not modify the CNN architecture, label mapping, class-weight formula, O/E definition, genomic grouping, MILP objective, Train/Validation/Test protocol, validation-only selection, test protocol, Grad-CAM mathematics, zero-axis definition, brightness baseline, or replicate-consistency definition. It must not add Task 2, AutoEncoder, anomaly detection, UMAP, HDBSCAN, or unknown-structure discovery. Raw data under `F:\Micro-C\micro-c数据` are read-only.

## Architecture

### Core module

`src/task1/pipeline.py` is the single source of truth. It defines:

- immutable path/configuration objects;
- the ordered Task 1 steps and their subprocess argument lists;
- environment and source-file validation;
- real-time terminal-and-file logging;
- fail-fast execution;
- smoke/full mode differences;
- test skipping and dry-run behavior;
- conservative resume validators;
- artifact validation after critical steps;
- validation-only best-input selection;
- final result reporting.

Commands are represented as argument lists and launched without `shell=True`, so paths containing spaces are safe. Every Python subprocess uses `sys.executable`, preserving the active conda or virtual environment.

### Python CLI

`scripts/run_all.py` is a thin cross-platform adapter. Its parser supports:

- `--mode {smoke,full}`, default `smoke`;
- `--skip-tests`;
- `--dry-run`;
- `--resume`;
- optional path overrides for project root, raw root, rep1, and rep2, with current project locations as defaults.

It creates `PipelineConfig`, calls the core pipeline, and maps pipeline errors to a non-zero process exit.

### PowerShell wrapper

`run_all.ps1` supports `-Mode`, `-SkipTests`, `-DryRun`, and `-Resume`. It resolves the current `python` command from the active environment, builds only the Python CLI arguments, invokes `scripts/run_all.py`, and propagates non-zero exit status. It contains no scientific step list.

## Data and output isolation

Full mode uses the canonical project paths:

- `data/processed/`;
- `data/splits/`;
- `outputs/task1/`.

Smoke mode uses an isolated pseudo-project root beneath `outputs/pipeline_runs/smoke/`:

- `outputs/pipeline_runs/smoke/data/processed/`;
- `outputs/pipeline_runs/smoke/data/splits/`;
- `outputs/pipeline_runs/smoke/outputs/task1/`;
- `outputs/pipeline_runs/smoke/outputs/qc/`.

Inventory and structure-extraction outputs are also redirected there. The existing scripts already expose the necessary path arguments. `finalize_task1.py --project-root` receives the smoke pseudo-project root, allowing its existing verifier to operate without writing to canonical formal results. Logs always go to canonical `outputs/logs/`.

Smoke runs use `--epochs 3 --patience 2` for both the three-input CNN comparison and zero-axis sensitivity. Full runs omit these arguments and therefore use the training scripts' formal defaults. Smoke prints a prominent warning that its results are not suitable for scientific reporting.

## Ordered execution

The pipeline order is fixed:

1. Environment Check
2. Step 0 — `python -m pytest -q`
3. Step 1 — `inspect_data.py`
4. Step 2 — `check_cool_files.py`
5. Step 3 — `find_structure_annotations.py`
6. Step 4 — `build_known_structure_dataset.py`, explicitly using rep1
7. Step 5 — `qc_known_structures.py`
8. Step 6 — `build_paired_known_dataset.py`, explicitly using rep1 and rep2
9. Step 7 — `build_group_split.py`
10. Step 8 — `run_task1_experiments.py`
11. Step 9 — `analyze_replicate_consistency.py`
12. Step 10 — `run_brightness_baseline.py`
13. Step 11 — `explain_task1.py`
14. Step 12 — resolve the one validation-selected input
15. Step 13 — `train_task1.py --exclude-zero-axis --mode both`
16. Step 14 — `finalize_task1.py --verify`
17. Step 15 — `python -m pytest -q`
18. Final summary

The orchestrator passes only options confirmed in the current script `argparse` definitions.

## Environment and source validation

Before scientific work, the core prints and logs `sys.executable`, Python version, project/raw roots, rep1/rep2 paths, mode, dry-run, resume, and test-skipping state. It verifies the two exact Cooler paths; missing files cause an explicit failure and candidate listing without guessing a replacement.

It checks imports for `numpy`, `pandas`, `scipy`, `sklearn`, `torch`, `cooler`, `matplotlib`, and `pytest`. Missing packages are listed as `Missing dependency: <name>` followed by `python -m pip install -r requirements.txt`; packages are never installed automatically. CUDA availability and GPU name are reported when available; unavailable CUDA is informational only.

Dry-run performs argument parsing, source-path validation, a non-mutating environment check, and command-plan construction. It does not create experiment directories, logs, datasets, checkpoints, or scientific outputs and never launches scientific subprocesses.

## Execution and logging

Each real run creates `outputs/logs/run_all_YYYYMMDD_HHMMSS.log`. A logger writes run metadata, command, start time, output, end time, elapsed duration, and PASS/FAIL/SKIP state. Subprocess stdout and stderr are merged and streamed line-by-line to both terminal and log. A non-zero return code raises `PipelineStepError`, records the failed step, and prevents every later step from running.

`--skip-tests` records both test steps as `SKIPPED by --skip-tests`. No other step is skipped by this flag.

## Artifact validation

Validators execute immediately after successful subprocesses:

- Structures: CSV exists, is non-empty, contains `type`, `chrom`, `start`, `end`, and `center`; class counts and total are printed. Total other than 344 emits a warning but does not alter data.
- Base rep1 dataset: raw/log1p/O/E arrays, labels, and metadata exist; arrays are four-dimensional and aligned in sample count; labels and metadata have the same count.
- QC: the output directory contains `qc_summary.csv` and at least one expected plot.
- Paired data: rep1/rep2 raw/log1p/O/E arrays and metadata exist; each input type has aligned shape; label arrays and ordered `structure_id` values match; paired metadata contains both replicates.
- Split: CSV contains `structure_id`, `type`, `genomic_group_id`, and `split`; train/val/test all exist; structure IDs are unique; each genomic group belongs to exactly one split.
- CNN comparison: `input_comparison.csv` contains raw/log1p/oe exactly once and each experiment directory has its checkpoint and primary metrics.
- Grad-CAM: detail and summary CSVs exist and at least one PNG exists.
- Finalization: existing `finalize_task1.py --verify` runs, then summary and manifest existence is confirmed.

## Validation-only model selection

`read_validation_selected_input(path)` reads only `input_type` and `best_input_type` from `input_comparison.csv`. Boolean parsing accepts normal CSV boolean representations. Exactly one row must be selected and its input must be one of `raw`, `log1p`, or `oe`. Zero selections, multiple selections, missing columns, or invalid inputs are fatal. Test metrics are neither required nor inspected, so changing them cannot alter the selected input.

The final displayed Test Accuracy and Test Macro-F1 are read only after selection from the selected input's locked `primary_structure_level_metrics.json`. They are reporting fields and never selection inputs.

## Conservative resume

Resume is allowed only for deterministic data-preparation steps with strong validators: annotation extraction, base rep1 dataset, QC, paired dataset, and group split. A step is skipped only when its validator passes against the active smoke/full workspace. Inventory and Cooler inspection are cheap and always rerun. Corrupt, incomplete, misaligned, or leaking artifacts fail validation and cause the corresponding step to rerun rather than being trusted.

CNN experiments, replicate consistency, brightness baseline, Grad-CAM, best-input resolution, zero-axis sensitivity, finalization, and tests are never skipped merely because outputs exist.

## Testing strategy

`tests/test_task1_pipeline.py` uses temporary directories, toy CSV/NumPy artifacts, fake subprocess runners, and monkeypatching. It does not train a CNN. Coverage includes CLI mode parsing, invalid mode, missing source files, fail-fast behavior, no execution after failure, validation-only selection and its error cases, test-metric independence, SkipTests, DryRun, smoke 3/2 arguments, absence of duplicated full training defaults, Resume validation and corruption, split leakage, artifact validation, and PowerShell argument mapping where practical.

Existing tests remain unchanged except where a new stable pipeline interface must be imported. `requirements.txt` adds `pytest`.

## Documentation and acceptance

README gains a “One-click Task 1 Reproduction” section covering environment creation, dependency installation, tests, smoke/full commands, Python commands, Resume, DryRun, SkipTests, execution-policy workarounds, smoke-result warnings, and the corrected paired-dataset command with both required Cooler arguments.

Acceptance requires:

1. all pytest tests pass;
2. full dry-run prints all ordered commands and performs no scientific write;
3. a real isolated smoke run reaches finalization and final pytest;
4. canonical `outputs/task1/` hashes for formal protected artifacts remain unchanged across the smoke run;
5. no full 60-epoch run is claimed unless actually executed;
6. final terminal output includes mode, validation-selected input, reporting-only Test Accuracy/Macro-F1, summary, manifest, Grad-CAM directory, checkpoint, log, and elapsed time.

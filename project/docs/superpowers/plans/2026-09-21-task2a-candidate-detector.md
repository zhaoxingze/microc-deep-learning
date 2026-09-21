# Task 2A Candidate Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a reproducible genome-wide background-autoencoder candidate detector through known-structure recall, without entering clustering or novelty claims.

**Architecture:** Focused `src/task2` modules own coordinates/scan, genome expected, background split, autoencoder, scoring, recall, reporting, and orchestration. The cross-platform `scripts/run_task2a.py` entry point calls the in-process pipeline, with full outputs canonical and smoke outputs isolated.

**Tech Stack:** Python 3.11+, NumPy memmap, pandas, Cooler, SciPy, scikit-learn, PyTorch, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-task2a-candidate-detector-design.md`

## Global Constraints

- Reuse `load_local_matrix`; never materialize the complete chromosome contact matrix or use image resize/interpolation.
- Baseline geometry is 6,400 bp / 800 bp step / 100 bp target bins / `1x64x64`.
- Expected is replicate-specific genome-wide target-bin expected, not Task 1 local O/E.
- AE checkpoint selection uses background validation MSE only; known recall is post-lock evaluation only.
- Primary scores use `i < j` and `|i-j|>2`; robust statistics use background train+validation only; weights remain 0.5/0.5.
- Smoke writes only below `outputs/pipeline_runs/task2a_smoke`; Task 1 results and raw inputs are read-only.
- Stop after Task 2A; no dimensionality reduction, clustering, or novelty conclusion.

## Review Focus

- Chromosome length not divisible by step or target size: keep only complete scan windows and compute expected denominators from valid target bins.
- Cooler pixels stored only in one triangle: accumulate each target pair once and divide by unordered target-pair counts.
- A background block with too few eligible windows: fail clearly rather than silently random-splitting overlapping windows.
- Zero MAD: robust scaling remains finite through epsilon and maps values relative to the median.
- Smoke region lacks a known class: report zero recall/no example without fabricating a sample.

---

### Task 1: Window coordinates, aligned scan, and genome expected

**Files:**
- Create: `src/task2/__init__.py`
- Create: `src/task2/window_scan.py`
- Create: `src/task2/expected.py`
- Create: `tests/test_task2_window_scan.py`
- Create: `tests/test_task2_expected.py`

**Interfaces:**
- Produces: `WindowCoordinate`, `generate_windows`, `stable_window_id`, `scan_replicates`, `compute_genome_expected`, `apply_expected`.
- Consumes: `load_local_matrix`, Cooler pixel selectors, output `Path` objects.

- [ ] Write tests asserting literal coordinates for a 20 kb toy chromosome, exclusion of partial final windows, stable IDs, rep1/rep2 ordered alignment, and output shape/zero-axis metadata.
- [ ] Run `python -m pytest tests/test_task2_window_scan.py -q`; expect import failure for `src.task2.window_scan`.
- [ ] Implement coordinate validation and memmapped aligned replicate scanning, calling `load_local_matrix` once per replicate/window.
- [ ] Write expected tests with hand-derived target-pair sums for a tiny Cooler-like pixel stream and a literal 3x3 O/E result, including zero expected.
- [ ] Run `python -m pytest tests/test_task2_expected.py -q`; expect missing expected functions.
- [ ] Implement chunked pixel accumulation by 100 bp target-bin distance and matrix application.
- [ ] Run both Task 2 files and the full suite; expect all pass.
- [ ] Commit as `feat: scan aligned task2 genome windows`.

### Task 2: Known exclusion and blocked background split

**Files:**
- Create: `src/task2/background.py`
- Create: `tests/test_task2_background.py`

**Interfaces:**
- Produces: `build_exclusion_intervals`, `mark_background_candidates`, `blocked_background_split`, `assert_no_background_overlap`.
- Consumes: scan metadata and structures with center/chrom.

- [ ] Write tests with literal intervals proving center ±(3200+margin), strict half-open overlap, cross-block removal, deterministic block assignment, and no train/val interval overlap.
- [ ] Run `python -m pytest tests/test_task2_background.py -q`; expect missing module/functions.
- [ ] Implement merged exclusions and whole-block split with an 80/20 seeded block permutation and explicit failure for fewer than two eligible blocks.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: build leakage-safe task2 background`.

### Task 3: Convolutional autoencoder and background-only training

**Files:**
- Create: `src/task2/autoencoder.py`
- Create: `tests/test_task2_autoencoder.py`

**Interfaces:**
- Produces: `BackgroundAutoencoder`, `WindowDataset`, `masked_reconstruction_error`, `train_autoencoder`, `score_reconstruction`.
- Consumes: aligned raw memmaps, background split rows, seed/device/training configuration.

- [ ] Write tests proving `1x64x64` round-trip shape, literal off-diagonal MSE, dataset log1p transformation, early stopping on validation loss, and checkpoint creation without known labels.
- [ ] Run `python -m pytest tests/test_task2_autoencoder.py -q`; expect missing module/functions.
- [ ] Implement the specified encoder/decoder, deterministic DataLoaders over both replicates, MSE training, validation-only selection, history CSV, and checkpoint loading.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: train task2 background autoencoder`.

### Task 4: Candidate scoring, regions, and recall evaluation

**Files:**
- Create: `src/task2/scoring.py`
- Create: `src/task2/recall.py`
- Create: `tests/test_task2_scoring.py`
- Create: `tests/test_task2_recall.py`

**Interfaces:**
- Produces: `off_diagonal_mask`, `density_features`, `RobustScale`, `fit_robust_scale`, `candidate_score`, `score_windows`, `annotate_known_overlap`, `merge_candidate_regions`, `recall_at_fractions`, `random_recall_baseline`, `score_ablation`.
- Consumes: raw arrays, expected vectors, AE reconstructions, background reference mask, known structures.

- [ ] Write scoring tests with hand-derived density/MSE values, zero-MAD finite z-score, invalid weight rejection, exact paired mean/difference, and zero-axis flags.
- [ ] Run `python -m pytest tests/test_task2_scoring.py -q`; expect missing module/functions.
- [ ] Implement fixed-mask density and shape scores, background-only robust scales, 0.5/0.5 replicate and paired score fields.
- [ ] Write recall tests proving known-center recall, ceil top-percent selection, deterministic random baseline, no-label empty-class handling, and literal region merge boundaries.
- [ ] Run `python -m pytest tests/test_task2_recall.py -q`; expect missing module/functions.
- [ ] Implement ranking/annotation, window and region recall, 100-repeat random baseline, and density/shape/combined ablation.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: score and evaluate task2 candidates`.

### Task 5: Reporting, figures, manifest, and summary

**Files:**
- Create: `src/task2/reporting.py`
- Create: `tests/test_task2_reporting.py`

**Interfaces:**
- Produces: `plot_recall_curve`, `plot_reconstruction_examples`, `plot_top_candidates`, `write_manifest`, `write_summary`, `validate_task2a_outputs`.
- Consumes: score/recall tables, arrays/reconstructions, configuration, input/output paths.

- [ ] Write tests using tiny arrays/tables that assert required CSV/JSON/Markdown/PNG artifacts, valid output hashes, smoke warning, and conservative readiness text.
- [ ] Run `python -m pytest tests/test_task2_reporting.py -q`; expect missing module/functions.
- [ ] Implement non-interactive matplotlib figures, UTF-8-sig CSV/summary, JSON manifest with input/output SHA-256, and zero-axis enrichment warnings.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: report task2a detector validation`.

### Task 6: One-command pipeline, dry-run, and resume

**Files:**
- Create: `src/task2/pipeline.py`
- Create: `scripts/run_task2a.py`
- Create: `tests/test_task2_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `Task2AConfig`, `Task2APaths`, `build_task2a_plan`, `execute_task2a`, CLI `parse_args/main`.
- Consumes: all Task 1–5 public interfaces.

- [ ] Write tests proving smoke/full path isolation, 100 kb/3 epoch smoke defaults, full chromosome/60 epoch defaults, dry-run no artifact creation, resume structural validation, fail-fast behavior, and CLI invalid mode rejection.
- [ ] Run `python -m pytest tests/test_task2_pipeline.py -q`; expect missing module/entry point.
- [ ] Implement scan→expected→background→train→score→evaluate→report state machine with Task 1 logger reuse, exact input validation, conservative resume, and actionable failure log path.
- [ ] Add README commands and formulas without changing Task 1 entry points.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: add one-click task2a pipeline`.

### Task 7: Integration validation and smoke hardening

**Files:**
- Modify only when a RED regression test proves a bug in files owned by Tasks 1–6.

**Interfaces:**
- Consumes: complete Task 2A pipeline.
- Produces: verified full dry-run, isolated smoke artifacts, protected Task 1/raw hash audit, and final evidence.

- [ ] Record SHA-256 for Task 1 protected artifacts and both Cooler inputs.
- [ ] Run `python -m compileall -q src scripts tests` and `python -m pytest -q`; expect success.
- [ ] Run `python scripts/run_task2a.py --mode full --dry-run`; expect complete plan and no scientific writes.
- [ ] Run `python scripts/run_task2a.py --mode smoke`; expect scan through final summary and smoke warning.
- [ ] For any integration failure, write the smallest regression test, observe RED, patch, observe GREEN, and rerun smoke.
- [ ] Recompute protected hashes; expect exact equality and all Task 2A smoke artifacts below the isolated root.
- [ ] Run final compile, full pytest, full dry-run, and `git diff --check`; expect success.
- [ ] Commit integration fixes as `fix: harden task2a smoke pipeline` only if needed.

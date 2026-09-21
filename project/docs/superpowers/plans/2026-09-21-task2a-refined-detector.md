# Task 2A.1 Refined Candidate Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cancellation-prone Task 2A fusion with a fixed OR-Max primary detector and a shape-normalized AutoEncoder branch, then evaluate it against coverage-matched random baselines in smoke and Full runs.

**Architecture:** Add focused preprocessing and evaluation modules while minimally extending the existing scoring, reporting, and pipeline interfaces. Preserve legacy outputs by routing `--detector-version refined` to separate output roots and reuse only validated immutable Full scan artifacts.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy, PyTorch, Cooler, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-task2a-refined-detector-design.md`

## Global Constraints

- Do not modify Task 1 code, Task 1 results, raw Cooler files, or legacy `outputs/task2a/` results.
- Keep window geometry, Expected(d), exclusion, blocked split, recall, zero-axis, and region merge definitions fixed.
- Use background-train-only per-replicate 99.5th-percentile O/E clipping, `oe_log_robust` normalization, unit scale for MAD ≤ 1e-8, and fixed normalized winsorization to [-10, 10].
- Apply fixed MEAN, OR_MAX, and POSITIVE_SUM rules without recall-based tuning; OR_MAX is primary.
- Select the AutoEncoder checkpoint from background validation loss only; Full maximum epochs 120, patience 15.
- Run at least 100 reproducible window-count and coverage-matched random repetitions.
- Stop after Task 2A.1 and report readiness without starting Task 2B–2D.

## Review Focus

- A background reference containing non-finite O/E values must still produce a finite positive clip or a clear validation error.
- A zero-MAD window must remain finite without hiding its constant shape representation.
- Coverage matching near chromosome boundaries must never generate coordinates outside the provided window universe.
- Empty diagnostic subsets must emit explicit NaN correlations and counts instead of crashing or fabricating values.
- Refined Full reuse must reject stale geometry, source paths, replicate order, or background leakage before training.

---

### Task 1: Shape preprocessing and AutoEncoder input

**Files:**
- Create: `src/task2/shape_preprocessing.py`
- Modify: `src/task2/autoencoder.py`
- Create: `tests/test_task2_shape_preprocessing.py`
- Modify: `tests/test_task2_autoencoder.py`

**Interfaces:**
- Produces `fit_background_oe_clips(rows, array_paths, expected, percentile=99.5) -> dict[str, float]`.
- Produces `preprocess_shape_window(raw, expected, clip_value, exclude_band=2, eps=1e-8) -> np.ndarray`.
- Extends `WindowDataset(..., input_type="oe_log_robust", expected=..., oe_clip_values=...)`.

- [ ] Write tests for O/E→clip→log1p→robust normalization, finite cleanup, zero MAD, and background-only clip fitting.
- [ ] Run the focused tests and observe failures caused by missing APIs.
- [ ] Implement minimal preprocessing and dataset integration without changing the AE architecture or loss.
- [ ] Run focused tests and the full suite; expect all pass.
- [ ] Commit as `feat: add shape-normalized task2 autoencoder input`.

### Task 2: Fixed OR fusion and refined score table

**Files:**
- Modify: `src/task2/scoring.py`
- Modify: `tests/test_task2_scoring.py`

**Interfaces:**
- Produces `fusion_scores(density_z, shape_z) -> dict[str, np.ndarray]` with `score_mean`, `score_or_max`, and `score_positive_sum`.
- Extends `score_windows(..., primary_fusion="or_max")` while keeping legacy columns available.

- [ ] Write failing tests for OR-Max, Positive-Sum, negative-branch non-cancellation, paired replicate fusion, and OR-controlled ranking.
- [ ] Run focused tests and confirm the old linear-only implementation fails.
- [ ] Implement all three fixed rules and compatibility aliases; do not add tunable weights.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: add fixed or fusion for task2 candidates`.

### Task 3: Coverage-matched baselines and detector diagnostics

**Files:**
- Create: `src/task2/refined_evaluation.py`
- Create: `tests/test_task2_refined_evaluation.py`

**Interfaces:**
- Produces `union_coverage_bp(intervals) -> int`.
- Produces `sample_coverage_matched(windows, target_bp, rng, tolerance=0.02) -> (DataFrame, int, bool)`.
- Produces `detector_comparison(scores, structures, fractions=...) -> DataFrame`.
- Produces `random_baseline_comparison(scores, structures, detector_table, repeats, seed, tolerance) -> DataFrame`.
- Produces `branch_correlations(scores, background_ids, top_fraction=0.10) -> DataFrame`.

- [ ] Write failing tests for union coverage, tolerance-aware sampling, seed reproducibility, five-method comparison, random summary schema, and empty-subset correlation behavior.
- [ ] Run focused tests and observe missing-module failures.
- [ ] Implement deterministic evaluation using existing center-based recall helpers.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: add coverage-matched task2 evaluation`.

### Task 4: Refined reporting and immutable output routing

**Files:**
- Modify: `src/task2/reporting.py`
- Modify: `src/task2/pipeline.py`
- Modify: `scripts/run_task2a.py`
- Modify: `tests/test_task2_reporting.py`
- Modify: `tests/test_task2_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Adds CLI `--detector-version {legacy,refined}` with `legacy` default for compatibility.
- Refined Full outputs to `outputs/task2a_refined`; refined smoke outputs below `outputs/pipeline_runs/task2a_refined_smoke`.
- Produces `branch_correlation.csv`, `detector_comparison.csv`, `random_baseline_comparison.csv`, `candidate_regions_or_max.csv`, refined figures, manifest, and summary.

- [ ] Write failing tests for output isolation, legacy preservation, 120/15 Full settings, refined manifest fields, correlation/scatter figures, two recall figures, and checkpoint independence from recall.
- [ ] Run focused tests and confirm versioned behavior is absent.
- [ ] Implement refined pipeline path while leaving the legacy path operational and old files untouched.
- [ ] Add exact CLI/README commands and strengthened reuse validation.
- [ ] Run focused and full tests; expect all pass.
- [ ] Commit as `feat: add versioned task2a refined pipeline`.

### Task 5: Smoke, Full, and evidence-based readiness report

**Files:**
- Modify only files from Tasks 1–4 if an observed integration failure receives a RED regression test.

**Interfaces:**
- Consumes validated `data/task2` artifacts and immutable `outputs/task2a` legacy evidence.
- Produces verified refined smoke and Full artifacts and the final Task 2A readiness decision.

- [ ] Hash raw inputs, Task 1 protected artifacts, legacy Full outputs, and reusable Task 2 data.
- [ ] Run compile and full pytest; expect success.
- [ ] Run `python scripts/run_task2a.py --mode smoke --detector-version refined`; expect complete isolated pipeline output.
- [ ] Validate refined smoke manifest, required schemas, finite AE inputs, and output isolation.
- [ ] Run `python scripts/run_task2a.py --mode full --detector-version refined --resume`; expect reuse validation followed by new AE/scoring/evaluation/reporting.
- [ ] Inspect refined Full metrics and ensure the summary answers all 18 requested questions without biological interpretation.
- [ ] Recompute protected hashes; expect exact equality.
- [ ] Run final compile, both pytest entry points, refined Full dry-run, manifest validation, and `git diff --check`.
- [ ] Commit integration fixes as `fix: harden refined task2a full run` only if RED tests require them.

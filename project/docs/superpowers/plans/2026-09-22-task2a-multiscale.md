# Task 2A.3 Multi-scale Candidate Detector Implementation Plan

> **For agentic workers:** execute inline with strict TDD and fresh verification. The user explicitly approved design through Full and requested no intermediate confirmation.

**Goal:** Build and execute the pre-registered three-scale Task 2A detector without changing Task 2A.2 scoring science or touching historical outputs.

**Architecture:** Focused `multiscale.py` owns scale geometry, scan identities, master splitting, leakage, fusion/NMS and audits. `multiscale_pipeline.py` orchestrates reusable primitives sequentially. `multiscale_reporting.py` produces the 30-question report. Existing CLI/config get minimal dispatch additions.

**Tech Stack:** Python 3.12, NumPy memmaps, pandas, SciPy, PyTorch, cooler, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-task2a-multiscale-design.md`

## Global constraints

- Scales are exactly small 3200/400, medium 6400/800, large 12800/1600 at target bin 100.
- Seed 20260920; Full 120 epochs/patience15; Smoke 3/patience2; random repeats100.
- Strict interval background exclusion with margin0 and one shared master block assignment.
- Each scale independently trains/calibrates; fusion is unweighted maximum after calibration.
- Formal regions are peak windows; coverage budgets are 5/10/20/30%; linear coordinates.
- Protocol/code/split are frozen before Full training and known recall; no post-result tuning.
- Never overwrite Task2A historical outputs or start Task2B.

## Review focus

- Same master block must prevent coordinate overlap even when scale windows differ.
- Medium cache identity conversion must not misalign replicate array indices.
- Expected slicing must be exactly 32/64/128 from the same length-128 vectors.
- Cross-scale NMS uses strict inequality and larger half-window; support never expands formal region.
- Random baselines and rescue analyses must use actual selected region unions at equal coverage.

### Task 1: Multiscale geometry, strict background, and leakage

**Files:** Create `src/task2/multiscale.py`; create `tests/test_task2_multiscale.py`.

- [x] Write failing tests for fixed configs, counts/coordinates, scale IDs, expected slices, interval exclusion, master assignments and cross-scale leakage.
- [x] Run targeted tests and confirm missing interfaces fail.
- [x] Implement minimal geometry/scan metadata/background/split functions.
- [x] Run targeted tests green.

### Task 2: Cross-scale regions, audits, rescue, readiness

**Files:** Extend `src/task2/multiscale.py`; extend `tests/test_task2_multiscale.py`.

- [x] Write failing tests for independent calibration marker validation, max fusion, NMS boundary, peak intervals, coverage, scale audit, rescue, random/readiness and no Task2B.
- [x] Run RED, implement, run GREEN.

### Task 3: Pipeline, report, and CLI

**Files:** Create `src/task2/multiscale_pipeline.py`, `src/task2/multiscale_reporting.py`; modify `src/task2/pipeline.py`, `scripts/run_task2a.py`, `README.md`; extend tests.

- [x] Write failing tests for paths, CLI, dry-run, output isolation, protected hashes, variable AE inputs and required report outputs.
- [x] Implement sequential scale orchestration and report/manifest generation.
- [x] Run targeted tests and the full historical suite.

### Task 4: Execution and audit

- [x] Run Full dry-run and verify no writes/historical changes.
- [x] Run Smoke; repair engineering defects only via RED-to-GREEN tests.
- [x] Freeze and run Full without result-driven tuning.
- [x] Validate manifests/hashes/required files/UTF-8 and run both pytest entry points.
- [x] Perform independent read-only review; fix Critical/Important engineering findings with regression tests only.
- [x] Report all requested results and stop at Task 2A.

## Execution record

- User supplied and pre-approved the entire design and execution sequence on 2026-09-22; no additional design gate is requested.
- Starting tree contains prior Task2A.2 uncommitted implementation/data plus unrelated `project.zip`; preserve all.
- Smoke completed after a deterministic shared-block feasibility repair; all 22 multiscale tests passed.
- Independent review required atomic scan manifests, per-structure rescue identities, and a persisted leakage audit; all were added with regression tests before Full.
- Full completed on 2026-09-23 with 1,850 candidate regions and `ready_for_task2b=false`; Task2B was not executed.
- Final verification: both pytest entry points passed 185 tests; manifest, scan SHA, frozen code SHA, 30-section summary, and 168 protected historical files validated.

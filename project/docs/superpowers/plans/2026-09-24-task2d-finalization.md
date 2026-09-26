# Task2D and Task2 Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Complete cross-replicate reproducibility validation and formally freeze the entire Task2 without altering Task2A/B/C or forcing a novel finding.

**Architecture:** A read-only input gate feeds a frozen protocol, one shared matrix measurement engine, postfit control and candidate analysis, a Task2D orchestrator, and a separate finalizer. Smoke is isolated; Full alone drives final Task2 tables and completion.

**Tech Stack:** Python, pandas, NumPy, SciPy, Matplotlib, Cooler, pytest, existing manifest utilities.

**Spec:** `docs/superpowers/specs/2026-09-24-task2d-finalization-design.md` and the complete pasted Task2D request.

## Global Constraints

- Do not edit Task2A/B/C files, train, rescan, tune PCA/HDBSCAN/proposals, or run Task3.
- Primary mask `i<j and |i-j|>2`; canonical 6400 bp only supplemental for candidates; controls canonical 6400 bp.
- Floors = valid Known canonical 5th percentiles; negative/nonfinite floors stop binary filtering. Strict cluster fraction >=0.8 among valid members.
- All outputs contain continuous metrics and provenance; empty strict lists are valid.
- Work in the specified saved project because frozen upstream artifacts and dirty user state are present locally; no commit/branch changes to user state.

## Review Focus

1. A candidate whose Task2A region spans scales must use its frozen representative scale/center, not the merged interval width. Test primary geometry from real candidate metadata.
2. Invalid expected distances and constant O/E vectors must stay invalid, never silently become zero correlation. Test masked pixel behavior and reason codes.
3. Known floor calibration must not see candidate values; test control-only floors and negative-floor failure.
4. Task2C noise and zero-overlap clustered candidates must survive selection even with zero strict novel clusters. Test pool/status precedence.
5. Resume/final verification must catch changed upstream hashes and missing output figures. Test hash validation and completion checks.

---

### Task 1: Read-only input gate and frozen protocol

**Files:** Create `src/task2d/inputs.py`, `src/task2d/__init__.py`, `tests/test_task2d_pipeline.py`.

**Interfaces:** `load_inputs(root) -> Task2DInputs`; `freeze_protocol(path, payload)`; `Task2DPaths(root, mode)`.

- [ ] Write tests constructing malformed alignments: candidate IDs reordered, duplicated Known IDs, missing completion, and a modified hashed upstream file must raise; smoke/full output roots differ. Run targeted pytest and observe RED.
- [ ] Implement manifest/completion verification using `validate_task2a_outputs`, metadata joins, exact IDs, source SHA256 snapshot, and immutable JSON freeze. Run targeted pytest and observe GREEN.
- [ ] Inspect actual source hashes and write Full `frozen_task2d_protocol.json` before any Full correlation. Include 5th-percentile, 80%, mask and all input hashes.

### Task 2: Shared matrix reproducibility engine

**Files:** Create `src/task2d/reproducibility.py`, `tests/test_task2d_reproducibility.py`.

**Interfaces:** `measure_pair(raw1, raw2, expected1, expected2) -> dict`; `read_pair(cool_paths, chrom, center, window_bp) -> tuple`.

- [ ] Write literal-array RED tests for shared center, 32/64/128 shape, per-replicate expected, upper mask and gap>2 mask, known Pearson/Spearman values, finite/constant failure, zero-axis warning, NRMSE/cosine, full-upper auxiliary fields.
- [ ] Implement raw O/E and pairwise finite masks, SciPy correlations, warning/status fields and one shared read call per replicate. Run targeted tests GREEN.

### Task 3: Controls, pools, threshold and final statuses

**Files:** Create `src/task2d/analysis.py`, extend `tests/test_task2d_reproducibility.py`.

**Interfaces:** `calibrate_known_floor(known) -> dict`; `annotate_candidates(candidate, summary, floors) -> DataFrame`; `cluster_reproducibility(candidate, summary) -> DataFrame`.

- [ ] Write RED tests with hand-calculated 5th percentiles, negative floor stop, pass AND logic, dynamic noise/clustered pools, Known-overlap precedence, 80% valid-member cluster gate, strict empty header, cluster/noise counts.
- [ ] Implement post-measurement annotations without modifying original labels or using candidates in floors. Run targeted tests GREEN.

### Task 4: Task2D pipeline and reproducible visual artifacts

**Files:** Create `src/task2d/reporting.py`, `src/task2d/pipeline.py`, `scripts/run_task2d.py`, `tests/test_task2d_pipeline.py`.

**Interfaces:** `execute_task2d(root, mode, dry_run, resume) -> Task2DResult`; CSV/plot writers; manifest.

- [ ] Write RED synthetic pipeline tests asserting same coordinates for both reps, all candidate/control rows, no upstream writes, Smoke/Full isolation, dry-run no correlation, and outputs/figures after successful run. Mock only Cooler read boundary.
- [ ] Implement control-first calibration, candidate measurements, CSV diagnostics, distribution figures, completion, summary, manifest, CLI. Run targeted tests GREEN.

### Task 5: Final candidate tables, common-scale heatmaps and Task2 closure

**Files:** Create `src/task2d/finalization.py`, `scripts/finalize_task2.py`, `tests/test_task2_finalization.py`; update `README.md`.

**Interfaces:** `finalize_task2(root, verify=True) -> Path`.

- [ ] Write RED tests for empty strict CSV with headers, nonempty follow-up rank, global 99.5% display scale, one heatmap per follow-up, 80% strict gate, all completion checks, `task2_complete=true` with `strict_novel_structure_detected=false`, altered upstream hash rejection.
- [ ] Implement output tables, shared O/E display scale, gallery/distribution figures, unified A/B/C/D summary, manifest and final completion. Run targeted tests GREEN.

### Task 6: Real-data acceptance

**Files:** Only Task2D/Task2-final generated outputs; no upstream edits.

- [ ] Run `python -m pytest -q`; repair any failure with a reproducing test and rerun.
- [ ] Run `python scripts/run_task2d.py --mode full --dry-run`, then Smoke, then Full; inspect controls and calibrated floors without changing protocol.
- [ ] Run `python scripts/finalize_task2.py --verify`; independently validate all manifests, expected/cooler hashes, completion and figure readability. Verify strict list may be empty and Task3 was not run.

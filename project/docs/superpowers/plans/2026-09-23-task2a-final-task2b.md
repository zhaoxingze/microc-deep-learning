# Task2A Final and Task2B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze six-channel Task2A proposals and complete Task2B representation, evaluation, Smoke and Full without entering Task2C.

**Architecture:** A pure proposal module feeds a finalization pipeline. A separate Task2B feature extractor consumes frozen candidate/reference centers and existing checkpoints; an evaluation module handles blocked probes, PCA, UMAP and consistency; the CLI orchestrates dry-run, smoke, full and manifest validation.

**Tech Stack:** Python 3.12, pandas/numpy, PyTorch, scikit-learn, scipy, cooler, matplotlib, umap-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-task2a-final-task2b-design.md`

## Global Constraints

- Proposal channels are exactly small/medium/large × density/shape, threshold paired empirical `p <= 0.05`.
- No new scales, AE training, Genome E(d) fitting, or Task2A detector optimization.
- Known labels cannot affect proposal threshold, ranking, NMS, merge or inclusion.
- Task2B fusion order is CNN, AE-small, AE-medium, AE-large; scaler fit is unlabeled Known+Background only.
- Seed 20260920; blocked probe CV; UMAP visualization only; no clustering or Task2C.
- Preserve prior Task1 and Task2A data/output files; write UTF-8 BOM CSV/Markdown and hashed manifests.
- Current workspace is required because the prior Task2A.3 code/data/output tree includes uncommitted user work. No branch switch or commit is needed for this authorized implementation; verify protected inputs by hash.

## Review Focus

- Paired p-value interpretation: geometric mean of existing replicate p-values must exactly match mean replicate anomaly and never read known labels.
- Transitive interval merge: overlapping A-B-C must form one component even if A and C do not overlap.
- Boundary matrices: padded candidate/reference windows must be handled explicitly and consistently at 3.2/6.4/12.8 kb.
- Probe leakage: grouped folds must have no shared 128 kb group and report prior CNN supervision as a limitation.
- Resume/integrity: stale or partial outputs cannot be silently accepted; source/checkpoint hashes and alignment must be checked.

---

### Task 1: Freeze independent Task2A proposals

**Files:** Create `src/task2/final_proposals.py`, `src/task2/final_pipeline.py`, `tests/test_task2a_final_proposals.py`.

**Interfaces:** `generate_channel_proposals(scores, threshold=0.05) -> DataFrame`; `merge_proposals(proposals) -> DataFrame`; `execute_task2a_final(project_root) -> FinalResult`.

- [ ] Step 1: Write tests for six-channel threshold and channel-local NMS, cross-scale non-suppression, transitive merge, deterministic representative tie-break, label-independent inclusion, fixed threshold, and recall-independent engineering gate.
- [ ] Step 2: Run `python -m pytest -q tests/test_task2a_final_proposals.py` and verify failure from missing implementation.
- [ ] Step 3: Implement the pure functions, then the finalization pipeline with frozen protocol written before final recall, annotation, recall, coverage, finalization JSON and hashed manifest.
- [ ] Step 4: Run targeted tests until green, then execute Task2A Final on existing Full scores. Validate required files, hashes, nonempty candidates and immutable historical outputs.

### Task 2: Extract frozen CNN/AE features

**Files:** Modify `src/task1/model.py`; create `src/task2b/features.py`, `tests/test_task2b_features.py`.

**Interfaces:** `SmallMicroCCNN.forward_features(x) -> Tensor`; `load_representation_models(paths) -> LoadedModels`; `extract_embeddings(centers, models, preprocessors) -> EmbeddingBlocks`.

- [ ] Step 1: Write tests proving CNN logits unchanged, hidden dimension from actual tensor, adaptive AE pooling fixed length for 32/64/128 inputs, selected Task1 input preprocessing, identical candidate/reference extraction, replicate averaging and cosine behavior.
- [ ] Step 2: Run `python -m pytest -q tests/test_task2b_features.py` and verify expected missing-interface failures.
- [ ] Step 3: Implement minimal model interface and batched feature extraction using existing checkpoints and preprocessing; keep model inference in eval/no-grad mode.
- [ ] Step 4: Run targeted tests and existing Task1 model tests.

### Task 3: Build references and representation evaluation

**Files:** Create `src/task2b/evaluation.py`, `tests/test_task2b_pipeline.py`.

**Interfaces:** `select_background_reference(...) -> DataFrame`; `fit_reference_scalers(...)`; `evaluate_blocked_probes(...) -> DataFrame`; `fit_visualizations(...)`.

- [ ] Step 1: Write tests for 344 blocked strict-background references, unlabeled reference-only scaler fit, zero group intersection in each probe fold, probe metric schema, deterministic PCA/UMAP seed, and no clustering calls.
- [ ] Step 2: Run targeted tests and verify RED.
- [ ] Step 3: Implement reference sampling, block scalers, GroupKFold probes, PCA and UMAP plotting, and similarity diagnostics.
- [ ] Step 4: Run targeted tests until green.

### Task 4: Orchestrate and report Task2B

**Files:** Create `src/task2b/pipeline.py`, `src/task2b/reporting.py`, `scripts/run_task2b.py`; modify `requirements.txt`, `README.md`; extend `tests/test_task2b_pipeline.py`.

**Interfaces:** `execute_task2b(mode, dry_run=False, resume=False, project_root=...) -> Task2BResult`.

- [ ] Step 1: Write tests for dry-run no mutations, Smoke/Full output isolation, candidate/reference metadata-array alignment, source hashes in manifest, resume validation, and Task2C not executed.
- [ ] Step 2: Run targeted tests and verify RED.
- [ ] Step 3: Implement orchestration, manifests and the 26-question summary. Add `umap-learn` compatible dependency range.
- [ ] Step 4: Run `python -m pytest -q`; fix failures.
- [ ] Step 5: Run `python scripts/run_task2b.py --mode full --dry-run`.
- [ ] Step 6: Run `python scripts/run_task2b.py --mode smoke`; verify smoke is explicitly non-scientific.
- [ ] Step 7: Run `python scripts/run_task2b.py --mode full`; verify all output hashes, shapes, metrics, and `task2b_complete` without Task2C.

### Task 5: Final audit

**Files:** Update plan execution record and reports only if findings require it.

- [ ] Step 1: Run `python -m pytest -q` after final code changes and inspect the full result.
- [ ] Step 2: Recheck Task2A frozen protocol, source hashes, final recall, Task2B manifest, Smoke/Full separation, and representative metadata alignment.
- [ ] Step 3: Review the implementation against the spec and disclose all scientific limitations, especially Task1 CNN supervision reuse.

## Execution record

- 2026-09-23: User supplied and approved the complete Task2A Final + Task2B method and explicitly required immediate implementation; extra design/plan approval would conflict with that instruction.

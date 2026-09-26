# Task 2A.2 calibrated detector implementation plan

> Native execution in the user's explicitly requested local project. Approval includes implementation, tests, smoke and Full; no further confirmation gate.

**Goal:** An independently calibrated, region-first Task 2A detector with honest coverage evaluation; stop at Task 2A regardless of readiness.

**Architecture:** Reuse scanner, expected vectors, shape preprocessing and AE. Add calibration, region construction, evaluation and a separate orchestration module dispatched by the existing CLI. Preserve legacy/refined output trees and original five data artifacts byte-for-byte.

**Tech stack:** Python, NumPy, pandas, scipy, PyTorch, matplotlib, pytest.

**Spec:** User attachment `40d6fdc4-e507-472a-bcd5-98a0182de40b/已粘贴的文本.txt`, read in full on 2026-09-22.

## Frozen scientific protocol (before any new Full recall)

- Linear chromosome, 6400 bp windows, step 800, 100 bp bins. Reuse validated original full scan, including smoke's first 100 kb subset; smoke does not write the original arrays.
- Whole background blocks, 128000 bp Full / 25600 smoke. Drop boundary-crossing windows. Seed 20260920, shuffle sorted eligible blocks once; nearest block counts to 70/15/15, clamp to nonempty splits. Known exclusion unchanged. Three-way pairwise coordinate audit mandatory.
- Train only sets clips (99.5 percentile) and AE gradient. Validation only selects checkpoint/early stop. Calibration only fits tail distributions. AE architecture/training loss/shape safeguards unchanged, Full 120 epochs/patience15.
- Per replicate and branch: inclusive upper tail p=(1+count(cal>=x))/(n+1); A=-log10(p). Average A over two replicates separately for each branch, then OR=max(paired density,paired shape). Optional Bonferroni is per-replicate only, not claimed as paired calibrated inference.
- Deterministic tie order: descending score, SHA256(seed|window_id) ascending; independent of labels and coordinates (avoids left-to-right saturation bias). Region ties use the peak's same tie key.
- NMS distance strictly <3200 bp, same chromosome; assign a suppressed window to the first (highest-ranked) nearby accepted peak. Retain only the contiguous 800-bp center component containing that peak as members; record discarded noncontiguous support. Region is min(start),max(end) of members, no transitive expansion between peaks. Regions may overlap; union counts once.
- Mixed branch iff both paired anomalies >= -log10(.05); otherwise larger branch, ties density. This tag never affects ranking.
- Coverage selection: descending whole regions until union >= requested budget; report actual and overshoot, never trim. Primary budgets5/10/20/30%, curve each1% through40%. Primary recall known-center in half-open union; auxiliary interval overlap.
- Coverage random: existing scan-window sampling with actual selected-region union target, tolerance2%,100 repeats; cache segment index. Region random: independent uniform valid integer linear starts, same chromosome and exact region-length multiset,100 repeats; report resulting union coverage. Seed fixed, summaries population std and empirical 2.5/97.5 quantiles (randomization reference intervals, not estimator confidence intervals).
- Readiness: Overall10 and20 each strictly above coverage-random97.5%; all three classes20 positive and >= respective region-random2.5%; zero-axis enrichment<=1 among windows whose centers are in selected20 union vs all scan windows; paired-replicate OR Spearman>0; abs(paired density/shape Spearman)<.9; train/val/cal coordinate-disjoint. All checks required. Smoke never authorizes Task2B. No automatic Task2B call in any case.
- Known audit best window is the highest ranked interval-overlapping window; all primary best-score fields belong to that same window. Also save independent branch maxima and containing-region diagnostics. Missed score medians reported separately for every coverage budget.
- No new Full recall is examined until code and protocol snapshot hashes are written. No scientific rule changes after evaluation.

## Review focus

1. Inclusive ties/saturated tail and deterministic label-independent NMS ordering.
2. Corrupt/stale hashes, missing rows, duplicates and mismatched replicate indices must reject cache.
3. Block boundary leakage and insufficient eligible blocks must fail explicitly.
4. Empty selections, absent smoke classes, interval boundary equality and random coverage mismatch remain explicit.
5. Region support cannot chain beyond the peak radius; random lengths must be identical, not merely similar means.

## Implementation steps and verification

- [x] Add failing tiny-data tests for three-way split, tails and aggregation; implement `background.py` additions and `calibration.py`; run targeted pytest.
- [x] Add failing region/coverage/random/audit/readiness tests; implement `region_detection.py` and `calibrated_evaluation.py`; run targeted pytest.
- [x] Add CLI/cache/output-isolation tests; implement `calibrated_pipeline.py`, `calibrated_reporting.py` and minimal existing CLI dispatch; run complete `python -m pytest -q`.
- [x] Run Full dry-run, smoke, repair functional failures via regression tests; verify protocol snapshot and reports.
- [x] Run exactly one valid Full scientific experiment with frozen rules, then hash-audit historical files and validate all outputs.
- [x] Review implementation and report every readiness outcome, limitations, test results and output paths. No scientific tuning and no Task2B.

## Execution record

- Initial state: main, only untracked `project/data/task2/`; existing code left intact except scoped additions/dispatch. No remote writes or commits requested this turn.
- First full test suite: 158 passed; initial smoke completed 118 windows/24 regions, 3 epochs, all outputs validated.
- Independent read-only review before Full: no Critical; Important coverage-validity gap reproduced and fixed with three RED-to-GREEN regression cases. Readiness's existing 10/20 coverage checks also require reached budgets and >=100 all-matched random repeats; unmatched draws remain in diagnostics and cannot certify readiness.
- Ruling: preserve approved inherited center exclusion. Actual interval audit finds 3 Train / 7 Validation / 0 Calibration partial known interval overlaps. Document as a limitation, never silently revise geometry. Added fail-fast actual-interval Calibration check (no resplit) with regression test. Cost/limitation: ordinary-background Train/Val interpretation is imperfect for long structures; this does not create Train/Val/Cal coordinate leakage.
- Review minors resolved in same pre-Full fix pass: report configured repeat count; preserve Z-score generic fields under explicit legacy names and expose calibrated paired diagnostics. Regression added for score semantics.
- Review did not judge future Full recall, biology, optimal thresholds, or dependence assumptions; those remain diagnostic/report limitations, not authority to tune.
- Full process interruption note: the desktop model switch terminated the first deterministic process at epoch 30, before calibration or known evaluation. A fresh run with the same frozen protocol reproduced epochs 1-30 exactly, completed all 120 epochs, and is the only valid Full scientific result.
- Final Full: AE best epoch 119, validation loss 0.345369810; 5795 windows and 1224 ranked regions. Manifest validation passed; both `python -m pytest -q` and direct `pytest -q` passed 163 tests. Historical/output hash audit passed for 142 protected files. Readiness false only because calibrated Overall recall did not exceed coverage-random 97.5% at either 10% or 20%; no Task2B executed.

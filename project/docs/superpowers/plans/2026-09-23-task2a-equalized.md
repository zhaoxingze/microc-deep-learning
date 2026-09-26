# Task2A.4 equalized calibration implementation plan

> Execution ruling: use the current workspace because the requested immutable Task2A.3 outputs are uncommitted inputs in this workspace. A separate worktree would not contain them. The cost is weaker filesystem isolation, mitigated by hashing every protected Task2A.3 data/output file before and after the run.

1. Add failing unit tests for common p-floor arithmetic, branch/replicate aggregation, validation, all ranking tie levels, unchanged NMS boundary, contribution counts, and the exact seven readiness checks.
2. Implement `src/task2/equalized.py` as pure deterministic post-processing functions and make targeted tests pass.
3. Add failing orchestration tests proving the pipeline consumes only existing Task2A.3 files, emits the required isolated artifacts, and never executes Task2B.
4. Implement `src/task2/equalized_pipeline.py` and `scripts/run_task2a_equalized.py`; reuse the established coverage/random/audit helpers without importing scan, expected, or AutoEncoder code.
5. Run targeted tests, both full pytest entrypoints, then the real Task2A.4 Full post-processing run.
6. Validate the manifest, protected hashes, score ceiling, scale contribution, random baselines, and each readiness item. Review the implementation and fix material findings.
7. Report the requested 17 values and stop without Task2B.

## Execution record

- 2026-09-23: Plan accepted directly from the user's complete Task2A.4 specification; no additional design gate required.
- 2026-09-23: Verified Task2A.3 inputs: calibration counts 1054/431/147 and original region peaks 1589/185/76.

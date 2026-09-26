# Task2A.4 cross-scale calibration equalization design

## Scope

This is a deterministic post-processing repair over the completed Task2A.3 Full run. It reads only the three existing per-scale `window_scores.csv` files and existing audit/evaluation metadata. It does not scan `.cool` files, recompute Genome E(d), train or load an AutoEncoder for inference, add a scale, tune a threshold, or use known labels in ranking.

## Frozen calibration rule

- Calibration sizes come from the Task2A.3 `scale_summary.csv` and must match the score tables' three scales.
- `common_p_floor = 1 / (min(calibration_n) + 1)`. For the completed run this is `1/148`.
- Every replicate/branch empirical p-value is replaced by `max(p_original, common_p_floor)`.
- Common anomaly is `-log10(p_common)`, so every scale has the same maximum anomaly.
- Replicate common OR is the maximum of density and shape common anomalies.
- Paired branch common anomaly preserves Task2A.3 aggregation: the mean of rep1 and rep2 common anomalies.
- Equalized scale score is the maximum of paired density and paired shape common anomalies.
- `replicate_min_score` is the minimum of the two replicate common OR scores.
- `original_uncapped_tail_score` is the prior Task2A.3 `scale_or_score`.

## Frozen ranking and regions

Ranking is descending by equalized scale score, then descending by replicate minimum score, then descending by original uncapped tail score. Remaining exact ties are ordered by chromosome, start, end, and stable window ID. Known labels do not participate.

Cross-scale NMS and formal region geometry are unchanged from Task2A.3: a window is suppressed when its center is strictly closer than half the larger of the two window sizes to an already selected peak on the same chromosome. The formal candidate region is the selected peak window. Coverage is the union of formal regions.

## Evaluation and gate

The frozen coverage budgets are 5%, 10%, 20%, and 30%. Coverage-matched and region-matched random baselines retain 100 repeats and seed 20260920. The final gate contains exactly:

1. 10% overall recall exceeds coverage-matched random 97.5 percentile.
2. 20% overall recall exceeds coverage-matched random 97.5 percentile.
3. CHIN, OPCID, and CHID recall are all positive at 20% coverage.
4. Zero-axis enrichment is at most one.
5. Rep1/rep2 Spearman is positive.
6. Existing split-leakage audit is clean.
7. Existing background-known-overlap audit is clean.

The region-matched random baseline remains an output but is not an extra readiness condition. The pipeline always records `task2b_executed=false` and stops after Task2A.4.

## Integrity and outputs

All files under `outputs/task2a_multiscale` and `data/task2_multiscale` are hashed before and after execution. Any change aborts the run. Outputs are isolated under `outputs/task2a_multiscale_equalized`, written as UTF-8 CSV/JSON/Markdown, and covered by a SHA256 manifest.

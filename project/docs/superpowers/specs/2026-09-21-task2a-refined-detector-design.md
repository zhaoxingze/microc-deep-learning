# Task 2A.1 Refined Candidate Detector Design

## Scope and frozen interfaces

This change refines Task 2A only. It does not perform PCA, UMAP, HDBSCAN,
clustering, latent-space analysis, biological interpretation, or any Task 2B–2D
work. The existing 6,400 bp / 800 bp / 100 bp aligned window scan, genome-wide
Expected(d), known-structure exclusion, blocked background split, center-based
known recall, zero-axis detection, region-merging principle, logging, and
manifest framework remain unchanged.

The legacy Full results under `outputs/task2a/` are immutable inputs to the
comparison. Refined Full results are written to `outputs/task2a_refined/`.
Refined smoke results are isolated under
`outputs/pipeline_runs/task2a_refined_smoke/`. The existing `data/task2/`
scan, expected vectors, and background split may be reused only after shape,
coordinate, replicate-order, source-path, parameter, and leakage validation.

## Refined shape representation

The density branch remains genome O/E mean over the upper triangle with
`i < j` and `|i-j| > 2`.

The AutoEncoder shape branch uses a fixed representation:

1. Compute genome-wide O/E from raw counts and the already-defined Expected(d).
2. Ignore non-finite values while fitting clipping. For transformation, map
   NaN and negative infinity to zero and positive infinity to the fitted clip.
3. Fit a separate 99.5th-percentile O/E clip for each replicate using only
   background-train windows and only valid upper-triangle pixels (`i < j`,
   `|i-j| > 2`). Known structures, validation windows, and candidate ranking do
   not participate.
4. Clip O/E to `[0, replicate_clip]` and apply `log1p`.
5. Within each window, estimate median and MAD from the same valid pixels and
   transform the whole 64×64 matrix as
   `(X - median_valid) / (1.4826 * MAD_valid + 1e-8)`.
6. Use unit scale when `MAD <= 1e-8`, then winsorize the normalized matrix to
   the fixed range `[-10, 10]` so sparse near-zero-MAD windows cannot dominate
   the MSE objective. This parameter is fixed before recall evaluation.
7. Replace any remaining non-finite value with zero.

The manifest names this mode `oe_log_robust`, records the percentile, source,
replicate clip values, normalization formula, and epsilon. The existing
convolutional AutoEncoder architecture and MSE objective remain unchanged.
Full training permits 120 epochs and patience 15; checkpoint selection uses
background validation reconstruction loss only.

## Fusion and paired scoring

For each replicate, density and off-diagonal reconstruction MSE are separately
converted to robust z-scores with parameters fitted only from background
train+validation windows, preserving the existing Task 2A reference rule.
Three fusion rules are fixed before evaluation:

- `MEAN = 0.5 * density_z + 0.5 * shape_z` (legacy comparison)
- `OR_MAX = max(density_z, shape_z)` (primary detector)
- `POSITIVE_SUM = max(0, density_z) + max(0, shape_z)` (auxiliary)

Each rule is applied to each replicate first, then rep1 and rep2 scores are
averaged to obtain the paired window score. No alpha/beta search or recall-based
tuning is allowed. The score table exposes per-replicate values and paired
`score_mean`, `score_or_max`, and `score_positive_sum`; OR-Max controls rank,
percentile, Top-candidate figures, zero-axis audit, and refined region merging.

## Evaluation and diagnostics

`detector_comparison.csv` evaluates Density-only, Shape-only, Legacy Mean,
OR-Max, and Positive-Sum at Top 1/2/5/10/20%. Each row records selected window
count, genomic interval union coverage, and overall/CHIN/OPCID/CHID recall.

The existing window-count random baseline remains. A required coverage-matched
baseline repeatedly samples available genomic windows until their interval
union is as close as possible to the detector's union coverage, accepting a
configured ±2% tolerance when attainable. It records achieved coverage and
success status rather than silently claiming an unattained match. Both
baselines use 100 repetitions and the fixed pipeline seed, and report mean,
standard deviation, 2.5th percentile, and 97.5th percentile. Region-matched
randomization is excluded from this refinement because it is optional and
would introduce a second spatial null model beyond the necessary repair.

`branch_correlation.csv` reports paired DensityZ versus ShapeZ Pearson and
Spearman correlations for all windows, background windows, known-center-overlap
windows, and the Top 10% OR-Max windows. `branch_scatter.png` visualizes these
paired values. Detector and random comparison curves are separate figures.

## Reporting and readiness

The refined manifest records preprocessing, fixed fusion rules, primary
detector, AE training outcome, score definitions, randomization controls,
branch correlations, and input/output hashes. The summary compares legacy and
refined Full evidence and explicitly answers all requested detector-quality
questions.

Readiness for Task 2B is evidence-based rather than an absolute recall cutoff.
The report checks whether OR-Max beats the coverage-matched 95% interval at a
useful candidate fraction, avoids near-zero class recall at Top 20%, is not
dominated by zero-axis windows, preserves positive replicate agreement, and
receives non-redundant information from the shape branch. Failure of this
combined check produces the exact conclusion `Task2A detector still needs
refinement.` No downstream task starts automatically.

## Validation

New tests cover fixed fusion, non-cancellation, O/E clipping and robust
normalization, zero-MAD handling, background-only fitting, interval union,
coverage sampling and reproducibility, branch correlations, detector tables,
versioned output isolation, manifest fields, and checkpoint independence from
known recall. All previous Task 1 and Task 2 tests remain green. Verification
runs pytest, refined smoke, then refined Full with validated reuse.

# Task 2A.3 Multi-scale Candidate Detector Design

## Intent and boundary

Test the pre-registered hypothesis that one 6.4 kb window is a scale mismatch for CHIN and CHID. The detector uses exactly three physical scales at the unchanged 100 bp bin resolution: small 3200/400, medium 6400/800, and large 12800/1600 bp window/step. Known labels are used only for strict background exclusion and final frozen evaluation. This task ends after Task 2A readiness reporting; it never starts Task 2B or tunes scales after recall is known.

## Reused science

Each scale retains the Task 2A.2 definitions: genome-wide E(d), Train-derived 99.5% O/E clipping, log1p and per-window robust normalization, the same convolutional autoencoder definition, Validation-only checkpoint choice, Calibration-only inclusive upper empirical tails with add-one smoothing, replicate-anomaly averaging, then max(Density, Shape). Expected is computed once per replicate through distance 127 and sliced to 32/64/128 values without changing its mathematics.

## Data and identities

All complete linear windows are stored as float32 `.npy` memmaps under `data/task2_multiscale/{small,medium,large}`. IDs are deterministic hashes of `scale|chrom|start|end`. Replicate metadata must have identical IDs, coordinates, and array indices. Full medium arrays may be reused from the SHA-validated Task 2A.2 cache; its metadata is rewritten with scale-aware IDs and an explicit source record. No dense whole-chromosome matrix is constructed.

## Background and leakage

Background eligibility is strict half-open interval non-overlap with every complete known interval, margin zero. A master list of genomic blocks is assigned with seed 20260920 to Train/Validation/Calibration near 70/15/15. The first seeded whole-block permutation for which every scale has nonempty Train/Validation/Calibration background is fixed; this feasibility condition uses no recall or class performance. A scale window may enter only if eligible and fully contained in one block; its split comes exclusively from that master assignment. Audits must show zero known overlap in every scale/split, zero same-scale split overlap, and zero cross-scale coordinate overlap. Any failure aborts Full without outcome-based resplitting.

## Scale and cross-scale detection

Each scale independently calibrates raw Density/Shape distributions and emits paired branch anomaly and scale OR. Single-scale evaluation uses its own scale-aware NMS and the peak window interval.

For MultiScale, all calibrated windows enter one pool and are sorted by score then the existing deterministic SHA tie key. A new candidate is suppressed when a higher-ranked peak on the same chromosome satisfies `abs(center difference) < max(window_bp_new, window_bp_peak)/2`. It is assigned to the highest-ranked qualifying peak. The formal candidate interval is exactly the peak `[start,end)`; support counts and support union are diagnostics only. MultiScale OR is therefore the maximum calibrated anomaly represented by the winning peak—no learned or hand-weighted scale combination.

## Evaluation and readiness

Small-only, Medium-only, Large-only, and MultiScale use identical 5/10/20/30% genomic union coverage budgets and known-center half-open recall. MultiScale receives 100 coverage-matched and 100 exact region-count/length-multiset random repetitions on the same linear chromosome domain. The structure audit records each scale's best interval-overlapping window and the first MultiScale budget selection; rescue summaries classify scale-specific center recall at 10/20/30%.

Readiness is frozen as: MultiScale Overall recall strictly exceeds coverage-random 97.5% at 10% and 20%; at 20% every class is positive and at least its region-random 2.5%; zero-axis enrichment among pooled windows whose centers fall in the selected 20% union is <=1; Spearman between Rep1/Rep2 scores at MultiScale peak windows is positive; all same/cross-scale split leakage audits pass; and every background known-overlap audit is zero. Invalid random matching fails closed. Every condition is required, and even success does not launch Task 2B.

## Isolation, reproducibility, and resources

Full writes only `data/task2_multiscale` and `outputs/task2a_multiscale`; Smoke writes beneath `outputs/pipeline_runs/task2a_multiscale_smoke`. Legacy, refined, and calibrated output/data hashes are captured before and verified after. `frozen_multiscale_protocol.json` records all rules, code hashes, splits, counts, environment, and seed before Full training/evaluation. Scales execute sequentially and release model/dataset objects with garbage collection before the next scale.

## Failure behavior

Schema, alignment, finite-value, expected-length, background purity, leakage, code-hash, output-hash, or random-validity failures abort explicitly. A scientifically unfavorable but technically valid run produces `ready_for_task2b=false`, complete diagnostics, and stops without changing the protocol.

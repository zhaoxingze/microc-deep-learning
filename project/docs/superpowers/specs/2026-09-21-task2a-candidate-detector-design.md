# Task 2A Genome-wide Candidate Detector Design

## Purpose and boundary

Task 2A builds a high-recall candidate-window detector for the WT Micro-C chromosome. It scans aligned rep1/rep2 windows, learns ordinary background with a convolutional autoencoder, combines reconstruction and contact-density anomaly scores, and evaluates whether ranked windows recover the 344 known CHIN/OPCID/CHID structures.

The deliverable stops at candidate windows, candidate regions, scores, and known-structure recall. It does not run UMAP, PCA clustering, K-means, HDBSCAN, Task 1 embeddings, or claim novel structures.

## Reused Task 1 contracts

- `src.data.load_microc.load_local_matrix` remains the only local Cooler reader and genomic aggregation implementation.
- `src.data.normalize.log1p_normalize` supplies the AE baseline input.
- `src.data.known_dataset.attach_structure_ids` supplies stable known-structure IDs.
- `src.task1.manifest.sha256_file` supplies hashes.
- `src.task1.pipeline.PipelineLogger` and `build_subprocess_environment` supply UTF-8 logging and deterministic CUDA subprocess configuration where applicable.
- Task 1 code and canonical outputs are read-only for Task 2A.

## Workspace and modes

Full outputs use `data/task2/` and `outputs/task2a/`. Smoke outputs use the isolated pseudo-root `outputs/pipeline_runs/task2a_smoke/`, including its own `data/task2/` and `outputs/task2a/`. Logs use `outputs/logs/task2a_TIMESTAMP.log`.

Smoke scans `[0, 100000)` and trains for 3 epochs with patience 2. Full scans the complete chromosome and uses 60 epochs with patience 10. Full is not required during implementation acceptance if its cost is materially larger than smoke.

## Window scanning

The baseline uses `window_bp=6400`, `step_bp=800`, and `target_bin_size=100`. Complete-window starts are `0, step_bp, ...` while `start + window_bp <= scan_end`; no scan window is padded. Each window is fetched independently with `load_local_matrix`, which aggregates 10 bp source contacts into a `1x64x64` target representation without image interpolation or chromosome-wide dense materialization.

`window_id` is `WIN_` plus the first 16 hexadecimal digits of SHA-256 over `chrom|start|end`. Both replicates use the identical coordinate plan and must have identical ordered IDs. Raw float32 arrays are written through NumPy open memmaps as:

- `genome_windows_rep1.npy`
- `genome_windows_rep2.npy`

Long metadata contains two rows per window with window ID, genomic coordinates, array index, replicate, zero-axis state, and source path. A scan manifest records shape and coordinate parameters.

For the 4,641,652 bp chromosome, the default complete-window plan contains 5,795 windows. Smoke contains 118 windows.

## Genome-wide expected

`compute_genome_expected` streams Cooler pixel chunks. Source bin IDs are mapped to global 100 bp target bins, counts are accumulated by target-bin distance for distances 0 through 63, and each sum is divided by the number of possible target-bin pairs at that distance. Missing pixel pairs therefore contribute zero through the denominator without constructing a dense chromosome matrix. The result is replicate-specific `E_rep(d)` on the 100 bp target scale.

`apply_expected` divides each local raw pixel by its replicate expected value at `|i-j|`, with zero output for zero expected. Expected vectors are saved in `genome_expected.npz` and are distinct from Task 1 local O/E.

## Background definition and leakage control

Each known structure excludes `[center - (window_bp/2 + margin), center + (window_bp/2 + margin))`, clipped to chromosome bounds. The default margin is 3,200 bp, so the default affected interval is center ±6,400 bp. A scan window is a background candidate only when it has no interval overlap with any exclusion interval.

Background train/validation split uses fixed 128,000 bp genomic blocks. Windows crossing a block boundary are excluded from AE training. Whole blocks are assigned by a seeded permutation to approximately 80% train and 20% validation, with at least one block per split. Consequently, no train window overlaps a validation window. A CSV records each window's eligibility, block, and split.

## Autoencoder

The baseline input is `log1p(raw)` for both biological replicates. Known recall is never used for architecture, optimizer, epoch, learning rate, or checkpoint selection.

Architecture:

- Conv 1→16, ReLU, MaxPool
- Conv 16→32, ReLU, MaxPool
- Conv 32→64, ReLU
- Conv 64→32, ReLU, bilinear upsample
- Conv 32→16, ReLU, bilinear upsample
- Conv 16→1 linear output

The output shape is `1x64x64`. Training uses Adam, learning rate `1e-3`, batch size 32, MSE loss, deterministic seed `20260920`, and early stopping solely on background validation MSE. The best checkpoint and train/validation loss curve are saved.

## Scores

The primary pixel mask is the upper triangle with `i < j` and `|i-j|>2`.

- `DensityScore_raw`: mean genome-wide O/E under that mask.
- `ShapeScore_raw`: mean squared AE reconstruction error under that mask.
- Auxiliary density fields: O/E mean, standard deviation, maximum, p95, and total under the same mask.

For each replicate and score family, robust scaling parameters come only from the union of background train and validation windows:

`z = (x - median) / (1.4826 * MAD + 1e-8)`.

The fixed baseline score is `0.5 * density_z + 0.5 * shape_z`. The weights are configurable but must be non-negative and sum to one. Paired score is the mean of rep1 and rep2 candidate scores. Score difference, minimum, maximum, Pearson, and Spearman are reporting fields only.

## Ranking, overlap, and regions

All windows are sorted by paired candidate score descending. Rank is one-based; percentile is `1 - (rank-1)/N`. Known overlap is annotated both by known center within the window (primary) and positive interval overlap (auxiliary).

`candidate_scores.csv` retains every window. `candidate_regions.csv` selects the top 10% by default and merges same-chromosome selected windows whose centers differ by no more than 3,200 bp. Region rows contain genomic span, peak score/window, member count, and known annotations. Regions are called candidates, never novel structures.

## Recall evaluation

Fractions are fixed at 1%, 2%, 5%, 10%, and 20%. Selection uses `ceil(N*fraction)` highest-scoring windows. A known structure is recalled when its center lies within at least one selected window. Outputs include overall and per-class recall for combined, density-only, and shape-only rankings.

Random baselines select the same number of windows without replacement for 100 seeded repeats and report mean, standard deviation, and empirical 2.5/97.5 percentiles. Known labels are used only after the AE checkpoint, scaling reference, and 0.5/0.5 score rule are locked.

## Visualization and artifact controls

The pipeline writes a recall curve; reconstruction examples for background, each available known class, and top anomalies; and paired heatmaps for the top 50 candidates. Missing class examples in smoke are reported rather than fabricated.

Every window records zero-axis status in both replicates. The summary reports the overall zero-axis fraction, top-fraction zero-axis fraction, enrichment ratio, and a warning when the top set is enriched.

## Manifest and summary

`task2a_manifest.json` records input hashes, coordinate parameters, expected definition, background exclusion/split, AE configuration, checkpoint criterion, fixed score definitions, robust statistics, seed, software versions, and output hashes.

`task2a_summary.md` answers the 18 requested methodological/result questions, states whether combined recall is better than the random baseline without retroactive tuning, describes limitations, and gives a conservative Task 2B readiness conclusion. Smoke summaries are explicitly non-scientific.

## Pipeline behavior

`python scripts/run_task2a.py --mode {smoke,full}` is the single entry point. `--dry-run` validates inputs and prints the plan without creating scientific artifacts. `--resume` skips only scan/expected/background stages after structural validation; AE training, scoring, recall evaluation, plots, manifest, and summary rerun.

Any exception fails fast, is written to the log, and prevents later stages. CLI errors identify the log path. No dependency is installed automatically.

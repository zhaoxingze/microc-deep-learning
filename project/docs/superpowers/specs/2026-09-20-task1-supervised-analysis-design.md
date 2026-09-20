# Micro-C Task 1 Supervised Analysis Design

## Objective

Complete Phase 1.5 and Phase 2 for the 344 known CHIN, OPCID, and CHID structures: paired WT rep1/rep2 data, leakage-free genomic-group splits, three fair CNN input experiments, independent structure-level test evaluation, replicate consistency, brightness baseline, Grad-CAM, zero-axis sensitivity, and reproducibility artifacts. Stop after Task 1; do not implement Task 2 discovery, AutoEncoder, UMAP, HDBSCAN, or whole-genome scanning.

## Existing Inputs

- Project root: `F:\Micro-C\project`
- Raw root: `F:\Micro-C\micro-c数据`
- WT candidates: `GSE272159_37C_rep1.mapq_30.10.cool` and `GSE272159_37C_rep2.mapq_30.10.cool`
- Structures: `data/processed/structures.csv`, 344 rows (CHIN 250, OPCID 68, CHID 26)
- Existing local fetch and genomic rebin: `src/data/load_microc.py`
- Existing normalization: `src/data/normalize.py`
- Existing rep1 arrays: raw, log1p, O/E, labels, and metadata
- Runtime: Python 3.12, PyTorch 2.14 CPU, SciPy 1.18, scikit-learn 1.9

## Stable Identity and Paired Data

`structure_id` is a deterministic SHA-256-derived identifier from normalized `type|chrom|start|end|center`, prefixed by the type. It is independent of row order and never uses UUIDs. Duplicate stable keys are a fatal error.

Rep1 and rep2 are rebuilt through the same reusable window-building function with `window_bp=6400`, source 10 bp bins, target 100 bp bins, and `balance=False`. Each replicate produces raw/log1p/O/E arrays in identical structure order. `paired_known_metadata.csv` has one row per structure-replicate pair and records structure ID, group ID, replicate, label, coordinates, array index, zero-axis counts, padding, availability, and source Cooler.

Missing replicate data is represented by `available=False`; no matrix is fabricated. The actual inputs are expected to contain all 344 structures in both replicates, and unexpected extraction failures are written as diagnostics and stop the primary paired build.

## Genomic Groups

Each structure window is `[center-3200, center+3200)`. Two windows overlap iff `a.start < b.end` and `b.start < a.end`; endpoint-only contact is not overlap. Groups are connected components of this interval-overlap graph, implemented by sorted interval merging with transitive closure. Group IDs are deterministic by chromosome and merged interval order.

All structures and both replicates in one `genomic_group_id` must share a split.

## MILP Split

The MILP assigns each of the 175 whole groups to exactly one of train, val, or test. Binary decision variables encode group assignment. Hard constraints enforce:

- exactly one split per group;
- every split contains at least one CHIN, OPCID, and CHID;
- no group can cross splits by construction.

The linear objective minimizes weighted absolute deviations between actual and target structure counts for total, CHIN, OPCID, and CHID at ratios 0.70/0.15/0.15. CHID deviations in val/test receive a modest higher objective weight as a soft minority-class protection, not an infeasibility-prone hard quota. Actual ratios and deviations are reported. A separate post-solve validator proves pairwise group intersections are empty and all structure IDs occur exactly once.

If `scipy.optimize.milp` is unavailable or fails, the script reports the reason and exits. It never silently switches to greedy splitting. A group-aware multi-restart greedy function may exist as an explicitly selected fallback, but is not invoked automatically.

## Training Data and Zero-Axis Policy

The main experiment keeps every available replicate. Train uses rep1 and rep2 as independent observations. Class weights are computed only from unique train structures: inverse frequency normalized to mean one.

`--train-replicates rep1` restricts training without changing the split. `--exclude-zero-axis` is used only for the post-selection sensitivity run. It marks a zero-axis replicate unavailable without modifying primary data. In sensitivity training, each replicate loss is weighted by `1 / available_replicates_for_structure`, so each structure has equal total contribution. In sensitivity validation/test, one available replicate is sufficient; structures with neither replicate are excluded and counts are reported.

## CNN and Fair Experiments

The model is a small three-block CNN:

1. Conv(1,32,3,pad=1), BatchNorm, ReLU, MaxPool
2. Conv(32,64,3,pad=1), BatchNorm, ReLU, MaxPool
3. Conv(64,128,3,pad=1), BatchNorm, ReLU
4. AdaptiveAvgPool(1), Flatten, Linear(128,64), ReLU, Dropout, Linear(64,3)

It returns logits without Softmax. Defaults are AdamW, learning rate 1e-3, batch size 32, 60 epochs, patience 10, and weighted cross-entropy. Before raw, log1p, and O/E experiments, the same seed is reset before constructing model, optimizer, scheduler (if any), and loaders.

Train loss is replicate-level weighted CE. Validation inference collects all available replicate logits, averages logits by `structure_id`, then computes both macro-F1 and CE loss from aggregated structure logits. Best epoch maximizes validation structure-level macro-F1; exact ties choose lower validation structure-level CE. The trainer accepts no test loader.

After all three checkpoints are locked, an independent evaluation function loads each checkpoint and evaluates test once. Best input type is chosen only from validation structure-level macro-F1, with validation loss as the tie-break. Test results never influence checkpoint, hyperparameters, or input selection.

## Metrics and Outputs

Primary metrics use one prediction per structure: accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1, confusion matrix, and row-normalized confusion matrix. Secondary files separately report rep1-only, rep2-only, and expanded replicate-level metrics.

Every experiment writes checkpoint, training history, training curve, primary classification report, confusion matrices, secondary metrics, predictions, and config. `input_comparison.csv` records all validation/test results and flags the validation-selected input.

## Replicate Consistency

For every structure, rep1 vs rep2 raw matrices are compared using Pearson and Spearman over:

- upper triangle including diagonal (`k=0`);
- upper triangle excluding the diagonal (`k=1`);
- entries with `|i-j|>2`.

The summary reports N, mean, median, standard deviation, Q1, and Q3 overall and by class.

## Brightness Baseline

Each raw replicate yields mean, standard deviation, maximum, total, median, and p95. A StandardScaler and multinomial Logistic Regression are fit only on train observations. Validation/test decision scores are averaged by structure exactly like CNN logits. This baseline is explanatory and never participates in CNN input selection.

## Grad-CAM

Grad-CAM targets the final Conv2d of the validation-selected CNN. For each class, it selects up to five correctly predicted test structures and up to three misclassified structures. Correct examples target the shared true/predicted class; incorrect examples target the predicted class. Each figure shows rep1 input/overlay and rep2 input/overlay; unavailable replicates remain blank.

For each replicate CAM, quantitative metrics include diagonal attention ratio for `|i-j|<=2`, zero-axis attention ratio from the raw-matrix zero row/column mask, and Pearson/Spearman CAM-input intensity correlation. Summaries are grouped by class and correctness. The narrative must not claim shape learning when attention is dominated by diagonal, zero-axis artifacts, or intensity.

## Sensitivity Analysis

Only after best input selection, retrain that input with zero-axis replicates excluded and structure-balanced replicate weights. Report actual train/val/test structure and replicate counts. Compare against the main best-input run without allowing sensitivity results to alter model selection.

## Reproducibility and Test Boundaries

`task1_manifest.json` records hashes for structures, split, coolers, checkpoints, parameters, MILP configuration, group/class counts, class weights, best epochs, validation and test metrics, versions, and device. CSV outputs use UTF-8 BOM.

Tests cover stable IDs, paired alignment, overlap transitivity, endpoint behavior, MILP integrity and classes, split reuse, train-only class weights, model shape, structure aggregation and loss, sensitivity weights, identical initialization, no-test trainer API, validation-only selectors, train-only scaler/model fit, and Grad-CAM shape.

## Completion Boundary

Task 1 is complete only when paired data, verified MILP split, all three CNN experiments, locked independent tests, metrics/plots/checkpoints, replicate consistency, brightness baseline, Grad-CAM with quantitative analysis, zero-axis sensitivity, summary, manifest, and full tests exist. Then stop without entering Task 2.

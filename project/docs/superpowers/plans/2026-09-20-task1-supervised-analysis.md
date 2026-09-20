# Micro-C Task 1 Supervised Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the complete leakage-free paired-replicate CHIN/OPCID/CHID supervised classification and interpretation pipeline.

**Architecture:** Reuse the existing local Cooler fetch and normalization functions, add deterministic paired-data and MILP split modules, then layer a testable task1 package for datasets, aggregation, CNN training, independent evaluation, diagnostics, and Grad-CAM. All validation/test decisions operate at structure level by averaging available replicate logits.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy MILP, PyTorch CPU, scikit-learn, matplotlib, seaborn, cooler, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-task1-supervised-analysis-design.md`

## Global Constraints

- Never modify raw files under `F:\Micro-C\micro-c数据`.
- Use only local genomic fetch; never materialize a full chromosome dense matrix.
- Genomic groups are indivisible; endpoint-only touching windows are not overlapping.
- Main split and all three CNN experiments use the same fixed `data/splits/group_split.csv`.
- Train uses rep1+rep2 by default; primary validation/test metrics average available replicate logits by structure.
- Class weights use unique train structures only; test data cannot affect any selection.
- Early stopping and input selection use validation structure-level Macro-F1, then lower validation structure CE loss.
- Main experiments retain zero-axis samples; exclusion is a post-selection sensitivity analysis.
- CSV outputs use UTF-8 BOM and all randomness is controlled by seed 20260920.
- This directory is not a Git repository. Replace commit steps with verified task checkpoints in the execution ledger; do not initialize Git without user authorization.
- Stop after Task 1; do not implement Task 2, AutoEncoder, UMAP, HDBSCAN, clustering, or genome-wide discovery.

## Review Focus

- A structure missing one replicate must aggregate the remaining logits and record the missing replicate; covered in Task 3 aggregation tests.
- A zero-axis sensitivity structure with two available replicates must contribute the same total train weight as one with a single replicate; covered in Task 3 weight tests.
- Two windows that only touch at an endpoint must be separate while transitive overlaps merge; covered in Task 2 group tests.
- A checkpoint/input selector must be invariant to arbitrary test metrics; covered in Task 4 selection tests.
- Grad-CAM must remain finite and 64×64 when a model input is all zeros; covered in Task 7 Grad-CAM tests.

---

### Task 1: Deterministic paired rep1/rep2 dataset

**Files:**
- Create: `src/data/known_dataset.py`
- Create: `scripts/build_paired_known_dataset.py`
- Create: `tests/test_task1_paired_dataset.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `load_local_matrix(cool_path, chrom, center, window_bp, target_bin_size, balance)`, `log1p_normalize(matrix)`, `compute_oe(matrix)`, `structures.csv`, two Cooler paths.
- Produces: `stable_structure_id(row) -> str`, `build_replicate(structures, cool_path, replicate, window_bp, target_bin_size) -> ReplicateBuild`, rep1/rep2 arrays, labels, per-replicate metadata, and `paired_known_metadata.csv`.

- [ ] **Step 1: Write failing stable-ID and paired-alignment tests**

```python
def test_stable_structure_id_ignores_row_order():
    row = {"type": "CHIN", "chrom": "MG1655", "start": 10, "end": 30, "center": 20}
    assert stable_structure_id(row) == stable_structure_id(dict(reversed(list(row.items()))))

def test_pair_metadata_aligns_same_structure_ids():
    paired = combine_replicate_metadata(rep1_fixture, rep2_fixture, groups_fixture)
    assert paired[paired.replicate == "rep1"].structure_id.tolist() == paired[paired.replicate == "rep2"].structure_id.tolist()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_task1_paired_dataset.py -q`
Expected: collection fails because `src.data.known_dataset` does not exist.

- [ ] **Step 3: Implement minimal reusable builder and CLI**

```python
def stable_structure_id(row: Mapping[str, object]) -> str:
    key = "|".join([str(row["type"]).upper(), str(row["chrom"]), str(int(row["start"])), str(int(row["end"])), _format_center(row["center"])])
    return f"{str(row['type']).upper()}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"

@dataclass(frozen=True)
class ReplicateBuild:
    arrays: dict[str, np.ndarray]
    labels: np.ndarray
    metadata: pd.DataFrame
```

Build both coolers through the same loop, fail on duplicate IDs/order/label mismatch, write float32 `(344,1,64,64)` arrays and int64 labels, and preserve all zero-axis/padding fields.

- [ ] **Step 4: Run focused and legacy suites**

Run: `python -m pytest tests/test_task1_paired_dataset.py tests/test_load_microc.py tests/test_normalize.py -q`
Expected: all tests pass.

- [ ] **Step 5: Build real paired data and verify hashes/shapes**

Run: `python scripts/build_paired_known_dataset.py --rep1-cool "F:\Micro-C\micro-c数据\GSE272159_37C_rep1.mapq_30.10.cool" --rep2-cool "F:\Micro-C\micro-c数据\GSE272159_37C_rep2.mapq_30.10.cool"`
Expected: both replicates report 344 samples, identical IDs/labels/order, and create paired metadata with 688 rows.

---

### Task 2: Connected genomic groups and MILP split

**Files:**
- Create: `src/task1/__init__.py`
- Create: `src/task1/splitting.py`
- Create: `scripts/build_group_split.py`
- Create: `tests/test_group_split.py`

**Interfaces:**
- Consumes: stable structure table/IDs from Task 1.
- Produces: `assign_genomic_groups(frame, window_bp=6400)`, `solve_group_milp(grouped, ratios, seed)`, `validate_group_split(split)`, `data/splits/group_split.csv`, `outputs/task1/split_summary.csv`.

- [ ] **Step 1: Write failing overlap and MILP tests**

```python
def test_transitive_overlap_merges_but_endpoint_touch_does_not():
    frame = fixture_centers([3200, 9599, 15999, 22400])
    grouped = assign_genomic_groups(frame, window_bp=6400)
    assert grouped.genomic_group_id.tolist() == ["group_0001", "group_0001", "group_0001", "group_0002"]

def test_milp_split_has_no_leakage_and_all_classes():
    split = solve_group_milp(group_fixture_with_all_classes(), (0.70, 0.15, 0.15), 20260920)
    validate_group_split(split)
    assert set(split.groupby("split").type.unique().map(set)) == {frozenset({"CHIN", "OPCID", "CHID"})}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_group_split.py -q`
Expected: collection fails because `src.task1.splitting` does not exist.

- [ ] **Step 3: Implement groups, MILP absolute-deviation objective, and independent validator**

```python
def assign_genomic_groups(frame: pd.DataFrame, window_bp: int = 6400) -> pd.DataFrame:
    ordered = frame.assign(window_start=frame.center-window_bp/2, window_end=frame.center+window_bp/2).sort_values(["chrom", "window_start", "window_end", "structure_id"])
    group_numbers, current_chrom, current_end, group_number = [], None, None, 0
    for row in ordered.itertuples():
        if row.chrom != current_chrom or current_end is None or row.window_start >= current_end:
            group_number += 1
            current_chrom, current_end = row.chrom, row.window_end
        else:
            current_end = max(current_end, row.window_end)
        group_numbers.append(f"group_{group_number:04d}")
    ordered["genomic_group_id"] = group_numbers
    return ordered.sort_index()

def solve_group_milp(frame: pd.DataFrame, ratios=(0.70, 0.15, 0.15), seed=20260920) -> pd.DataFrame:
    groups = summarize_groups(frame)
    c, integrality, bounds, constraints = build_assignment_problem(groups, ratios)
    result = scipy.optimize.milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": 120})
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")
    return decode_assignment(frame, groups, result.x)

def validate_group_split(frame: pd.DataFrame) -> None:
    group_split_counts = frame.groupby("genomic_group_id").split.nunique()
    if int(group_split_counts.max()) != 1:
        raise ValueError("Genomic group leakage detected")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Expected train, val, and test splits")
```

Use binary group×split variables, equality assignment constraints, per-split/per-class lower bounds of one, and auxiliary positive/negative deviations for total and class targets. Weight val/test CHID deviations above other soft deviations. Reject non-success solver statuses with the exact SciPy message.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_group_split.py -q`
Expected: all overlap, endpoint, stability, class-presence, and leakage tests pass.

- [ ] **Step 5: Run the real split and stop at the training gate for inspection**

Run: `python scripts/build_group_split.py`
Expected: 175 groups, 344 structures, three nonempty classes per split, pairwise group intersections empty, actual ratios/deviations printed and saved.

---

### Task 3: Task1 datasets, aggregation, metrics, and CNN

**Files:**
- Create: `src/task1/dataset.py`
- Create: `src/task1/aggregation.py`
- Create: `src/task1/metrics.py`
- Create: `src/task1/model.py`
- Create: `tests/test_task1_dataset.py`
- Create: `tests/test_task1_model.py`

**Interfaces:**
- Consumes: paired arrays/metadata and fixed group split.
- Produces: `PairedMicroCDataset`, `compute_structure_class_weights`, `replicate_balance_weights`, `aggregate_structure_logits`, `structure_cross_entropy`, `classification_metrics`, `SmallMicroCCNN`.

- [ ] **Step 1: Write failing dataset/aggregation/model tests**

```python
def test_aggregate_logits_uses_available_replicates_only():
    logits = torch.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    ids = ["A", "A", "B"]
    out, labels, order = aggregate_structure_logits(logits, torch.tensor([0,0,2]), ids)
    assert torch.equal(out, torch.tensor([[1.,1.,0.],[0.,0.,3.]]))

def test_structure_balanced_weights_sum_to_one_per_structure():
    weights = replicate_balance_weights(["A", "A", "B"])
    assert weights.tolist() == [0.5, 0.5, 1.0]

def test_cnn_returns_logits_for_three_classes():
    assert SmallMicroCCNN()(torch.zeros(4,1,64,64)).shape == (4,3)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_task1_dataset.py tests/test_task1_model.py -q`
Expected: imports fail because Task 3 modules do not exist.

- [ ] **Step 3: Implement the focused modules**

```python
def compute_structure_class_weights(train_structures: pd.DataFrame, classes=(0,1,2)) -> torch.Tensor:
    unique = train_structures.drop_duplicates("structure_id")
    counts = unique.label_id.value_counts().reindex(classes).to_numpy()
    inverse = len(unique) / (len(classes) * counts)
    return torch.tensor(inverse, dtype=torch.float32)
```

The dataset returns matrix, label, structure ID, replicate, sample weight, and availability. Metrics fix class order CHIN/OPCID/CHID and expose flat summary fields plus report frames.

- [ ] **Step 4: Run focused then full suite**

Run: `python -m pytest tests/test_task1_dataset.py tests/test_task1_model.py -q && python -m pytest -q`
Expected: all tests pass.

---

### Task 4: Trainer, independent evaluator, and experiment reporting

**Files:**
- Create: `src/task1/trainer.py`
- Create: `src/task1/evaluation.py`
- Create: `src/task1/reporting.py`
- Create: `scripts/train_task1.py`
- Create: `scripts/run_task1_experiments.py`
- Create: `tests/test_task1_training_protocol.py`

**Interfaces:**
- Consumes: Task 3 datasets/model/aggregation/metrics.
- Produces: `set_reproducible_seed(seed)`, `train_experiment(model, optimizer, train_loader, val_loader, class_weights, epochs, patience, checkpoint_path, device)`, `choose_better_epoch(candidate, incumbent)`, `evaluate_locked_checkpoint(checkpoint_path, model_factory, eval_loader, class_weights, device)`, `select_best_input(comparison)`, experiment directories and `input_comparison.csv`.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_train_signature_has_no_test_loader():
    assert "test_loader" not in inspect.signature(train_experiment).parameters

def test_epoch_tie_break_uses_structure_validation_loss():
    assert choose_better_epoch({"val_macro_f1": .5, "val_loss": .8}, {"val_macro_f1": .5, "val_loss": .7}) is True

def test_input_selection_ignores_test_metrics():
    rows = [{"input_type":"raw","best_val_macro_f1":.8,"best_val_loss":.5,"test_macro_f1":0}, {"input_type":"oe","best_val_macro_f1":.7,"best_val_loss":.4,"test_macro_f1":1}]
    assert select_best_input(pd.DataFrame(rows)) == "raw"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_task1_training_protocol.py -q`
Expected: imports fail because trainer/evaluation modules do not exist.

- [ ] **Step 3: Implement deterministic training and locked evaluation**

```python
def train_experiment(model, optimizer, train_loader, val_loader, class_weights, epochs, patience, checkpoint_path, device):
    best, history, stale = None, [], 0
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, optimizer, train_loader, class_weights, device)
        val_stats = evaluate_structure_level(model, val_loader, class_weights, device)
        candidate = {"epoch": epoch, **val_stats}
        if best is None or choose_better_epoch(candidate, best):
            best, stale = candidate, 0
            save_checkpoint(checkpoint_path, model, optimizer, best)
        else:
            stale += 1
        history.append({"epoch": epoch, **train_stats, **prefix_keys(val_stats, "val_")})
        if stale >= patience:
            break
    return best, pd.DataFrame(history)

def evaluate_locked_checkpoint(checkpoint_path, model_factory, eval_loader, class_weights, device):
    model = model_factory().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate_structure_level(model, eval_loader, class_weights, device)
```

Compute validation CE only after structure-logit aggregation. Reset seed before every model/optimizer/loader initialization. Save checkpoint metadata proving the selected epoch and never pass test objects into the trainer.

- [ ] **Step 4: Implement CSV/PNG outputs and CLI orchestration**

Generate primary and secondary reports, confusion matrices, normalized matrices, predictions, training history, and curves. Run three inputs sequentially with identical configuration, select best input from validation only, then evaluate locked checkpoints once.

- [ ] **Step 5: Run protocol and full tests**

Run: `python -m pytest tests/test_task1_training_protocol.py -q && python -m pytest -q`
Expected: all tests pass and identical-seed model states are byte-equal before training.

- [ ] **Step 6: Run raw/log1p/OE experiments**

Run: `python scripts/run_task1_experiments.py --epochs 60 --batch-size 32 --lr 0.001 --patience 10 --seed 20260920`
Expected: three locked checkpoints and complete output directories; comparison selects one best input solely by validation structure Macro-F1.

---

### Task 5: Replicate consistency analysis

**Files:**
- Create: `src/task1/consistency.py`
- Create: `scripts/analyze_replicate_consistency.py`
- Create: `tests/test_replicate_consistency.py`

**Interfaces:**
- Consumes: paired raw arrays and metadata.
- Produces: `matrix_correlations(rep1, rep2)`, replicate consistency detail and summary CSVs.

- [ ] **Step 1: Write and run failing hand-derived correlation tests**

```python
def test_correlation_masks_have_expected_counts():
    masks = triangle_masks(4, diagonal_band=2)
    assert masks["upper"].sum() == 10
    assert masks["offdiag"].sum() == 6
    assert masks["exclude_band2"].sum() == 1
```

Run: `python -m pytest tests/test_replicate_consistency.py -q`
Expected: import failure for the missing consistency module.

- [ ] **Step 2: Implement correlations, summaries, and CLI**

Use finite Pearson/Spearman calculations with explicit NaN for constant vectors and summary N that counts finite values only.

- [ ] **Step 3: Verify and run real analysis**

Run: `python -m pytest tests/test_replicate_consistency.py -q && python scripts/analyze_replicate_consistency.py`
Expected: 344 detail rows and overall plus three class summaries.

---

### Task 6: Brightness Logistic Regression baseline

**Files:**
- Create: `src/task1/brightness.py`
- Create: `scripts/run_brightness_baseline.py`
- Create: `tests/test_brightness_baseline.py`

**Interfaces:**
- Consumes: paired raw arrays/metadata and fixed split.
- Produces: feature extraction, train-only StandardScaler/LogisticRegression fit, structure-level decision-score evaluation, reports and plots.

- [ ] **Step 1: Write and run failing feature and fit-boundary tests**

```python
def test_intensity_features_are_hand_computed():
    x = np.array([[0., 2.], [4., 6.]])
    assert extract_intensity_features(x)[["mean","max","total"]].tolist() == [3., 6., 12.]

def test_scaler_mean_uses_train_rows_only():
    fitted = fit_brightness_model(train_fixture, train_labels)
    assert np.allclose(fitted.scaler.mean_, train_fixture.mean(axis=0))
```

Run: `python -m pytest tests/test_brightness_baseline.py -q`
Expected: import failure for the missing brightness module.

- [ ] **Step 2: Implement baseline and reports**

Fit scaler/model only from train observations; average val/test decision scores by structure; use the same metrics/plots as CNN without entering input selection.

- [ ] **Step 3: Verify and run baseline**

Run: `python -m pytest tests/test_brightness_baseline.py -q && python scripts/run_brightness_baseline.py`
Expected: primary report, confusion matrix, feature detail/distribution, and metrics files exist.

---

### Task 7: Grad-CAM and quantitative attention analysis

**Files:**
- Create: `src/task1/gradcam.py`
- Create: `scripts/explain_task1.py`
- Create: `tests/test_task1_gradcam.py`

**Interfaces:**
- Consumes: validation-selected checkpoint/input, paired test data, raw matrices for zero-axis masks.
- Produces: `GradCAM`, `cam_quantitative_metrics`, per-structure figures, `gradcam_metrics.csv`, and `gradcam_summary.csv`.

- [ ] **Step 1: Write and run failing Grad-CAM tests**

```python
def test_gradcam_is_finite_64_square_even_for_zero_input():
    model = SmallMicroCCNN().eval()
    cam = GradCAM(model, model.target_layer).compute(torch.zeros(1,1,64,64), target_class=0)
    assert cam.shape == (64,64)
    assert np.isfinite(cam).all()

def test_diagonal_attention_ratio_literal():
    cam = np.ones((5,5))
    assert diagonal_attention_ratio(cam, band=0) == pytest.approx(5/25)
```

Run: `python -m pytest tests/test_task1_gradcam.py -q`
Expected: import failure for the missing Grad-CAM module.

- [ ] **Step 2: Implement hooks, CAM resizing/normalization, metrics, and figure selection**

Use forward/backward hooks on the final convolution, ReLU weighted activation sum, bilinear interpolation to 64×64, and safe all-zero normalization. Generate 2×2 replicate figures and quantitative summaries by class/correctness.

- [ ] **Step 3: Verify and run explanations**

Run: `python -m pytest tests/test_task1_gradcam.py -q && python scripts/explain_task1.py`
Expected: finite CAM tests pass; each class has available correct figures and actual misclassification figures without fabrication.

---

### Task 8: Zero-axis sensitivity, manifest, summary, and completion verification

**Files:**
- Modify: `scripts/run_task1_experiments.py`
- Create: `src/task1/manifest.py`
- Create: `scripts/finalize_task1.py`
- Create: `tests/test_task1_manifest.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all Task 1 outputs and best validation-selected input.
- Produces: sensitivity checkpoint/reports, `task1_manifest.json`, `task1_summary.md`, final requirement audit.

- [ ] **Step 1: Write and run failing manifest/completion tests**

```python
def test_manifest_hashes_existing_split_and_checkpoints(tmp_path):
    manifest = build_manifest(inputs_fixture)
    assert manifest["group_split_sha256"] == sha256(inputs_fixture.split_path)
    assert set(manifest["experiments"]) == {"raw", "log1p", "oe"}

def test_sensitivity_counts_structures_and_replicates_separately():
    counts = participation_counts(sensitivity_metadata_fixture)
    assert counts["val"]["structures"] == 2
    assert counts["val"]["replicates"] == 3
```

Run: `python -m pytest tests/test_task1_manifest.py -q`
Expected: import failure for the missing manifest module.

- [ ] **Step 2: Run best-input zero-axis sensitivity experiment**

Run: `python scripts/train_task1.py --input-type <validation-selected> --exclude-zero-axis --output-dir outputs/task1/zero_axis_sensitivity`
Expected: structure-balanced train loss, actual participation counts, locked checkpoint, and independent test results.

- [ ] **Step 3: Implement manifest/summary generation and README commands**

Generate the 24-answer scientific summary from recorded CSV/JSON artifacts, include hashes/versions/device, compare brightness/raw/OE, quantify attention and sensitivity, and state completion from an explicit artifact checklist.

- [ ] **Step 4: Run complete fresh verification**

Run: `python -m compileall -q src scripts tests && python -m pytest -q && pytest -q && python scripts/finalize_task1.py --verify`
Expected: both pytest entry points pass with zero failures; artifact audit reports every required item present and internally consistent.

- [ ] **Step 5: Perform final code/data review**

Review all new modules against the spec, inspect every test/test-selection boundary, open representative plots, verify manifest hashes, and record any rulings or deferred minor findings in the execution ledger before the final report.

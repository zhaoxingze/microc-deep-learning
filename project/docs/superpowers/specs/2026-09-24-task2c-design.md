# Task2C Unsupervised Grouping Design

## Intent and scope

Group the frozen Task2B candidate and Known paired-average fused embeddings by structural morphology. Produce an auditable descriptive cluster inventory and possible *novel-like cluster candidates* for later Task2D validation. Do not infer confirmed novel structures or run Task2D. Task2A and Task2B are read-only inputs and their scores are not clustering features.

## Frozen inputs and primary population

Validate Task2A Final and Task2B manifests/completion and hashes before work. Match 231 candidate embeddings to candidate metadata by row and ID and 344 Known reference embeddings to unique reference IDs. The 344 Background reference embeddings are excluded from both PCA fitting and HDBSCAN fitting, but projected later for proximity diagnostics. Candidate rep1 and rep2 embeddings are not independent fit samples; the Task2B paired-average matrix is the sole candidate input. Already-standardized 256D vectors are not scaled again. Known type labels and candidate known-overlap annotations are isolated until after HDBSCAN fit.

## Fixed clustering protocol

Freeze a JSON protocol before computing cluster composition. Fit a full-SVD PCA to candidate+Known vectors without labels; choose the smallest number of components with cumulative explained variance at least 0.95. Fit sklearn.cluster.HDBSCAN on that PCA space with `min_cluster_size=5`, `min_samples=5`, `metric='euclidean'`, `cluster_selection_method='eom'`, `cluster_selection_epsilon=0.0`, `allow_single_cluster=False`. Do not tune from Known composition, silhouette, UMAP appearance, or proposal scores. Sensitivity diagnostics may additionally run only `8/5` and `5/8`, without changing the primary labels.

## Post-fit analysis

After fit, attach Known type and candidate genomic-known overlap metadata. Keep noise (`-1`) in membership and candidate tables but never call it a novel-like cluster. For each non-noise cluster compute counts by source and Known type, probability summary, fused-embedding centroid, true medoid nearest the centroid, and a candidate representative nearest the centroid where available. Annotate Known-dominant at type purity >=0.5, otherwise mixed-known. A cluster is novel-like only if it has >=5 candidates, zero Known members, and zero candidate known-center and known-interval overlaps. Nearest Known distances and Background-to-cluster proximity are diagnostics, never filters. Priority is a simple Task2D handoff label, not a learned score.

## Outputs and presentation

Full writes only under `outputs/task2c/`; Smoke only under `outputs/pipeline_runs/task2c_smoke/`. Persist the frozen protocol, PCA model/variance, cluster membership, candidate clusters, summary/representatives/centroids, nearest-known/background diagnostics, optional fixed sensitivity audit, PCA and UMAP figures, uniform medium-scale paired-O/E cluster heatmap overviews, manifest, summary, and completion. Reuse Task2B UMAP coordinates only for plotting primary labels; never cluster in UMAP. Heatmap matrices are local visualization reads, not new proposal scanning or formal replicate validation. Summary states prior CNN supervision, Background reuse/overlap, known-label development history, and lack of biological validation.

## Errors and tests

Fail closed on hash mismatch, alignment mismatch, missing Known type, nonfinite/mismatched vector, unsupported sklearn HDBSCAN, invalid frozen protocol, or incomplete outputs. Test that labels, Background, proposal scores, and replicate rows cannot enter the fit vectors; verify 95% PCA rule, fixed HDBSCAN parameters, post-fit annotations, medoid/centroid, noise, diagnostics, plot alignment, output isolation, and no Task2D artifact. Run full pytest, dry-run, Smoke, Full, then revalidate output manifests and unchanged input hashes.

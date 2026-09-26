# Task2B Structure Representation

> FULL: descriptive representation analysis; no novelty claim.

## Task2A final proposal context

1. Task2A is frozen as an independent six-channel high-recall proposal generator because repeated tuning on the same 344 known structures risks validation overfitting. It proposes regions; it does not determine novelty.
2. Final candidate regions: `231` (this full run embeds `231`).
3. Genomic union coverage: `2141200` bp, fraction `0.461301`.
4. Final center-based Known Recall: Overall `0.476744`, CHIN `0.420000`, OPCID `0.705882`, CHID `0.423077`.
5. Recall is measured on known structures already used repeatedly during development; it does not establish novel-structure precision or full-genome discovery sensitivity.
6. The threshold and six-channel merge were frozen before this last evaluation. Known labels will no longer tune Task2A.

## Representation and references

7. Fused = standardized Task1 CNN hidden feature + standardized Small/Medium/Large AE encoder pooled features.
8–12. Dimensions: CNN `64`, Small AE `64`, Medium AE `64`, Large AE `64`, Fused `256`.
Reference Known `344`; strict blocked Background `344`. Candidate/reference matrices use one identical extraction and preprocessing function.
Background selection audit (full reference): strict eligible pool `1059`; maximum mutually nonoverlapping 12.8 kb windows `178`; selected minimum center gap `4800` bp; adjacent overlapping pairs `231`; previously used in AE training `234`. These correlations are limitations, not independent background observations.
Scalers were fit on the unlabeled Known+Background reference features only; candidates were transformed afterward.

## Genomic GroupKFold known-vs-background probe

13–19. 5-fold mean ± std (random Balanced Accuracy reference ≈ 0.5):
- CNN-only: Accuracy 0.7918 ± 0.0285; Balanced Accuracy 0.7957 ± 0.0311; ROC-AUC 0.8998 ± 0.0208; PR-AUC 0.9109 ± 0.0269
- AE-only: Accuracy 0.8138 ± 0.0151; Balanced Accuracy 0.8200 ± 0.0197; ROC-AUC 0.9052 ± 0.0225; PR-AUC 0.9064 ± 0.0279
- Fused: Accuracy 0.8344 ± 0.0151; Balanced Accuracy 0.8401 ± 0.0197; ROC-AUC 0.9254 ± 0.0143; PR-AUC 0.9339 ± 0.0140
GroupKFold uses 128 kb genomic center blocks, then purges training intervals crossing each test fold boundary; both group and interval overlap audits are zero. The probe is diagnostic, not a detector or a feature-selection loop. The frozen Task1 CNN was trained earlier using these same known structures, so CNN-only and Fused probe results are not independent generalization estimates. AE-only is less affected by this prior supervision but its background references partly reuse AE training windows.

## PCA and UMAP

20. PCA explained variance: PC1 `0.364488`, PC2 `0.192070`, cumulative `0.556558`.
21. PCA map: known/background centroid separation relative to within-group spread `0.221`; candidates occupy `4` of four quadrants.
22. UMAP map: known/background centroid separation relative to spread `0.380`; candidates occupy `4` of four quadrants. UMAP is visualization only.
23. Candidate embeddings span `4` UMAP quadrants; this is descriptive, not a cluster or novelty decision.

## Replicate stability and limitations

24. Mean rep1/rep2 cosine: CNN `0.678991`, AE Small `0.882693`, Medium `0.940090`, Large `0.953278`, Fused `0.892723`.
Padded matrix extraction records: `0` across candidate and reference replicate windows.
25. Main limits: prior supervised CNN exposure to known references, partial AE-training reuse among background references, correlated overlapping background windows, class imbalance/limited CHID count, coordinate-block CV on one chromosome, padded edge windows, and two-replicate data. Neither PCA nor UMAP validates novel structures.
26. Technical conditions for Task2C: `met`; `task2b_complete=true`. This is not a scientific claim of new structure discovery.

No HDBSCAN, KMeans, DBSCAN, cluster labeling, or Task2C operation was executed.

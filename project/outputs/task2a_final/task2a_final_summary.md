# Task2A Final: frozen high-recall proposal generator

Task2A now proposes regions for representation and grouping; it does not determine novelty. The fixed threshold and rules were frozen before final known recall.

## Frozen protocol and output

- Six independent scale × branch channels: `['small_density', 'small_shape', 'medium_density', 'medium_shape', 'large_density', 'large_shape']`
- Fixed paired p-equivalent score threshold: `<= 0.05`
Each replicate p-value is empirical on background calibration data. Their geometric mean is a fixed paired ranking score, **not** a separately calibrated paired empirical p-value; the threshold must not be interpreted as a formal paired 5% test.
Observed fraction of calibration-background windows below this fixed score threshold (before NMS):

| channel        |   n_calibration_windows |   n_below_fixed_threshold |   observed_tail_fraction |   fixed_threshold |
|:---------------|------------------------:|--------------------------:|-------------------------:|------------------:|
| small_density  |                    1054 |                        52 |                0.0493359 |              0.05 |
| small_shape    |                    1054 |                        39 |                0.0370019 |              0.05 |
| medium_density |                     431 |                        20 |                0.0464037 |              0.05 |
| medium_shape   |                     431 |                        15 |                0.0348028 |              0.05 |
| large_density  |                     147 |                         6 |                0.0408163 |              0.05 |
| large_shape    |                     147 |                         7 |                0.047619  |              0.05 |

- Channel peaks: `{'small_density': 244, 'small_shape': 226, 'medium_density': 114, 'medium_shape': 72, 'large_density': 51, 'large_shape': 69}`
- Proposals before cross-channel merge: `776`
- Candidate regions after transitive overlap merge: `231`
- Candidate genomic union coverage: `2141200` bp / `4641652` bp = `0.461301278`
- Regions with a known center: `92`
- Regions with a known interval overlap: `119`
These overlap fractions describe the candidate pool and are not calibrated precision, because unlabeled candidates cannot be declared false positives.

## Final known-structure recall (center-based primary)

| type    |   n_known |   center_recall |   interval_overlap_recall |
|:--------|----------:|----------------:|--------------------------:|
| overall |       344 |        0.476744 |                  0.625    |
| CHIN    |       250 |        0.42     |                  0.556    |
| OPCID   |        68 |        0.705882 |                  0.852941 |
| CHID    |        26 |        0.423077 |                  0.692308 |

## Historical experiments and limitation

Single-scale anomaly, mean fusion, OR fusion, calibrated OR, multi-scale winner-takes-all, and equalized cross-scale ranking were tested. Global winner-takes-all exposed branch competition, scale multiplicity bias, NMS order dependence, unequal calibration resolution, and candidate-selection bias.
The same 344 known structures have been reused during method development. Further tuning to raise their recall would weaken their value as validation. This final recall is an experimental outcome and limitation, not a threshold to optimize or a Task2B performance gate.

## Engineering readiness

| check                               | passed   |
|:------------------------------------|:---------|
| proposal_protocol_frozen            | True     |
| candidate_regions_valid             | True     |
| task2a_artifacts_complete           | True     |
| rep1_rep2_accessible                | True     |
| no_train_val_calibration_leakage    | True     |
| known_labels_not_used_for_proposals | True     |

- ready_for_task2b: `true`

No Task2C clustering or novelty claim is made here.

# Current Model vs. Reference Paper

Last updated: 2026-08-05 — full audit-style comparison added; v4.4 remains research-only and selected model remains `v3.2.2-cuda-graph-dual-runtime`

Reference: [Residual Neural Network for Precise 6-DoF Capsule Endoscope Localization Using Electromagnetic Induction](https://doi.org/10.1109/TIM.2026.3655917). Paper values below come from Section II–III, Eq. (8)–(9), Table II, and Table III.

Detailed Vietnamese report with figure-by-figure interpretation:
[BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md](BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md).

Canonical generated comparison with 30 point-by-point rows, 13 figures, 6
machine-readable tables and a SHA-256 manifest:
[full_pipeline_comparison_2026_08_05/REPORT.md](../reports/full_pipeline_comparison_2026_08_05/REPORT.md).

Vietnamese 16-slide presentation and exact speaker script:
[presentation deck](../reports/presentation_paper_comparison_2026_08_06/SLIDE_DECK.pdf) ·
[speaker script](../reports/presentation_paper_comparison_2026_08_06/SLIDE_SCRIPT.md).

## Protocol comparison

| Item | Reference paper | Current project | Interpretation |
|---|---|---|---|
| Task | 9 EMF amplitudes → 6-DoF capsule pose | Same physical inverse-localization task | Aligned at task level |
| Workspace | 500 × 500 × 500 mm | 100 × 100 × 100 mm | Numerical accuracy is not directly comparable across volumes |
| Core network | One width-512 network with 7 residual blocks | Selected v3.2.2 is a 5-model component ensemble; experimental v4 added one width-512, 7-block C3 network | Paper architecture is retained, but current component rules are new |
| Cross-generation calibration | No learned generation adapter reported | v4 C3 used an identity-initialized per-domain 9-channel gain/bias adapter before temporal features | Implemented and evaluated; v4 was not selected after final test |
| Temporal input | Single EMF row | Causal W3; one branch also receives two `Δrow` values | New relative to paper |
| Time-aware continuous input | Not used by the reported ResNet input | v4 treats each programmed robot row as one fixed `Δt`; if hardware timestamps exist, only causal `Δt / median(Δt_train)` is used; absolute time is forbidden | New relative to paper and designed for realtime streams |
| Row-time rule | None | Each observed row is one unit step; only `Δrow`, never absolute row index | Avoids scripted-trajectory phase leakage |
| Orientation target | `cos(roll,pitch,yaw)` | Continuous rotation-6D projected to SO(3) | Current representation preserves sign |
| Synthetic training | 40³ positions × 10³ orientations = 64 million model-generated samples | Fold-specific fitted forward model generated synthetic W3 samples at a 1:1 ratio with real train rows; synthetic rows never entered validation/test | Much smaller, calibration-gated synthetic augmentation |
| Synthetic→calibration order | Initial analytical synthetic training; 150 real poses fit physical TX parameters; calibrated forward/inverse models are retrained, but the retraining dataset is not fully specified | v4.4 analogue uses 107 fixed pure-synthetic epochs followed by real-only supervised calibration; its synthetic generator was already fit on train-fold real data | Same high-level neural order, not an exact reproduction of physical-calibration order |
| Real calibration | 150 random poses used to identify physical forward-model parameters, followed by retraining | Nine non-test sessions, 6,019 real rows; physics fit used up to 1,200 rows/fold and 1,800 for deployment | Different calibration strategy and coverage |
| Development/test | Paper: 100-pose five-loop dynamic helix for real-system benchmark | Three leave-one-`con_rot`-session-out development folds plus one-time 3-session `cyl_rot` final test (3,012 rows) | Final test is reported, but protocols remain different |
| Orientation coverage | Roll, pitch, yaw varied 20°–160° | Mainly pitch; roll/yaw variation is very narrow | Current orientation result is an easier/narrower regime |
| Locked test | Paper reports real benchmark | `cyl_rot` was opened once on 2026-08-04 after bundle freeze; a receipt prevents a second opening | v4 failed the balanced final gate; no tuning is allowed on this test |
| New sealed test | Not applicable | None remains after the one-time `cyl_rot` opening | A newly acquired later session is required for the next changed model |
| Physics consistency | Reconstructed EMF error below 3.4% | Fold held-out EMF correlation 0.878–0.918 and RMSE/mean 0.390–0.474; final `cyl_rot` mean channel MAPE is 106.19% even from ground-truth pose | Augmentation gate passed on non-rotating development folds, but the forward model does not generalize to the opened rotating final sessions |

## Numerical comparison

| Metric | Reference paper real system | Selected v3.2.2 development CV | Selected v3.2.2 one-time `cyl_rot` test | Can be compared directly? |
|---|---:|---:|---:|---|
| Position axis-style RMSE, Eq. (8) | 1.90 mm | 0.7231 mm | 1.1222 mm | Same algebra, but no: workspace, path, training, and test differ |
| Euler-component direction RMSE, Eq. (9) | 3.55° | 0.2436° | 1.3801° | Algebraically similar, but motion/workspace differ |
| SO(3) geodesic RMSE | Not reported | 0.4219° | 2.3904° | No paper counterpart |
| Position Euclidean p95 | Not reported | 2.0667 mm | 3.1047 mm | No paper counterpart |
| SO(3) geodesic p95 | Not reported | 0.6587° | 5.0578° | No paper counterpart |
| Batch-one inference | 0.82 ms, i9-10980XE + RTX 3090 system | CPU 1.290 ms; RTX 5060 Ti CUDA Graph 0.858 ms; 5 models | Same selected runtime | Different hardware/statistic; not a direct speed claim |

The current values must not be described as outperforming the paper. The paper
evaluates a 500 mm workspace with broad simultaneous roll/pitch/yaw motion,
whereas current CV covers a 100 mm workspace and primarily pitch rotation.

The v4 hybrid passed development CV but failed its frozen one-time final gate.
On `cyl_rot`, it improved SO(3) RMSE from 2.3904° to 1.8653° and p95 from
5.0578° to 3.7461°, but position RMSE regressed from 1.1222 to 1.2335 mm and
position p95 from 3.1047 to 3.4725 mm. The all-four-metric rule therefore keeps
v3.2.2 selected. Choosing a new position/orientation component rule from these
test results would be leakage.

A later development-only ablation retrained seed-42 C3 at synthetic/real ratios
`0.25` and `0.50`, retaining the predeclared `w=0.20` blend. Both improved all
four pooled metrics over the matched-fold v3.2.2 reference, but failed the
per-session robustness gate because fold-s3 SO(3) p95 regressed by `8.08%` and
`6.13%`, respectively. The earlier `1.00` ratio was still the only ratio to
pass development (`2.71%` worst-session regression), although it remains
rejected by its historical one-time final result. The ablation did not read or
rescore `cyl_rot`; no deployment decision changed.

| Synthetic / real | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 | Development gate |
|---:|---:|---:|---:|---:|---|
| 0, v3.2.2 reference | 0.7474 mm | 2.3539 mm | 0.4602° | 0.8580° | reference |
| 0.25 | 0.7401 mm | 2.2653 mm | 0.3588° | 0.6242° | reject: session gate |
| 0.50 | 0.7298 mm | 2.3057 mm | 0.3798° | 0.6594° | reject: session gate |
| 1.00 | 0.7327 mm | 2.2613 mm | 0.3586° | 0.6362° | pass development only |

The subsequent pure-synthetic→real experiment was more promising for
position. With a robust development-only rule
`XYZ=0.8*v3.2.2+0.2*C3` and
`rotation6D=0.7*v3.2.2+0.3*C3`, all three seeds passed aggregate and session
accuracy gates:

| Seed | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 | Composite gain |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.6878 mm | 2.0470 mm | 0.4048° | 0.7384° | 11.78% |
| 7 | 0.7077 mm | 2.1137 mm | 0.4045° | 0.7495° | 10.12% |
| 123 | 0.7021 mm | 2.0971 mm | 0.4096° | 0.7395° | 10.49% |

This remains a research candidate. Candidate-specific CPU p95 was 2.1648 ms,
while GPU p95 was 1.7234 ms and failed the predeclared 1.5 ms limit. There is
also no new unopened final set. `cyl_rot` was not rescored and the selected
deployment remains v3.2.2.

The reproducible paper-style report is in
`reports/paper_style_2026_08_04/REPORT.md`. Its Fig. 9 ground-truth-pose
control shows that EMF reconstruction remains poor even when inverse-network
pose error is removed. This points to forward-model/calibration mismatch on
rotation and must be investigated using new rotating development sessions,
not by fitting to the opened `cyl_rot` labels.

## Current selected component rule

```text
XYZ = mean(
  raw-W3 seed42,
  raw-W3 seed7,
  raw-W3 seed123,
  unit-row-time W3 seed42
)

orientation = current-EMF + past/current-log-ratio W3 seed42
```

The unit-row-time branch improved the three-fold mean position-axis RMSE from
0.7460 mm to 0.7231 mm. It was not selected for orientation because its mean
SO(3) RMSE was 0.4884°, worse than the selected log-ratio branch at 0.4219°.

Runtime v3.2.1 leaves all five checkpoints and every prediction unchanged. It
keeps component aggregation in Torch, skips a chordal SO(3) SVD whose result was
immediately overwritten, and traces/freezes the complete ensemble once at
startup. Three 5,000-iteration repeats selected 6 intra-op and 1 inter-op thread
on the i5-14600K; median-run p95 fell from 2.151 ms to 1.290 ms (about 775.1 Hz).
All 1,001 checked production outputs matched eager execution exactly. CUDA is
supported, but no GPU result is reported because the NVIDIA driver was not
accessible during this audit.

Runtime v3.2.2 corrects that visibility statement: the managed sandbox hid the
NVIDIA device nodes, while host-side execution confirmed an RTX 5060 Ti 16 GB,
driver 580.173.02 and CUDA 13.0. Three CUDA Graph runs measured p95 values
0.8670, 0.8581 and 0.8530 ms with per-row pinned-host input copies; the median
repetition is 0.8581 ms (about 1,165 Hz), 33.5% lower than the CPU p95. CPU and
GPU pose outputs differed by at most
0.00007 over 1,001 development rows, with materially identical metrics. CUDA
Graph capture/startup, feature construction, uncertainty post-processing,
acquisition and TCP/IP are excluded.

## Mandatory update rule

After every model, feature, calibration, split, or deployment change:

1. Append a dated entry to `docs/EXPERIMENT_LEDGER.md`, including failed trials.
2. Append one row to `docs/MODEL_UPGRADE_HISTORY.csv`.
3. Update the current-project column and numerical table in this file.
4. Update `checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json` only if the selected deployment changes.
5. Keep the test status and strict-blindness qualification explicit.

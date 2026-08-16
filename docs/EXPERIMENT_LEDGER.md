# Experiment Ledger

This file records decisions, data roles, metrics, and immutable artifacts. Add a
new dated entry whenever data, protocol, feature representation, model, or test
state changes.

## 2026-08-03 — v4 cross-generation implementation (no real experiment yet)

Implemented:

- automatic timestamp-order session discovery with the newest session sealed;
- strict canonical schema/units, reset-aware segmentation, file hashes,
  quarantine and 100 mm workspace checks;
- fixed-row-delta robot acquisitions without mandatory timestamp columns, plus
  per-generation train-only delta-time medians and causal `dt_ratio`; absolute
  time and absolute row index are never features;
- B1/C1/C2/C3/C4 ResNet candidates, identity-initialized generation adapter,
  session/generation/spatial sampler and conditional G1 GRU;
- frozen-v3.2 development baseline/component ablations, disjoint-CV pooling,
  four-metric quality gate, candidate freeze and one-time sealed-test guards;
- transport-neutral realtime JSONL predictor with CPU TorchScript, GPU CUDA
  Graph, reset/gap state handling, confidence and dispersion.

Verification:

```text
23 v3+v4 unit/smoke tests passed, including headerless fixed-row-time input,
frozen-v3.2-on-v4 baseline and
quality-gate accept/reject paths.
One-epoch synthetic v4 train/evaluate passed on CPU.
The C2 W3/47-feature model completed a CUDA Graph forward smoke test on the
RTX 5060 Ti with finite `(1, 9)` output.
No historical cyl_rot or new sealed prediction/evaluation was executed.
```

Baseline leakage guard: old `con_rot` development sessions use their original
fold-specific v3 checkpoints; the all-old-data deployment bundle is used only
for genuinely unseen new-generation development sessions.

Superseded assumption (corrected by the 2026-08-04 entry below):

```text
This entry originally waited for new_calib_2026_08_03/. The user later
confirmed that new_calib/ already contains the newest robot calibration files.
Those files were used by the completed physics-hybrid experiment below. Do not
treat the missing directory as a current blocker.
```

## 2026-08-02 — repository fidelity and leakage audit

Decision:

- Core paper equations and residual architecture are substantially present.
- The repository is not a complete reproduction of the paper.
- Historical no-leak path avoids exact row overlap but still has trajectory and
  private-selection contamination.
- Paper-style cosine/arccos orientation is invalid for signed new-calibration
  angles.
- The new 100 mm acquisition must use a separate v3 protocol.

Key actions:

- Preserved the historical paper-oriented pipeline.
- Started a parallel v3 path rather than rewriting old checkpoint semantics.
- Corrected pooled RMSE aggregation.
- Made calibration-v2 defaults safer.

## 2026-08-02 — new calibration v3 single-frame baseline

Data:

```text
Source directory: new_calib/
Raw rows:        9,033
Valid rows:      9,031
Quarantined:     2
Workspace:       100 x 100 x 100 mm
```

Representation:

```text
Input:  9 EMF values
Output: XYZ + rotation-6D
Loss:   normalized position MSE + SO(3) geodesic loss
```

Primary family-held-out protocol:

```text
Train: con_rot + cyl_norot
Dev:   con_norot
Test:  cyl_rot, unopened
```

Development result:

```text
Position axis RMSE:       1.4279 mm
Position Euclidean RMS:   2.4731 mm
SO(3) geodesic RMSE:      1.0684 deg
Best epoch:               20
Training stopped:         70
```

This baseline was not selected as the temporal comparison baseline because its
development family did not rotate.

## 2026-08-02 — temporal v3 matched protocol

Protocol:

```text
Train:
  con_norot sessions 1-3
  cyl_norot sessions 1-3
  con_rot sessions 1-2

Rotating dev:
  con_rot session 3 / conrot3_data.csv

Locked test:
  cyl_rot sessions 1-3
```

Counts:

```text
Train: 5,018
Dev:   1,001
Test:  3,012
```

Candidate results:

| Candidate | Input dimension | Position axis RMSE | SO(3) RMSE | Position p95 | SO(3) p95 |
|---|---:|---:|---:|---:|---:|
| M0 matched single | 9 | 1.2378 mm | 1.0367 deg | 2.7945 mm | 2.3458 deg |
| M1 W3 | 27 | **0.5908 mm** | **0.2465 deg** | **1.5069 mm** | **0.3589 deg** |
| M1 W5 | 45 | 0.6455 mm | 0.3059 deg | 1.9103 mm | 0.4428 deg |
| M1 W9 | 81 | 0.6840 mm | 0.2677 deg | 1.8802 mm | 0.3860 deg |
| M3 single position + W3 orientation | 9 + 27 | 1.2378 mm | 0.2465 deg | 2.7945 mm | 0.3589 deg |

Decision:

- Select M1 W3 because it dominates M0, W5, W9, and M3 on both primary RMSE
  metrics and both p95 metrics.
- Reject M3 for this dataset because the temporal model itself has better
  position accuracy.
- Keep timestamp support inactive until timestamped acquisitions exist.
- Keep `cyl_rot` locked.

Frozen artifact:

```text
Path:
  checkpoints/new_calib_v3_temporal/frozen_w3_h0_seed42/

Checkpoint SHA-256:
  9872e73819a2a67e1c0eabe8337d035bde2227e2cc99412a46d8f654effb864d

Protocol config SHA-256:
  91b7d401976c64ad834e002e7664f29b77d618df7e0ffe316e758e25cd0941b1

Training config SHA-256:
  4c33f5b0fdcec4e83cfb6d6bcd58598232b52ea9bfe8c2dd2079f7796352c660
```

Frozen manifest:

```text
checkpoints/new_calib_v3_temporal/frozen_w3_h0_seed42/freeze_manifest.json
```

## 2026-08-02 — cross-session calibration robustness upgrade

The earlier `conrot3_data.csv` result was optimistic, so selection was repeated
with three leave-one-`con_rot`-acquisition-session-out folds. `cyl_rot` was not
predicted or scored.

Raw W3 seed-42 results:

| Development session | Position axis RMSE | SO(3) RMSE | Position p95 | SO(3) p95 |
|---|---:|---:|---:|---:|
| 1 | 1.0512 mm | 0.4543 deg | 3.2104 mm | 0.9320 deg |
| 2 | 0.6768 mm | 0.5968 deg | 2.1190 mm | 0.9863 deg |
| 3 | 0.6702 mm | 0.2659 deg | 1.6338 mm | 0.3386 deg |
| Mean | 0.7994 mm | 0.4390 deg | 2.3211 mm | 0.7523 deg |

This session spread is now treated as the main calibration-generalization
uncertainty; the former single-session 0.5908 mm result is historical evidence,
not the current expected accuracy.

Rejected ablations:

- Strong gain/offset augmentation: fold-1 position 2.4750 mm.
- Mild gain/offset augmentation: fold-1 position 1.7480 mm.
- Current-EMF plus log-ratio features for XYZ: mean position 1.0089 mm.
- SO(3) mean of the three raw seeds: mean orientation 0.4691 deg, worse than
  raw seed 42 at 0.4390 deg.

Reason: absolute EMF amplitude is essential position information. Enforced
amplitude invariance is not a valid calibration correction for this dataset.

## 2026-08-02 — selected cross-feature component ensemble

Selected development-only component rule:

```text
XYZ:         arithmetic mean of raw-W3 seeds 42, 7, 123
Orientation: current absolute EMF + past/current log-ratio, seed 42
Rotation:    continuous rotation-6D, evaluated on SO(3)
```

Mean of three development folds:

```text
Position axis RMSE: 0.7460 mm
Position p95:       2.0927 mm
SO(3) RMSE:         0.4219 deg
SO(3) p95:          0.6587 deg
```

Compared with raw W3 seed 42, the selected rule improves mean position-axis
RMSE by about 6.7%, SO(3) RMSE by about 3.9%, position p95 by about 9.8%, and
SO(3) p95 by about 12.4%. These are development-CV improvements, not final-test
claims.

## 2026-08-02 — target-free temporal boundaries

The first temporal implementation used an XYZ-jump threshold to identify robot
trajectory restarts. This did not put XYZ values into model features, but it was
target-informed preprocessing and required an unavailable runtime signal.

It was replaced by:

- known acquisition period in `configs/new_calib_v3.yaml`: 500 rows for
  no-rotation and 1,000 rows for rotation;
- source-row discontinuities caused by quarantined invalid rows;
- explicit reset events during live inference.

`segment_position_jump_mm` is now `null`. Old and new train/validation/test NPZ
arrays were checked and were identical (NaNs compared as equal) for fold 3, so
existing trained weights remain numerically consistent with the corrected
target-free rule.

Strict-blindness qualification: raw test labels are present locally and the
preparation audit previously inspected target-derived restart boundaries. No
model prediction, error, or test metric was used for selection, but a
publication-grade final claim should preferably use a newly acquired sealed
test after this protocol is frozen.

## 2026-08-02 — fixed-epoch deployment training

After CV selection, all 6,019 non-test samples were used for fixed-epoch
training with no validation file and no checkpoint selection:

| Component | Seed | Fixed epochs | Checkpoint SHA-256 |
|---|---:|---:|---|
| Raw position member | 42 | 165 | `132b2039...c947341` |
| Raw position member | 7 | 130 | `beef41bc...d986359` |
| Raw position member | 123 | 118 | `5012f46a...8f3ec20` |
| Log-ratio orientation | 42 | 132 | `7ecf6544...e0be91` |

Full hashes and the frozen component rule are recorded in
`checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json`.

Batch-one CPU benchmark for all four models plus aggregation, four Torch
threads: p95 1.704 ms (about 587 Hz). Sensor and TCP/IP latency are excluded.

## 2026-08-02 — unit-row timestamp ablation and v3.2 selection

Request: treat every observed row as one timestamp. To prevent trajectory-phase
leakage, the experiment uses only causal differences:

```text
row time:       t = source_row
model feature:  Δrow only
normal step:    1
padding/reset:  0
absolute t:     never used
W3 input:       27 EMF values + 2 delta-row values = 29
```

Raw W3 plus unit-row-time seed-42 results:

| Development session | Position axis RMSE | SO(3) RMSE | Position p95 | SO(3) p95 |
|---|---:|---:|---:|---:|
| 1 | 1.0618 mm | 0.4667 deg | 3.5338 mm | 0.9409 deg |
| 2 | 0.5733 mm | 0.7281 deg | 1.6577 mm | 1.1615 deg |
| 3 | 0.6232 mm | 0.2703 deg | 1.7870 mm | 0.4437 deg |
| Mean | 0.7528 mm | 0.4884 deg | 2.3262 mm | 0.8487 deg |

Decision by component:

- Do not replace the orientation branch: row-time orientation is worse than
  log-ratio orientation (0.4884° versus 0.4219° mean SO(3) RMSE).
- Add row-time seed 42 as a fourth XYZ ensemble member because its errors are
  complementary to the three raw seeds.

Selected v3.2 three-fold mean:

```text
Position axis RMSE:             0.7231 mm
Position Euclidean p95:         2.0667 mm
SO(3) geodesic RMSE:            0.4219 deg
SO(3) geodesic p95:             0.6587 deg
Euler-component RMSE, paper Eq. (9) form: 0.2436 deg
```

Relative to v3.1, XYZ RMSE improves by about 3.1%; orientation is unchanged.
The deployment row-time model was trained on all 6,019 non-test samples for 79
fixed epochs (the median best epoch across folds). Checkpoint:

```text
checkpoints/new_calib_v3_temporal_deployment/w3_rowtime_seed42/
SHA-256: 1514e7ef883f5eb1236ad54c624323bc3f49f62dd5002d658ac8623e7f7dc384
```

Five-model CPU batch-one benchmark, four Torch threads: p95 2.151 ms (about
465 Hz), excluding acquisition and TCP/IP. The locked test was not predicted
or scored.

The paper comparison is maintained in `docs/PAPER_COMPARISON_TABLE.md`; the
machine-readable version history is `docs/MODEL_UPGRADE_HISTORY.csv`.

## 2026-08-02 — v3.2.1 hardware-neutral runtime fast-path

The model was not retrained and the five selected checkpoint hashes/component
rule remain unchanged. Inference now keeps component aggregation in Torch and,
when the dedicated orientation checkpoint is present, skips the chordal SO(3)
mean/SVD that the old path calculated and then overwrote.

CPU batch-one sweep on Intel Core i5-14600K, PyTorch 2.11.0+cu130, 4,000 timed
iterations per selected thread count, inter-op threads fixed at 1:

| Intra-op threads | p95 latency | p95 rate |
|---:|---:|---:|
| 4 | 1.8984 ms | 526.8 Hz |
| 5 | 1.8377 ms | 544.2 Hz |
| 6 | **1.7776 ms** | **562.5 Hz** |
| 7 | 1.9204 ms | 520.7 Hz |

This eager fast-path selected 6/1. The complete selected ensemble was then
traced and frozen once at startup. Exact production-wrapper repeats, 5,000
timed iterations each:

| Backend/profile | Repetition p95 values | Median p95 |
|---|---|---:|
| TorchScript, 5/1 | 1.3521, 1.3882, 1.3159 ms | 1.3521 ms |
| TorchScript, 6/1 | 1.2941, 1.3020, 1.2597 ms | **1.2941 ms** |

Thread selection above used the same architecture on CV checkpoints. A final
three-repeat confirmation using the exact five frozen deployment checkpoints
gave p95 values 1.2988, 1.2820, and 1.2902 ms; the median repetition is
**1.2902 ms (775.1 Hz)**. This improves median-run p95 latency by 40.03% from
the original 2.1513 ms measurement. Scope
remains network forward plus aggregation/member outputs only; one-time tracing,
feature construction, acquisition, and TCP/IP are excluded. Eager and
TorchScript prediction CSVs were identical on all 1,001 replay rows (maximum
absolute difference 0.0), including model-disagreement and safety fields.
Regression tests: 12 passed.

CUDA code paths remain supported through `--device auto/cuda`, but GPU was not
benchmarked: `torch.cuda.is_available()` returned false and `nvidia-smi` could
not communicate with the NVIDIA driver. Explicit `--device cuda` now raises a
clear error throughout current v3 train/evaluate/predict paths instead of
silently falling back to CPU. Artifacts:

Correction: those availability results applied only inside the managed sandbox;
the following v3.2.2 entry records the successful host-side GPU discovery and
benchmark.

```text
results/new_calib_v3_temporal_deployment/runtime_selection_v3_2_1.json
results/new_calib_v3_temporal_deployment/inference_benchmark_deployment_cpu6_torchscript_rep3.json
```

Test impact: none. No locked-test prediction or metric was produced.

## 2026-08-02 — v3.2.2 CUDA visibility correction and dual runtime

The earlier CUDA failure was caused by the managed sandbox hiding
`/dev/nvidia0` and `/dev/nvidiactl`, not by missing host hardware or driver.
Approved host-side inspection confirmed:

```text
GPU: NVIDIA GeForce RTX 5060 Ti, 16,311 MiB
Driver: 580.173.02
Driver CUDA: 13.0
PyTorch: 2.11.0+cu130
torch.cuda.is_available(): true
```

Exact five-checkpoint deployment, batch one, 4,000 timed iterations:

| Runtime | p95 | Rate |
|---|---:|---:|
| CPU TorchScript 6/1, median repetition | 1.2902 ms | 775.1 Hz |
| CUDA eager probe | 1.7275 ms | 578.9 Hz |
| CUDA TorchScript probe | 1.1720 ms | 853.3 Hz |
| CUDA Graph + pinned H2D rep 1 | 0.8670 ms | 1,153.3 Hz |
| CUDA Graph + pinned H2D rep 2 (median) | **0.8581 ms** | **1,165.4 Hz** |
| CUDA Graph + pinned H2D rep 3 | 0.8530 ms | 1,172.4 Hz |

CUDA Graph reduces p95 by 33.5% relative to the selected CPU fallback. Per-row
pinned-host-to-device input and final device-to-host pose output are included;
capture/startup, feature construction and uncertainty post-processing are
excluded. A CPU-versus-CUDA TorchScript comparison on
1,001 development rows had maximum absolute pose-output difference 0.00007 and
mean absolute difference 0.0000014; pose metrics were materially identical.

Selected deployment policy:

- GPU training is allowed and recommended for speed.
- Ordinary CUDA CSV prediction uses TorchScript.
- The future batch-one realtime/TCP service should capture CUDA Graph once at
  startup, then replay it for each causal row.
- CPU TorchScript 6/1 remains the portable no-GPU fallback.

No checkpoint, feature, split, CV metric, or test state changed. The locked test
was not predicted or scored. Artifacts:

```text
results/new_calib_v3_temporal_deployment/runtime_selection_v3_2_2.json
results/new_calib_v3_temporal_deployment/inference_benchmark_deployment_gpu_cuda_graph_host_pinned_rep2.json
```

## Historical test-state snapshot from 2026-08-02 (superseded)

As of 2026-08-02:

```text
cyl_rot test opened: NO
test_metrics.csv present: NO
test_predictions.csv present: NO
```

Interpret `opened: NO` as no model-quality inference/evaluation on the test.
The strict-blindness qualification above still applies.
All v3 evaluators now refuse `--split test` unless the one-time
`--confirm_open_locked_test` flag is supplied explicitly.

This snapshot is retained for audit history. The 2026-08-04 entry below records
the later hash-frozen one-time opening and is the current test state.

## Pending acquisition/validation work

- Add monotonic hardware timestamps synchronized with robot poses.
- Record sampling rate, dropped frames, and clock offset.
- Verify nine-channel mapping and polarity.
- Verify rigid robot/world-to-TX coordinate transform.
- Collect broad roll and yaw motion; the current acquisition varies mainly in
  position and pitch.
- Collect a new independent final test for any model changed after `cyl_rot` is
  opened.

## 2026-08-04 — v4 physics-hybrid CV, freeze, and one-time final rejection

### Corrected input assumption

The newest available calibration data are the 12 headerless files already in
`new_calib/`; there is no separate `new_calib_2026_08_03/` directory. Each row
is one fixed robot-programmed delta-t and contains nine EMF channels followed
by `X,Y,Z,roll,pitch,yaw`. Absolute timestamp and absolute row index were not
used as model features. File/reset padding has delta-row 0; ordinary causal
transitions have delta-row 1.

Nine non-rotating sessions (6,019 valid rows) were used for development and
deployment training. The three `cyl_rot` sessions (3,012 rows) remained locked
through all architecture, weight, epoch, and runtime choices.

### Fold-specific physics gate

The old physical config was rejected for this calibration frame. A new
forward model was fit only on the real training sources of each fold, with up
to 1,200 evenly sampled rows, four starts, and a 100 x 100 x 100 mm workspace.

| Held-out fold | EMF correlation | RMSE/mean(|EMF|) | Gate |
|---|---:|---:|---|
| con_rot s1 | 0.914858 | 0.394478 | pass |
| con_rot s2 | 0.878366 | 0.473933 | pass |
| con_rot s3 | 0.918091 | 0.390480 | pass |

Every calibration metadata file records `test_labels_loaded: false` and hashes
the same three protected `cyl_rot` sources.

### Hybrid training protocol

Candidate C3 retained width 512, seven residual blocks, rotation-6D output,
combined W3 raw/log-ratio/delta-row features, an identity-initialized domain
gain/bias adapter, and balanced session/spatial sampling. Each fold added one
physics-synthetic W3 sample per real train sample. Validation stayed real-only.

Training was staged:

1. real + synthetic pretraining;
2. real-only fine-tuning from the pretrain weights;
3. feature and target normalization recomputed from real-only train data.

Fold-3 controls showed why staging was retained:

| Fold-3 candidate | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| Direct mixed real+synthetic | 0.7966 mm | 2.4372 mm | 0.4983 deg | 0.8604 deg |
| Real-only C3 from scratch | 0.8083 mm | 2.1542 mm | 0.2440 deg | 0.3234 deg |
| Staged + real normalization rebase | 0.6706 mm | 1.8644 mm | 0.2386 deg | 0.3874 deg |

The v4 component was complementary to frozen v3.2.2. Screening selected a
runtime-friendly fixed rule on development CV:

```text
position = 0.80 * v3.2.2 + 0.20 * hybrid-v4
orientation = equal rotation-6D mean then Gram-Schmidt projection
```

The earlier fold-3 pilot weight 0.30 was rejected after s2 position p95
regressed by 7.6%. Weight 0.20 was then treated as a new development-CV
candidate and checked with seeds 42, 7, and 123. It was not changed after the
test opened.

### Development CV results

Pooled metrics were recomputed from all row-level out-of-fold predictions,
not averaged from fold summaries:

| Seed/rule | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| Frozen v3.2.2 baseline | 0.747421 mm | 2.353941 mm | 0.460227 deg | 0.858034 deg |
| Hybrid w=0.20, seed 42 | 0.732697 mm | 2.261255 mm | 0.358576 deg | 0.636209 deg |
| Hybrid w=0.20, seed 7 | 0.716511 mm | 2.212483 mm | 0.382881 deg | 0.670030 deg |
| Hybrid w=0.20, seed 123 | 0.731978 mm | 2.256041 mm | 0.354239 deg | 0.619090 deg |

Seed-42 development gate passed all checks: the four-metric geometric
composite improved 14.12%; no session p95 regressed more than 2.71%.

### Deployment and realtime artifacts

The deployment physics config was fit on 1,800 samples from all nine non-test
sessions only after verifying the three CV calibration gates. It generated
6,019 synthetic samples for pretraining. Fixed epochs were the seed-42 median
best CV epochs: 107 pretrain and 68 real-only fine-tune.

```text
v4 checkpoint:
checkpoints/new_calib_v4_hybrid/C3/deployment_seed42_finetune_real_rebase
best.pt SHA-256:
1c4423fa13ebc035853493682a069656d2fc81af28bc4374d5115d3afc9f57e2

frozen bundle:
checkpoints/frozen_v4_hybrid_w20_seed42/freeze_manifest.json
```

Full JSONL predictor latency, including causal feature construction, all six
networks, blend, uncertainty, and device transfer:

| Runtime | p95 |
|---|---:|
| CPU TorchScript, 6 threads | 1.9923 ms |
| RTX 5060 Ti CUDA Graph | 1.4226 ms |

Packing diagnostics into one CUDA output tensor reduced GPU p95 from 1.5653 to
1.4226 ms without changing predictions. CPU/GPU maximum pose difference on
100 non-test real rows was 0.000061; shifting all absolute timestamps changed
pose by exactly 0.0. Regression suite: 24/24 passed.

### One-time final test

The bundle, v3.2.2 manifest, v4 checkpoint, rule, configs, CV metrics, runtime,
and parity artifacts were hash-frozen before the test was materialized. The
one-time receipt is:

```text
checkpoints/frozen_v4_hybrid_w20_seed42/sealed_open_receipt.json
sealed_test.npz SHA-256:
22bec463c1397b86760bf3fe3408601a14c6e22836633ecaf9f1e5ba70639615
```

Final pooled result:

| Frozen model/rule | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| v3.2.2 | 1.122164 mm | 3.104672 mm | 2.390419 deg | 5.057797 deg |
| v3.2.2 + hybrid-v4 w=0.20 | 1.233463 mm | 3.472531 mm | 1.865280 deg | 3.746090 deg |

The hybrid improved orientation RMSE by 21.97% and orientation p95 by 25.93%,
but regressed position RMSE by 9.92% and position p95 by 11.85%. Position p95
regressed in every final session by 6.7% to 14.5%. The immutable all-four
final gate therefore failed:

```text
results/new_calib_v4_final/final_quality_gate.json
accepted: false
decision: retain_v3_2_2_baseline
```

No alternate test-derived blend, orientation-only component choice, or
fine-tuning is permitted. The experimental v4 checkpoint is retained for
research, but it is not the selected deployment.

### Required next acquisition

Collect at least three new rotating development sessions with broader
roll/pitch/yaw coverage and a separate later sealed session. The next model
may investigate preserving the v3 position branch while improving orientation,
but that rule must be selected using only new development data. The opened
`cyl_rot` set can remain a historical benchmark, not a model-selection set.

## 2026-08-04 — Paper-style frozen-result report v1

Added `reporting/build_paper_style_report.py` to regenerate structural
analogues of the reference paper's Figs. 4, 7, 8, 9 and Tables I–III. The
script reads only existing frozen histories, predictions, summaries, and the
already opened one-time final evaluation. It performs no training,
calibration, model selection, or deployment change.

Outputs are recorded in `reports/paper_style_2026_08_04/` as PNG/PDF figures,
CSV/Markdown tables, a linked `REPORT.md`, and a SHA-256 manifest. Missing
current-data FCN, KAN, and nonlinear-optimization baselines are explicitly not
fabricated; the paper's Table II values are shown only as non-comparable
context.

The Fig. 9 analogue includes a ground-truth-pose forward-reconstruction
control. Mean per-channel MAPE on the final rotating sessions was 106.19% from
ground-truth pose, 107.16% from v3.2.2 pose, and 105.39% from v4 pose. Because
the control is already poor, the dominant discrepancy is the fitted forward
model/channel correction on `cyl_rot`, not inverse-network pose error. This is
a reporting diagnostic on an opened test and cannot authorize tuning. The
selected deployment remains v3.2.2.

After visual comparison with the paper, Fig. 8 was changed from a smoothed
3,012-row pooled display to 100 uniformly sampled, unsmoothed poses from the
clean third final session. This matches the paper's plot density but not its
trajectory geometry: current data is a constant-radius five-loop cylinder,
while the paper used a tapered five-loop helix. Metrics and deployment
decisions are unchanged.

A detailed Vietnamese audit was added at
`docs/BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md`. It separates faithful
reproduction, new deployed ideas, experimental/rejected ideas, missing paper
baselines, remaining leakage risks, and the permitted claim for each figure.

## 2026-08-04 — post-final development-only synthetic-ratio ablation

Question tested: can a smaller amount of physics-synthetic data regularize C3
better than the earlier 1:1 real/synthetic stage?

Leakage boundary:

- only the three existing leave-one-`con_rot`-session-out development folds
  were loaded and scored;
- every validation set remained 100% real;
- synthetic rows were created only inside each fold's training side using the
  already gated fold-specific physics config;
- the fixed seed-42 C3 staged protocol and previously declared `w=0.20` blend
  were retained;
- `cyl_rot` was not reopened, read, scored, or used to select a ratio.

Latency was not rebenchmarked for rejected screening weights. The quality gate
uses the prior measured ratio-1.00 runtime as an explicitly marked
architecture-matched proxy because model count, tensor shapes, feature path,
blend and backend are identical. This is sufficient to reject the candidates,
but is not a candidate-specific latency measurement.

Pooled out-of-fold result:

| Synthetic / real | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 | Composite gain | Worst session p95 regression | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0, v3.2.2 reference | 0.747421 mm | 2.353941 mm | 0.460227 deg | 0.858034 deg | 0.00% | 0.00% | reference |
| 0.25 | 0.740104 mm | 2.265264 mm | 0.358824 deg | 0.624217 deg | 14.26% | +8.08% | reject |
| 0.50 | 0.729814 mm | 2.305677 mm | 0.379807 deg | 0.659359 deg | 11.75% | +6.13% | reject |
| 1.00, prior candidate | 0.732697 mm | 2.261255 mm | 0.358576 deg | 0.636209 deg | 14.12% | +2.71% | pass development |

Both new ratios improved all four aggregate metrics against the matched-fold
v3.2.2 baseline, but failed the predeclared per-session gate: fold s3 SO(3)
p95 regressed by more than 5%. Neither ratio dominates 1.00 across all four
metrics. Ratio 2.00 was not trained because the lower-ratio screen supplied no
evidence that increasing synthetic exposure would improve the balanced gate,
while the forward-model mismatch remains unresolved.

The old ratio-1.00 candidate remains the best balanced development setting,
but its hash-frozen one-time final result already failed position metrics.
Therefore no changed model is deployable and v3.2.2 remains selected. A new
rotating development acquisition and a later sealed session are required for
the next model decision.

Reproducible artifacts:

```text
results/new_calib_v4_hybrid_ratio025/quality_gate.json
results/new_calib_v4_hybrid_ratio050/quality_gate.json
reports/synthetic_ratio_ablation_2026_08_04/
evaluation/summarize_synthetic_ratio_ablation_v4.py
```

## 2026-08-05 — pure-synthetic pretraining then real-only calibration

Motivation: the reference paper first trains from analytical synthetic data,
then estimates physical-system parameters from 150 real poses and retrains.
The paper does not fully specify whether retraining directly mixes real pairs
or regenerates the full calibrated synthetic dataset. This experiment tests
the neural data order without claiming an exact reproduction.

Protocol per grouped `con_rot` fold:

1. Reuse the fold-specific forward model fitted and gated only on train-fold
   real rows.
2. Train C3 for 107 fixed epochs on one synthetic sample per available real
   row, with zero real samples in Stage 1 and no validation-based epoch choice.
3. Transfer weights to real-only Stage 2, reset feature/target normalization,
   and fine-tune against real train rows.
4. Evaluate only the held-out real `con_rot` session. No historical `cyl_rot`
   label or prediction is loaded or rescored.

Seed-42 fixed equal-orientation result:

| Protocol | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| Mixed 1× → real, prior v4.2 | 0.732697 mm | 2.261255 mm | 0.358576 deg | 0.636209 deg |
| Synthetic-only 1× → real | 0.687766 mm | 2.046979 mm | 0.380011 deg | 0.689200 deg |

Pure-synthetic pretraining improves position substantially but the old 50/50
orientation blend fails the per-session gate for seeds 7 and 123. A
development-only robustness screen selected a lower C3 orientation weight:

```text
position = 0.80 * v3.2.2 + 0.20 * C3
rotation6D = 0.70 * v3.2.2 + 0.30 * C3, then Gram-Schmidt
```

| Seed | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 | Composite gain | Worst session p95 regression | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.687766 mm | 2.046979 mm | 0.404780 deg | 0.738389 deg | 11.78% | -0.69% | pass |
| 7 | 0.707701 mm | 2.113703 mm | 0.404460 deg | 0.749455 deg | 10.12% | +1.78% | pass |
| 123 | 0.702089 mm | 2.097148 mm | 0.409627 deg | 0.739544 deg | 10.49% | +1.54% | pass |

The research deployment checkpoint used all 6,019 non-test rows: 107 fixed
pure-synthetic epochs followed by 108 fixed real-only epochs. Runtime parity
passed on 100 non-test rows; maximum CPU/GPU pose difference was 0.00003052 and
absolute timestamp shift difference was 0.0.

Three candidate-specific 2,000-run measurements produced median p95 latency:

```text
CPU TorchScript: 2.164829 ms  (passes 3.0 ms)
GPU CUDA Graph:  1.723428 ms  (fails 1.5 ms)
```

The old v4 checkpoint also measured 1.638843 ms GPU p95 on the current host,
above its historical 1.422557 ms, so current-host jitter is material. The
predeclared limit is nevertheless unchanged. The final development gate fails
on GPU latency, and there is no new unopened sealed test. Selected deployment
therefore remains v3.2.2; the new checkpoint is explicitly research-only.

Artifacts:

```text
reports/synth_then_calib_2026_08_05/
results/new_calib_v4_synth_then_calib/final_development_gate_seed42.json
checkpoints/new_calib_v4_synth_then_calib/C3/research_candidate_manifest.json
```

## 2026-08-05 — canonical full paper-vs-current comparison report

Scope: reporting only. No checkpoint training, final inference, calibration,
blend selection or deployment mutation was performed. The generator reads
frozen summaries plus the non-test deployment array for coverage statistics;
it does not read raw final labels or final prediction rows.

Outputs:

- 30-row point-by-point comparison covering physics, synthetic data,
  calibration, temporal features, network, split/leakage, metrics, runtime and
  realtime integration;
- 13 PNG/PDF figures, including parallel pipelines, architecture, leakage
  boundaries, data scale, three-seed development results, one-time final
  decision, non-head-to-head paper context, runtime and physics diagnostics;
- six CSV/Markdown tables and a source/output SHA-256 manifest;
- explicit claim matrix separating selected v3.2.2, final-rejected mixed v4.2
  and research-only v4.4 synthetic→real.

Verified conclusions:

- core task/backbone are close to the paper, but the full pipeline is not an
  exact reproduction;
- current protocol is more explicit against session/window/test leakage than
  the protocol disclosed by the paper, without claiming the paper leaked;
- current evidence is weaker in workspace/orientation/synthetic coverage and
  physics consistency;
- v4.4 improves all four development metrics for seeds 42/7/123, but has no
  valid final result, fails the predeclared GPU latency gate, and remains
  research-only;
- selected deployment remains `v3.2.2-cuda-graph-dual-runtime`.

Artifacts:

```text
reporting/build_full_pipeline_comparison.py
reports/full_pipeline_comparison_2026_08_05/REPORT.md
reports/full_pipeline_comparison_2026_08_05/manifest.json
```

## 2026-08-06 — Vietnamese paper-comparison presentation deck

Scope: reporting only. The slide generator reads the frozen figures from the
canonical full comparison; it does not read raw final labels/predictions,
train, infer, calibrate, select a model, or alter deployment.

The 16-slide, 15–18 minute narrative follows one decision-oriented sequence:

1. honest verdict and common 9-EMF→6-DoF task;
2. side-by-side pipelines and data/coverage mismatch;
3. synthetic/calibration order and architecture changes;
4. leakage boundaries, development evidence and one-time final decision;
5. non-head-to-head paper context, Fig. 8 trajectory explanation and physics gap;
6. realtime TCP/IP architecture and gated plan for the next calibration set.

The script includes exact Vietnamese narration, slide transitions, an 8-minute
short route and answers to likely questions about v4.4 selection, timestamp,
synthetic scale, paper comparability and the robot's role.

Artifacts:

```text
reporting/build_paper_comparison_presentation.py
reports/presentation_paper_comparison_2026_08_06/SLIDE_DECK.pdf
reports/presentation_paper_comparison_2026_08_06/SLIDE_SCRIPT.md
reports/presentation_paper_comparison_2026_08_06/SLIDE_OUTLINE.csv
reports/presentation_paper_comparison_2026_08_06/manifest.json
```

Decision: presentation-only update; selected deployment remains
`v3.2.2-cuda-graph-dual-runtime`, mixed v4.2 remains final-rejected and v4.4
remains research-only.

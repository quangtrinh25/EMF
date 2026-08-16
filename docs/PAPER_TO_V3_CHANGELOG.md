# Paper-to-v3 Technical Changelog

Reference paper:
`Residual_Neural_Network_for_Precise_6-DoF_Capsule_Endoscope_Localization_Using_Electromagnetic_Induction.pdf`

This document separates faithful reproduction components, system-specific
changes, and new research directions. The v3 results must not be described as a
direct reproduction or a direct numerical improvement over the paper unless an
equivalent evaluation protocol is collected.

## Method comparison

| Dimension | Paper | Historical repository path | New calibration v3 |
|---|---|---|---|
| Workspace | 500 x 500 x 500 mm | Approximately 130 mm calibrated bounds | Nominal 100 x 100 x 100 mm |
| EMF input | 9 peak amplitudes | 9 peak amplitudes | 9 amplitudes per row; 27 for selected 3-row causal window |
| Network core | 7 residual blocks, width 512 | Preserved | Preserved; input/output dimensions generalized |
| Orientation target | `cos(roll,pitch,yaw)` | Preserved | Continuous rotation-6D |
| Orientation sign | Restricted branch | Lost by cosine/arccos | Preserved |
| Orientation metric | Euler component RMSE | Unsigned Euler component RMSE | SO(3) geodesic plus wrapped per-axis diagnostics |
| Training data | 64M deterministic synthetic grid | Smaller random synthetic/hybrid data | Supervised grouped real acquisitions; physics augmentation gated |
| Split | Random 70/15/15 synthetic; distinct real test | Interleaved row dev in older path | File/session split before windows |
| Temporal context | None | Five-row experimental model | Causal W3; optional unit-delta-row branch for XYZ ensemble |
| Timestamp | None | Row order only | Optional monotonic timestamp; delta-time only |
| Forecast horizon | None | None | Configurable `horizon_steps` |
| Model selection | Paper protocol not fully recoverable | Private set influenced ensemble selection | 3-fold leave-one-rotating-session-out CV; test frozen |
| RMSE aggregation | Headline paper values | Arithmetic average of file RMSE in older pipeline | Pooled from sample squared errors |
| Test integrity | Paper real test | Historical private set was inspected repeatedly | `cyl_rot` locked and absent from result artifacts |

## Faithfully retained from the paper

- Three TX by three RX channel structure and nine EMF measurements.
- `Rz(yaw) @ Ry(pitch) @ Rx(roll)` convention.
- Dipole-field forward-model structure.
- Residual network core: width 512, seven residual blocks, two linear layers per
  block, LeakyReLU slope 0.01.
- Single-frame model remains available as M0 and as a paper-oriented baseline.

## Necessary system-specific changes

### 100 mm workspace

The project workspace is physically 100 x 100 x 100 mm. It is not rescaled to
the paper's 500 mm cube. The nominal center and tolerance are explicit in
`configs/new_calib_v3.yaml`, and predictions outside this workspace are flagged
rather than silently clipped.

### Signed orientation

The new data contains roll near 179 degrees and yaw near -120 degrees. The
paper-style cosine target cannot recover sign. v3 predicts the first two columns
of a rotation matrix, projects them to SO(3) with Gram-Schmidt, and reports a
parameterization-independent geodesic error.

### Grouped evaluation

Older row-modulo splits placed adjacent samples from the same trajectory into
train and dev. v3 assigns acquisition sessions before creating samples or
windows. The final `cyl_rot` family is never used in normalization, augmentation,
checkpoint selection, window selection, or ensemble selection.

Temporal boundaries are defined by the known acquisition row period and
invalid-row gaps. XYZ labels are not used to decide where a model window starts
or resets. Live inference requires an acquisition reset event; replay of the
legacy scans may use their known 500/1,000-row period.

### Correct pooled metrics

The older summary averaged file-level RMSE values. v3 pools sample squared
errors before taking the square root and additionally reports Euclidean mean,
median, p95, and SO(3) percentiles.

## New directions introduced by this project

### M1: causal temporal localization

The network consumes only current and past EMF rows. Window 3 was selected over
1, 5, and 9 on rotating development data. No centered/future window is used,
so online inference is causal.

### M2: timestamp-aware temporal localization

The ingestion and inference code accepts monotonic timestamps and adds only
consecutive delta-time features. This supports sampling jitter and missing-time
awareness without exposing absolute time or scripted trajectory phase. M2 is
implemented but not activated because the current files have no timestamps.

### Forecasting / latency compensation

`horizon_steps > 0` aligns past EMF windows to a future pose target. This is a
new task relative to the paper and must be reported separately from current-pose
localization.

### M3: component ensemble

The original project idea of single-frame position plus temporal orientation is
preserved and evaluated under a clean matched protocol. It was not selected for
the current data because temporal W3 also improved position substantially.

### M4: multi-seed position ensemble

Three independently initialized W3 ResNets are averaged in millimetres for XYZ.
Across three development folds this reduced mean position-axis RMSE from 0.7994
mm for seed 42 alone to 0.7460 mm. Rotation is not averaged from these three
models because doing so degraded cross-fold SO(3) error.

### M5: calibration-aware temporal orientation branch

The orientation branch keeps the current nine absolute EMF amplitudes and adds
clipped log-ratios of each past channel to its current value. This preserves
position-bearing absolute amplitude while exposing a gain-reduced motion cue.
It was rejected for XYZ but selected for orientation after three-fold CV.

### M6: unit-row-time position diversity

Each observed row is treated as one discrete time step, but the network receives
only the two causal W3 differences `Δrow`, never the absolute row index. The
standalone branch did not improve orientation, but adding it as a fourth XYZ
ensemble member reduced mean CV position-axis RMSE from 0.7460 to 0.7231 mm.

### M7: hardware-neutral inference fast path

The v3.2.1 runtime keeps the five selected models and component rule unchanged.
It aggregates position tensors before the CPU transfer, avoids the overwritten
chordal SO(3) calculation, and traces/freezes the full ensemble once at startup.
Three 5,000-iteration repeats selected 6 intra-op and 1 inter-op thread on the
i5-14600K, reducing median-run p95 batch-one model latency from 2.151 ms to
1.290 ms. Eager and TorchScript produced identical full CSV outputs on 1,001
rows. CPU, automatic device selection, and explicit CUDA remain supported;
explicit CUDA now fails clearly instead of silently falling back to CPU.

### M8: CUDA Graph dual-runtime profile

Host-side inspection found the RTX 5060 Ti that the managed sandbox had hidden.
The same five frozen checkpoints run with CUDA TorchScript for file prediction
and with a captured CUDA Graph for future batch-one realtime inference. Three
4,000-iteration CUDA Graph runs including pinned-host input copies gave a
median-repetition p95 of 0.8581 ms (about 1,165 Hz), versus 1.2902 ms for the
portable CPU profile. CPU/GPU pose
outputs differed by at most 0.00007 over 1,001 development rows and their
metrics were materially identical. This changes runtime only, not model weights,
features, split, or locked-test status.

### Fixed-epoch all-non-test deployment training

After feature, component, seed, and epoch choices were frozen by CV, deployment
models were trained on all 6,019 non-test samples for fixed epochs. There is no
validation file and no best-checkpoint selection in this stage. The 3,012
`cyl_rot` samples remain excluded from training and normalization.

### Data and checkpoint governance

- Explicit acquisition manifest.
- Malformed-row quarantine without altering raw CSVs.
- OOD and workspace warnings at inference.
- Test-locked protocol.
- Frozen checkpoint bundle containing configs, protocol, metrics, history, and
  SHA-256 hashes.

## Historical ideas retained but not currently enabled

- Multi-start physical calibration with per-channel gain/bias correction.
- Synthetic-real hybrid training.
- EMF noise/gain augmentation.
- Output correction.
- FCN, KAN, and optimization baselines.
- Single-frame/temporal ensemble.

They remain in the repository. Physics/hybrid augmentation is intentionally
gated for the new coordinate frame until channel mapping, polarity, and the
robot-to-TX transform are verified.

## Current evidence

Mean of three leave-one-`con_rot`-session-out development folds:

| Candidate | Position axis RMSE | SO(3) RMSE | Position p95 | SO(3) p95 |
|---|---:|---:|---:|---:|
| Raw W3 seed 42 | 0.7994 mm | 0.4390 deg | 2.3211 mm | 0.7523 deg |
| Raw W3 three-seed mean for both heads | **0.7460 mm** | 0.4691 deg | **2.0927 mm** | 0.7896 deg |
| Log-ratio W3 seed 42 | 1.0089 mm | **0.4219 deg** | 3.0536 mm | **0.6587 deg** |
| Raw row-time W3 seed 42 | 0.7528 mm | 0.4884 deg | 2.3262 mm | 0.8487 deg |
| v3.1: raw three-seed XYZ + log-ratio orientation | 0.7460 mm | **0.4219 deg** | 2.0927 mm | **0.6587 deg** |
| Selected v3.2: add row-time as fourth XYZ member | **0.7231 mm** | **0.4219 deg** | **2.0667 mm** | **0.6587 deg** |

These values are arithmetic means of fold-level metrics; full per-fold results
are in `results/new_calib_v3_temporal_cv/`. They estimate acquisition-session
repeatability within `con_rot`, not generalization to the locked capsule/family,
and are not direct paper comparisons.

Strong and mild session gain/offset augmentation were rejected: on fold 1 they
increased position-axis RMSE to 2.4750 mm and 1.7480 mm respectively, versus
1.0512 mm without them. EMF amplitude carries essential position information,
so enforced amplitude invariance is harmful for this dataset.

A protocol-aware comparison with the reference paper is maintained in
`docs/PAPER_COMPARISON_TABLE.md`; numerical values must not be presented as a
direct superiority claim because workspace and orientation coverage differ.

## Claims that remain unsupported

- Broad roll/yaw generalization: current spans are approximately 0.03 and 0.04
  degrees; pitch spans 10 degrees.
- Full paper reproduction: paper grid, ablations, baselines, hardware, wireless
  acquisition, physics consistency, and latency claims are not all reproduced.
- Direct superiority over the paper's 1.90 mm / 3.55 degree result: domains and
  orientation ranges differ.
- Valid physics augmentation in the new frame: hardware-validation gates are
  still false.
- Final held-out performance: `cyl_rot` has no prediction or metric artifacts
  and remains unopened for model-quality assessment.

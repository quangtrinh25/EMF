# Runbook: EMF New Calibration v3

Last updated: 2026-08-02

This is the operational source of truth for the 100 x 100 x 100 mm workspace,
signed rotation-6D model, and causal temporal pipeline. The historical
paper-reproduction commands remain available, but they are not the recommended
path for `new_calib/`.

## Current selected deployment candidate

```text
Model:             Cross-feature component ensemble
Runtime:           v3.2.1 tensor fast-path; model weights remain v3.2
Window:            3 causal rows [t-2, t-1, t]
Prediction target: capsule pose(t), horizon_steps=0
Position:          mean of raw-W3 seeds 42, 7, 123 + unit-row-time seed 42
Orientation:       current EMF + past/current log-ratio, seed 42
Output:            XYZ mm + continuous rotation-6D -> signed Euler
Training:          all 6,019 non-test samples, fixed CV-selected epochs
Timestamp feature: unit delta-row in one XYZ branch; no hardware timestamp yet
Test status:       LOCKED / NO TEST METRICS OR PREDICTIONS
```

This means the test has not been used for model-quality inference or scoring.
Because raw labels exist locally and an earlier preprocessing audit inspected
restart boundaries, use a newly acquired sealed test for the strongest
publication claim; see the leakage qualification in the experiment ledger.

Deployment checkpoints and SHA-256:

```text
raw seed42:  132b2039b22a6e4d8e9c19460b44083664ccccd08ed557bea1e6af789c947341
raw seed7:   beef41bc9b06023f8e3d78c99446cfbeafde9c32dcc12df50f61d1db2d986359
raw seed123: 5012f46a3162727d08919d214e0bd24075c5a7daa32c34babe4d92d0b8f3ec20
log seed42:  7ecf6544f57d64edc519d80a015486abfadcca53f854a4d5985209ced1e0be91
row seed42:  1514e7ef883f5eb1236ad54c624323bc3f49f62dd5002d658ac8623e7f7dc384
```

The immutable selection/training record is
`checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json`.
The former `frozen_w3_h0_seed42` is retained as a historical single-fold
candidate and must not be confused with the current deployment ensemble.

## 1. Environment check

Run from the repository root:

```bash
# From the cloned repository root:
python3 --version
python3 -c "import torch, numpy, pandas, scipy, yaml; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

Run the regression tests before a new experiment:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q calibration datasets evaluation inference models physics training tests pipeline.py
```

## 2. Validate the raw 100 mm dataset

```bash
python3 datasets/prepare_new_calib.py \
  --config configs/new_calib_v3.yaml \
  --out_dir data/new_calib_v3
```

Inspect before training:

```text
data/new_calib_v3/dataset_summary.json
data/new_calib_v3/data_quality.csv
data/new_calib_v3/quarantined_rows.csv
data/new_calib_v3/split_manifest.csv
```

Expected state for the 2026-08-02 acquisition:

```text
Raw rows:          9,033
Valid rows:        9,031
Quarantined rows:  2
Nominal workspace: 100 x 100 x 100 mm
```

Never edit the two source CSVs to hide malformed rows. Keep the original files
and let the preparation scripts record the quarantine decision.

## 3. Prepare leave-one-rotating-session-out development folds

Each fold leaves one complete `con_rot` acquisition for development. Every
`cyl_rot` acquisition remains assigned to the locked test. Run all three folds:

```bash
for SESSION in 1 2 3; do
  python3 datasets/prepare_temporal_v3.py \
    --config configs/new_calib_v3.yaml \
    --out_dir data/new_calib_v3_temporal_cv/dev_s${SESSION} \
    --window_size 3 \
    --horizon_steps 0 \
    --include_dt no \
    --feature_mode raw_window \
    --dev_con_rot_session ${SESSION}
done
```

Expected counts:

```text
Train: 5,017 / 5,019 / 5,018
Dev:   1,002 / 1,000 / 1,001
Test:  3,012
```

Preparation is safe: windows are constructed after role assignment and never
cross a file, acquisition, invalid-row gap, or session boundary. Boundaries use
the known acquisition row period (500 no-rotation, 1,000 rotation), not XYZ
labels. `segment_position_jump_mm` is `null` in the production config.

## 4. Smoke test after code changes

```bash
python3 pipeline.py new-calib-v3-temporal \
  --mode quick \
  --window_size 3 \
  --horizon_steps 0 \
  --include_dt auto \
  --feature_mode raw_window \
  --device cpu
```

Quick mode uses a small model for two epochs. Its metrics are wiring checks and
must not be reported as model quality.

## 5. Historical matched M0/M1/M3 single-development run

This older convenience command trains a matched nine-input single-frame baseline, the temporal
model, and the position/orientation ensemble. It evaluates validation only by
default.

```bash
python3 pipeline.py new-calib-v3-temporal \
  --mode full \
  --window_size 3 \
  --horizon_steps 0 \
  --include_dt auto \
  --device cpu
```

Do not add `--open-test` during model development.

Main outputs:

```text
checkpoints/new_calib_v3_temporal/w3_h0_auto/single/
checkpoints/new_calib_v3_temporal/w3_h0_auto/temporal/
results/new_calib_v3_temporal/w3_h0_auto/single/val_metrics.csv
results/new_calib_v3_temporal/w3_h0_auto/temporal/val_metrics.csv
results/new_calib_v3_temporal/w3_h0_auto/ensemble/val_metrics.csv
```

Do not use this single-session result as the final accuracy estimate. The
three-fold CV in the experiment ledger is the current selection evidence.

## 6. Optional window ablation

Only compare candidates on the rotating development acquisition. The completed
ablation used windows 3, 5, and 9. For another window:

```bash
python3 pipeline.py new-calib-v3-temporal \
  --mode full \
  --window_size 7 \
  --horizon_steps 0 \
  --include_dt auto \
  --device cpu
```

Record every tried candidate in
`results/new_calib_v3_temporal/TEMPORAL_SELECTION.md`. Do not silently discard a
failed candidate.

### Reproduce the selected three-fold comparison

For each prepared raw fold, train seeds 42, 7, and 123. The example below is
fold 1/seed 42; change both identifiers for the other eight runs:

```bash
python3 training/train_v3.py \
  --data_dir data/new_calib_v3_temporal_cv/dev_s1/temporal \
  --checkpoint_dir checkpoints/new_calib_v3_temporal_cv/dev_s1_seed42 \
  --config configs/training_v3.yaml \
  --epochs 200 --device cpu --seed 42

python3 evaluation/evaluate_v3.py \
  --data_dir data/new_calib_v3_temporal_cv/dev_s1/temporal \
  --checkpoint_dir checkpoints/new_calib_v3_temporal_cv/dev_s1_seed42 \
  --out_dir results/new_calib_v3_temporal_cv/dev_s1_seed42 \
  --split val --device cpu
```

Prepare and train the calibration-aware orientation candidate with
`--feature_mode current_plus_log_ratio`; its three fold seed-42 checkpoints are
under `checkpoints/new_calib_v3_temporal_cv_logratio/`.

Evaluate the selected component rule on a fold with:

```bash
python3 evaluation/multiseed_ensemble_v3.py \
  --data_dir data/new_calib_v3_temporal_cv/dev_s1/temporal \
  --checkpoint_dirs \
    checkpoints/new_calib_v3_temporal_cv/dev_s1_seed42 \
    checkpoints/new_calib_v3_temporal_cv/dev_s1_seed7 \
    checkpoints/new_calib_v3_temporal_cv/dev_s1_seed123 \
  --orientation_data_dir \
    data/new_calib_v3_temporal_cv_logratio/dev_s1/temporal \
  --orientation_checkpoint_dir \
    checkpoints/new_calib_v3_temporal_cv_logratio/dev_s1_seed42 \
  --out_dir results/new_calib_v3_temporal_cv/dev_s1_crossfeature_component \
  --split val --device cpu
```

### Rebuild deployment checkpoints after CV choices are frozen

Prepare all 6,019 non-test samples without creating a fake validation split:

```bash
python3 datasets/prepare_temporal_v3.py \
  --config configs/new_calib_v3.yaml \
  --out_dir data/new_calib_v3_temporal_deployment/w3_raw \
  --window_size 3 --horizon_steps 0 --include_dt no \
  --feature_mode raw_window --deployment_train_all_non_test

python3 datasets/prepare_temporal_v3.py \
  --config configs/new_calib_v3.yaml \
  --out_dir data/new_calib_v3_temporal_deployment/w3_logratio \
  --window_size 3 --horizon_steps 0 --include_dt no \
  --feature_mode current_plus_log_ratio --deployment_train_all_non_test

python3 datasets/prepare_temporal_v3.py \
  --config configs/new_calib_v3.yaml \
  --out_dir data/new_calib_v3_temporal_deployment/w3_rowtime \
  --window_size 3 --horizon_steps 0 --include_dt yes \
  --feature_mode raw_window --row_as_timestamp \
  --deployment_train_all_non_test
```

Train exact fixed epochs; do not add validation or early selection:

```bash
python3 training/train_v3.py --fixed_epochs_no_val --epochs 165 --seed 42 \
  --device cpu --config configs/training_v3.yaml \
  --data_dir data/new_calib_v3_temporal_deployment/w3_raw/temporal \
  --checkpoint_dir checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed42

python3 training/train_v3.py --fixed_epochs_no_val --epochs 130 --seed 7 \
  --device cpu --config configs/training_v3.yaml \
  --data_dir data/new_calib_v3_temporal_deployment/w3_raw/temporal \
  --checkpoint_dir checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed7

python3 training/train_v3.py --fixed_epochs_no_val --epochs 118 --seed 123 \
  --device cpu --config configs/training_v3.yaml \
  --data_dir data/new_calib_v3_temporal_deployment/w3_raw/temporal \
  --checkpoint_dir checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed123

python3 training/train_v3.py --fixed_epochs_no_val --epochs 132 --seed 42 \
  --device cpu --config configs/training_v3.yaml \
  --data_dir data/new_calib_v3_temporal_deployment/w3_logratio/temporal \
  --checkpoint_dir checkpoints/new_calib_v3_temporal_deployment/w3_logratio_seed42

python3 training/train_v3.py --fixed_epochs_no_val --epochs 79 --seed 42 \
  --device cpu --config configs/training_v3.yaml \
  --data_dir data/new_calib_v3_temporal_deployment/w3_rowtime/temporal \
  --checkpoint_dir checkpoints/new_calib_v3_temporal_deployment/w3_rowtime_seed42
```

## 7. Historical single-model freeze command

This command documents how the former single-fold candidate was frozen. The
current five-model deployment is frozen by `deployment_manifest.json` instead.
Freeze only after window, loss, seed, augmentation, and stopping policy are no
longer being changed:

```bash
python3 evaluation/freeze_v3_candidate.py \
  --checkpoint_dir checkpoints/new_calib_v3_temporal/w3_h0_auto/temporal \
  --data_dir data/new_calib_v3_temporal_runs/w3_h0_auto/temporal \
  --val_metrics results/new_calib_v3_temporal/w3_h0_auto/temporal/val_metrics.csv \
  --out_dir checkpoints/new_calib_v3_temporal/frozen_w3_h0_seed42 \
  --selection_reason "Window 3 dominates M0, W5, W9, and M3 on rotating-dev; test remained unopened."
```

The freeze tool refuses to overwrite an existing frozen directory and records
SHA-256 hashes for the checkpoint, normalization, configs, protocol, history,
and validation result.

## 8. Predict capsule pose with the selected ensemble

For a current headerless 15-column calibration CSV:

```bash
python3 pipeline.py predict-v3-ensemble \
  --input new_calib/conrot3_data.csv \
  --output results/capsule_pose_ensemble.csv \
  --checkpoint_dirs \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed42 \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed7 \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed123 \
  --aux_position_checkpoint_dirs \
    checkpoints/new_calib_v3_temporal_deployment/w3_rowtime_seed42 \
  --orientation_checkpoint_dir \
    checkpoints/new_calib_v3_temporal_deployment/w3_logratio_seed42 \
  --reset_every_rows 1000 \
  --input_unit mV \
  --header no \
  --device cpu \
  --backend torchscript \
  --cpu_threads 6 \
  --cpu_interop_threads 1
```

`--reset_every_rows 1000` is only for replaying the old rotating robot files.
For a live stream, send an explicit reset event/column when the acquisition
starts or reconnects. Do not detect resets from the unknown XYZ target.

The inference command constructs both causal feature views and reports:

- signed Euler angles converted from rotation-6D;
- optional metrics when labels exist;
- input OOD fraction relative to training normalization;
- predictions outside the nominal 100 mm workspace.
- model-disagreement values for position and orientation;
- warm-up rows that must not be sent to downstream robot control.

CPU batch-one benchmark (TorchScript, 6 intra-op, 1 inter-op thread; network
plus aggregation only): median-run p95 1.290 ms, about 775.1 Hz across three
5,000-iteration repeats. This excludes one-time trace/freeze startup, feature
construction, sensor acquisition, preprocessing transport, and TCP/IP latency.
The v3.2.1 fast-path has exactly the same selected checkpoints and produced a
maximum full-output difference of 0 on a 1,001-row eager/TorchScript check.

Device choice:

```text
--device cpu   portable baseline; use the measured 6/1 thread profile on this PC
--device auto  use CUDA if PyTorch can access it, otherwise use CPU
--device cuda  require CUDA; stop rather than silently falling back to CPU
--backend auto CPU/CUDA -> TorchScript for ordinary prediction
```

For GPU file prediction, use the same command with `--device cuda --backend
torchscript` and omit the CPU thread flags. The host contains an RTX 5060 Ti
16 GB with driver 580.173.02/CUDA 13.0. The earlier failed visibility check was
inside a managed sandbox that hid `/dev/nvidia*`; normal host-terminal execution
sees CUDA. Full v3 training likewise accepts `--device cuda`; explicit CUDA
fails clearly if the calling environment really cannot access it.
Training on GPU does not require GPU deployment: checkpoint loading uses
device mapping, so a GPU-trained model can run later with `--device cpu`.

Reproduce the selected CPU benchmark without touching the locked test:

```bash
PYTHONPATH=. python3 evaluation/time_inference_v3_ensemble.py \
  --data_dir data/new_calib_v3_temporal_cv/dev_s3/temporal \
  --checkpoint_dirs \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed42 \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed7 \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed123 \
  --aux_position_data_dirs \
    data/new_calib_v3_temporal_cv_rowtime/dev_s3/temporal \
  --aux_position_checkpoint_dirs \
    checkpoints/new_calib_v3_temporal_deployment/w3_rowtime_seed42 \
  --orientation_data_dir \
    data/new_calib_v3_temporal_cv_logratio/dev_s3/temporal \
  --orientation_checkpoint_dir \
    checkpoints/new_calib_v3_temporal_deployment/w3_logratio_seed42 \
  --split val --iterations 5000 --warmup 300 \
  --device cpu --backend torchscript_ensemble \
  --threads 6 --interop_threads 1
```

Run it three times and report the median of the three p95 values. This benchmark
uses only development features; it neither predicts nor scores `cyl_rot`.

For the future batch-one realtime service, reproduce the GPU profile by changing
the benchmark tail to:

```bash
--iterations 4000 --warmup 300 \
--device cuda --backend cuda_graph \
--input_residency host_pinned \
--threads 6 --interop_threads 1
```

Run it three times. The measured p95 values were 0.8670, 0.8581 and 0.8530 ms;
median 0.8581 ms (about 1,165 Hz). Per-row pinned-host input copy is included;
CUDA Graph capture/startup and uncertainty post-processing are excluded. Capture
must occur once when the future service starts, before sending poses over TCP/IP.

## 9. Timestamped acquisition and delta-time features

Current CSVs have no timestamp. Do not use row index, Unix time, elapsed
trajectory fraction, or `t/T` as an input feature.

New acquisition files should use a header and a monotonic hardware timestamp:

```text
timestamp_ns,EMF1,...,EMF9,X,Y,Z,roll,pitch,yaw
```

For each timestamped file entry in `configs/new_calib_v3.yaml`, add:

```yaml
header: true
timestamp_column: timestamp_ns
timestamp_unit: ns
```

Then prepare/train with:

```bash
python3 pipeline.py new-calib-v3-temporal \
  --mode full \
  --window_size 3 \
  --horizon_steps 0 \
  --include_dt auto
```

`auto` adds only consecutive `delta_t` values when every acquisition has valid
timestamps. It refuses mixed timestamped/non-timestamped protocols. Absolute
timestamps never enter the model.

Timestamped inference example:

```bash
python3 pipeline.py predict-v3 \
  --input timestamped.csv \
  --output results/timestamped_predictions.csv \
  --checkpoint_dir checkpoints/timestamp_candidate \
  --header yes \
  --timestamp_column timestamp_ns \
  --timestamp_unit ns
```

Synchronize robot pose and EMF clocks before training. If they are sampled on
different clocks, interpolate the robot pose to the EMF timestamp and record
the estimated clock offset.

The selected raw and log-ratio checkpoints have `dt_feature_count=0`; merely
adding a hardware timestamp does not change their prediction. To exploit real
sampling intervals, collect timestamped calibration data and retrain a new CV
candidate with delta-time features. Never feed absolute wall-clock time or
trajectory progress into the network.

Exception: the selected auxiliary XYZ model intentionally uses the experimental
`delta_row_steps_only` feature. It treats each observed row as one time step and
receives only `[Δrow(t-2,t-1), Δrow(t-1,t)]`; it never receives absolute row
number. The predictor detects this requirement from checkpoint metadata, so no
extra inference flag is needed.

## 10. Forecasting / latency compensation

To predict `N` samples after the most recent EMF input:

```bash
python3 pipeline.py new-calib-v3-temporal \
  --mode full \
  --window_size 3 \
  --horizon_steps 2 \
  --include_dt auto
```

Report the horizon in milliseconds only when sample rate or timestamps are
known. Otherwise report it as samples.

### Planned realtime TCP/IP boundary

At runtime the localization model consumes EMF, not robot pose. The robot pose
is a supervised label only during calibration. A future TCP service should use
this logical message flow:

```text
EMF acquisition -> {sequence, sensor_timestamp_ns, emf[9], reset}
localizer       -> {sequence, pose_xyz_mm, pose_rpy_deg,
                    position_dispersion_mm, orientation_dispersion_deg,
                    warmup, ood_fraction, outside_workspace}
robot controller <- consumes the localized capsule pose
```

Use a length-prefixed binary or JSON frame, sequence numbers, explicit units,
and an acquisition-reset flag. Reject stale/out-of-order messages. The robot
must ignore rows marked warm-up, and safety thresholds for OOD/disagreement
must be calibrated on independent data before closed-loop actuation. TCP/IP
transport is not implemented yet; the predictor and output fields are ready
for that later service layer.

## 11. Opening the locked test exactly once

> Do not run this section while any architecture, window, seed, loss,
> augmentation, or selection rule is still being tuned.

Use the four fixed-epoch deployment checkpoints directly; do not retrain in the
test command:

```bash
python3 evaluation/multiseed_ensemble_v3.py \
  --data_dir data/new_calib_v3_temporal_deployment/w3_raw/temporal \
  --checkpoint_dirs \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed42 \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed7 \
    checkpoints/new_calib_v3_temporal_deployment/w3_raw_seed123 \
  --aux_position_data_dirs \
    data/new_calib_v3_temporal_deployment/w3_rowtime/temporal \
  --aux_position_checkpoint_dirs \
    checkpoints/new_calib_v3_temporal_deployment/w3_rowtime_seed42 \
  --orientation_data_dir \
    data/new_calib_v3_temporal_deployment/w3_logratio/temporal \
  --orientation_checkpoint_dir \
    checkpoints/new_calib_v3_temporal_deployment/w3_logratio_seed42 \
  --out_dir results/new_calib_v3_temporal_deployment/final_test \
  --split test \
  --confirm_open_locked_test \
  --device cpu
```

Immediately after the run:

1. Record the result in `docs/EXPERIMENT_LEDGER.md`.
2. Record checkpoint/config hashes.
3. Mark the test as opened.
4. Do not tune on that test result.
5. Any later model change requires a new, independently acquired final test.

## 12. Physics/hybrid gate

Do not enable physics-based synthetic augmentation for the new frame until all
three fields below are verified in `configs/new_calib_v3.yaml`:

```yaml
channel_mapping_verified: true
channel_polarity_verified: true
pose_frame_transform_verified: true
```

Supervised temporal training is allowed while these remain false. A claim of a
validated physical forward model is not.

## 13. Backup and version control

At audit time this workspace was not a valid Git worktree, so Git history could
not be used to reconstruct past decisions. Before collecting another dataset or
opening final test, back up at least:

```text
docs/
configs/new_calib_v3.yaml
configs/training_v3.yaml
checkpoints/new_calib_v3_temporal/frozen_w3_h0_seed42/
checkpoints/new_calib_v3_temporal_deployment/
results/new_calib_v3_temporal_cv/
results/new_calib_v3_temporal_cv_logratio/
data/new_calib_v3_temporal_deployment/w3_raw/temporal/protocol.json
data/new_calib_v3_temporal_deployment/w3_logratio/temporal/protocol.json
data/new_calib_v3_temporal_deployment/w3_rowtime/temporal/protocol.json
```

Prefer placing the project in a real version-controlled repository and commit
code/config/document changes separately from generated datasets and checkpoints.

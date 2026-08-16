# EMF Capsule Localization Pipeline

This repository trains, evaluates, reports, and runs an EMF capsule-localization
model from 9 EMF channels.

The leakage-safe cross-generation v4 upgrade is implemented. Its complete
data, CUDA training, quality-gate, sealed-test, and realtime command sequence is
in [`docs/V4_UPGRADE_RUNBOOK.md`](docs/V4_UPGRADE_RUNBOOK.md). The existing v4
candidates did not pass every frozen deployment gate, so v3.2.2 remains the
selected model; v4.4 remains research-only pending newly collected rotating
development sessions and a later sealed test.

This public source tree intentionally excludes raw robot calibration data,
sealed-test rows, generated datasets, checkpoints, predictions, virtual
environments, and the reference-paper PDF. Put authorized calibration CSVs in
`new_calib/` and regenerate all data/model artifacts locally. Do not commit a
sealed test before its one-time evaluation protocol is complete.

## Start here

The current recommended system is the 100 mm signed-orientation temporal v3
pipeline. Use these documents as the source of truth:

- [`docs/RUNBOOK_NEW_CALIB_V3.md`](docs/RUNBOOK_NEW_CALIB_V3.md): exact commands
  from raw data validation through frozen inference and one-time final test.
- [`docs/PAPER_TO_V3_CHANGELOG.md`](docs/PAPER_TO_V3_CHANGELOG.md): what was
  retained, corrected, or added relative to the paper and historical pipeline.
- [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md): dated decisions,
  metrics, hashes, and the current locked-test state.
- [`docs/PAPER_COMPARISON_TABLE.md`](docs/PAPER_COMPARISON_TABLE.md): protocol-
  aware table comparing the current system with the reference paper.
- [`docs/MODEL_UPGRADE_HISTORY.csv`](docs/MODEL_UPGRADE_HISTORY.csv): one-row-
  per-upgrade metric history.

Current selected deployment manifest:

```text
checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json
```

Do not open `cyl_rot` test until the runbook's freeze checklist is satisfied.

## New calibration v3: 100 mm workspace and signed orientation

The recommended workflow for `new_calib/` is separate from the historical
paper-reproduction workflow. It keeps the physical workspace at exactly
`100 x 100 x 100 mm`, groups all repeated files from one trajectory family in
the same split, and predicts `XYZ + rotation-6D`. Reported orientation error is
the signed SO(3) geodesic distance rather than `arccos(cos(Euler))`.

Validate all files and create the four family-held-out folds:

```bash
python3 datasets/prepare_new_calib.py
```

Run an end-to-end smoke test on the frozen primary protocol:

```bash
python3 pipeline.py new-calib-v3 --mode quick
```

Train the full seven-block model. This evaluates validation only and leaves the
held-out `cyl_rot` family unopened:

```bash
python3 pipeline.py new-calib-v3 --mode full
```

Do not use the convenience `--open-test` path for the selected deployment.
Follow the one-time guarded ensemble command in the runbook instead.

Predict signed Euler angles from a raw CSV using a v3 checkpoint:

```bash
python3 pipeline.py predict-v3 \
  --input new_calib/connorot_data.csv \
  --output results/new_calib_v3/predictions.csv \
  --checkpoint_dir checkpoints/new_calib_v3/fold_cyl_rot \
  --header no
```

The protocol and acquisition manifest are in
`configs/new_calib_v3.yaml`. Its hardware-validation gates intentionally block
any claim that the old dipole calibration is valid in the new frame. Verify
channel mapping, channel polarity and the robot-to-transmitter rigid transform
before enabling synthetic physics augmentation.

Important limitation: the current files span the 100 mm position cube but only
exercise pitch materially; roll and yaw remain almost constant. The v3 output
is capable of signed full 6-DoF, but the current dataset does not validate broad
roll/yaw generalization.

## Temporal v3 and timestamp-ready workflow

The temporal workflow creates causal windows inside each acquisition only. Its
matched protocol uses sessions 1-2 of `con_rot` for training, session 3 as the
rotating repeatability-development set, and keeps every `cyl_rot` acquisition
locked as the generalization test.

Run the complete M0/M1/M3 comparison without opening test:

```bash
python3 pipeline.py new-calib-v3-temporal \
  --mode full \
  --window_size 5 \
  --horizon_steps 0 \
  --include_dt auto
```

The window ablation selected a three-row causal window. The current deployment
uses three raw-W3 XYZ members, one unit-row-time W3 XYZ member, and one
log-ratio orientation member:

```text
checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json
```

Predict with it:

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
  --header no \
  --device cpu \
  --backend torchscript \
  --cpu_threads 6 \
  --cpu_interop_threads 1
```

This CPU profile measured 1.290 ms median-run p95 for batch-one network
inference plus ensemble aggregation on the current i5-14600K (about 775.1 Hz).
`--backend auto` selects TorchScript on both CPU and CUDA. Use
`--device auto` to select CUDA when available and CPU otherwise. Use `--device
cuda` when GPU execution is mandatory; it stops with an explicit error if
PyTorch cannot access the NVIDIA driver instead of silently using CPU. The
selected model and all checked predictions are unchanged by this optimization.

The host also has an RTX 5060 Ti 16 GB. Three CUDA Graph batch-one runs including
per-row pinned-host input copies measured 0.858 ms median-run p95 (about 1,165
Hz), while CUDA TorchScript file prediction
measured 1.172 ms p95 in the initial probe. The managed tool sandbox hides the
GPU device nodes, but normal host-terminal execution sees CUDA correctly.

Current acquisitions have no hardware timestamp column. The selected auxiliary
XYZ branch treats every observed row as one unit step and uses only `delta-row`
features, never the absolute row index. For a future timestamped CSV,
add `header: true`, `timestamp_column`, and `timestamp_unit` to its file entry in
`configs/new_calib_v3.yaml`. Inference then accepts, for example:

```bash
python3 pipeline.py predict-v3 \
  --input timestamped.csv \
  --output results/timestamped_predictions.csv \
  --checkpoint_dir checkpoints/your_timestamp_model \
  --header yes \
  --timestamp_column timestamp_ns \
  --timestamp_unit ns
```

Only consecutive `delta_t` values enter the model. Absolute time, row index,
and normalized trajectory phase are intentionally excluded to prevent temporal
shortcut leakage. Set `--horizon_steps N` to train a latency-compensation model
that predicts pose `N` samples after the latest EMF row.

## Basic Commands

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Train the full no-leak balanced model:

```bash
python3 pipeline.py noleak-balanced
```

Train the time-aware temporal model:

```bash
python3 pipeline.py temporal-balanced
```

Run a fast smoke test:

```bash
python3 pipeline.py noleak-balanced --mode quick
python3 pipeline.py temporal-balanced --mode quick
```

Run the private real test using the saved final model:

```bash
python3 pipeline.py private-test
```

Create 9-EMF-only private files and predict their poses:

```bash
python3 pipeline.py private-9emf
```

Create 9-EMF-only private files and predict with the temporal model:

```bash
python3 pipeline.py private-9emf-temporal
```

Run the ensemble private test, using single-row position and temporal orientation:

```bash
python3 pipeline.py ensemble-private
```

Create 9-EMF-only private files and predict with the ensemble:

```bash
python3 pipeline.py private-9emf-ensemble
```

Run only the raw private test:

```bash
python3 pipeline.py private-test --mode raw
```

Run only the corrected private test:

```bash
python3 pipeline.py private-test --mode corrected
```

Predict pose from a CSV:

```bash
python3 pipeline.py predict \
  --input your_file.csv \
  --output results/predicted_capsule_pose.csv
```

For raw calibration-style CSVs in `data/raw_calibration/`, use millivolts:

```bash
python3 pipeline.py predict \
  --input data/raw_calibration/Set12_cyl_spi_rot_2.csv \
  --output results/predicted_capsule_pose.csv \
  --input_unit mV \
  --header no
```

The prediction output contains:

```text
sample_index,
pred_X_mm, pred_Y_mm, pred_Z_mm,
pred_roll_deg, pred_pitch_deg, pred_yaw_deg
```

If the input also contains target pose columns, a metrics file is written next to the prediction CSV.

## What Each Basic Command Writes

| Command | Main outputs |
|---|---|
| `python3 pipeline.py noleak-balanced` | `checkpoints/noleak_balanced_easy/final/`, `results/noleak_balanced_easy/final_summary.md` |
| `python3 pipeline.py temporal-balanced` | `checkpoints/temporal_balanced_easy/final/`, `results/temporal_balanced_easy/final_summary.md` |
| `python3 pipeline.py noleak-balanced --mode quick` | `checkpoints/noleak_balanced_easy_quick/final/`, `results/noleak_balanced_easy_quick/final_summary.md` |
| `python3 pipeline.py temporal-balanced --mode quick` | `checkpoints/temporal_balanced_easy_quick/final/`, `results/temporal_balanced_easy_quick/final_summary.md` |
| `python3 pipeline.py private-test` | `results/private_test/private_real_pose_metrics.csv`, corrected metrics if available |
| `python3 pipeline.py private-9emf` | `results/private_9emf_inputs/`, `results/private_9emf_predictions/` |
| `python3 pipeline.py private-9emf-temporal` | `results/private_9emf_temporal_inputs/`, `results/private_9emf_temporal_predictions/` |
| `python3 pipeline.py ensemble-private` | `results/ensemble_private/ensemble_pose_metrics.csv`, `results/ensemble_private/ensemble_pose_predictions.csv` |
| `python3 pipeline.py private-9emf-ensemble` | `results/private_9emf_ensemble_inputs/`, `results/private_9emf_ensemble_predictions/` |
| `python3 pipeline.py predict --input your_file.csv --output results/predicted_capsule_pose.csv` | `results/predicted_capsule_pose.csv`, optional metrics CSV |

The deployment checkpoint folder is:

```text
checkpoints/noleak_balanced_easy/final/
```

Keep the whole folder, not only `resnet_best.pt`, because prediction also needs the normalization files:

```text
resnet_best.pt
emf_mean.npy
emf_std.npy
pose_mean.npy
pose_std.npy
pose_output_correction.npz
```

## Easy No-Leak Workflow

The main command is:

```bash
python3 pipeline.py noleak-balanced
```

Default split:

| Role | Files |
|---|---|
| Calibration, hybrid training, real dev source | `set10_cyl_no_rot`, `set11_con_spi_rot` |
| Private final test | `set13_con_spi_no_rot`, `set12_cyl_spi_rot` |

The command does:

```text
1. Create deterministic real train/dev/private CSV splits
2. Check that source rows do not leak across roles
3. Tune augmentation/loss settings on real-dev only
4. Train the final balanced model on all non-private rows
5. Fit output correction from non-private rows
6. Evaluate private test rows once
7. Write summary files
```

Modes:

| Command | Purpose |
|---|---|
| `python3 pipeline.py noleak-balanced` | Full workflow with tuning grid |
| `python3 pipeline.py noleak-balanced --mode quick` | Small smoke test: 2 epochs, 2048 synthetic samples, CPU |
| `python3 pipeline.py noleak-balanced --mode final-only` | Skip tuning and train one default balanced model |

Default output paths:

```text
configs/physics_calibrated_v2_noleak_balanced_easy.yaml
data/real_splits/noleak_balanced_easy/
data/synthetic_noleak_balanced_easy/
data/hybrid_noleak_balanced_easy/
checkpoints/noleak_balanced_easy/final/
results/noleak_balanced_easy/
```

Key final outputs:

```text
results/noleak_balanced_easy/tuning_summary.csv
results/noleak_balanced_easy/private_real_pose_metrics.csv
results/noleak_balanced_easy/private_real_pose_metrics_corrected.csv
results/noleak_balanced_easy/final_summary.md
checkpoints/noleak_balanced_easy/final/resnet_best.pt
checkpoints/noleak_balanced_easy/final/pose_output_correction.npz
```

## Temporal Workflow

The temporal command is:

```bash
python3 pipeline.py temporal-balanced
```

It treats CSV row order as time order. For each row `t`, the model predicts the pose at `t` from the current and previous EMF rows:

```text
[EMF(t-4), EMF(t-3), EMF(t-2), EMF(t-1), EMF(t)]
```

Default temporal input size:

```text
5 rows * 9 EMF channels = 45 features
```

The first rows are padded by repeating the first EMF row. Windows are built inside each CSV file only, so they never cross trajectory boundaries.

Temporal outputs:

```text
checkpoints/temporal_balanced_easy/final/resnet_best.pt
checkpoints/temporal_balanced_easy/final/model_metadata.json
results/temporal_balanced_easy/private_real_pose_metrics.csv
results/temporal_balanced_easy/final_summary.md
```

## Model

Input:

```text
EMF1, EMF2, EMF3, EMF4, EMF5, EMF6, EMF7, EMF8, EMF9
```

Output:

```text
X_mm, Y_mm, Z_mm, roll_deg, pitch_deg, yaw_deg
```

Internally, the neural network predicts:

```text
X_mm, Y_mm, Z_mm, cos_roll, cos_pitch, cos_yaw
```

Architecture:

| Item | Value |
|---|---:|
| Input channels | 9 |
| Hidden size | 512 |
| Residual blocks | 7 |
| Output values | 6 |
| Trainable parameters | 3,748,742 |

## Data Protocol

Raw local calibration data is stored under:

```text
data/raw_calibration/
```

File mapping:

| Dataset key | CSV file | Rows |
|---|---|---:|
| `set10_cyl_no_rot` | `set10_2304_cyl_no_rot.csv` | 100 |
| `set11_con_spi_rot` | `Set11_con_spi_rot_2.csv` | 200 |
| `set12_cyl_spi_rot` | `Set12_cyl_spi_rot_2.csv` | 200 |
| `set13_con_spi_no_rot` | `Set13_con_spi_no_rot.csv` | 100 |

Raw calibration CSV format:

```text
EMF1..EMF9, X, Y, Z, roll, pitch, yaw
```

EMF values in these files are millivolts. The pipeline converts them to volts internally.

## Detailed Commands

Most users should use:

```bash
python3 pipeline.py private-test
```

Equivalent advanced raw command:

```bash
python3 pipeline.py evaluate-real \
  --checkpoint_dir checkpoints/noleak_balanced_easy/final \
  --out_dir results/private_test/raw \
  --eval_files set13_con_spi_no_rot set12_cyl_spi_rot
```

Equivalent advanced corrected command:

```bash
python3 pipeline.py evaluate-real \
  --checkpoint_dir checkpoints/noleak_balanced_easy/final \
  --out_dir results/private_test/corrected \
  --eval_files set13_con_spi_no_rot set12_cyl_spi_rot \
  --pose_correction checkpoints/noleak_balanced_easy/final/pose_output_correction.npz
```

Use explicit CSV paths:

```bash
python3 pipeline.py evaluate-real \
  --checkpoint_dir checkpoints/noleak_balanced_easy/final \
  --out_dir results/noleak_balanced_easy/dev_check \
  --eval_csvs \
    set10_dev=data/real_splits/noleak_balanced_easy/real_dev/set10_cyl_no_rot.csv \
    set11_dev=data/real_splits/noleak_balanced_easy/real_dev/set11_con_spi_rot.csv
```

## Notes

- The no-leak private test is the defensible benchmark.
- The synthetic test checks whether the neural inverse model learned the generated calibrated physics mapping.
- Older all-data adapted artifacts are archived under `rublsh/`; those can be useful for deployment on the same machine, but they are not a clean held-out benchmark.

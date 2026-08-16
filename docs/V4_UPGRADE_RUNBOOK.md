# v4 Physics-Hybrid Calibration Runbook — 100 mm Workspace

Last updated: 2026-08-04

## Current decision

- Raw calibration source: `new_calib/`, 12 headerless robot CSV files.
- Development/train: nine non-rotating `con_rot`/`cyl_norot` sessions, 6,019 valid rows.
- One-time final test: three `cyl_rot` sessions, 3,012 rows.
- The final test was opened exactly once after the v4 bundle was hash-frozen.
- v4 passed development CV but failed the final balanced gate because position
  error increased. The selected deployable model therefore remains v3.2.2.
- Do not tune another model or blend weight against the opened `cyl_rot` data.
  A changed model requires newly collected rotating development sessions and a
  newly sealed final test.

Final decision artifact:

```text
results/new_calib_v4_final/final_quality_gate.json
decision: retain_v3_2_2_baseline
```

## Data and time semantics

Each CSV row is one robot-programmed fixed `delta-t`:

```text
EMF1..EMF9,X,Y,Z,roll,pitch,yaw
```

There is no hardware timestamp column in the current files. The causal W3
feature uses three rows and two delta-row values:

```text
normal row transition: 1
padding after reset:   0
absolute timestamp:    never a model feature
C3 packed input:       27 EMF + 2 delta-row = 29
C3 internal features:  27 raw + 18 log-ratio + 2 delta-row = 47
```

At realtime inference, `timestamp_ns` may be sent for packet ordering, reset,
and future TCP/IP monitoring. Changing its absolute offset must not change the
pose. It is not used as a predictor feature for this calibration generation.

## Leakage boundary

The protected files are:

```text
new_calib/cylspirot.csv
new_calib/cylspin2rot_data.csv
new_calib/cylspirot3.csv
```

Development folds are `real_current__con_rot__s1`, `s2`, and `s3`. Causal
windows are constructed only after the fold split and never cross a file or
reset. Physics parameters are fit only on each fold's training sessions.
Synthetic samples are appended only to train; validation remains 100% real.

The one-time receipt now exists at:

```text
checkpoints/frozen_v4_hybrid_w20_seed42/sealed_open_receipt.json
```

The loader will refuse a second sealed opening.

## Reproduce development folds

List the legal folds without loading protected labels:

```bash
./.venv/bin/python datasets/prepare_multigeneration_v4.py \
  --config configs/new_calib_v4.yaml --list_folds
```

Prepare one C3 real fold:

```bash
./.venv/bin/python datasets/prepare_multigeneration_v4.py \
  --config configs/new_calib_v4.yaml --candidate C3 \
  --fold real_current__con_rot__s1 \
  --out_dir data/new_calib_v4_real/C3/dev_s1
```

## Fit and gate the fold physics generator

```bash
./.venv/bin/python -u calibration/calibrate_v4_fold.py \
  --config configs/new_calib_v4.yaml \
  --fold real_current__con_rot__s1 \
  --out_config configs/physics_v4_dev_s1.yaml \
  --out_dir results/physics_v4/dev_s1 \
  --max_fit_samples 1200 --starts 4 --max_nfev 800 --seed 42
```

Required physics gate:

```text
held-out flat correlation >= 0.80
held-out RMSE / mean(|EMF|) <= 0.50
test_labels_loaded == false
```

Observed held-out values:

| Fold | Correlation | RMSE/mean | Synthetic allowed |
|---|---:|---:|---|
| s1 | 0.9149 | 0.3945 | yes |
| s2 | 0.8784 | 0.4739 | yes |
| s3 | 0.9181 | 0.3905 | yes |

## Build staged hybrid training data

```bash
./.venv/bin/python datasets/build_hybrid_v4.py \
  --real_data_dir data/new_calib_v4_real/C3/dev_s1 \
  --physics_config configs/physics_v4_dev_s1.yaml \
  --out_dir data/new_calib_v4_hybrid/C3/dev_s1 \
  --synthetic_ratio 1.0 --seed 42

./.venv/bin/python datasets/prepare_real_finetune_v4.py \
  --real_data_dir data/new_calib_v4_real/C3/dev_s1 \
  --hybrid_data_dir data/new_calib_v4_hybrid/C3/dev_s1 \
  --out_dir data/new_calib_v4_hybrid/C3/dev_s1_finetune_real
```

Stage 1 contains equal counts of real and physics-synthetic train rows. Stage
2 contains real train rows only but preserves the model's two-generation
layout so weights transfer safely.

## Train and evaluate one CV model

```bash
./.venv/bin/python -u training/train_v4.py \
  --data_dir data/new_calib_v4_hybrid/C3/dev_s1 \
  --checkpoint_dir checkpoints/new_calib_v4_hybrid/C3/dev_s1_seed42 \
  --config configs/training_v4.yaml --seed 42 --device cuda

./.venv/bin/python -u training/train_v4.py \
  --data_dir data/new_calib_v4_hybrid/C3/dev_s1_finetune_real \
  --checkpoint_dir checkpoints/new_calib_v4_hybrid/C3/dev_s1_seed42_finetune_real_rebase \
  --config configs/training_v4_finetune.yaml --seed 42 --device cuda \
  --init_checkpoint checkpoints/new_calib_v4_hybrid/C3/dev_s1_seed42 \
  --reset_normalization_on_init

./.venv/bin/python evaluation/evaluate_v4.py \
  --data_dir data/new_calib_v4_hybrid/C3/dev_s1_finetune_real \
  --checkpoint_dir checkpoints/new_calib_v4_hybrid/C3/dev_s1_seed42_finetune_real_rebase \
  --out_dir results/new_calib_v4_hybrid/C3/dev_s1_seed42_finetune_real_rebase \
  --split val --device cuda
```

Seeds 42, 7, and 123 were run on all three folds. The screened ensemble rule
was frozen as:

```text
position = 0.80 * v3.2.2 + 0.20 * hybrid-v4
orientation = equal rotation-6D mean, then Gram-Schmidt projection
```

The runtime-friendly rotation-6D mean was numerically equivalent to the
two-rotation chordal/SVD mean on development predictions.

## Development result

Pooled row-level CV for seed 42:

| Model | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| v3.2.2 baseline | 0.7474 mm | 2.3539 mm | 0.4602 deg | 0.8580 deg |
| v4 hybrid ensemble w=0.20 | 0.7327 mm | 2.2613 mm | 0.3586 deg | 0.6362 deg |

Seeds 7 and 123 also improved all four pooled metrics. Development quality
gate composite improvement was 14.12%; worst session p95 regression was 2.71%.

## Development-only synthetic-ratio ablation after final opening

The already-opened `cyl_rot` final cannot be used again for tuning. A legal
seed-42 screen therefore changed only the Stage-1 synthetic/real ratio and
evaluated the same three grouped `con_rot` development folds. The fixed
`w=0.20` blend and all gate thresholds were retained.

To build another ratio, change both the output prefix and
`--synthetic_ratio`; validation remains real-only:

```bash
./.venv/bin/python datasets/build_hybrid_v4.py \
  --real_data_dir data/new_calib_v4_real/C3/dev_s1 \
  --physics_config configs/physics_v4_dev_s1.yaml \
  --out_dir data/new_calib_v4_hybrid_ratio025/C3/dev_s1 \
  --synthetic_ratio 0.25 --seed 42
```

The completed ratios use these separate data/checkpoint/result prefixes:

```text
data/new_calib_v4_hybrid_ratio025
checkpoints/new_calib_v4_hybrid_ratio025
results/new_calib_v4_hybrid_ratio025

data/new_calib_v4_hybrid_ratio050
checkpoints/new_calib_v4_hybrid_ratio050
results/new_calib_v4_hybrid_ratio050
```

The train, real-only fine-tune, evaluation and fold aggregation commands are
identical to the 1.00 procedure above, with the matching prefix. Official gate
artifacts are:

```text
results/new_calib_v4_hybrid_ratio025/quality_gate.json
results/new_calib_v4_hybrid_ratio050/quality_gate.json
```

Because both screening candidates were rejected before deployment training,
their latency check uses the measured ratio-1.00 runtime only as an explicitly
marked architecture-matched proxy. The graph, six-network operation count,
feature path and `w=0.20` blend are identical; there is no claim that the two
rejected checkpoints received a new candidate-specific latency benchmark.

Rebuild the combined table and figure without reading final data:

```bash
./.venv/bin/python evaluation/summarize_synthetic_ratio_ablation_v4.py
```

| Synthetic / real | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 | Worst session p95 regression | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 0, v3.2.2 | 0.7474 mm | 2.3539 mm | 0.4602 deg | 0.8580 deg | 0.00% | reference |
| 0.25 | 0.7401 mm | 2.2653 mm | 0.3588 deg | 0.6242 deg | +8.08% | reject |
| 0.50 | 0.7298 mm | 2.3057 mm | 0.3798 deg | 0.6594 deg | +6.13% | reject |
| 1.00 | 0.7327 mm | 2.2613 mm | 0.3586 deg | 0.6362 deg | +2.71% | pass development |

Ratios 0.25 and 0.50 fail because fold-s3 SO(3) p95 exceeds the allowed 5%
regression. Ratio 1.00 remains the balanced development winner, but it already
failed its frozen final position gate. Do not deploy any v4 ratio from this
table and do not use the opened final to choose a task-specific component.

## Pure synthetic pretraining then real-only calibration

This paper-order analogue uses the new builder mode `--train_mode
synthetic_only`. For each fold, Stage 1 contains no real samples and is trained
for a fixed 107 epochs without validation selection:

```bash
./.venv/bin/python datasets/build_hybrid_v4.py \
  --real_data_dir data/new_calib_v4_real/C3/dev_s1 \
  --physics_config configs/physics_v4_dev_s1.yaml \
  --out_dir data/new_calib_v4_synth_then_calib/C3/dev_s1_pretrain \
  --synthetic_ratio 1.0 --train_mode synthetic_only --seed 42

./.venv/bin/python -u training/train_v4.py \
  --data_dir data/new_calib_v4_synth_then_calib/C3/dev_s1_pretrain \
  --checkpoint_dir checkpoints/new_calib_v4_synth_then_calib/C3/dev_s1_pretrain_seed42 \
  --config configs/training_v4.yaml --seed 42 --device cuda \
  --epochs 107 --fixed_epochs_no_val
```

Then prepare and train the real-only calibration stage:

```bash
./.venv/bin/python datasets/prepare_real_finetune_v4.py \
  --real_data_dir data/new_calib_v4_real/C3/dev_s1 \
  --hybrid_data_dir data/new_calib_v4_synth_then_calib/C3/dev_s1_pretrain \
  --out_dir data/new_calib_v4_synth_then_calib/C3/dev_s1_finetune_real

./.venv/bin/python -u training/train_v4.py \
  --data_dir data/new_calib_v4_synth_then_calib/C3/dev_s1_finetune_real \
  --checkpoint_dir checkpoints/new_calib_v4_synth_then_calib/C3/dev_s1_finetune_real_rebase_seed42 \
  --config configs/training_v4_finetune.yaml --seed 42 --device cuda \
  --init_checkpoint checkpoints/new_calib_v4_synth_then_calib/C3/dev_s1_pretrain_seed42 \
  --reset_normalization_on_init
```

Run folds s1/s2/s3 for seeds 42, 7 and 123. The retained development rule is:

```text
position = 0.80 * v3.2.2 + 0.20 * C3
rotation6D = 0.70 * v3.2.2 + 0.30 * C3, then Gram-Schmidt
```

All three seeds pass the four aggregate accuracy metrics and per-session p95
gate. Rebuild the report with:

```bash
./.venv/bin/python evaluation/summarize_synth_then_calib_v4.py
```

Research-only runtime command:

```bash
./.venv/bin/python inference/realtime_predictor_hybrid_v4.py \
  --hybrid_checkpoint_dir \
    checkpoints/new_calib_v4_synth_then_calib/C3/research_deployment_finetune_real_seed42 \
  --hybrid_position_weight 0.20 --hybrid_orientation_weight 0.30 \
  --generation real_current --device auto --backend auto
```

Do not treat this as the selected pose source. Candidate-specific median p95 is
2.1648 ms CPU and 1.7234 ms GPU; the GPU misses the 1.5 ms gate. No unopened
sealed test exists, so v3.2.2 remains the deployment model.

## Deployment training

The deployment physics fit is authorized only by the three passing CV
metadata files:

```bash
./.venv/bin/python -u calibration/calibrate_v4_fold.py \
  --config configs/new_calib_v4.yaml --fold deployment \
  --out_config configs/physics_v4_deployment.yaml \
  --out_dir results/physics_v4/deployment \
  --max_fit_samples 1800 --starts 4 --max_nfev 800 --seed 42 \
  --cv_metadata \
    results/physics_v4/dev_s1/calibration_metadata.json \
    results/physics_v4/dev_s2/calibration_metadata.json \
    results/physics_v4/dev_s3/calibration_metadata.json
```

Deployment used all 6,019 non-test real rows. Median CV epochs were 107 for
hybrid pretraining and 68 for real-only fine-tuning. Checkpoint:

```text
checkpoints/new_calib_v4_hybrid/C3/deployment_seed42_finetune_real_rebase
SHA-256: 1c4423fa13ebc035853493682a069656d2fc81af28bc4374d5115d3afc9f57e2
```

## Runtime, parity, and development gate

```bash
./.venv/bin/python evaluation/benchmark_hybrid_runtime_v4.py \
  --hybrid_checkpoint_dir \
    checkpoints/new_calib_v4_hybrid/C3/deployment_seed42_finetune_real_rebase \
  --runs 2000 --warmup 100 \
  --out results/new_calib_v4_hybrid/deployment/runtime_seed42_w20_optimized.json
```

Measured full predictor p95, including feature construction, six networks,
blend, confidence/dispersion, and device transfer:

```text
CPU TorchScript: 1.9923 ms
GPU CUDA Graph:  1.4226 ms
```

CPU/GPU maximum pose difference over 100 real development rows was
0.000061; absolute timestamp shift difference was 0.0. All 24 regression tests
passed.

## One-time final result — do not rerun or tune against it

| Model | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| frozen v3.2.2 | 1.1222 mm | 3.1047 mm | 2.3904 deg | 5.0578 deg |
| frozen v4 hybrid w=0.20 | 1.2335 mm | 3.4725 mm | 1.8653 deg | 3.7461 deg |

The hybrid improved orientation by about 22–26% but regressed position RMSE by
9.9% and position p95 by 11.8%. It therefore failed the all-four-metric final
gate. The selected deployment remains v3.2.2.

## Realtime interfaces

Run the selected v3.2.2 localizer as a stateful JSONL process:

```bash
./.venv/bin/python inference/realtime_predictor_v32.py \
  --device auto --backend auto --input_unit mV
```

The selected JSONL wrapper was smoke-tested on CPU TorchScript and GPU CUDA
Graph; maximum pose difference over a three-row causal sequence was 0.000031.

The experimental rejected hybrid can still be replayed for research, but must
not be used or presented as the selected pose source:

```bash
./.venv/bin/python inference/realtime_predictor_hybrid_v4.py \
  --hybrid_checkpoint_dir \
    checkpoints/new_calib_v4_hybrid/C3/deployment_seed42_finetune_real_rebase \
  --hybrid_position_weight 0.20 --generation real_current \
  --device auto --backend auto
```

Input JSONL:

```json
{"timestamp_ns": 123456789000, "reset": false, "emf": [1,2,3,4,5,6,7,8,9]}
```

EMF defaults to mV. Send `reset:true` for the first row after connection,
robot-program restart, or sensor reset. TCP/IP transport remains outside the
training loop; it should forward this request/response schema without adding
target-derived state.

## Rebuild paper-style figures and tables

This command reads frozen histories, predictions, summaries, and the already
opened one-time final result. It performs reporting only: it does not train,
calibrate, choose a model, or change the frozen deployment decision.

```bash
./.venv/bin/python reporting/build_paper_style_report.py \
  --out_dir reports/paper_style_2026_08_04
```

Generated outputs:

```text
reports/paper_style_2026_08_04/REPORT.md
reports/paper_style_2026_08_04/report_manifest.json
reports/paper_style_2026_08_04/figures/fig4_ablation_and_seed_robustness.{png,pdf}
reports/paper_style_2026_08_04/figures/fig7_physics_calibration_effect.{png,pdf}
reports/paper_style_2026_08_04/figures/fig8_final_trajectory_orientation_errors.{png,pdf}
reports/paper_style_2026_08_04/figures/fig9_physics_consistency.{png,pdf}
reports/paper_style_2026_08_04/tables/table1_scenario_performance.{csv,md}
reports/paper_style_2026_08_04/tables/table2_real_system_benchmark.{csv,md}
reports/paper_style_2026_08_04/tables/table3_method_comparison.{csv,md}
reports/paper_style_2026_08_04/tables/table4_physics_reconstruction_error.{csv,md}
```

The full Vietnamese pipeline-by-pipeline comparison, including a description
and claim boundary for every figure, is maintained in
`docs/BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md`.

These are structural analogues of the paper's Figs. 4, 7, 8, 9 and Tables
I–III, not direct reproductions. The paper uses a 500 mm cube; this system uses
a 100 mm cube and a different acquisition protocol. Missing current-data
FCN/KAN/optimization baselines are not fabricated.

The Fig. 8 analogue displays 100 uniformly sampled poses from the clean third
`cyl_rot` session and plots unsmoothed errors, matching the paper's display
density without changing the recorded geometry. The current robot programmed
a five-loop cylinder with radius `50.000 ± 0.038 mm`; the paper evaluated 100
poses on a tapered five-loop helix. Therefore the current trajectory must stay
cylindrical in an honest result plot. A tapered plot requires a new tapered
robot acquisition, not a plotting transformation.

Fig. 9 includes a reconstruction control computed from ground-truth pose. Its
large error on `cyl_rot` shows that the current fitted forward model/channel
correction does not generalize to the rotating final sessions. This figure is
a calibration diagnostic, not evidence matching the paper's reconstruction
accuracy. Do not tune on these opened final labels; resolve it with newly
collected rotating development data.

## Rebuild the canonical full comparison

This command rebuilds the point-by-point audit, side-by-side pipeline and
architecture diagrams, leakage boundary, data coverage, development/final
results, runtime context, physics audit, figure mapping and claim matrix. It
reads frozen summaries only and does not rerun final inference or change model
selection.

```bash
./.venv/bin/python reporting/build_full_pipeline_comparison.py \
  --out_dir reports/full_pipeline_comparison_2026_08_05
```

Primary outputs:

```text
reports/full_pipeline_comparison_2026_08_05/REPORT.md
reports/full_pipeline_comparison_2026_08_05/manifest.json
reports/full_pipeline_comparison_2026_08_05/figures/
reports/full_pipeline_comparison_2026_08_05/tables/
```

The report is canonical for communication. The deployment decision remains in
`results/new_calib_v4_final/final_decision_manifest.json`.

## Rebuild the Vietnamese presentation deck

The slide generator consumes only the frozen comparison figures. It produces
16:9 PNG slides, a combined 16-page PDF, a 15–18 minute speaker script, an
8-minute shortened route, likely Q&A answers, an outline CSV and a hash
manifest.

```bash
./.venv/bin/python reporting/build_paper_comparison_presentation.py \
  --out_dir reports/presentation_paper_comparison_2026_08_06
```

Outputs:

```text
reports/presentation_paper_comparison_2026_08_06/SLIDE_DECK.pdf
reports/presentation_paper_comparison_2026_08_06/SLIDE_SCRIPT.md
reports/presentation_paper_comparison_2026_08_06/SLIDE_OUTLINE.csv
reports/presentation_paper_comparison_2026_08_06/slides/
reports/presentation_paper_comparison_2026_08_06/manifest.json
```

The phrases `development-only`, `not head-to-head`, and `research-only` are
deliberate claim boundaries and should remain in any shortened presentation.

## Next legitimate experiment

1. Collect at least three new non-test rotating calibration sessions with
   broad pitch/roll/yaw coverage.
2. Keep an additional, later rotating session sealed as the new final test.
3. Refit physics and repeat grouped CV using only the new development sessions.
4. Investigate a two-head/component policy in CV: preserve v3 position while
   improving orientation. Do not justify its weights using the opened test.
5. Record the programmed physical delta-t in metadata; use hardware timestamp
   only for dropout/reset unless actual sample intervals vary.

## Record keeping

After every attempt, including rejection, update:

- `docs/EXPERIMENT_LEDGER.md`
- `docs/MODEL_UPGRADE_HISTORY.csv`
- `docs/PAPER_COMPARISON_TABLE.md`

Never overwrite raw CSVs or historical experiment artifacts.

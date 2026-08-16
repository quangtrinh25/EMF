import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
FILE_KEYS = ["set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"]


def run(args):
    print("+", " ".join(args), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def generate(config, out_dir, n_samples):
    run([
        "datasets/generator.py",
        "--config", config,
        "--out_dir", out_dir,
        "--n_samples", str(n_samples),
    ])


def train(
    data_dir,
    checkpoint_dir,
    model_type,
    epochs,
    batch_size,
    device,
    orientation_weight=None,
    init_checkpoint=None,
    lr=None,
    temporal_window=None,
    temporal_mode=None,
    feature_layout=None,
):
    cmd = [
        "training/train.py",
        "--data_dir", data_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--model_type", model_type,
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    if device is not None:
        cmd += ["--device", device]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    if orientation_weight is not None:
        cmd += ["--orientation_weight", str(orientation_weight)]
    if init_checkpoint is not None:
        cmd += ["--init_checkpoint", init_checkpoint]
    if temporal_window is not None:
        cmd += ["--temporal_window", str(temporal_window)]
    if temporal_mode is not None:
        cmd += ["--temporal_mode", temporal_mode]
    if feature_layout is not None:
        cmd += ["--feature_layout", feature_layout]
    run(cmd)


def evaluate(data_dir, checkpoint_dir, model_type):
    run([
        "evaluation/evaluate.py",
        "--data_dir", data_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--model_type", model_type,
    ])


def pose_report():
    run(["evaluation/pose_report.py"])


def fine_tune_orientation(args):
    train(
        args.data_dir,
        args.checkpoint_dir,
        "resnet",
        args.epochs,
        args.batch_size,
        args.device,
        args.orientation_weight,
        args.init_checkpoint,
        args.lr,
    )
    run([
        "evaluation/evaluate.py",
        "--data_dir", args.data_dir,
        "--checkpoint_dir", args.checkpoint_dir,
        "--model_type", "resnet",
        "--results_dir", args.results_dir,
        "--label", args.label,
    ])


def calibrate_pose_outputs():
    run(["evaluation/calibrate_pose_outputs.py"])


def calibrate_v2(args):
    cmd = [
        "calibration/calibrate_v2.py",
        "--data_dir", args.data_dir,
        "--out_config", args.out_config,
        "--out_dir", args.out_dir,
        "--starts", str(args.starts),
        "--max_nfev", str(args.max_nfev),
        "--fit_mode", args.fit_mode,
    ]
    if getattr(args, "fit_files", None) is not None:
        cmd += ["--fit_files", *args.fit_files]
    if getattr(args, "report_files", None) is not None:
        cmd += ["--report_files", *args.report_files]
    if getattr(args, "fit_csvs", None) is not None:
        cmd += ["--fit_csvs", *args.fit_csvs]
    if getattr(args, "report_csvs", None) is not None:
        cmd += ["--report_csvs", *args.report_csvs]
    if getattr(args, "workspace_from", None) is not None:
        cmd += ["--workspace_from", args.workspace_from]
    run(cmd)


def build_hybrid_dataset(args):
    out_dir = getattr(args, "hybrid_dir", args.out_dir)
    cmd = [
        "datasets/build_hybrid_dataset.py",
        "--synthetic_dir", args.synthetic_dir,
        "--real_dir", args.real_dir,
        "--out_dir", out_dir,
        "--augment_copies", str(args.augment_copies),
        "--noise_std_mV", str(args.noise_std_mV),
        "--gain_drift", str(args.gain_drift),
    ]
    if getattr(args, "real_files", None) is not None:
        cmd += ["--real_files", *args.real_files]
    if getattr(args, "real_csvs", None) is not None:
        cmd += ["--real_csvs", *args.real_csvs]
    run(cmd)


def evaluate_real_csv(args):
    out_dir = getattr(args, "real_eval_out_dir", args.out_dir)
    cmd = [
        "evaluation/evaluate_real_csv.py",
        "--data_dir", args.data_dir,
        "--checkpoint_dir", args.checkpoint_dir,
        "--model_type", args.model_type,
        "--out_dir", out_dir,
    ]
    if getattr(args, "eval_files", None) is not None:
        cmd += ["--eval_files", *args.eval_files]
    if getattr(args, "eval_csvs", None) is not None:
        cmd += ["--eval_csvs", *args.eval_csvs]
    if getattr(args, "pose_correction", None) is not None:
        cmd += ["--pose_correction", args.pose_correction]
    run(cmd)


def verify_calibration_params(args):
    run([
        "calibration/verify_calibration_params.py",
        "--config", args.config,
        "--data_dir", args.data_dir,
        "--out_dir", args.out_dir,
    ])


def calibrated_v2(args):
    calibrate_v2(args)
    generate(args.out_config, args.synthetic_dir, args.n_samples)
    build_hybrid_dataset(args)
    train(args.hybrid_dir, args.checkpoint_dir, "resnet", args.epochs, args.batch_size, args.device, args.orientation_weight)
    evaluate(args.hybrid_dir, args.checkpoint_dir, "resnet")
    evaluate_real_csv(args)


def named_csv_args(base_dir, role, keys):
    return [f"{key}={Path(base_dir) / role / (key + '.csv')}" for key in keys]


def read_metric(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No metrics found in {path}")
    total = sum(int(float(row["num_samples"])) for row in rows)
    # Pool squared errors before taking the square root.  Averaging file RMSEs
    # underestimates the true sample-level RMSE when file errors differ.
    pos = (sum(float(row["position_rmse_mm"]) ** 2 * int(float(row["num_samples"])) for row in rows) / total) ** 0.5
    ori = (sum(float(row["orientation_rmse_deg"]) ** 2 * int(float(row["num_samples"])) for row in rows) / total) ** 0.5
    return total, pos, ori


def new_calib_v3(args):
    with open(args.config) as stream:
        protocol = yaml.safe_load(stream)
    fold = args.fold or protocol.get("primary_fold")
    if fold not in protocol.get("folds", {}):
        raise ValueError(f"Unknown fold {fold!r}; available: {sorted(protocol.get('folds', {}))}")

    run([
        "datasets/prepare_new_calib.py",
        "--config", args.config,
        "--out_dir", args.prepared_dir,
        "--folds", fold,
    ])
    data_dir = str(Path(args.prepared_dir) / f"fold_{fold}")
    run_name = f"fold_{fold}" + ("_quick" if args.mode == "quick" else "")
    checkpoint_dir = str(Path(args.checkpoint_root) / run_name)
    results_dir = str(Path(args.results_root) / run_name)

    if args.mode == "quick":
        epochs = args.epochs or 2
        hidden = args.hidden or 64
        blocks = args.residual_blocks or 2
        device = args.device or "cpu"
    else:
        epochs = args.epochs
        hidden = args.hidden
        blocks = args.residual_blocks
        device = args.device

    train_cmd = [
        "training/train_v3.py",
        "--data_dir", data_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--config", args.training_config,
    ]
    for option, value in (
        ("--epochs", epochs),
        ("--batch_size", args.batch_size),
        ("--device", device),
        ("--hidden", hidden),
        ("--residual_blocks", blocks),
        ("--seed", args.seed),
    ):
        if value is not None:
            train_cmd += [option, str(value)]
    run(train_cmd)

    eval_device = device or "cpu"
    run([
        "evaluation/evaluate_v3.py",
        "--data_dir", data_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--out_dir", results_dir,
        "--split", "val",
        "--device", eval_device,
    ])
    if args.open_test:
        run([
            "evaluation/evaluate_v3.py",
            "--data_dir", data_dir,
            "--checkpoint_dir", checkpoint_dir,
            "--out_dir", results_dir,
            "--split", "test",
            "--device", eval_device,
            "--confirm_open_locked_test",
        ])
    else:
        print("Test fold remains unopened. Re-run with --open-test only after freezing the model.")


def predict_v3(args):
    cmd = [
        "inference/predict_capsule_pose_v3.py",
        "--input", args.input,
        "--output", args.output,
        "--checkpoint_dir", args.checkpoint_dir,
        "--input_unit", args.input_unit,
        "--header", args.header,
        "--device", args.device,
        "--backend", args.backend,
    ]
    if args.metrics_out:
        cmd += ["--metrics_out", args.metrics_out]
    if getattr(args, "timestamp_column", None):
        cmd += ["--timestamp_column", args.timestamp_column, "--timestamp_unit", args.timestamp_unit]
    if getattr(args, "reset_column", None):
        cmd += ["--reset_column", args.reset_column]
    if getattr(args, "reset_every_rows", None):
        cmd += ["--reset_every_rows", str(args.reset_every_rows)]
    if getattr(args, "row_as_timestamp", False):
        cmd += ["--row_as_timestamp"]
    if getattr(args, "cpu_threads", None) is not None:
        cmd += [
            "--cpu_threads", str(args.cpu_threads),
            "--cpu_interop_threads", str(args.cpu_interop_threads),
        ]
    run(cmd)


def predict_v3_ensemble(args):
    cmd = [
        "inference/ensemble_predict_capsule_pose_v3.py",
        "--input", args.input,
        "--output", args.output,
        "--checkpoint_dirs", *args.checkpoint_dirs,
        "--input_unit", args.input_unit,
        "--header", args.header,
        "--device", args.device,
        "--backend", args.backend,
    ]
    if args.orientation_model_index is not None:
        cmd += ["--orientation_model_index", str(args.orientation_model_index)]
    if args.aux_position_checkpoint_dirs:
        cmd += ["--aux_position_checkpoint_dirs", *args.aux_position_checkpoint_dirs]
    if args.orientation_checkpoint_dir:
        cmd += ["--orientation_checkpoint_dir", args.orientation_checkpoint_dir]
    if args.metrics_out:
        cmd += ["--metrics_out", args.metrics_out]
    if args.metadata_out:
        cmd += ["--metadata_out", args.metadata_out]
    if args.timestamp_column:
        cmd += ["--timestamp_column", args.timestamp_column, "--timestamp_unit", args.timestamp_unit]
    if args.reset_column:
        cmd += ["--reset_column", args.reset_column]
    if args.reset_every_rows:
        cmd += ["--reset_every_rows", str(args.reset_every_rows)]
    if args.row_as_timestamp:
        cmd += ["--row_as_timestamp"]
    if args.cpu_threads is not None:
        cmd += [
            "--cpu_threads", str(args.cpu_threads),
            "--cpu_interop_threads", str(args.cpu_interop_threads),
        ]
    run(cmd)


def temporal_v3_train_command(data_dir, checkpoint_dir, args, quick):
    cmd = [
        "training/train_v3.py",
        "--data_dir", data_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--config", args.training_config,
    ]
    epochs = args.epochs if args.epochs is not None else (2 if quick else None)
    hidden = args.hidden if args.hidden is not None else (64 if quick else None)
    blocks = args.residual_blocks if args.residual_blocks is not None else (2 if quick else None)
    device = args.device if args.device is not None else ("cpu" if quick else None)
    for option, value in (
        ("--epochs", epochs),
        ("--batch_size", args.batch_size),
        ("--device", device),
        ("--hidden", hidden),
        ("--residual_blocks", blocks),
        ("--seed", args.seed),
    ):
        if value is not None:
            cmd += [option, str(value)]
    run(cmd)


def new_calib_v3_temporal(args):
    feature_tag = {
        "raw_window": "raw",
        "current_plus_log_ratio": "logratio",
    }[args.feature_mode]
    run_name = f"w{args.window_size}_h{args.horizon_steps}_{args.include_dt}_{feature_tag}"
    if args.row_as_timestamp:
        run_name += "_rowtime"
    if args.mode == "quick":
        run_name += "_quick"
    prepared_dir = str(Path(args.prepared_root) / run_name)
    prepare_cmd = [
        "datasets/prepare_temporal_v3.py",
        "--config", args.config,
        "--out_dir", prepared_dir,
        "--window_size", str(args.window_size),
        "--horizon_steps", str(args.horizon_steps),
        "--include_dt", args.include_dt,
        "--feature_mode", args.feature_mode,
    ]
    if args.row_as_timestamp:
        prepare_cmd += ["--row_as_timestamp"]
    run(prepare_cmd)

    checkpoint_root = Path(args.checkpoint_root) / run_name
    results_root = Path(args.results_root) / run_name
    single_data = str(Path(prepared_dir) / "single")
    temporal_data = str(Path(prepared_dir) / "temporal")
    single_checkpoint = str(checkpoint_root / "single")
    temporal_checkpoint = str(checkpoint_root / "temporal")
    quick = args.mode == "quick"
    temporal_v3_train_command(single_data, single_checkpoint, args, quick)
    temporal_v3_train_command(temporal_data, temporal_checkpoint, args, quick)

    eval_device = args.device or "cpu"
    for label, data_dir, checkpoint in (
        ("single", single_data, single_checkpoint),
        ("temporal", temporal_data, temporal_checkpoint),
    ):
        run([
            "evaluation/evaluate_v3.py",
            "--data_dir", data_dir,
            "--checkpoint_dir", checkpoint,
            "--out_dir", str(results_root / label),
            "--split", "val",
            "--device", eval_device,
        ])
    run([
        "evaluation/ensemble_v3.py",
        "--data_dir", temporal_data,
        "--position_checkpoint_dir", single_checkpoint,
        "--orientation_checkpoint_dir", temporal_checkpoint,
        "--out_dir", str(results_root / "ensemble"),
        "--split", "val",
        "--device", eval_device,
    ])

    if args.open_test:
        for label, data_dir, checkpoint in (
            ("single", single_data, single_checkpoint),
            ("temporal", temporal_data, temporal_checkpoint),
        ):
            run([
                "evaluation/evaluate_v3.py",
                "--data_dir", data_dir,
                "--checkpoint_dir", checkpoint,
                "--out_dir", str(results_root / label),
                "--split", "test",
                "--device", eval_device,
                "--confirm_open_locked_test",
            ])
        run([
            "evaluation/ensemble_v3.py",
            "--data_dir", temporal_data,
            "--position_checkpoint_dir", single_checkpoint,
            "--orientation_checkpoint_dir", temporal_checkpoint,
            "--out_dir", str(results_root / "ensemble"),
            "--split", "test",
            "--device", eval_device,
            "--confirm_open_locked_test",
        ])
    else:
        print("Temporal test remains unopened. Select M0/M1/M3 on rotating dev only.")


def write_summary_markdown(path, rows, final_raw, final_corrected, selected_id):
    lines = [
        "# No-Leak Balanced Easy Summary",
        "",
        f"Selected candidate: `{selected_id}`",
        "",
        "## Tuning",
        "",
        "| Candidate | Dev Position RMSE (mm) | Dev Orientation RMSE (deg) | Dev Score |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | {float(row['dev_position_rmse_mm']):.4f} | "
            f"{float(row['dev_orientation_rmse_deg']):.4f} | {float(row['dev_score']):.4f} |"
        )
    lines.extend([
        "",
        "## Private Test",
        "",
        "| Result | Samples | Position RMSE (mm) | Orientation RMSE (deg) |",
        "|---|---:|---:|---:|",
        f"| Raw | {final_raw[0]} | {final_raw[1]:.4f} | {final_raw[2]:.4f} |",
        f"| Corrected | {final_corrected[0]} | {final_corrected[1]:.4f} | {final_corrected[2]:.4f} |",
        "",
        "Private files: `set13_con_spi_no_rot`, `set12_cyl_spi_rot`.",
    ])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_temporal_summary(path, metrics, checkpoint_dir, window_size, augment_copies):
    lines = [
        "# Temporal Balanced Easy Summary",
        "",
        f"Checkpoint: `{checkpoint_dir}`",
        f"Window size: `{window_size}` past rows",
        f"Augment copies: `{augment_copies}`",
        "",
        "## Private Test",
        "",
        "| Samples | Position RMSE (mm) | Orientation RMSE (deg) |",
        "|---:|---:|---:|",
        f"| {metrics[0]} | {metrics[1]:.4f} | {metrics[2]:.4f} |",
        "",
        "Private files: `set13_con_spi_no_rot`, `set12_cyl_spi_rot`.",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fit_pose_correction(checkpoint_dir, train_csvs, eval_csvs, out_path, metrics_out):
    cmd = [
        "evaluation/fit_pose_output_correction.py",
        "--checkpoint_dir", checkpoint_dir,
        "--train_csvs", *train_csvs,
        "--out", out_path,
        "--metrics_out", metrics_out,
    ]
    if eval_csvs:
        cmd += ["--eval_csvs", *eval_csvs]
    run(cmd)


def evaluate_real_csv_paths(checkpoint_dir, out_dir, eval_csvs, pose_correction=None):
    cmd = [
        "evaluation/evaluate_real_csv.py",
        "--checkpoint_dir", checkpoint_dir,
        "--out_dir", out_dir,
        "--eval_csvs", *eval_csvs,
    ]
    if pose_correction is not None:
        cmd += ["--pose_correction", pose_correction]
    run(cmd)


def noleak_candidates(mode):
    base = [
        {"candidate": "base", "orientation_weight": 4.0, "augment_copies": 50, "noise_std_mV": 0.5, "gain_drift": 0.02},
        {"candidate": "ori2", "orientation_weight": 2.0, "augment_copies": 50, "noise_std_mV": 0.5, "gain_drift": 0.02},
        {"candidate": "ori8", "orientation_weight": 8.0, "augment_copies": 50, "noise_std_mV": 0.5, "gain_drift": 0.02},
        {"candidate": "aug100", "orientation_weight": 4.0, "augment_copies": 100, "noise_std_mV": 0.5, "gain_drift": 0.02},
        {"candidate": "noise02", "orientation_weight": 4.0, "augment_copies": 50, "noise_std_mV": 0.2, "gain_drift": 0.02},
        {"candidate": "noise10", "orientation_weight": 4.0, "augment_copies": 50, "noise_std_mV": 1.0, "gain_drift": 0.02},
        {"candidate": "gain01", "orientation_weight": 4.0, "augment_copies": 50, "noise_std_mV": 0.5, "gain_drift": 0.01},
        {"candidate": "gain04", "orientation_weight": 4.0, "augment_copies": 50, "noise_std_mV": 0.5, "gain_drift": 0.04},
    ]
    if mode in {"quick", "final-only"}:
        return base[:1]
    return base


def noleak_balanced(args):
    quick = args.mode == "quick"
    base_name = args.name + ("_quick" if quick else "")
    split_dir = f"data/real_splits/{base_name}"
    results_dir = f"results/{base_name}"
    checkpoint_root = f"checkpoints/{base_name}"
    train_keys = ["set10_cyl_no_rot", "set11_con_spi_rot"]
    private_keys = ["set13_con_spi_no_rot", "set12_cyl_spi_rot"]
    n_samples = 2048 if quick else args.n_samples
    epochs = 2 if quick else args.epochs
    batch_size = 256 if quick else args.batch_size
    device = "cpu" if quick else args.device
    starts = 1 if quick else args.starts
    max_nfev = 100 if quick else args.max_nfev

    run([
        "datasets/make_real_split.py",
        "--raw_dir", args.data_dir,
        "--out_dir", split_dir,
        "--train_files", *train_keys,
        "--private_files", *private_keys,
        "--dev_mod", str(args.dev_mod),
        "--dev_remainder", str(args.dev_remainder),
    ])
    run(["evaluation/check_split_leakage.py", "--metadata", f"{split_dir}/metadata.csv"])

    train_csvs = named_csv_args(split_dir, "calib_train", train_keys)
    dev_csvs = named_csv_args(split_dir, "real_dev", train_keys)
    private_csvs = named_csv_args(split_dir, "private_test", private_keys)

    candidates = noleak_candidates(args.mode)
    if args.mode == "final-only":
        selected = candidates[0]
        tuning_rows = [{
            **selected,
            "dev_position_rmse_mm": float("nan"),
            "dev_orientation_rmse_deg": float("nan"),
            "dev_score": float("nan"),
            "checkpoint_dir": "",
        }]
    else:
        tuning_config = f"configs/physics_calibrated_v2_{base_name}_tuning.yaml"
        tuning_synth = f"data/synthetic_{base_name}_tuning"
        tuning_calib_results = f"{results_dir}/tuning_calibration"
        run([
            "calibration/calibrate_v2.py",
            "--data_dir", args.data_dir,
            "--out_config", tuning_config,
            "--out_dir", tuning_calib_results,
            "--starts", str(starts),
            "--max_nfev", str(max_nfev),
            "--fit_csvs", *train_csvs,
            "--report_csvs", *train_csvs,
            "--workspace_from", "fit",
        ])
        generate(tuning_config, tuning_synth, n_samples)

        tuning_rows = []
        for candidate in candidates:
            candidate_id = candidate["candidate"]
            hybrid_dir = f"data/hybrid_{base_name}_tuning/{candidate_id}"
            ckpt_dir = f"{checkpoint_root}/tuning/{candidate_id}"
            dev_results = f"{results_dir}/tuning/{candidate_id}/dev_raw"
            run([
                "datasets/build_hybrid_dataset.py",
                "--synthetic_dir", tuning_synth,
                "--out_dir", hybrid_dir,
                "--real_csvs", *train_csvs,
                "--augment_copies", str(candidate["augment_copies"]),
                "--noise_std_mV", str(candidate["noise_std_mV"]),
                "--gain_drift", str(candidate["gain_drift"]),
            ])
            train(hybrid_dir, ckpt_dir, "resnet", epochs, batch_size, device, candidate["orientation_weight"])
            evaluate_real_csv_paths(ckpt_dir, dev_results, dev_csvs)
            dev_metric_path = f"{dev_results}/real_csv_pose_metrics.csv"
            _, dev_pos, dev_ori = read_metric(dev_metric_path)
            tuning_rows.append({
                **candidate,
                "dev_position_rmse_mm": dev_pos,
                "dev_orientation_rmse_deg": dev_ori,
                "dev_score": dev_pos + dev_ori,
                "checkpoint_dir": ckpt_dir,
            })

        tuning_rows.sort(key=lambda row: (row["dev_score"], row["dev_position_rmse_mm"]))
        selected = tuning_rows[0]
    os.makedirs(results_dir, exist_ok=True)
    tuning_summary = f"{results_dir}/tuning_summary.csv"
    with open(tuning_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tuning_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tuning_rows)

    final_config = f"configs/physics_calibrated_v2_{base_name}.yaml"
    final_synth = f"data/synthetic_{base_name}"
    final_hybrid = f"data/hybrid_{base_name}"
    final_ckpt = f"{checkpoint_root}/final"
    final_calib_results = f"{results_dir}/final_calibration"
    full_nonprivate_csvs = [f"{key}={Path(args.data_dir) / filename}" for key, filename in {
        "set10_cyl_no_rot": "set10_2304_cyl_no_rot.csv",
        "set11_con_spi_rot": "Set11_con_spi_rot_2.csv",
    }.items()]
    run([
        "calibration/calibrate_v2.py",
        "--data_dir", args.data_dir,
        "--out_config", final_config,
        "--out_dir", final_calib_results,
        "--starts", str(starts),
        "--max_nfev", str(max_nfev),
        "--fit_csvs", *full_nonprivate_csvs,
        "--report_csvs", *full_nonprivate_csvs,
        "--workspace_from", "fit",
    ])
    generate(final_config, final_synth, n_samples)
    run([
        "datasets/build_hybrid_dataset.py",
        "--synthetic_dir", final_synth,
        "--out_dir", final_hybrid,
        "--real_csvs", *full_nonprivate_csvs,
        "--augment_copies", str(int(selected["augment_copies"])),
        "--noise_std_mV", str(selected["noise_std_mV"]),
        "--gain_drift", str(selected["gain_drift"]),
    ])
    train(final_hybrid, final_ckpt, "resnet", epochs, batch_size, device, float(selected["orientation_weight"]))
    run([
        "evaluation/evaluate.py",
        "--data_dir", final_hybrid,
        "--checkpoint_dir", final_ckpt,
        "--model_type", "resnet",
        "--results_dir", results_dir,
        "--label", "synthetic_test",
    ])

    raw_private_dir = f"{results_dir}/private_raw"
    evaluate_real_csv_paths(final_ckpt, raw_private_dir, private_csvs)
    final_raw = read_metric(f"{raw_private_dir}/real_csv_pose_metrics.csv")

    correction_path = f"{final_ckpt}/pose_output_correction.npz"
    correction_metrics = f"{results_dir}/pose_output_correction_metrics.csv"
    fit_pose_correction(final_ckpt, full_nonprivate_csvs, dev_csvs, correction_path, correction_metrics)
    corrected_private_dir = f"{results_dir}/private_corrected"
    evaluate_real_csv_paths(final_ckpt, corrected_private_dir, private_csvs, pose_correction=correction_path)
    final_corrected = read_metric(f"{corrected_private_dir}/real_csv_pose_metrics.csv")

    # Keep stable filenames for the README/test plan.
    for src, dst in [
        (f"{raw_private_dir}/real_csv_pose_metrics.csv", f"{results_dir}/private_real_pose_metrics.csv"),
        (f"{raw_private_dir}/real_csv_pose_predictions.csv", f"{results_dir}/private_real_pose_predictions.csv"),
        (f"{corrected_private_dir}/real_csv_pose_metrics.csv", f"{results_dir}/private_real_pose_metrics_corrected.csv"),
        (f"{corrected_private_dir}/real_csv_pose_predictions.csv", f"{results_dir}/private_real_pose_predictions_corrected.csv"),
    ]:
        shutil.copyfile(src, dst)

    write_summary_markdown(
        f"{results_dir}/final_summary.md",
        tuning_rows,
        final_raw,
        final_corrected,
        selected["candidate"],
    )
    print(f"Selected candidate: {selected['candidate']}")
    print(f"Final raw private RMSE: position={final_raw[1]:.4f} mm orientation={final_raw[2]:.4f} deg")
    print(f"Final corrected private RMSE: position={final_corrected[1]:.4f} mm orientation={final_corrected[2]:.4f} deg")


def temporal_balanced(args):
    quick = args.mode == "quick"
    base_name = args.name + ("_quick" if quick else "")
    split_dir = f"data/real_splits/{base_name}"
    data_dir = f"data/temporal_{base_name}"
    checkpoint_dir = f"checkpoints/{base_name}/final"
    results_dir = f"results/{base_name}"
    train_keys = ["set10_cyl_no_rot", "set11_con_spi_rot"]
    private_keys = ["set13_con_spi_no_rot", "set12_cyl_spi_rot"]

    epochs = 2 if quick else args.epochs
    augment_copies = 5 if quick else args.augment_copies
    device = "cpu" if quick else args.device

    run([
        "datasets/make_real_split.py",
        "--raw_dir", args.data_dir,
        "--out_dir", split_dir,
        "--train_files", *train_keys,
        "--private_files", *private_keys,
        "--dev_mod", str(args.dev_mod),
        "--dev_remainder", str(args.dev_remainder),
    ])
    run(["evaluation/check_split_leakage.py", "--metadata", f"{split_dir}/metadata.csv"])

    train_csvs = named_csv_args(split_dir, "calib_train", train_keys)
    dev_csvs = named_csv_args(split_dir, "real_dev", train_keys)
    private_csvs = named_csv_args(split_dir, "private_test", private_keys)

    run([
        "datasets/build_temporal_dataset.py",
        "--train_csvs", *train_csvs,
        "--val_csvs", *dev_csvs,
        "--out_dir", data_dir,
        "--window_size", str(args.window_size),
        "--augment_copies", str(augment_copies),
        "--noise_std_mV", str(args.noise_std_mV),
        "--gain_drift", str(args.gain_drift),
    ])
    train(
        data_dir,
        checkpoint_dir,
        "resnet",
        epochs,
        args.batch_size,
        device,
        args.orientation_weight,
        temporal_window=args.window_size,
        temporal_mode="past",
        feature_layout="oldest_to_current",
    )

    private_dir = f"{results_dir}/private_raw"
    evaluate_real_csv_paths(checkpoint_dir, private_dir, private_csvs)
    final_metrics = read_metric(f"{private_dir}/real_csv_pose_metrics.csv")
    os.makedirs(results_dir, exist_ok=True)
    shutil.copyfile(f"{private_dir}/real_csv_pose_metrics.csv", f"{results_dir}/private_real_pose_metrics.csv")
    shutil.copyfile(f"{private_dir}/real_csv_pose_predictions.csv", f"{results_dir}/private_real_pose_predictions.csv")
    write_temporal_summary(
        f"{results_dir}/final_summary.md",
        final_metrics,
        checkpoint_dir,
        args.window_size,
        augment_copies,
    )
    print(f"Temporal private RMSE: position={final_metrics[1]:.4f} mm orientation={final_metrics[2]:.4f} deg")


def predict(args):
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir == "auto":
        candidates = [
            Path("checkpoints/noleak_balanced_easy/final"),
            Path("checkpoints/noleak_balanced_easy_quick/final"),
            Path("checkpoints/custom_system_v2_noleak_balanced"),
            Path("checkpoints/custom_system_v2_noleak_norot"),
        ]
        for candidate in candidates:
            if (candidate / "resnet_best.pt").exists() or (candidate / "best.pt").exists():
                checkpoint_dir = str(candidate)
                break
        else:
            checkpoint_dir = str(candidates[0])
    pose_correction = args.pose_correction
    if pose_correction == "auto":
        candidate = Path(checkpoint_dir) / "pose_output_correction.npz"
        pose_correction = str(candidate) if candidate.exists() else None
    elif pose_correction == "none":
        pose_correction = None
    cmd = [
        "inference/predict_capsule_pose.py",
        "--input", args.input,
        "--output", args.output,
        "--checkpoint_dir", checkpoint_dir,
        "--input_unit", args.input_unit,
        "--header", args.header,
        "--device", args.device,
    ]
    if args.metrics_out is not None:
        cmd += ["--metrics_out", args.metrics_out]
    if pose_correction is not None:
        cmd += ["--pose_correction", pose_correction]
    run(cmd)


def private_test(args):
    out_dir = Path(args.out_dir)
    eval_files = args.eval_files

    correction = args.pose_correction
    if correction == "auto":
        candidate = Path(args.checkpoint_dir) / "pose_output_correction.npz"
        correction = str(candidate) if candidate.exists() else None
    elif correction == "none":
        correction = None

    def run_private(label, pose_correction=None):
        target_dir = out_dir / label
        cmd = [
            "evaluation/evaluate_real_csv.py",
            "--checkpoint_dir", args.checkpoint_dir,
            "--model_type", "resnet",
            "--out_dir", str(target_dir),
            "--eval_files", *eval_files,
        ]
        if pose_correction is not None:
            cmd += ["--pose_correction", pose_correction]
        run(cmd)
        return target_dir

    os.makedirs(out_dir, exist_ok=True)
    if args.mode in {"both", "raw"}:
        raw_dir = run_private("raw")
        shutil.copyfile(raw_dir / "real_csv_pose_metrics.csv", out_dir / "private_real_pose_metrics.csv")
        shutil.copyfile(raw_dir / "real_csv_pose_predictions.csv", out_dir / "private_real_pose_predictions.csv")

    if args.mode in {"both", "corrected"}:
        if correction is None:
            if args.mode == "corrected":
                raise FileNotFoundError(
                    f"No pose correction found for {args.checkpoint_dir}. "
                    "Pass --pose_correction PATH or use --mode raw."
                )
            print("No pose correction found; skipped corrected private test.")
        else:
            corrected_dir = run_private("corrected", pose_correction=correction)
            shutil.copyfile(
                corrected_dir / "real_csv_pose_metrics.csv",
                out_dir / "private_real_pose_metrics_corrected.csv",
            )
            shutil.copyfile(
                corrected_dir / "real_csv_pose_predictions.csv",
                out_dir / "private_real_pose_predictions_corrected.csv",
            )

    print(f"Private test outputs written to {out_dir}")


def write_first_columns(src, dst, n_columns):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(src, newline="") as f_in, open(dst, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for line_no, row in enumerate(reader, start=1):
            if len(row) < n_columns:
                raise ValueError(f"{src}:{line_no} has {len(row)} columns, expected at least {n_columns}")
            writer.writerow(row[:n_columns])


def write_raw_prediction_format(emf_path, prediction_path, out_path):
    with open(prediction_path, newline="") as f:
        predictions = list(csv.DictReader(f))

    pose_cols = [
        "pred_X_mm", "pred_Y_mm", "pred_Z_mm",
        "pred_roll_deg", "pred_pitch_deg", "pred_yaw_deg",
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(emf_path, newline="") as f_in, open(out_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        count = 0
        for count, row in enumerate(reader, start=1):
            pred = predictions[count - 1]
            writer.writerow(row[:9] + [pred[col] for col in pose_cols])
    if count != len(predictions):
        raise RuntimeError(f"Row mismatch: {emf_path} has {count} rows, {prediction_path} has {len(predictions)} predictions")


def private_9emf(args):
    raw_files = {
        "set13_con_spi_no_rot": "Set13_con_spi_no_rot.csv",
        "set12_cyl_spi_rot": "Set12_cyl_spi_rot_2.csv",
    }
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir == "auto_temporal":
        candidates = [
            Path("checkpoints/temporal_balanced_easy/final"),
            Path("checkpoints/temporal_balanced_easy_quick/final"),
        ]
        for candidate in candidates:
            if (candidate / "resnet_best.pt").exists() or (candidate / "best.pt").exists():
                checkpoint_dir = str(candidate)
                break
        else:
            checkpoint_dir = str(candidates[0])

    pose_correction = args.pose_correction
    if pose_correction == "auto":
        candidate = Path(checkpoint_dir) / "pose_output_correction.npz"
        pose_correction = str(candidate) if candidate.exists() else None
    elif pose_correction == "none":
        pose_correction = None

    for key, filename in raw_files.items():
        emf_path = input_dir / f"{key}_9emf.csv"
        prediction_path = out_dir / f"{key}_predictions.csv"
        raw_format_path = out_dir / f"{key}_raw_format_predictions.csv"

        write_first_columns(Path(args.data_dir) / filename, emf_path, 9)
        cmd = [
            "inference/predict_capsule_pose.py",
            "--input", str(emf_path),
            "--output", str(prediction_path),
            "--checkpoint_dir", checkpoint_dir,
            "--input_unit", "mV",
            "--header", "no",
            "--device", args.device,
        ]
        if pose_correction is not None:
            cmd += ["--pose_correction", pose_correction]
        run(cmd)
        write_raw_prediction_format(emf_path, prediction_path, raw_format_path)

    print(f"9-EMF private inputs written to {input_dir}")
    print(f"Private pose predictions written to {out_dir}")


def ensemble_private(args):
    run([
        "evaluation/ensemble_private.py",
        "--position_checkpoint_dir", args.position_checkpoint_dir,
        "--orientation_checkpoint_dir", args.orientation_checkpoint_dir,
        "--out_dir", args.out_dir,
        "--eval_files", *args.eval_files,
    ])


def private_9emf_ensemble(args):
    raw_files = {
        "set13_con_spi_no_rot": "Set13_con_spi_no_rot.csv",
        "set12_cyl_spi_rot": "Set12_cyl_spi_rot_2.csv",
    }
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in raw_files.items():
        emf_path = input_dir / f"{key}_9emf.csv"
        prediction_path = out_dir / f"{key}_predictions.csv"
        raw_format_path = out_dir / f"{key}_raw_format_predictions.csv"
        write_first_columns(Path(args.data_dir) / filename, emf_path, 9)
        run([
            "inference/ensemble_predict_capsule_pose.py",
            "--input", str(emf_path),
            "--output", str(prediction_path),
            "--position_checkpoint_dir", args.position_checkpoint_dir,
            "--orientation_checkpoint_dir", args.orientation_checkpoint_dir,
            "--input_unit", "mV",
            "--header", "no",
            "--device", args.device,
        ])
        write_raw_prediction_format(emf_path, prediction_path, raw_format_path)

    print(f"9-EMF ensemble private inputs written to {input_dir}")
    print(f"Ensemble private pose predictions written to {out_dir}")


def make_paper_plots(args):
    scripts = [
        "plots/figure4_ablation.py",
        "plots/figure7_calibration.py",
        "plots/figure8_benchmarking.py",
        "plots/figure9_physics.py",
    ]
    if args.include_simulated:
        scripts.insert(1, "plots/figure5_simulated.py")
    for script in scripts:
        run([script])


def reproduce(args):
    generate(args.config, args.data_dir, args.n_samples)
    for model_type in args.models:
        train(args.data_dir, args.checkpoint_dir, model_type, args.epochs, args.batch_size, args.device, args.orientation_weight)
    evaluate(args.data_dir, args.checkpoint_dir, "resnet")


def calibrate_and_retrain(args):
    run(["calibration/calibrate.py"])
    generate(args.config, args.data_dir, args.n_samples)
    train(args.data_dir, args.checkpoint_dir, "resnet", args.epochs, args.batch_size, args.device, args.orientation_weight)
    evaluate(args.data_dir, args.checkpoint_dir, "resnet")


def smoke(args):
    generate(args.config, args.data_dir, args.n_samples)
    train(args.data_dir, args.checkpoint_dir, "resnet", args.epochs, args.batch_size, args.device, args.orientation_weight)
    evaluate(args.data_dir, args.checkpoint_dir, "resnet")


def main():
    parser = argparse.ArgumentParser(
        description="Clean EMF capsule localization pipeline based on plan.md and the residual-NN paper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_p = subparsers.add_parser("smoke", help="Small end-to-end synthetic run.")
    smoke_p.add_argument("--config", default="configs/physics_nominal.yaml")
    smoke_p.add_argument("--data_dir", default="data/smoke_nominal")
    smoke_p.add_argument("--checkpoint_dir", default="checkpoints/smoke")
    smoke_p.add_argument("--n_samples", type=int, default=2048)
    smoke_p.add_argument("--epochs", type=int, default=2)
    smoke_p.add_argument("--batch_size", type=int, default=256)
    smoke_p.add_argument("--device", default="cpu")
    smoke_p.add_argument("--orientation_weight", type=float, default=None)
    smoke_p.set_defaults(func=smoke)

    repro_p = subparsers.add_parser("reproduce", help="Stages 1-8: synthetic paper reproduction.")
    repro_p.add_argument("--config", default="configs/physics_nominal.yaml")
    repro_p.add_argument("--data_dir", default="data/synthetic_nominal")
    repro_p.add_argument("--checkpoint_dir", default="checkpoints/paper_reproduction")
    repro_p.add_argument("--n_samples", type=int, default=640_000)
    repro_p.add_argument("--models", nargs="+", default=["resnet"], choices=["resnet", "fcn", "kan"])
    repro_p.add_argument("--epochs", type=int, default=None)
    repro_p.add_argument("--batch_size", type=int, default=None)
    repro_p.add_argument("--device", default=None)
    repro_p.add_argument("--orientation_weight", type=float, default=None)
    repro_p.set_defaults(func=reproduce)

    calib_p = subparsers.add_parser("calibrated", help="Stage 9: fit hardware parameters, regenerate data, retrain.")
    calib_p.add_argument("--config", default="configs/physics_calibrated.yaml")
    calib_p.add_argument("--data_dir", default="data/synthetic_calibrated")
    calib_p.add_argument("--checkpoint_dir", default="checkpoints/custom_system")
    calib_p.add_argument("--n_samples", type=int, default=640_000)
    calib_p.add_argument("--epochs", type=int, default=None)
    calib_p.add_argument("--batch_size", type=int, default=None)
    calib_p.add_argument("--device", default=None)
    calib_p.add_argument("--orientation_weight", type=float, default=None)
    calib_p.set_defaults(func=calibrate_and_retrain)

    eval_p = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint on a test split.")
    eval_p.add_argument("--data_dir", default="data/synthetic_nominal")
    eval_p.add_argument("--checkpoint_dir", default="checkpoints/paper_reproduction")
    eval_p.add_argument("--model_type", default="resnet", choices=["resnet", "fcn", "kan"])
    eval_p.set_defaults(func=lambda args: evaluate(args.data_dir, args.checkpoint_dir, args.model_type))

    report_p = subparsers.add_parser("pose-report", help="Save before/after position and orientation metrics.")
    report_p.set_defaults(func=lambda args: pose_report())

    ft_p = subparsers.add_parser("fine-tune-orientation", help="Fine-tune calibrated ResNet with stronger orientation loss.")
    ft_p.add_argument("--data_dir", default="data/synthetic_calibrated")
    ft_p.add_argument("--checkpoint_dir", default="checkpoints/custom_system_oriented")
    ft_p.add_argument("--init_checkpoint", default="checkpoints/custom_system/resnet_best.pt")
    ft_p.add_argument("--epochs", type=int, default=50)
    ft_p.add_argument("--batch_size", type=int, default=1024)
    ft_p.add_argument("--device", default=None)
    ft_p.add_argument("--lr", type=float, default=1e-4)
    ft_p.add_argument("--orientation_weight", type=float, default=8.0)
    ft_p.add_argument("--results_dir", default="results/pose")
    ft_p.add_argument("--label", default="after_calibration_oriented")
    ft_p.set_defaults(func=fine_tune_orientation)

    output_cal_p = subparsers.add_parser("calibrate-pose-outputs", help="Post-calibrate predicted angles using the validation split.")
    output_cal_p.set_defaults(func=lambda args: calibrate_pose_outputs())

    calib_v2_p = subparsers.add_parser("calibrate-v2", help="Robust multi-start calibration with channel gain/bias correction.")
    calib_v2_p.add_argument("--data_dir", default="data/raw_calibration")
    calib_v2_p.add_argument("--out_config", default="configs/physics_calibrated_v2.yaml")
    calib_v2_p.add_argument("--out_dir", default="results/calibration_v2")
    calib_v2_p.add_argument("--starts", type=int, default=12)
    calib_v2_p.add_argument("--max_nfev", type=int, default=5000)
    calib_v2_p.add_argument("--fit_mode", choices=["all", "no_rot"], default="all")
    calib_v2_p.add_argument("--fit_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    calib_v2_p.add_argument("--report_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    calib_v2_p.add_argument("--fit_csvs", nargs="+", default=None, help="Explicit calibration CSVs as name=path.")
    calib_v2_p.add_argument("--report_csvs", nargs="+", default=None, help="Explicit report CSVs as name=path.")
    calib_v2_p.add_argument("--workspace_from", choices=["all", "fit"], default="all")
    calib_v2_p.set_defaults(func=calibrate_v2)

    hybrid_p = subparsers.add_parser("build-hybrid", help="Mix calibrated synthetic data with augmented real calibration CSV samples.")
    hybrid_p.add_argument("--synthetic_dir", default="data/synthetic_calibrated_v2")
    hybrid_p.add_argument("--real_dir", default="data/raw_calibration")
    hybrid_p.add_argument("--out_dir", default="data/hybrid_calibrated_v2")
    hybrid_p.add_argument("--augment_copies", type=int, default=50)
    hybrid_p.add_argument("--noise_std_mV", type=float, default=0.5)
    hybrid_p.add_argument("--gain_drift", type=float, default=0.02)
    hybrid_p.add_argument("--real_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    hybrid_p.add_argument("--real_csvs", nargs="+", default=None, help="Explicit real CSVs as name=path.")
    hybrid_p.set_defaults(func=build_hybrid_dataset)

    real_eval_p = subparsers.add_parser("evaluate-real", help="Evaluate a checkpoint directly on the raw calibration CSV files.")
    real_eval_p.add_argument("--data_dir", default="data/raw_calibration")
    real_eval_p.add_argument("--checkpoint_dir", default="checkpoints/custom_system_v2")
    real_eval_p.add_argument("--model_type", default="resnet", choices=["resnet", "fcn", "kan"])
    real_eval_p.add_argument("--out_dir", default="results/pose_v2")
    real_eval_p.add_argument("--eval_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    real_eval_p.add_argument("--eval_csvs", nargs="+", default=None, help="Explicit eval CSVs as name=path.")
    real_eval_p.add_argument("--pose_correction", default=None)
    real_eval_p.set_defaults(func=evaluate_real_csv)

    noleak_p = subparsers.add_parser("noleak-balanced", help="Easy no-leak balanced split, tuning, final training, and private evaluation.")
    noleak_p.add_argument("--mode", choices=["full", "quick", "final-only"], default="full")
    noleak_p.add_argument("--name", default="noleak_balanced_easy")
    noleak_p.add_argument("--data_dir", default="data/raw_calibration")
    noleak_p.add_argument("--n_samples", type=int, default=256_000)
    noleak_p.add_argument("--epochs", type=int, default=120)
    noleak_p.add_argument("--batch_size", type=int, default=1024)
    noleak_p.add_argument("--device", default=None)
    noleak_p.add_argument("--starts", type=int, default=12)
    noleak_p.add_argument("--max_nfev", type=int, default=5000)
    noleak_p.add_argument("--dev_mod", type=int, default=5)
    noleak_p.add_argument("--dev_remainder", type=int, default=0)
    noleak_p.set_defaults(func=noleak_balanced)

    temporal_p = subparsers.add_parser("temporal-balanced", help="Train and evaluate a past-window temporal ResNet on real no-leak splits.")
    temporal_p.add_argument("--mode", choices=["full", "quick"], default="full")
    temporal_p.add_argument("--name", default="temporal_balanced_easy")
    temporal_p.add_argument("--data_dir", default="data/raw_calibration")
    temporal_p.add_argument("--window_size", type=int, default=5)
    temporal_p.add_argument("--epochs", type=int, default=120)
    temporal_p.add_argument("--batch_size", type=int, default=256)
    temporal_p.add_argument("--device", default=None)
    temporal_p.add_argument("--augment_copies", type=int, default=200)
    temporal_p.add_argument("--noise_std_mV", type=float, default=0.5)
    temporal_p.add_argument("--gain_drift", type=float, default=0.02)
    temporal_p.add_argument("--orientation_weight", type=float, default=4.0)
    temporal_p.add_argument("--dev_mod", type=int, default=5)
    temporal_p.add_argument("--dev_remainder", type=int, default=0)
    temporal_p.set_defaults(func=temporal_balanced)

    predict_p = subparsers.add_parser("predict", help="Easy prediction from an EMF CSV.")
    predict_p.add_argument("--input", required=True)
    predict_p.add_argument("--output", default="results/predicted_capsule_pose.csv")
    predict_p.add_argument("--metrics_out", default=None)
    predict_p.add_argument("--checkpoint_dir", default="auto")
    predict_p.add_argument("--input_unit", choices=["mV", "V"], default="mV")
    predict_p.add_argument("--header", choices=["auto", "yes", "no"], default="auto")
    predict_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict_p.add_argument("--pose_correction", default="auto", help="Correction .npz, 'auto' to use checkpoint correction if present, or 'none'.")
    predict_p.set_defaults(func=predict)

    private_p = subparsers.add_parser("private-test", help="Run the default no-leak private real test.")
    private_p.add_argument("--checkpoint_dir", default="checkpoints/noleak_balanced_easy/final")
    private_p.add_argument("--out_dir", default="results/private_test")
    private_p.add_argument("--mode", choices=["both", "raw", "corrected"], default="both")
    private_p.add_argument("--pose_correction", default="auto", help="Correction .npz, 'auto' to use checkpoint correction if present, or 'none'.")
    private_p.add_argument("--eval_files", nargs="+", default=["set13_con_spi_no_rot", "set12_cyl_spi_rot"], choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    private_p.set_defaults(func=private_test)

    private_9emf_p = subparsers.add_parser("private-9emf", help="Create 9-EMF-only private inputs and predict private poses.")
    private_9emf_p.add_argument("--data_dir", default="data/raw_calibration")
    private_9emf_p.add_argument("--checkpoint_dir", default="checkpoints/noleak_balanced_easy/final")
    private_9emf_p.add_argument("--input_dir", default="results/private_9emf_inputs")
    private_9emf_p.add_argument("--out_dir", default="results/private_9emf_predictions")
    private_9emf_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    private_9emf_p.add_argument("--pose_correction", default="auto", help="Correction .npz, 'auto' to use checkpoint correction if present, or 'none'.")
    private_9emf_p.set_defaults(func=private_9emf)

    private_9emf_temporal_p = subparsers.add_parser("private-9emf-temporal", help="Create 9-EMF-only private inputs and predict poses with the temporal model.")
    private_9emf_temporal_p.add_argument("--data_dir", default="data/raw_calibration")
    private_9emf_temporal_p.add_argument("--checkpoint_dir", default="auto_temporal")
    private_9emf_temporal_p.add_argument("--input_dir", default="results/private_9emf_temporal_inputs")
    private_9emf_temporal_p.add_argument("--out_dir", default="results/private_9emf_temporal_predictions")
    private_9emf_temporal_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    private_9emf_temporal_p.add_argument("--pose_correction", default="auto", help="Correction .npz, 'auto' to use checkpoint correction if present, or 'none'.")
    private_9emf_temporal_p.set_defaults(func=private_9emf)

    ensemble_private_p = subparsers.add_parser("ensemble-private", help="Private test with position from single-row model and orientation from temporal model.")
    ensemble_private_p.add_argument("--position_checkpoint_dir", default="checkpoints/noleak_balanced_easy/final")
    ensemble_private_p.add_argument("--orientation_checkpoint_dir", default="checkpoints/temporal_balanced_easy/final")
    ensemble_private_p.add_argument("--out_dir", default="results/ensemble_private")
    ensemble_private_p.add_argument("--eval_files", nargs="+", default=["set13_con_spi_no_rot", "set12_cyl_spi_rot"], choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    ensemble_private_p.set_defaults(func=ensemble_private)

    private_9emf_ensemble_p = subparsers.add_parser("private-9emf-ensemble", help="Create 9-EMF private files and predict with position+temporal-orientation ensemble.")
    private_9emf_ensemble_p.add_argument("--data_dir", default="data/raw_calibration")
    private_9emf_ensemble_p.add_argument("--position_checkpoint_dir", default="checkpoints/noleak_balanced_easy/final")
    private_9emf_ensemble_p.add_argument("--orientation_checkpoint_dir", default="checkpoints/temporal_balanced_easy/final")
    private_9emf_ensemble_p.add_argument("--input_dir", default="results/private_9emf_ensemble_inputs")
    private_9emf_ensemble_p.add_argument("--out_dir", default="results/private_9emf_ensemble_predictions")
    private_9emf_ensemble_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    private_9emf_ensemble_p.set_defaults(func=private_9emf_ensemble)

    verify_calib_p = subparsers.add_parser("verify-calib", help="Evaluate any calibrated parameter YAML on the raw calibration CSV files.")
    verify_calib_p.add_argument("--config", default="configs/physics_calibrated_v2.yaml")
    verify_calib_p.add_argument("--data_dir", default="data/raw_calibration")
    verify_calib_p.add_argument("--out_dir", default="results/calibration_own_params")
    verify_calib_p.set_defaults(func=verify_calibration_params)

    full_v2_p = subparsers.add_parser("calibrated-v2", help="Run v2 calibration, local synthetic generation, hybrid training, and real CSV evaluation.")
    full_v2_p.add_argument("--data_dir", default="data/raw_calibration")
    full_v2_p.add_argument("--out_config", default="configs/physics_calibrated_v2.yaml")
    full_v2_p.add_argument("--out_dir", default="results/calibration_v2")
    full_v2_p.add_argument("--starts", type=int, default=12)
    full_v2_p.add_argument("--max_nfev", type=int, default=5000)
    full_v2_p.add_argument("--fit_mode", choices=["all", "no_rot"], default="all")
    full_v2_p.add_argument("--fit_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    full_v2_p.add_argument("--report_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    full_v2_p.add_argument("--fit_csvs", nargs="+", default=None, help="Explicit calibration CSVs as name=path.")
    full_v2_p.add_argument("--report_csvs", nargs="+", default=None, help="Explicit report CSVs as name=path.")
    full_v2_p.add_argument("--workspace_from", choices=["all", "fit"], default="all")
    full_v2_p.add_argument("--synthetic_dir", default="data/synthetic_calibrated_v2")
    full_v2_p.add_argument("--real_dir", default="data/raw_calibration")
    full_v2_p.add_argument("--hybrid_dir", "--out_hybrid_dir", dest="hybrid_dir", default="data/hybrid_calibrated_v2")
    full_v2_p.add_argument("--real_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    full_v2_p.add_argument("--real_csvs", nargs="+", default=None, help="Explicit real CSVs as name=path.")
    full_v2_p.add_argument("--augment_copies", type=int, default=50)
    full_v2_p.add_argument("--noise_std_mV", type=float, default=0.5)
    full_v2_p.add_argument("--gain_drift", type=float, default=0.02)
    full_v2_p.add_argument("--checkpoint_dir", default="checkpoints/custom_system_v2")
    full_v2_p.add_argument("--real_eval_out_dir", default="results/pose_v2")
    full_v2_p.add_argument("--eval_files", nargs="+", default=None, choices=[
        "set10_cyl_no_rot", "set11_con_spi_rot", "set12_cyl_spi_rot", "set13_con_spi_no_rot"
    ])
    full_v2_p.add_argument("--eval_csvs", nargs="+", default=None, help="Explicit eval CSVs as name=path.")
    full_v2_p.add_argument("--pose_correction", default=None)
    full_v2_p.add_argument("--n_samples", type=int, default=256_000)
    full_v2_p.add_argument("--epochs", type=int, default=120)
    full_v2_p.add_argument("--batch_size", type=int, default=1024)
    full_v2_p.add_argument("--device", default=None)
    full_v2_p.add_argument("--orientation_weight", type=float, default=4.0)
    full_v2_p.add_argument("--model_type", default="resnet", choices=["resnet"])
    full_v2_p.set_defaults(func=calibrated_v2)

    v3_p = subparsers.add_parser(
        "new-calib-v3",
        help="Prepare a trajectory-safe 100 mm fold and train signed rotation-6D ResNet.",
    )
    v3_p.add_argument("--config", default="configs/new_calib_v3.yaml")
    v3_p.add_argument("--training_config", default="configs/training_v3.yaml")
    v3_p.add_argument("--prepared_dir", default="data/new_calib_v3")
    v3_p.add_argument("--checkpoint_root", default="checkpoints/new_calib_v3")
    v3_p.add_argument("--results_root", default="results/new_calib_v3")
    v3_p.add_argument("--fold", choices=["con_norot", "con_rot", "cyl_norot", "cyl_rot"], default=None)
    v3_p.add_argument("--mode", choices=["full", "quick"], default="full")
    v3_p.add_argument("--epochs", type=int, default=None)
    v3_p.add_argument("--batch_size", type=int, default=None)
    v3_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    v3_p.add_argument("--hidden", type=int, default=None)
    v3_p.add_argument("--residual_blocks", type=int, default=None)
    v3_p.add_argument("--seed", type=int, default=None)
    v3_p.add_argument(
        "--open-test",
        action="store_true",
        help="Evaluate the held-out family after model choices are frozen.",
    )
    v3_p.set_defaults(func=new_calib_v3)

    predict_v3_p = subparsers.add_parser("predict-v3", help="Predict signed XYZ+Euler pose with a v3 checkpoint.")
    predict_v3_p.add_argument("--input", required=True)
    predict_v3_p.add_argument("--output", default="results/predicted_capsule_pose_v3.csv")
    predict_v3_p.add_argument("--checkpoint_dir", required=True)
    predict_v3_p.add_argument("--metrics_out", default=None)
    predict_v3_p.add_argument("--input_unit", choices=["mV", "V"], default="mV")
    predict_v3_p.add_argument("--header", choices=["auto", "yes", "no"], default="auto")
    predict_v3_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict_v3_p.add_argument("--backend", choices=["auto", "eager", "torchscript"], default="auto")
    predict_v3_p.add_argument("--cpu_threads", type=int, default=None)
    predict_v3_p.add_argument("--cpu_interop_threads", type=int, default=1)
    predict_v3_p.add_argument("--timestamp_column", default=None)
    predict_v3_p.add_argument("--timestamp_unit", choices=["s", "ms", "us", "ns"], default="s")
    predict_v3_p.add_argument("--reset_column", default=None)
    predict_v3_p.add_argument("--reset_every_rows", type=int, default=None)
    predict_v3_p.add_argument("--row_as_timestamp", action="store_true")
    predict_v3_p.set_defaults(func=predict_v3)

    predict_v3_ensemble_p = subparsers.add_parser(
        "predict-v3-ensemble", help="Predict capsule pose with a compatible multi-seed v3 ensemble.",
    )
    predict_v3_ensemble_p.add_argument("--input", required=True)
    predict_v3_ensemble_p.add_argument("--output", default="results/predicted_capsule_pose_v3_ensemble.csv")
    predict_v3_ensemble_p.add_argument("--checkpoint_dirs", nargs="+", required=True)
    predict_v3_ensemble_p.add_argument("--aux_position_checkpoint_dirs", nargs="*", default=[])
    predict_v3_ensemble_p.add_argument("--orientation_model_index", type=int, default=None)
    predict_v3_ensemble_p.add_argument("--orientation_checkpoint_dir", default=None)
    predict_v3_ensemble_p.add_argument("--metrics_out", default=None)
    predict_v3_ensemble_p.add_argument("--metadata_out", default=None)
    predict_v3_ensemble_p.add_argument("--input_unit", choices=["mV", "V"], default="mV")
    predict_v3_ensemble_p.add_argument("--header", choices=["auto", "yes", "no"], default="auto")
    predict_v3_ensemble_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict_v3_ensemble_p.add_argument("--backend", choices=["auto", "eager", "torchscript"], default="auto")
    predict_v3_ensemble_p.add_argument("--cpu_threads", type=int, default=None)
    predict_v3_ensemble_p.add_argument("--cpu_interop_threads", type=int, default=1)
    predict_v3_ensemble_p.add_argument("--timestamp_column", default=None)
    predict_v3_ensemble_p.add_argument("--timestamp_unit", choices=["s", "ms", "us", "ns"], default="s")
    predict_v3_ensemble_p.add_argument("--reset_column", default=None)
    predict_v3_ensemble_p.add_argument("--reset_every_rows", type=int, default=None)
    predict_v3_ensemble_p.add_argument("--row_as_timestamp", action="store_true")
    predict_v3_ensemble_p.set_defaults(func=predict_v3_ensemble)

    temporal_v3_p = subparsers.add_parser(
        "new-calib-v3-temporal",
        help="Train matched single/causal-temporal v3 models and dev-only ensemble.",
    )
    temporal_v3_p.add_argument("--config", default="configs/new_calib_v3.yaml")
    temporal_v3_p.add_argument("--training_config", default="configs/training_v3.yaml")
    temporal_v3_p.add_argument("--prepared_root", default="data/new_calib_v3_temporal_runs")
    temporal_v3_p.add_argument("--checkpoint_root", default="checkpoints/new_calib_v3_temporal")
    temporal_v3_p.add_argument("--results_root", default="results/new_calib_v3_temporal")
    temporal_v3_p.add_argument("--mode", choices=["full", "quick"], default="full")
    temporal_v3_p.add_argument("--window_size", type=int, default=5)
    temporal_v3_p.add_argument("--horizon_steps", type=int, default=0)
    temporal_v3_p.add_argument("--include_dt", choices=["auto", "yes", "no"], default="auto")
    temporal_v3_p.add_argument(
        "--feature_mode",
        choices=["raw_window", "current_plus_log_ratio"],
        default="raw_window",
    )
    temporal_v3_p.add_argument("--row_as_timestamp", action="store_true")
    temporal_v3_p.add_argument("--epochs", type=int, default=None)
    temporal_v3_p.add_argument("--batch_size", type=int, default=None)
    temporal_v3_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    temporal_v3_p.add_argument("--hidden", type=int, default=None)
    temporal_v3_p.add_argument("--residual_blocks", type=int, default=None)
    temporal_v3_p.add_argument("--seed", type=int, default=None)
    temporal_v3_p.add_argument(
        "--open-test", action="store_true",
        help="Evaluate all frozen temporal candidates on the locked test family.",
    )
    temporal_v3_p.set_defaults(func=new_calib_v3_temporal)

    plot_p = subparsers.add_parser("paper-plots", help="Generate paper-style plots from plot.md.")
    plot_p.add_argument("--include_simulated", action="store_true", help="Also run Figure 5; this can be slower because it runs optimization baselines.")
    plot_p.set_defaults(func=make_paper_plots)

    export_p = subparsers.add_parser("export", help="Export the ResNet checkpoint to ONNX.")
    export_p.add_argument("--checkpoint_dir", default="checkpoints/custom_system")
    export_p.set_defaults(func=lambda args: run(["production/export_onnx.py", "--checkpoint_dir", args.checkpoint_dir]))

    args = parser.parse_args()
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()

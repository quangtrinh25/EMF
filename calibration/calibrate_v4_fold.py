"""Fit a physics generator on real training sessions for one leakage-safe fold."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.calibrate_v2 import (
    base_theta,
    lower_bounds,
    predict_corrected,
    randomize_theta,
    residual,
    unpack_theta,
    upper_bounds,
)
from calibration.report_utils import default_rx_params
from datasets.prepare_multigeneration_v4 import (
    available_folds,
    discover_sources,
    load_source,
    sha256_file,
    validate_workspace,
)


def sample_sources_evenly(sources, maximum, seed):
    rng = np.random.default_rng(seed)
    per_source = max(1, int(np.ceil(maximum / len(sources))))
    emf, poses, source_names = [], [], []
    for source in sources:
        count = min(len(source["emf"]), per_source)
        indices = rng.choice(len(source["emf"]), count, replace=False)
        emf.append(source["emf"][indices])
        poses.append(source["poses"][indices])
        source_names.extend([source["source_file"]] * count)
    emf = np.vstack(emf)
    poses = np.vstack(poses)
    source_names = np.asarray(source_names)
    if len(emf) > maximum:
        indices = rng.choice(len(emf), maximum, replace=False)
        emf, poses, source_names = emf[indices], poses[indices], source_names[indices]
    return emf, poses, source_names


def physical_metrics(measured, predicted):
    error = measured - predicted
    flat_corr = (
        float(np.corrcoef(measured.ravel(), predicted.ravel())[0, 1])
        if np.std(measured) > 0 and np.std(predicted) > 0 else None
    )
    mean_mv = float(np.mean(np.abs(measured)) * 1e3)
    rmse_mv = float(np.sqrt(np.mean(error ** 2)) * 1e3)
    return {
        "samples": int(len(measured)),
        "rmse_mV": rmse_mv,
        "mae_mV": float(np.mean(np.abs(error)) * 1e3),
        "mean_abs_measured_mV": mean_mv,
        "rmse_over_mean_fraction": rmse_mv / max(mean_mv, 1e-12),
        "flat_correlation": flat_corr,
    }


def generator_workspace(config, train_poses):
    center = np.asarray(config["workspace"]["center_mm"], dtype=float)
    size = np.asarray(config["workspace"]["size_mm"], dtype=float)
    lower, upper = center - size / 2.0, center + size / 2.0
    mins, maxs = train_poses[:, 3:].min(axis=0), train_poses[:, 3:].max(axis=0)
    margins = np.asarray([2.0, 2.0, 2.0])
    return {
        "x_mm": [float(lower[0]), float(upper[0])],
        "y_mm": [float(lower[1]), float(upper[1])],
        "z_mm": [float(lower[2]), float(upper[2])],
        "orientation_deg": {
            "roll": [float(mins[0] - margins[0]), float(maxs[0] + margins[0])],
            "pitch": [float(mins[1] - margins[1]), float(maxs[1] + margins[1])],
            "yaw": [float(mins[2] - margins[2]), float(maxs[2] + margins[2])],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fit and gate a fold-specific v4 physics generator.")
    parser.add_argument("--config", default="configs/new_calib_v4.yaml")
    parser.add_argument("--fold", required=True)
    parser.add_argument("--out_config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_fit_samples", type=int, default=1200)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--max_nfev", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum_dev_correlation", type=float, default=0.80)
    parser.add_argument("--maximum_dev_rmse_over_mean", type=float, default=0.50)
    parser.add_argument(
        "--cv_metadata", nargs="*", default=[],
        help=(
            "Required for --fold deployment: metadata from every held-out "
            "physics calibration fold that authorized synthetic generation."
        ),
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    descriptors = discover_sources(config)
    folds = available_folds(descriptors)
    eligible = [item for item in descriptors if not item["locked"] and not item["sealed"]]
    cv_gate_metadata = []
    if args.fold == "deployment":
        if len(args.cv_metadata) < 3:
            raise ValueError("Deployment calibration requires metadata from at least three CV folds")
        expected_locked = {
            (item["source_file"], item["sha256"]) for item in descriptors if item["locked"]
        }
        for raw_path in args.cv_metadata:
            with open(raw_path) as stream:
                row = json.load(stream)
            observed_locked = {
                (item["file"], item["sha256"]) for item in row.get("locked_test_sources", [])
            }
            if row.get("synthetic_allowed") is not True or row.get("test_labels_loaded") is not False:
                raise RuntimeError(f"CV physics gate is not valid: {raw_path}")
            if observed_locked != expected_locked:
                raise RuntimeError(f"Locked-test hashes differ in CV metadata: {raw_path}")
            cv_gate_metadata.append({"path": str(raw_path), **row})
        if len({row["fold"] for row in cv_gate_metadata}) != len(cv_gate_metadata):
            raise ValueError("Deployment CV metadata contains duplicate folds")
        dev_descriptor = None
        train_descriptors = eligible
    else:
        if args.fold not in folds:
            raise ValueError(f"Unknown development fold {args.fold!r}")
        dev_descriptor = folds[args.fold]
        train_descriptors = [item for item in eligible if item is not dev_descriptor]
    train_sources = [load_source(item) for item in train_descriptors]
    dev_source = None if dev_descriptor is None else load_source(dev_descriptor)
    for source in [*train_sources, *([] if dev_source is None else [dev_source])]:
        validate_workspace(source, config)

    emf_fit, poses_fit, sampled_sources = sample_sources_evenly(
        train_sources, args.max_fit_samples, args.seed,
    )
    rx = default_rx_params()
    channel_scale = np.maximum(np.std(emf_fit, axis=0), 0.004)
    base = base_theta(poses_fit)
    rng = np.random.default_rng(args.seed)
    attempts = []
    best = None
    for start in range(args.starts):
        x0 = randomize_theta(base, rng, first=start == 0)
        result = least_squares(
            residual, x0=x0, bounds=(lower_bounds(), upper_bounds()),
            args=(emf_fit, poses_fit, rx, channel_scale), method="trf",
            loss="soft_l1", f_scale=1.0, max_nfev=args.max_nfev,
            ftol=1e-9, xtol=1e-9, gtol=1e-9,
        )
        predicted = predict_corrected(result.x, poses_fit, rx)
        metrics = physical_metrics(emf_fit, predicted)
        attempts.append({
            "start": start, "success": bool(result.success), "nfev": int(result.nfev),
            "cost": float(result.cost), **metrics,
        })
        print(
            f"start={start} nfev={result.nfev} fit_rmse={metrics['rmse_mV']:.3f}mV "
            f"corr={metrics['flat_correlation']:.4f}", flush=True,
        )
        if best is None or metrics["rmse_mV"] < best[0]:
            best = (metrics["rmse_mV"], result)

    tx = unpack_theta(best[1].x)
    train_prediction = predict_corrected(best[1].x, poses_fit, rx)
    train_metrics = physical_metrics(emf_fit, train_prediction)
    if dev_source is None:
        cv_metrics = [row["dev_metrics"] for row in cv_gate_metadata]
        total_samples = sum(int(row["samples"]) for row in cv_metrics)
        dev_metrics = {
            "samples": total_samples,
            "rmse_mV": max(float(row["rmse_mV"]) for row in cv_metrics),
            "mae_mV": max(float(row["mae_mV"]) for row in cv_metrics),
            "mean_abs_measured_mV": sum(
                float(row["mean_abs_measured_mV"]) * int(row["samples"]) for row in cv_metrics
            ) / total_samples,
            "rmse_over_mean_fraction": max(
                float(row["rmse_over_mean_fraction"]) for row in cv_metrics
            ),
            "flat_correlation": min(float(row["flat_correlation"]) for row in cv_metrics),
        }
    else:
        dev_prediction = predict_corrected(best[1].x, dev_source["poses"], rx)
        dev_metrics = physical_metrics(dev_source["emf"], dev_prediction)
    synthetic_allowed = (
        dev_metrics["flat_correlation"] is not None
        and dev_metrics["flat_correlation"] >= args.minimum_dev_correlation
        and dev_metrics["rmse_over_mean_fraction"] <= args.maximum_dev_rmse_over_mean
    )
    physics_config = {
        "schema_version": 4,
        "tx": tx,
        "rx": rx,
        "workspace": generator_workspace(config, np.vstack([source["poses"] for source in train_sources])),
        "calibration_gate": {
            "synthetic_allowed": bool(synthetic_allowed),
            "minimum_dev_correlation": args.minimum_dev_correlation,
            "maximum_dev_rmse_over_mean": args.maximum_dev_rmse_over_mean,
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
            "validation_source": (
                "three_fold_held_out_cv" if dev_source is None else "current_held_out_fold"
            ),
        },
    }
    out_config = Path(args.out_config)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config, "w") as stream:
        yaml.safe_dump(physics_config, stream, sort_keys=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(attempts).to_csv(out_dir / "calibration_starts.csv", index=False)
    metadata = {
        "schema_version": 4,
        "fold": args.fold,
        "fit_samples": int(len(emf_fit)),
        "fit_sample_count_by_file": {
            name: int(np.sum(sampled_sources == name)) for name in sorted(set(sampled_sources))
        },
        "train_sources": [
            {"file": item["source_file"], "sha256": item["sha256"]} for item in train_descriptors
        ],
        "dev_source": (
            None if dev_descriptor is None
            else {"file": dev_descriptor["source_file"], "sha256": dev_descriptor["sha256"]}
        ),
        "cv_gate_sources": [
            {"fold": row["fold"], "path": row["path"], "dev_metrics": row["dev_metrics"]}
            for row in cv_gate_metadata
        ],
        "locked_test_sources": [
            {"file": item["source_file"], "sha256": item["sha256"]}
            for item in descriptors if item["locked"]
        ],
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "synthetic_allowed": bool(synthetic_allowed),
        "out_config": str(out_config),
        "out_config_sha256": sha256_file(out_config),
        "test_labels_loaded": False,
    }
    with open(out_dir / "calibration_metadata.json", "w") as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps(metadata, indent=2))
    if not synthetic_allowed:
        raise RuntimeError("Physics generator failed the held-out calibration gate; synthetic training is forbidden")


if __name__ == "__main__":
    main()

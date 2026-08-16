"""Evaluate frozen v3.2.2 component rules on v4 development folds only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_v3 import load_model
from evaluation.v4_common import resolve_device, sha256_file
from physics.rotation_repr import mean_rotation_6d, so3_geodesic_deg, target_v3_to_pose_deg
from training.metrics import pose_metrics_v3


def logratio_features(raw_window, eps=1e-4):
    current = raw_window[:, -1, :]
    ratio = np.log((raw_window[:, :-1, :] + eps) / (current[:, None, :] + eps))
    return np.concatenate([current, np.clip(ratio, -5.0, 5.0).reshape(len(raw_window), -1)], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen v3.2 components on a v4 validation fold.")
    parser.add_argument("--data_dir", required=True, help="A v4 B1/W3 fold directory")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--deployment_manifest", default="checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json")
    parser.add_argument("--position_raw_indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--exclude_rowtime", action="store_true")
    parser.add_argument("--orientation", choices=["log", "raw7_log", "raw123_log"], default="log")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--split", choices=["val", "sealed_test"], default="val")
    parser.add_argument("--confirm_open_sealed_test", action="store_true")
    parser.add_argument("--frozen_manifest", default=None)
    args = parser.parse_args()
    with open(Path(args.data_dir) / "protocol.json") as stream:
        protocol = json.load(stream)
    if args.split == "val" and protocol.get("sealed_test_opened") is not False:
        raise RuntimeError("Frozen v3.2 development evaluator refuses opened sealed-test data")
    if args.split == "sealed_test":
        if not args.confirm_open_sealed_test or args.frozen_manifest is None:
            raise RuntimeError("Sealed v3.2 baseline requires explicit confirmation and frozen manifest")
        if protocol.get("sealed_test_opened") is not True:
            raise RuntimeError("Sealed dataset protocol does not record an authorized opening")
        with open(args.frozen_manifest) as stream:
            frozen = json.load(stream)
        receipt_path = Path(args.frozen_manifest).parent / "sealed_open_receipt.json"
        with open(receipt_path) as stream:
            receipt = json.load(stream)
        if receipt.get("one_time_open_complete") is not True:
            raise RuntimeError("Sealed-open receipt is incomplete")
        if Path(receipt["dataset_dir"]).resolve() != Path(args.data_dir).resolve():
            raise RuntimeError("Sealed baseline dataset differs from the one-time receipt")
        if frozen.get("bundle", {}).get("v32_manifest_sha256") != sha256_file(args.deployment_manifest):
            raise RuntimeError("v3.2 deployment manifest differs from the frozen bundle")
    if int(protocol["window_size"]) != 3:
        raise ValueError("Frozen v3.2 evaluator requires a W3 v4 fold")
    with open(args.deployment_manifest) as stream:
        deployment = json.load(stream)
    if deployment.get("test_opened") is not False:
        raise RuntimeError("Frozen v3.2 manifest no longer records a closed historical test")
    fold_parts = str(protocol.get("fold", "")).split("__")
    old_oof_session = None
    if len(fold_parts) == 3 and fold_parts[1] == "con_rot":
        old_oof_session = int(fold_parts[2].removeprefix("s"))
    if old_oof_session is None:
        # Deployment v3.2 has never seen any new-generation session, so it is
        # a valid frozen baseline for new-session folds.
        raw_paths = [item["path"] for item in deployment["models"] if item["role"] == "position_member"]
        rowtime_path = next(item["path"] for item in deployment["models"] if item["role"] == "position_member_unit_row_time")
        log_path = next(item["path"] for item in deployment["models"] if item["role"] == "orientation")
        checkpoint_source = "frozen_v3_2_deployment_for_unseen_new_generation"
    else:
        # The deployment checkpoint includes every old non-test session. Use
        # the historical fold-specific checkpoints for genuinely out-of-fold
        # predictions on old con_rot sessions.
        raw_paths = [
            f"checkpoints/new_calib_v3_temporal_cv/dev_s{old_oof_session}_seed{seed}"
            for seed in (42, 7, 123)
        ]
        rowtime_path = f"checkpoints/new_calib_v3_temporal_cv_rowtime/dev_s{old_oof_session}_seed42"
        log_path = f"checkpoints/new_calib_v3_temporal_cv_logratio/dev_s{old_oof_session}_seed42"
        checkpoint_source = f"historical_v3_2_out_of_fold_dev_session_{old_oof_session}"
    if any(index < 0 or index >= len(raw_paths) for index in args.position_raw_indices):
        raise ValueError("position_raw_indices is outside the three frozen raw members")
    device = resolve_device(args.device)
    raw_models = [load_model(path, device)[0] for path in raw_paths]
    rowtime_model = load_model(rowtime_path, device)[0]
    log_model = load_model(log_path, device)[0]
    with np.load(Path(args.data_dir) / f"{args.split}.npz") as loaded:
        data = {key: loaded[key] for key in loaded.files}
    packed = data["emf"].astype(np.float32)
    raw_window = packed[:, :27].reshape(-1, 3, 9)
    raw_features = packed[:, :27]
    rowtime_features = packed[:, :29]
    log_features = logratio_features(raw_window)
    raw_tensor = torch.as_tensor(raw_features, device=device)
    rowtime_tensor = torch.as_tensor(rowtime_features, device=device)
    log_tensor = torch.as_tensor(log_features, device=device)
    with torch.inference_mode():
        raw_pred = [model(raw_tensor, return_normalized=False).cpu().numpy() for model in raw_models]
        rowtime_pred = rowtime_model(rowtime_tensor, return_normalized=False).cpu().numpy()
        log_pred = log_model(log_tensor, return_normalized=False).cpu().numpy()
    position_members = [raw_pred[index] for index in args.position_raw_indices]
    if not args.exclude_rowtime:
        position_members.append(rowtime_pred)
    position = np.mean(np.stack(position_members)[:, :, :3], axis=0)
    if args.orientation == "log":
        orientation = log_pred[:, 3:]
    else:
        raw_index = 1 if args.orientation == "raw7_log" else 2
        orientation = mean_rotation_6d(np.stack([raw_pred[raw_index][:, 3:], log_pred[:, 3:]]))
    pred = np.column_stack([position, orientation]).astype(np.float32)
    target = data["target"]
    aggregate = pose_metrics_v3(pred, target)
    sessions = {}
    session_keys = data["session_key"].astype(str)
    for key in sorted(set(session_keys)):
        mask = session_keys == key
        sessions[key] = pose_metrics_v3(pred[mask], target[mask])
    pred_pose = target_v3_to_pose_deg(pred)
    target_pose = target_v3_to_pose_deg(target)
    predictions = pd.DataFrame({
        "generation": data["generation"].astype(str),
        "session_key": session_keys,
        "source_file": data["source_file"].astype(str),
        "source_row": data["source_row"],
        "target_X_mm": target_pose[:, 0], "target_Y_mm": target_pose[:, 1], "target_Z_mm": target_pose[:, 2],
        "target_roll_deg": target_pose[:, 3], "target_pitch_deg": target_pose[:, 4], "target_yaw_deg": target_pose[:, 5],
        "pred_X_mm": pred_pose[:, 0], "pred_Y_mm": pred_pose[:, 1], "pred_Z_mm": pred_pose[:, 2],
        "pred_roll_deg": pred_pose[:, 3], "pred_pitch_deg": pred_pose[:, 4], "pred_yaw_deg": pred_pose[:, 5],
        "position_euclidean_error_mm": np.linalg.norm(pred[:, :3] - target[:, :3], axis=1),
        "orientation_geodesic_error_deg": so3_geodesic_deg(pred[:, 3:], target[:, 3:]),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_name = "val_predictions.csv" if args.split == "val" else "sealed_test_predictions.csv"
    predictions.to_csv(out_dir / prediction_name, index=False)
    summary = {
        "schema_version": 4,
        "split": "v4_development_only" if args.split == "val" else "sealed_test",
        "rule": {
            "position_raw_indices": args.position_raw_indices,
            "include_rowtime": not args.exclude_rowtime,
            "orientation": args.orientation,
        },
        "checkpoint_source": checkpoint_source,
        "aggregate": aggregate,
        "sessions": sessions,
        "historical_locked_test_opened": args.split == "sealed_test",
        "new_sealed_test_opened": args.split == "sealed_test",
    }
    with open(out_dir / "summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

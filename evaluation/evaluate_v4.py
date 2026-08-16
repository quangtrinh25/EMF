"""Evaluate v4 validation folds and guarded one-time sealed tests."""

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

from evaluation.v4_common import checkpoint_signature, load_model_v4, resolve_device
from physics.rotation_repr import pose_deg_to_target_v3, so3_geodesic_deg, target_v3_to_pose_deg
from training.metrics import pose_metrics_v3


def require_sealed_authorization(data_dir, checkpoint_dir, confirmed, frozen_manifest):
    if not confirmed or frozen_manifest is None:
        raise RuntimeError(
            "Sealed test refused: pass --confirm_open_sealed_test and the frozen manifest"
        )
    with open(Path(data_dir) / "protocol.json") as stream:
        protocol = json.load(stream)
    if protocol.get("sealed_test_opened") is not True:
        raise RuntimeError("Dataset protocol does not record an authorized sealed-test opening")
    with open(frozen_manifest) as stream:
        manifest = json.load(stream)
    receipt_path = Path(frozen_manifest).parent / "sealed_open_receipt.json"
    if not receipt_path.exists():
        raise RuntimeError("The one-time sealed-open receipt is missing")
    with open(receipt_path) as stream:
        receipt = json.load(stream)
    if receipt.get("one_time_open_complete") is not True:
        raise RuntimeError("The sealed-open receipt is incomplete")
    if Path(receipt.get("dataset_dir", "")).resolve() != Path(data_dir).resolve():
        raise RuntimeError("Sealed evaluator data directory differs from the one-time receipt")
    signature = checkpoint_signature(checkpoint_dir)
    if manifest.get("frozen") is not True or manifest.get("model_sha256") != signature["best_pt_sha256"]:
        raise RuntimeError("Checkpoint does not match the frozen candidate")


def metric_row(scope, pred, target):
    return {"scope": scope, **pose_metrics_v3(pred, target)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate v4 with grouped diagnostics.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", choices=["val", "sealed_test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--confirm_open_sealed_test", action="store_true")
    parser.add_argument("--frozen_manifest", default=None)
    args = parser.parse_args()

    if args.split == "sealed_test":
        require_sealed_authorization(
            args.data_dir, args.checkpoint_dir, args.confirm_open_sealed_test,
            args.frozen_manifest,
        )
    data_path = Path(args.data_dir) / f"{args.split}.npz"
    with np.load(data_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    device = resolve_device(args.device)
    model, metadata = load_model_v4(args.checkpoint_dir, device)
    x = torch.as_tensor(data["emf"], dtype=torch.float32, device=device)
    generation = torch.as_tensor(data["generation_index"], dtype=torch.long, device=device)
    with torch.inference_mode():
        pred = model(x, generation, return_normalized=False).cpu().numpy()
    target = data["target"]
    rows = [metric_row("pooled", pred, target)]
    generation_names = data["generation"].astype(str)
    session_keys = data["session_key"].astype(str)
    source_files = data["source_file"].astype(str)
    for name in sorted(set(generation_names)):
        mask = generation_names == name
        rows.append(metric_row(f"generation:{name}", pred[mask], target[mask]))
    for key in sorted(set(session_keys)):
        mask = session_keys == key
        rows.append(metric_row(f"session:{key}", pred[mask], target[mask]))

    workspace = metadata["workspace"]
    center = np.asarray(workspace["center_mm"], dtype=float)
    half = np.asarray(workspace["size_mm"], dtype=float) / 2.0
    lower, upper = center - half, center + half
    boundary_distance = np.min(
        np.concatenate([target[:, :3] - lower, upper - target[:, :3]], axis=1), axis=1,
    )
    boundary_mask = boundary_distance <= 10.0
    for label, mask in (("boundary_le_10mm", boundary_mask), ("interior_gt_10mm", ~boundary_mask)):
        if np.any(mask):
            rows.append(metric_row(label, pred[mask], target[mask]))
    warmup = data["warmup"].astype(bool) if "warmup" in data else np.zeros(len(target), dtype=bool)
    if np.any(warmup):
        rows.append(metric_row("warmup", pred[warmup], target[warmup]))
    if np.any(~warmup):
        rows.append(metric_row("steady_state", pred[~warmup], target[~warmup]))

    pred_pose = target_v3_to_pose_deg(pred)
    target_pose = target_v3_to_pose_deg(target)
    position_error = np.linalg.norm(pred[:, :3] - target[:, :3], axis=1)
    orientation_error = so3_geodesic_deg(pred[:, 3:], target[:, 3:])
    predictions = pd.DataFrame({
        "generation": generation_names,
        "session_key": session_keys,
        "source_file": source_files,
        "source_row": data["source_row"],
        "warmup": warmup,
        "boundary_distance_mm": boundary_distance,
        "target_X_mm": target_pose[:, 0], "target_Y_mm": target_pose[:, 1], "target_Z_mm": target_pose[:, 2],
        "target_roll_deg": target_pose[:, 3], "target_pitch_deg": target_pose[:, 4], "target_yaw_deg": target_pose[:, 5],
        "pred_X_mm": pred_pose[:, 0], "pred_Y_mm": pred_pose[:, 1], "pred_Z_mm": pred_pose[:, 2],
        "pred_roll_deg": pred_pose[:, 3], "pred_pitch_deg": pred_pose[:, 4], "pred_yaw_deg": pred_pose[:, 5],
        "position_euclidean_error_mm": position_error,
        "orientation_geodesic_error_deg": orientation_error,
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / f"{args.split}_metrics.csv", index=False)
    predictions.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)
    pooled = metrics.loc[metrics["scope"] == "pooled"].iloc[0].to_dict()
    sessions = {
        row["scope"].removeprefix("session:"): row
        for row in metrics.to_dict("records") if row["scope"].startswith("session:")
    }
    summary = {
        "schema_version": 4,
        "split": args.split,
        "aggregate": pooled,
        "sessions": sessions,
        "checkpoint_signature": checkpoint_signature(args.checkpoint_dir),
        "sealed_evaluation": args.split == "sealed_test",
    }
    with open(out_dir / "summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

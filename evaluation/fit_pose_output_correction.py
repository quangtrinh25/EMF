import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import COLS, extract_emf_and_pose
from evaluation.evaluate_real_csv import load_model, parse_named_csvs, prepare_emf_features
from datasets.temporal_windows import load_temporal_csvs
from training.metrics import orientation_rmse, position_rmse


def angles_from_cos(values):
    return np.degrees(np.arccos(np.clip(values, -1.0, 1.0)))


def target_from_pose_deg(poses):
    target = np.zeros((len(poses), 6), dtype=np.float32)
    target[:, :3] = poses[:, :3]
    target[:, 3:] = np.cos(np.radians(poses[:, 3:]))
    return target


def load_csvs(items):
    frames = []
    for _, path in parse_named_csvs(items):
        frames.append(pd.read_csv(path, header=None, names=COLS))
    return pd.concat(frames, ignore_index=True)


def predict_frame(df, model, device, metadata=None):
    emf_v, poses = extract_emf_and_pose(df)
    features = prepare_emf_features(emf_v, metadata or {})
    target = target_from_pose_deg(poses)
    with torch.no_grad():
        pred = model(torch.tensor(features, dtype=torch.float32, device=device), return_normalized=False).cpu().numpy()
    return pred, target


def predict_csv_items(items, model, device, metadata=None):
    metadata = metadata or {}
    if metadata.get("temporal_window"):
        features, target, _ = load_temporal_csvs(items, int(metadata["temporal_window"]))
        with torch.no_grad():
            pred = model(torch.tensor(features, dtype=torch.float32, device=device), return_normalized=False).cpu().numpy()
        return pred, target
    df = load_csvs(items)
    return predict_frame(df, model, device, metadata=metadata)


def pose_features(pred):
    return np.column_stack([pred[:, :3], angles_from_cos(pred[:, 3:])])


def pose_targets(target):
    return np.column_stack([target[:, :3], angles_from_cos(target[:, 3:])])


def fit_affine(features, targets, alpha):
    x_aug = np.column_stack([features, np.ones(len(features))])
    regularizer = np.eye(x_aug.shape[1]) * alpha
    regularizer[-1, -1] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ targets)
    return beta[:-1], beta[-1]


def apply_correction(pred, coef, intercept):
    corrected_pose = pose_features(pred) @ coef + intercept
    corrected_pose[:, 3:] = np.clip(corrected_pose[:, 3:], 0.0, 180.0)
    corrected = np.zeros_like(pred)
    corrected[:, :3] = corrected_pose[:, :3]
    corrected[:, 3:] = np.cos(np.radians(corrected_pose[:, 3:]))
    return corrected


def metrics_row(label, pred, target):
    return {
        "label": label,
        "num_samples": len(pred),
        "position_rmse_mm": position_rmse(pred, target),
        "orientation_rmse_deg": orientation_rmse(pred, target),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit affine output correction from real CSV predictions.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model_type", default="resnet", choices=["resnet", "fcn", "kan"])
    parser.add_argument("--train_csvs", nargs="+", required=True, help="Training CSVs as name=path.")
    parser.add_argument("--eval_csvs", nargs="+", default=None, help="Optional eval CSVs as name=path.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--metrics_out", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_path, metadata = load_model(args.checkpoint_dir, args.model_type, device)
    train_pred, train_target = predict_csv_items(args.train_csvs, model, device, metadata=metadata)

    coef, intercept = fit_affine(pose_features(train_pred), pose_targets(train_target), args.alpha)
    out_path = args.out or os.path.join(args.checkpoint_dir, "pose_output_correction.npz")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        coef=coef,
        intercept=intercept,
        alpha=float(args.alpha),
        checkpoint=ckpt_path,
    )

    rows = [
        metrics_row("train_raw", train_pred, train_target),
        metrics_row("train_corrected", apply_correction(train_pred, coef, intercept), train_target),
    ]
    if args.eval_csvs is not None:
        eval_pred, eval_target = predict_csv_items(args.eval_csvs, model, device, metadata=metadata)
        rows.append(metrics_row("eval_raw", eval_pred, eval_target))
        rows.append(metrics_row("eval_corrected", apply_correction(eval_pred, coef, intercept), eval_target))

    metrics = pd.DataFrame(rows)
    if args.metrics_out is not None:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        metrics.to_csv(args.metrics_out, index=False)

    print(f"Saved pose correction to {out_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

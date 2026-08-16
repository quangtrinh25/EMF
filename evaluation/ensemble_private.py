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

from calibration.report_utils import COLS, FILE_MAP, extract_emf_and_pose
from evaluation.evaluate_real_csv import (
    angles_from_cos,
    load_model,
    parse_named_csvs,
    prepare_emf_features,
    target_from_pose_deg,
)
from training.metrics import orientation_rmse, position_rmse


def predict_with_checkpoint(df, checkpoint_dir, device):
    model, ckpt_path, metadata = load_model(checkpoint_dir, "resnet", device)
    emf_v, poses = extract_emf_and_pose(df)
    features = prepare_emf_features(emf_v, metadata)
    with torch.no_grad():
        pred = model(torch.tensor(features, dtype=torch.float32, device=device), return_normalized=False).cpu().numpy()
    return pred, target_from_pose_deg(poses), ckpt_path


def ensemble_prediction(position_pred, orientation_pred):
    pred = np.zeros_like(position_pred)
    pred[:, :3] = position_pred[:, :3]
    pred[:, 3:] = orientation_pred[:, 3:]
    return pred


def eval_sources(args):
    if args.eval_csvs is not None:
        return [(name, path, path) for name, path in parse_named_csvs(args.eval_csvs)]
    return [(key, FILE_MAP[key], os.path.join(args.data_dir, FILE_MAP[key])) for key in args.eval_files]


def write_results(rows, pred_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "ensemble_pose_metrics.csv")
    predictions_path = os.path.join(out_dir, "ensemble_pose_predictions.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(pred_rows).to_csv(predictions_path, index=False)
    return metrics_path, predictions_path


def main():
    parser = argparse.ArgumentParser(description="Ensemble position from single-row ResNet and orientation from temporal ResNet.")
    parser.add_argument("--data_dir", default="data/raw_calibration")
    parser.add_argument("--position_checkpoint_dir", default="checkpoints/noleak_balanced_easy/final")
    parser.add_argument("--orientation_checkpoint_dir", default="checkpoints/temporal_balanced_easy/final")
    parser.add_argument("--out_dir", default="results/ensemble_private")
    parser.add_argument("--eval_files", nargs="+", default=["set13_con_spi_no_rot", "set12_cyl_spi_rot"], choices=list(FILE_MAP.keys()))
    parser.add_argument("--eval_csvs", nargs="+", default=None, help="Explicit eval CSVs as name=path.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    pred_rows = []

    for key, filename, path in eval_sources(args):
        df = pd.read_csv(path, header=None, names=COLS)
        pos_pred, target, pos_ckpt = predict_with_checkpoint(df, args.position_checkpoint_dir, device)
        ori_pred, _, ori_ckpt = predict_with_checkpoint(df, args.orientation_checkpoint_dir, device)
        pred = ensemble_prediction(pos_pred, ori_pred)
        rows.append({
            "dataset": key,
            "filename": filename,
            "num_samples": len(df),
            "position_checkpoint": pos_ckpt,
            "orientation_checkpoint": ori_ckpt,
            "position_rmse_mm": position_rmse(pred, target),
            "orientation_rmse_deg": orientation_rmse(pred, target),
        })
        pred_angles = angles_from_cos(pred[:, 3:])
        target_angles = angles_from_cos(target[:, 3:])
        for i in range(len(df)):
            pred_rows.append({
                "dataset": key,
                "sample_index": i,
                "target_X_mm": target[i, 0],
                "target_Y_mm": target[i, 1],
                "target_Z_mm": target[i, 2],
                "pred_X_mm": pred[i, 0],
                "pred_Y_mm": pred[i, 1],
                "pred_Z_mm": pred[i, 2],
                "target_roll_deg": target_angles[i, 0],
                "target_pitch_deg": target_angles[i, 1],
                "target_yaw_deg": target_angles[i, 2],
                "pred_roll_deg": pred_angles[i, 0],
                "pred_pitch_deg": pred_angles[i, 1],
                "pred_yaw_deg": pred_angles[i, 2],
            })

    metrics_path, predictions_path = write_results(rows, pred_rows, args.out_dir)
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()

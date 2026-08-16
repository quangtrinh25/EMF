import argparse
import json
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
from evaluation.evaluate import build_model
from datasets.temporal_windows import build_past_windows
from training.metrics import orientation_rmse, position_rmse


def angles_from_cos(values):
    return np.degrees(np.arccos(np.clip(values, -1.0, 1.0)))


def parse_named_csvs(items):
    parsed = []
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"Expected name=path CSV argument, got: {item}")
        name, path = item.split('=', 1)
        if not name or not path:
            raise ValueError(f"Expected name=path CSV argument, got: {item}")
        parsed.append((name, path))
    return parsed


def target_from_pose_deg(poses):
    target = np.zeros((len(poses), 6), dtype=np.float32)
    target[:, :3] = poses[:, :3]
    target[:, 3:] = np.cos(np.radians(poses[:, 3:]))
    return target


def load_metadata(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "model_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_model(checkpoint_dir, model_type, device):
    metadata = load_metadata(checkpoint_dir)
    mean_emf = np.load(os.path.join(checkpoint_dir, 'emf_mean.npy'))
    input_dim = int(metadata.get("input_dim", len(mean_emf)))
    model = build_model(model_type, input_dim=input_dim).to(device)
    ckpt_path = os.path.join(checkpoint_dir, f'{model_type}_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.set_normalization(
        mean_emf,
        np.load(os.path.join(checkpoint_dir, 'emf_std.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_std.npy')),
    )
    model.eval()
    return model, ckpt_path, metadata


def load_pose_correction(path):
    if path is None:
        return None
    data = np.load(path)
    return {
        'coef': data['coef'],
        'intercept': data['intercept'],
    }


def apply_pose_correction(pred, correction):
    if correction is None:
        return pred
    pred_angles = angles_from_cos(pred[:, 3:])
    features = np.column_stack([pred[:, :3], pred_angles])
    corrected_pose = features @ correction['coef'] + correction['intercept']
    corrected_pose[:, 3:] = np.clip(corrected_pose[:, 3:], 0.0, 180.0)
    corrected = np.zeros_like(pred)
    corrected[:, :3] = corrected_pose[:, :3]
    corrected[:, 3:] = np.cos(np.radians(corrected_pose[:, 3:]))
    return corrected


def prepare_emf_features(emf_v, metadata):
    window = metadata.get("temporal_window")
    if window:
        if metadata.get("temporal_mode", "past") != "past":
            raise ValueError("Only past temporal windows are supported.")
        return build_past_windows(emf_v, int(window))
    return emf_v


def evaluate_frame(df, model, device, metadata=None, correction=None):
    metadata = metadata or {}
    emf_v, poses = extract_emf_and_pose(df)
    features = prepare_emf_features(emf_v, metadata)
    target = target_from_pose_deg(poses)
    with torch.no_grad():
        pred = model(torch.tensor(features, dtype=torch.float32, device=device), return_normalized=False).cpu().numpy()
    pred = apply_pose_correction(pred, correction)
    return pred, target, {
        'position_rmse_mm': position_rmse(pred, target),
        'orientation_rmse_deg': orientation_rmse(pred, target),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw_calibration')
    parser.add_argument('--checkpoint_dir', default='checkpoints/custom_system_v2')
    parser.add_argument('--model_type', default='resnet', choices=['resnet', 'fcn', 'kan'])
    parser.add_argument('--out_dir', default='results/pose_v2')
    parser.add_argument('--eval_files', nargs='+', default=list(FILE_MAP.keys()), choices=list(FILE_MAP.keys()))
    parser.add_argument('--eval_csvs', nargs='+', default=None, help='Explicit eval CSVs as name=path.')
    parser.add_argument('--pose_correction', default=None, help='Optional pose output correction .npz.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt_path, metadata = load_model(args.checkpoint_dir, args.model_type, device)
    correction = load_pose_correction(args.pose_correction)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    pred_rows = []
    if args.eval_csvs is not None:
        eval_sources = [(name, path, path) for name, path in parse_named_csvs(args.eval_csvs)]
    else:
        eval_sources = [(key, FILE_MAP[key], os.path.join(args.data_dir, FILE_MAP[key])) for key in args.eval_files]

    for key, filename, path in eval_sources:
        df = pd.read_csv(path, header=None, names=COLS)
        pred, target, metrics = evaluate_frame(df, model, device, metadata=metadata, correction=correction)
        rows.append({
            'dataset': key,
            'filename': filename,
            'num_samples': len(df),
            'checkpoint': ckpt_path,
            'pose_correction': args.pose_correction or '',
            **metrics,
        })
        pred_angles = angles_from_cos(pred[:, 3:])
        target_angles = angles_from_cos(target[:, 3:])
        for i in range(len(df)):
            pred_rows.append({
                'dataset': key,
                'sample_index': i,
                'target_X_mm': target[i, 0],
                'target_Y_mm': target[i, 1],
                'target_Z_mm': target[i, 2],
                'pred_X_mm': pred[i, 0],
                'pred_Y_mm': pred[i, 1],
                'pred_Z_mm': pred[i, 2],
                'target_roll_deg': target_angles[i, 0],
                'target_pitch_deg': target_angles[i, 1],
                'target_yaw_deg': target_angles[i, 2],
                'pred_roll_deg': pred_angles[i, 0],
                'pred_pitch_deg': pred_angles[i, 1],
                'pred_yaw_deg': pred_angles[i, 2],
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, 'real_csv_pose_metrics.csv'), index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(args.out_dir, 'real_csv_pose_predictions.csv'), index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()

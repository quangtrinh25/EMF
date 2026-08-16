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

from models.residual_net import ResidualNet
from physics.rotation_repr import (
    so3_geodesic_deg,
    target_v3_to_pose_deg,
    wrapped_angle_difference_deg,
)
from training.metrics import pose_metrics_v3


def load_model(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / 'model_metadata.json') as stream:
        metadata = json.load(stream)
    if metadata.get('target_representation') != 'position_mm+rotation_6d':
        raise ValueError("Checkpoint is not a v3 signed-orientation model")
    model = ResidualNet(
        input_dim=int(metadata['input_dim']),
        output_dim=int(metadata['output_dim']),
        hidden=int(metadata['hidden']),
        n_blocks=int(metadata['residual_blocks']),
        slope=float(metadata.get('leaky_relu_slope', 0.01)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / 'best.pt', map_location=device))
    model.set_normalization(
        np.load(checkpoint_dir / 'emf_mean.npy'),
        np.load(checkpoint_dir / 'emf_std.npy'),
        np.load(checkpoint_dir / 'pose_mean.npy'),
        np.load(checkpoint_dir / 'pose_std.npy'),
    )
    model.eval()
    return model, metadata


def metric_row(label, pred, target):
    return {'scope': label, **pose_metrics_v3(pred, target)}


def require_test_confirmation(split, confirmed):
    if split == 'test' and not confirmed:
        raise RuntimeError(
            "Locked v3 test refused. Add --confirm_open_locked_test only after "
            "the deployment manifest and all model choices are frozen."
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate v3 with pooled XYZ and SO(3) metrics.")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--confirm_open_locked_test', action='store_true')
    args = parser.parse_args()
    require_test_confirmation(args.split, args.confirm_open_locked_test)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA device. "
            "Check the NVIDIA driver, or use --device cpu/auto."
        )
    device = torch.device(
        'cuda' if args.device in {'auto', 'cuda'} and torch.cuda.is_available() else 'cpu'
    )
    data = np.load(Path(args.data_dir) / f'{args.split}.npz')
    emf = data['emf'].astype(np.float32)
    target = data['target'].astype(np.float32)
    model, metadata = load_model(args.checkpoint_dir, device)
    with torch.inference_mode():
        pred = model(torch.tensor(emf, device=device), return_normalized=False).cpu().numpy()

    rows = [metric_row('pooled', pred, target)]
    families = data['family'].astype(str)
    source_files = data['source_file'].astype(str)
    for family in sorted(set(families)):
        mask = families == family
        rows.append(metric_row(f'family:{family}', pred[mask], target[mask]))
    for source_file in sorted(set(source_files)):
        mask = source_files == source_file
        rows.append(metric_row(f'file:{source_file}', pred[mask], target[mask]))

    pred_pose = target_v3_to_pose_deg(pred)
    target_pose = target_v3_to_pose_deg(target)
    xyz_error = np.linalg.norm(pred[:, :3] - target[:, :3], axis=1)
    orientation_error = so3_geodesic_deg(pred[:, 3:], target[:, 3:])
    euler_error = wrapped_angle_difference_deg(pred_pose[:, 3:], target_pose[:, 3:])
    if 'target_source_row' in data:
        source_row = data['target_source_row']
        input_source_row = data['input_source_row']
    else:
        source_row = data['source_row']
        input_source_row = source_row
    predictions = pd.DataFrame({
        'family': families,
        'source_file': source_files,
        'input_source_row': input_source_row,
        'target_source_row': source_row,
        'target_X_mm': target_pose[:, 0],
        'target_Y_mm': target_pose[:, 1],
        'target_Z_mm': target_pose[:, 2],
        'pred_X_mm': pred_pose[:, 0],
        'pred_Y_mm': pred_pose[:, 1],
        'pred_Z_mm': pred_pose[:, 2],
        'position_euclidean_error_mm': xyz_error,
        'target_roll_deg': target_pose[:, 3],
        'target_pitch_deg': target_pose[:, 4],
        'target_yaw_deg': target_pose[:, 5],
        'pred_roll_deg': pred_pose[:, 3],
        'pred_pitch_deg': pred_pose[:, 4],
        'pred_yaw_deg': pred_pose[:, 5],
        'roll_wrapped_error_deg': euler_error[:, 0],
        'pitch_wrapped_error_deg': euler_error[:, 1],
        'yaw_wrapped_error_deg': euler_error[:, 2],
        'orientation_geodesic_error_deg': orientation_error,
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / f'{args.split}_metrics.csv', index=False)
    predictions.to_csv(out_dir / f'{args.split}_predictions.csv', index=False)
    with open(out_dir / 'evaluation_metadata.json', 'w') as stream:
        json.dump({
            'split': args.split,
            'data_dir': args.data_dir,
            'checkpoint_dir': args.checkpoint_dir,
            'model_metadata': metadata,
            'metric_aggregation': 'pooled_from_sample_errors',
        }, stream, indent=2)
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()

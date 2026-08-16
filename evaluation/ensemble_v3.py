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

from evaluation.evaluate_v3 import load_model, metric_row, require_test_confirmation
from physics.rotation_repr import (
    so3_geodesic_deg,
    target_v3_to_pose_deg,
    wrapped_angle_difference_deg,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate v3 single-position + temporal-orientation ensemble.")
    parser.add_argument('--data_dir', required=True, help="Temporal dataset directory containing current_emf.")
    parser.add_argument('--position_checkpoint_dir', required=True)
    parser.add_argument('--orientation_checkpoint_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
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
    if 'current_emf' not in data:
        raise ValueError("Temporal dataset has no current_emf field for the single-frame component")
    temporal_features = data['emf'].astype(np.float32)
    current_emf = data['current_emf'].astype(np.float32)
    target = data['target'].astype(np.float32)
    position_model, position_metadata = load_model(args.position_checkpoint_dir, device)
    orientation_model, orientation_metadata = load_model(args.orientation_checkpoint_dir, device)
    if int(position_metadata['input_dim']) != 9:
        raise ValueError("Position component must be the matched nine-input baseline")
    if int(orientation_metadata['input_dim']) != temporal_features.shape[1]:
        raise ValueError("Orientation checkpoint does not match temporal feature dimensions")

    with torch.inference_mode():
        position_pred = position_model(
            torch.tensor(current_emf, device=device), return_normalized=False,
        ).cpu().numpy()
        orientation_pred = orientation_model(
            torch.tensor(temporal_features, device=device), return_normalized=False,
        ).cpu().numpy()
    ensemble = np.column_stack([position_pred[:, :3], orientation_pred[:, 3:]])

    families = data['family'].astype(str)
    source_files = data['source_file'].astype(str)
    rows = [metric_row('pooled', ensemble, target)]
    for family in sorted(set(families)):
        mask = families == family
        rows.append(metric_row(f'family:{family}', ensemble[mask], target[mask]))
    for source_file in sorted(set(source_files)):
        mask = source_files == source_file
        rows.append(metric_row(f'file:{source_file}', ensemble[mask], target[mask]))

    pred_pose = target_v3_to_pose_deg(ensemble)
    target_pose = target_v3_to_pose_deg(target)
    xyz_error = np.linalg.norm(ensemble[:, :3] - target[:, :3], axis=1)
    orientation_error = so3_geodesic_deg(ensemble[:, 3:], target[:, 3:])
    euler_error = wrapped_angle_difference_deg(pred_pose[:, 3:], target_pose[:, 3:])
    predictions = pd.DataFrame({
        'family': families,
        'source_file': source_files,
        'input_source_row': data['input_source_row'],
        'target_source_row': data['target_source_row'],
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
    with open(out_dir / 'ensemble_metadata.json', 'w') as stream:
        json.dump({
            'split': args.split,
            'position_checkpoint_dir': args.position_checkpoint_dir,
            'orientation_checkpoint_dir': args.orientation_checkpoint_dir,
            'selection_rule': 'single_frame_position_plus_temporal_rotation6d',
            'metric_aggregation': 'pooled_from_sample_errors',
            'test_opened': args.split == 'test',
        }, stream, indent=2)
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()

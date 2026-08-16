import argparse
import hashlib
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
    mean_rotation_6d,
    so3_geodesic_deg,
    target_v3_to_pose_deg,
    wrapped_angle_difference_deg,
)


COMPATIBILITY_KEYS = (
    'input_dim', 'output_dim', 'target_representation', 'feature_layout',
    'temporal_window', 'temporal_feature_mode', 'horizon_steps', 'dt_feature_count',
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensemble_targets(predictions, orientation_model_index=None):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 3 or predictions.shape[2] != 9:
        raise ValueError(f"Expected predictions shaped (models, samples, 9), got {predictions.shape}")
    position = np.mean(predictions[:, :, :3], axis=0)
    orientation = (
        mean_rotation_6d(predictions[:, :, 3:])
        if orientation_model_index is None
        else predictions[int(orientation_model_index), :, 3:]
    )
    return np.column_stack([position, orientation]).astype(np.float32)


def safe_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a multi-seed v3 ensemble on one locked protocol split.")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--checkpoint_dirs', nargs='+', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--confirm_open_locked_test', action='store_true')
    parser.add_argument(
        '--orientation_model_index', type=int, default=None,
        help='Use orientation from this checkpoint index; default is the SO(3) ensemble mean.',
    )
    parser.add_argument('--orientation_data_dir', default=None)
    parser.add_argument('--orientation_checkpoint_dir', default=None)
    parser.add_argument('--aux_position_data_dirs', nargs='*', default=[])
    parser.add_argument('--aux_position_checkpoint_dirs', nargs='*', default=[])
    args = parser.parse_args()
    require_test_confirmation(args.split, args.confirm_open_locked_test)

    if len(args.checkpoint_dirs) < 2:
        raise ValueError("At least two independently trained checkpoints are required")
    if args.orientation_model_index is not None and not (
        0 <= args.orientation_model_index < len(args.checkpoint_dirs)
    ):
        raise ValueError("orientation_model_index is outside checkpoint_dirs")
    if bool(args.orientation_data_dir) != bool(args.orientation_checkpoint_dir):
        raise ValueError("orientation_data_dir and orientation_checkpoint_dir must be supplied together")
    if len(args.aux_position_data_dirs) != len(args.aux_position_checkpoint_dirs):
        raise ValueError("aux_position_data_dirs and aux_position_checkpoint_dirs must have equal length")
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA device. "
            "Check the NVIDIA driver, or use --device cpu/auto."
        )
    device = torch.device(
        'cuda' if args.device in {'auto', 'cuda'} and torch.cuda.is_available() else 'cpu'
    )
    data = np.load(Path(args.data_dir) / f'{args.split}.npz')
    features = data['emf'].astype(np.float32)
    target = data['target'].astype(np.float32)

    predictions = []
    metadata_rows = []
    reference = None
    for checkpoint_dir_raw in args.checkpoint_dirs:
        checkpoint_dir = Path(checkpoint_dir_raw)
        model, metadata = load_model(checkpoint_dir, device)
        signature = {
            key: (
                metadata.get(key, 'raw_window')
                if key == 'temporal_feature_mode'
                else metadata.get(key)
            )
            for key in COMPATIBILITY_KEYS
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(
                f"Incompatible checkpoint {checkpoint_dir}: {signature} != {reference}"
            )
        with torch.inference_mode():
            pred = model(torch.tensor(features, device=device), return_normalized=False).cpu().numpy()
        predictions.append(pred)
        metadata_rows.append({
            'checkpoint_dir': str(checkpoint_dir),
            'seed': metadata.get('seed'),
            'best_pt_sha256': sha256(checkpoint_dir / 'best.pt'),
        })

    stacked = np.stack(predictions)
    aux_position_predictions = []
    aux_position_records = []
    for aux_data_dir, aux_checkpoint_dir_raw in zip(
        args.aux_position_data_dirs, args.aux_position_checkpoint_dirs,
    ):
        aux_data = np.load(Path(aux_data_dir) / f'{args.split}.npz')
        for key in ('target', 'family', 'source_file', 'input_source_row', 'target_source_row'):
            if key in data and key in aux_data and not np.array_equal(
                data[key], aux_data[key], equal_nan=data[key].dtype.kind in 'fc',
            ):
                raise ValueError(f"Base and auxiliary position datasets are not aligned on {key}")
        aux_checkpoint_dir = Path(aux_checkpoint_dir_raw)
        aux_model, aux_metadata = load_model(aux_checkpoint_dir, device)
        with torch.inference_mode():
            aux_pred = aux_model(
                torch.tensor(aux_data['emf'].astype(np.float32), device=device),
                return_normalized=False,
            ).cpu().numpy()
        aux_position_predictions.append(aux_pred)
        aux_position_records.append({
            'data_dir': aux_data_dir,
            'checkpoint_dir': str(aux_checkpoint_dir),
            'seed': aux_metadata.get('seed'),
            'best_pt_sha256': sha256(aux_checkpoint_dir / 'best.pt'),
            'feature_layout': aux_metadata.get('feature_layout'),
            'timestamp_feature': aux_metadata.get('timestamp_feature'),
        })

    ensemble = ensemble_targets(stacked, args.orientation_model_index)
    position_stacked = (
        np.concatenate([stacked, np.stack(aux_position_predictions)], axis=0)
        if aux_position_predictions else stacked
    )
    ensemble[:, :3] = np.mean(position_stacked[:, :, :3], axis=0)
    orientation_model_record = None
    orientation_predictions = None
    if args.orientation_checkpoint_dir:
        orientation_data = np.load(Path(args.orientation_data_dir) / f'{args.split}.npz')
        for key in ('target', 'family', 'source_file', 'input_source_row', 'target_source_row'):
            if key in data and key in orientation_data and not np.array_equal(
                data[key], orientation_data[key], equal_nan=data[key].dtype.kind in 'fc',
            ):
                raise ValueError(f"Position and orientation datasets are not aligned on {key}")
        orientation_model, orientation_metadata = load_model(args.orientation_checkpoint_dir, device)
        with torch.inference_mode():
            orientation_predictions = orientation_model(
                torch.tensor(orientation_data['emf'].astype(np.float32), device=device),
                return_normalized=False,
            ).cpu().numpy()
        ensemble[:, 3:] = orientation_predictions[:, 3:]
        orientation_checkpoint = Path(args.orientation_checkpoint_dir)
        orientation_model_record = {
            'checkpoint_dir': str(orientation_checkpoint),
            'seed': orientation_metadata.get('seed'),
            'best_pt_sha256': sha256(orientation_checkpoint / 'best.pt'),
            'feature_layout': orientation_metadata.get('feature_layout'),
            'temporal_feature_mode': orientation_metadata.get('temporal_feature_mode'),
        }
    position_dispersion = np.sqrt(np.mean(
        np.sum((position_stacked[:, :, :3] - ensemble[None, :, :3]) ** 2, axis=2), axis=0,
    ))
    orientation_members = list(stacked)
    if orientation_predictions is not None:
        orientation_members.append(orientation_predictions)
    orientation_dispersion = np.sqrt(np.mean(np.stack([
        so3_geodesic_deg(model_pred[:, 3:], ensemble[:, 3:]) ** 2
        for model_pred in orientation_members
    ]), axis=0))

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
    source_row = data['target_source_row'] if 'target_source_row' in data else data['source_row']
    input_source_row = data['input_source_row'] if 'input_source_row' in data else source_row
    frame = pd.DataFrame({
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
        'position_ensemble_dispersion_mm': position_dispersion,
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
        'orientation_ensemble_dispersion_deg': orientation_dispersion,
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f'{args.split}_metrics.csv', index=False)
    frame.to_csv(out_dir / f'{args.split}_predictions.csv', index=False)
    report = {
        'split': args.split,
        'data_dir': args.data_dir,
        'models': metadata_rows,
        'auxiliary_position_models': aux_position_records,
        'external_orientation_model': orientation_model_record,
        'compatibility_signature': reference,
        'position_aggregation': 'arithmetic_mean_mm',
        'orientation_aggregation': (
            'external_orientation_checkpoint'
            if orientation_model_record is not None
            else (
                'chordal_matrix_mean_projected_to_SO3'
                if args.orientation_model_index is None
                else f"checkpoint_index_{args.orientation_model_index}"
            )
        ),
        'uncertainty_is_model_disagreement_not_calibrated_probability': True,
        'position_dispersion_error_correlation': safe_correlation(position_dispersion, xyz_error),
        'orientation_dispersion_error_correlation': safe_correlation(orientation_dispersion, orientation_error),
    }
    with open(out_dir / 'ensemble_metadata.json', 'w') as stream:
        json.dump(report, stream, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

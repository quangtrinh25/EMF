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

from evaluation.evaluate_v3 import load_model
from evaluation.multiseed_ensemble_v3 import COMPATIBILITY_KEYS, ensemble_targets
from inference.predict_capsule_pose_v3 import extract, prepare_features, read_frame
from physics.rotation_repr import pose_deg_to_target_v3, so3_geodesic_deg, target_v3_to_pose_deg
from training.metrics import pose_metrics_v3


class SelectedComponentEnsemble(torch.nn.Module):
    """Frozen v3.2 component rule with member outputs for uncertainty."""
    def __init__(self, position_models, aux_position_models, orientation_model):
        super().__init__()
        self.position_models = torch.nn.ModuleList(position_models)
        self.aux_position_models = torch.nn.ModuleList(aux_position_models)
        self.orientation_model = orientation_model

    def forward(self, position_features, orientation_features, aux_position_features):
        base_predictions = [
            model(position_features, return_normalized=False)
            for model in self.position_models
        ]
        aux_predictions = [
            model(aux_position_features[index], return_normalized=False)
            for index, model in enumerate(self.aux_position_models)
        ]
        base_stacked = torch.stack(base_predictions)
        position_stacked = (
            torch.cat([base_stacked, torch.stack(aux_predictions)], dim=0)
            if aux_predictions else base_stacked
        )
        orientation_prediction = self.orientation_model(
            orientation_features, return_normalized=False,
        )
        ensemble = torch.cat([
            position_stacked[:, :, :3].mean(dim=0),
            orientation_prediction[:, 3:],
        ], dim=1)
        return ensemble, base_stacked, position_stacked, orientation_prediction


def normalized_signature(metadata):
    return {
        key: metadata.get(key, 'raw_window') if key == 'temporal_feature_mode' else metadata.get(key)
        for key in COMPATIBILITY_KEYS
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested):
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access a CUDA device. "
                "Check the NVIDIA driver, or use --device cpu/auto."
            )
        return torch.device('cuda')
    if requested == 'auto' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description="Predict capsule 6-DoF pose with a multi-seed v3 ensemble.")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint_dirs', nargs='+', required=True)
    parser.add_argument('--aux_position_checkpoint_dirs', nargs='*', default=[])
    parser.add_argument('--orientation_model_index', type=int, default=None)
    parser.add_argument('--orientation_checkpoint_dir', default=None)
    parser.add_argument('--input_unit', choices=['mV', 'V'], default='mV')
    parser.add_argument('--header', choices=['auto', 'yes', 'no'], default='auto')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--backend', choices=['auto', 'eager', 'torchscript'], default='auto')
    parser.add_argument('--cpu_threads', type=int, default=None)
    parser.add_argument('--cpu_interop_threads', type=int, default=1)
    parser.add_argument('--metrics_out', default=None)
    parser.add_argument('--metadata_out', default=None)
    parser.add_argument('--timestamp_column', default=None)
    parser.add_argument('--timestamp_unit', choices=['s', 'ms', 'us', 'ns'], default='s')
    parser.add_argument('--reset_column', default=None)
    parser.add_argument('--reset_every_rows', type=int, default=None)
    parser.add_argument('--row_as_timestamp', action='store_true')
    args = parser.parse_args()

    if args.cpu_threads is not None:
        if args.cpu_threads < 1 or args.cpu_interop_threads < 1:
            raise ValueError("CPU thread counts must be positive")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_interop_threads)

    if len(args.checkpoint_dirs) < 2:
        raise ValueError("At least two checkpoint directories are required")
    if args.orientation_model_index is not None and not (
        0 <= args.orientation_model_index < len(args.checkpoint_dirs)
    ):
        raise ValueError("orientation_model_index is outside checkpoint_dirs")

    frame = read_frame(args.input, args.header)
    emf, target_pose, source_rows, timestamps_s, reset_flags = extract(
        frame, args.timestamp_column, args.timestamp_unit, args.reset_column,
    )
    gap_reset = np.r_[True, np.diff(source_rows) > 1] if len(source_rows) else np.zeros(0, dtype=bool)
    reset_flags = gap_reset if reset_flags is None else (reset_flags | gap_reset)
    if args.row_as_timestamp:
        if timestamps_s is not None:
            raise ValueError("Choose a real timestamp column or --row_as_timestamp, not both")
        timestamps_s = source_rows.astype(float)
    if args.reset_every_rows is not None:
        if args.reset_every_rows < 1:
            raise ValueError("reset_every_rows must be positive")
        periodic_reset = (source_rows > 0) & (source_rows % args.reset_every_rows == 0)
        reset_flags = periodic_reset if reset_flags is None else (reset_flags | periodic_reset)
    if args.input_unit == 'mV':
        emf *= 1e-3
    device = resolve_device(args.device)
    runtime_backend = 'torchscript' if args.backend == 'auto' else args.backend

    models = []
    metadata_rows = []
    reference_signature = None
    reference_metadata = None
    for checkpoint_dir_raw in args.checkpoint_dirs:
        checkpoint_dir = Path(checkpoint_dir_raw)
        model, metadata = load_model(checkpoint_dir, device)
        signature = normalized_signature(metadata)
        if reference_signature is None:
            reference_signature = signature
            reference_metadata = metadata
        elif signature != reference_signature:
            raise ValueError(f"Incompatible checkpoint {checkpoint_dir}: {signature} != {reference_signature}")
        models.append(model)
        metadata_rows.append({
            'checkpoint_dir': str(checkpoint_dir),
            'seed': metadata.get('seed'),
            'best_pt_sha256': file_sha256(checkpoint_dir / 'best.pt'),
        })

    external_orientation_model = None
    external_orientation_metadata = None
    external_orientation_record = None
    if args.orientation_checkpoint_dir:
        orientation_checkpoint = Path(args.orientation_checkpoint_dir)
        external_orientation_model, external_orientation_metadata = load_model(
            orientation_checkpoint, device,
        )
        if int(external_orientation_metadata.get('horizon_steps', 0)) != int(
            reference_metadata.get('horizon_steps', 0)
        ):
            raise ValueError("Position and orientation checkpoints have different horizons")
        if int(external_orientation_metadata.get('temporal_window', 0)) != int(
            reference_metadata.get('temporal_window', 0)
        ):
            raise ValueError("Position and orientation checkpoints have different window sizes")
        external_orientation_record = {
            'checkpoint_dir': str(orientation_checkpoint),
            'seed': external_orientation_metadata.get('seed'),
            'best_pt_sha256': file_sha256(orientation_checkpoint / 'best.pt'),
            'feature_layout': external_orientation_metadata.get('feature_layout'),
            'temporal_feature_mode': external_orientation_metadata.get('temporal_feature_mode'),
        }

    aux_position_models = []
    aux_position_metadata = []
    aux_position_records = []
    for aux_checkpoint_raw in args.aux_position_checkpoint_dirs:
        aux_checkpoint = Path(aux_checkpoint_raw)
        aux_model, aux_metadata = load_model(aux_checkpoint, device)
        if int(aux_metadata.get('horizon_steps', 0)) != int(reference_metadata.get('horizon_steps', 0)):
            raise ValueError("Base and auxiliary position checkpoints have different horizons")
        if int(aux_metadata.get('temporal_window', 0)) != int(reference_metadata.get('temporal_window', 0)):
            raise ValueError("Base and auxiliary position checkpoints have different window sizes")
        aux_position_models.append(aux_model)
        aux_position_metadata.append(aux_metadata)
        aux_position_records.append({
            'checkpoint_dir': str(aux_checkpoint),
            'seed': aux_metadata.get('seed'),
            'best_pt_sha256': file_sha256(aux_checkpoint / 'best.pt'),
            'feature_layout': aux_metadata.get('feature_layout'),
            'timestamp_feature': aux_metadata.get('timestamp_feature'),
        })

    needs_row_time = reference_metadata.get('timestamp_source') == 'unit_step_source_row_experiment'
    if external_orientation_metadata is not None:
        needs_row_time |= external_orientation_metadata.get('timestamp_source') == 'unit_step_source_row_experiment'
    needs_row_time |= any(
        metadata.get('timestamp_source') == 'unit_step_source_row_experiment'
        for metadata in aux_position_metadata
    )
    if timestamps_s is None and needs_row_time:
        timestamps_s = source_rows.astype(float)

    features, warmup = prepare_features(
        emf, reference_metadata, timestamps_s, reset_flags, return_warmup=True,
    )
    orientation_features = None
    if external_orientation_model is not None:
        orientation_features = prepare_features(
            emf, external_orientation_metadata, timestamps_s, reset_flags,
        )
    aux_position_features = [
        prepare_features(emf, metadata, timestamps_s, reset_flags)
        for metadata in aux_position_metadata
    ]
    horizon = int(reference_metadata.get('horizon_steps', 0))
    if horizon > 0:
        if len(features) <= horizon:
            raise ValueError("Input is shorter than the checkpoint prediction horizon")
        features = features[:-horizon]
        if orientation_features is not None:
            orientation_features = orientation_features[:-horizon]
        aux_position_features = [values[:-horizon] for values in aux_position_features]
        source_rows = source_rows[:-horizon]
        warmup = warmup[:-horizon]
        if target_pose is not None:
            target_pose = target_pose[horizon:]

    feature_tensor = torch.tensor(features, device=device)
    if len(feature_tensor) == 0:
        raise ValueError("Input contains no valid EMF rows")
    aux_position_feature_tensors = [
        torch.tensor(values, device=device) for values in aux_position_features
    ]
    orientation_tensor = (
        None
        if orientation_features is None
        else torch.tensor(orientation_features, device=device)
    )
    external_orientation_prediction_tensor = None
    per_model_ood = []
    for model in models:
        mean = model.emf_mean.detach().cpu().numpy()
        std = model.emf_std.detach().cpu().numpy()
        per_model_ood.append(np.mean(np.abs((features - mean) / (std + 1e-8)) > 3.0, axis=1))
    for model, values in zip(aux_position_models, aux_position_features):
        mean = model.emf_mean.detach().cpu().numpy()
        std = model.emf_std.detach().cpu().numpy()
        per_model_ood.append(np.mean(np.abs((values - mean) / (std + 1e-8)) > 3.0, axis=1))
    if external_orientation_model is not None:
        mean = external_orientation_model.emf_mean.detach().cpu().numpy()
        std = external_orientation_model.emf_std.detach().cpu().numpy()
        per_model_ood.append(np.mean(
            np.abs((orientation_features - mean) / (std + 1e-8)) > 3.0, axis=1,
        ))

    with torch.inference_mode():
        if external_orientation_model is not None:
            runtime_model = SelectedComponentEnsemble(
                models, aux_position_models, external_orientation_model,
            ).eval()
            if runtime_backend == 'torchscript':
                runtime_model = torch.jit.trace(
                    runtime_model,
                    (
                        feature_tensor[:1], orientation_tensor[:1],
                        tuple(values[:1] for values in aux_position_feature_tensors),
                    ),
                    check_trace=False,
                )
                runtime_model = torch.jit.freeze(runtime_model.eval())
            (
                ensemble_tensor, stacked_tensor, position_stacked_tensor,
                external_orientation_prediction_tensor,
            ) = runtime_model(
                feature_tensor, orientation_tensor, tuple(aux_position_feature_tensors),
            )
            ensemble = ensemble_tensor.cpu().numpy()
        else:
            prediction_tensors = [
                model(feature_tensor, return_normalized=False) for model in models
            ]
            aux_position_prediction_tensors = [
                model(values, return_normalized=False)
                for model, values in zip(aux_position_models, aux_position_feature_tensors)
            ]
            stacked_tensor = torch.stack(prediction_tensors)
            position_stacked_tensor = torch.cat(
                [stacked_tensor, torch.stack(aux_position_prediction_tensors)], dim=0,
            ) if aux_position_prediction_tensors else stacked_tensor
            stacked_for_ensemble = stacked_tensor.cpu().numpy()
            ensemble = ensemble_targets(stacked_for_ensemble, args.orientation_model_index)
            ensemble[:, :3] = position_stacked_tensor[:, :, :3].mean(dim=0).cpu().numpy()
    stacked = stacked_tensor.cpu().numpy()
    position_stacked = position_stacked_tensor.cpu().numpy()
    external_orientation_prediction = (
        None
        if external_orientation_prediction_tensor is None
        else external_orientation_prediction_tensor.cpu().numpy()
    )
    pred_pose = target_v3_to_pose_deg(ensemble)
    position_dispersion = np.sqrt(np.mean(
        np.sum((position_stacked[:, :, :3] - ensemble[None, :, :3]) ** 2, axis=2), axis=0,
    ))
    orientation_members = list(stacked)
    if external_orientation_prediction is not None:
        orientation_members.append(external_orientation_prediction)
    orientation_dispersion = np.sqrt(np.mean(np.stack([
        so3_geodesic_deg(pred[:, 3:], ensemble[:, 3:]) ** 2 for pred in orientation_members
    ]), axis=0))
    ood_fraction = np.max(np.stack(per_model_ood), axis=0)

    workspace = reference_metadata.get('workspace')
    outside_workspace = np.zeros(len(pred_pose), dtype=bool)
    if workspace:
        center = np.asarray(workspace['center_mm'], dtype=float)
        half_size = np.asarray(workspace['size_mm'], dtype=float) / 2.0
        tolerance = float(workspace.get('tolerance_mm', 0.0))
        outside_workspace = np.any(
            (pred_pose[:, :3] < center - half_size - tolerance)
            | (pred_pose[:, :3] > center + half_size + tolerance), axis=1,
        )

    output = pd.DataFrame({
        'source_row': source_rows,
        'pred_X_mm': pred_pose[:, 0],
        'pred_Y_mm': pred_pose[:, 1],
        'pred_Z_mm': pred_pose[:, 2],
        'pred_roll_deg': pred_pose[:, 3],
        'pred_pitch_deg': pred_pose[:, 4],
        'pred_yaw_deg': pred_pose[:, 5],
        'position_ensemble_dispersion_mm': position_dispersion,
        'orientation_ensemble_dispersion_deg': orientation_dispersion,
        'input_ood_feature_fraction_max_model': ood_fraction,
        'outside_nominal_workspace': outside_workspace,
        'warmup': warmup,
    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    if target_pose is not None:
        metrics = pose_metrics_v3(ensemble, pose_deg_to_target_v3(target_pose))
        metrics_path = Path(args.metrics_out) if args.metrics_out else output_path.with_name(output_path.stem + '_metrics.json')
        with open(metrics_path, 'w') as stream:
            json.dump(metrics, stream, indent=2)
        print(json.dumps(metrics, indent=2))

    report = {
        'models': metadata_rows,
        'auxiliary_position_models': aux_position_records,
        'external_orientation_model': external_orientation_record,
        'compatibility_signature': reference_signature,
        'position_aggregation': 'arithmetic_mean_mm',
        'orientation_aggregation': (
            'external_orientation_checkpoint'
            if external_orientation_record is not None
            else (
                'chordal_matrix_mean_projected_to_SO3'
                if args.orientation_model_index is None
                else f'checkpoint_index_{args.orientation_model_index}'
            )
        ),
        'runtime_device': str(device),
        'runtime_backend': runtime_backend,
        'cpu_threads': torch.get_num_threads() if device.type == 'cpu' else None,
        'cpu_interop_threads': torch.get_num_interop_threads() if device.type == 'cpu' else None,
        'external_orientation_fast_path': external_orientation_record is not None,
        'torchscript_startup_trace_excluded_from_steady_state_benchmark': True,
        'uncertainty_is_model_disagreement_not_calibrated_probability': True,
        'warmup_rows': int(np.sum(warmup)),
        'outside_workspace_rows': int(np.sum(outside_workspace)),
    }
    metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_name(output_path.stem + '_metadata.json')
    with open(metadata_path, 'w') as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {len(output)} ensemble capsule-pose predictions to {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

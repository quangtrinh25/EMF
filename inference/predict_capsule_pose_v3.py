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
from physics.rotation_repr import pose_deg_to_target_v3, target_v3_to_pose_deg
from training.metrics import pose_metrics_v3


POSE_COLS = ['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model(features, return_normalized=False)


def read_frame(path, header):
    if header == 'yes':
        return pd.read_csv(path)
    if header == 'no':
        return pd.read_csv(path, header=None)
    first = pd.read_csv(path, nrows=0)
    return pd.read_csv(path) if any(str(col).startswith('EMF') for col in first.columns) else pd.read_csv(path, header=None)


def extract(frame, timestamp_column=None, timestamp_unit='s', reset_column=None):
    emf_names = [f'EMF{i}' for i in range(1, 10)]
    if all(name in frame.columns for name in emf_names):
        emf = frame[emf_names].to_numpy(dtype=np.float32)
        pose = frame[POSE_COLS].to_numpy(dtype=np.float32) if all(name in frame.columns for name in POSE_COLS) else None
    else:
        if frame.shape[1] < 9:
            raise ValueError("Input must contain at least nine EMF columns")
        emf = frame.iloc[:, :9].to_numpy(dtype=np.float32)
        pose = frame.iloc[:, 9:15].to_numpy(dtype=np.float32) if frame.shape[1] >= 15 else None
    valid = np.isfinite(emf).all(axis=1)
    if pose is not None:
        valid &= np.isfinite(pose).all(axis=1)
    timestamps_s = None
    if timestamp_column is not None:
        if timestamp_column not in frame.columns:
            raise ValueError(f"Timestamp column {timestamp_column!r} not found")
        factors = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
        timestamp = pd.to_numeric(frame[timestamp_column], errors='coerce').to_numpy(dtype=float)
        valid &= np.isfinite(timestamp)
        timestamps_s = timestamp[valid] * factors[timestamp_unit]
    reset_flags = None
    if reset_column is not None:
        if reset_column not in frame.columns:
            raise ValueError(f"Reset column {reset_column!r} not found")
        reset_values = frame[reset_column]
        if reset_values.dtype == bool:
            reset_flags = reset_values.to_numpy()[valid]
        else:
            reset_flags = reset_values.astype(str).str.strip().str.lower().isin(
                {'1', 'true', 'yes', 'y', 'reset'},
            ).to_numpy()[valid]
    if timestamps_s is not None and len(timestamps_s) > 1:
        non_increasing = np.diff(timestamps_s) <= 0
        reset_after_jump = (
            reset_flags[1:] if reset_flags is not None
            else np.zeros(len(timestamps_s) - 1, dtype=bool)
        )
        if np.any(non_increasing & ~reset_after_jump):
            raise ValueError(
                "Timestamps must be strictly increasing inside each acquisition segment"
            )
    return (
        emf[valid], None if pose is None else pose[valid], np.flatnonzero(valid),
        timestamps_s, reset_flags,
    )


def prepare_features(emf, metadata, timestamps_s=None, reset_flags=None, return_warmup=False):
    window = metadata.get('temporal_window')
    if not window:
        warmup = np.zeros(len(emf), dtype=bool)
        return (emf, warmup) if return_warmup else emf
    window = int(window)
    feature_mode = metadata.get('temporal_feature_mode', 'raw_window')
    dt_count = int(metadata.get('dt_feature_count', 0))
    if dt_count:
        if timestamps_s is None:
            raise ValueError("This checkpoint requires timestamp-derived delta_t features")
    if reset_flags is None:
        reset_flags = np.zeros(len(emf), dtype=bool)
    else:
        reset_flags = np.asarray(reset_flags, dtype=bool)
        if reset_flags.shape != (len(emf),):
            raise ValueError("reset_flags must have one value per EMF row")
    reset_flags = reset_flags.copy()
    if len(reset_flags):
        reset_flags[0] = True

    feature_rows = []
    warmup = np.zeros(len(emf), dtype=bool)
    emf_buffer = []
    time_buffer = []
    for index, row in enumerate(emf):
        if reset_flags[index]:
            emf_buffer = []
            time_buffer = []
        emf_buffer.append(row)
        if timestamps_s is not None:
            time_buffer.append(timestamps_s[index])
        if len(emf_buffer) > window:
            emf_buffer.pop(0)
            if time_buffer:
                time_buffer.pop(0)
        warmup[index] = len(emf_buffer) < window
        padded_emf = [emf_buffer[0]] * (window - len(emf_buffer)) + emf_buffer
        window_emf = np.stack(padded_emf)
        if feature_mode == 'raw_window':
            values = window_emf.reshape(-1)
        elif feature_mode == 'current_plus_log_ratio':
            current = window_emf[-1]
            eps_v = 1e-4
            log_ratio = np.log((window_emf[:-1] + eps_v) / (current[None, :] + eps_v))
            values = np.concatenate([
                current, np.clip(log_ratio, -5.0, 5.0).reshape(-1),
            ])
        else:
            raise ValueError(f"Unsupported temporal feature mode: {feature_mode}")
        if dt_count:
            padded_time = [time_buffer[0]] * (window - len(time_buffer)) + time_buffer
            values = np.concatenate([values, np.diff(padded_time)])
        feature_rows.append(values.astype(np.float32))
    features = np.stack(feature_rows) if feature_rows else np.empty((0, int(metadata['input_dim'])), dtype=np.float32)
    return (features, warmup) if return_warmup else features


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
    parser = argparse.ArgumentParser(description="Predict signed 6-DoF pose with a v3 checkpoint.")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--input_unit', choices=['mV', 'V'], default='mV')
    parser.add_argument('--header', choices=['auto', 'yes', 'no'], default='auto')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--backend', choices=['auto', 'eager', 'torchscript'], default='auto')
    parser.add_argument('--cpu_threads', type=int, default=None)
    parser.add_argument('--cpu_interop_threads', type=int, default=1)
    parser.add_argument('--metrics_out', default=None)
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
    model, metadata = load_model(args.checkpoint_dir, device)
    if timestamps_s is None and metadata.get('timestamp_source') == 'unit_step_source_row_experiment':
        timestamps_s = source_rows.astype(float)
    features, warmup = prepare_features(
        emf, metadata, timestamps_s, reset_flags, return_warmup=True,
    )
    horizon = int(metadata.get('horizon_steps', 0))
    if target_pose is not None and horizon > 0:
        if len(features) <= horizon:
            raise ValueError("Input is shorter than the checkpoint prediction horizon")
        features = features[:-horizon]
        target_pose = target_pose[horizon:]
        source_rows = source_rows[:-horizon]
        warmup = warmup[:-horizon]
    feature_tensor = torch.tensor(features, device=device)
    if len(feature_tensor) == 0:
        raise ValueError("Input contains no valid EMF rows")
    with torch.inference_mode():
        if runtime_backend == 'torchscript':
            runtime_model = torch.jit.trace(
                InferenceWrapper(model), feature_tensor[:1], check_trace=False,
            )
            runtime_model = torch.jit.freeze(runtime_model.eval())
            pred_target = runtime_model(feature_tensor).cpu().numpy()
        else:
            pred_target = model(
                feature_tensor, return_normalized=False,
            ).cpu().numpy()
    pred_pose = target_v3_to_pose_deg(pred_target)
    emf_mean = model.emf_mean.detach().cpu().numpy()
    emf_std = model.emf_std.detach().cpu().numpy()
    ood_fraction = float(np.mean(np.abs((features - emf_mean) / (emf_std + 1e-8)) > 3.0))
    workspace = metadata.get('workspace')
    outside_workspace = np.zeros(len(pred_pose), dtype=bool)
    if workspace:
        center = np.asarray(workspace['center_mm'], dtype=float)
        half_size = np.asarray(workspace['size_mm'], dtype=float) / 2.0
        tolerance = float(workspace.get('tolerance_mm', 0.0))
        outside_workspace = np.any(
            (pred_pose[:, :3] < center - half_size - tolerance)
            | (pred_pose[:, :3] > center + half_size + tolerance),
            axis=1,
        )
    output = pd.DataFrame({
        'source_row': source_rows,
        'pred_X_mm': pred_pose[:, 0],
        'pred_Y_mm': pred_pose[:, 1],
        'pred_Z_mm': pred_pose[:, 2],
        'pred_roll_deg': pred_pose[:, 3],
        'pred_pitch_deg': pred_pose[:, 4],
        'pred_yaw_deg': pred_pose[:, 5],
        'outside_nominal_workspace': outside_workspace,
        'warmup': warmup,
    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    if target_pose is not None:
        metrics = pose_metrics_v3(pred_target, pose_deg_to_target_v3(target_pose))
        metrics_path = Path(args.metrics_out) if args.metrics_out else output_path.with_name(output_path.stem + '_metrics.json')
        with open(metrics_path, 'w') as stream:
            json.dump(metrics, stream, indent=2)
        print(json.dumps(metrics, indent=2))
    print(f"Saved {len(output)} signed-pose predictions to {output_path}")
    print(f"Checkpoint representation: {metadata['target_representation']}")
    print(f"Runtime: device={device}, backend={runtime_backend}")
    if metadata.get('temporal_window'):
        print(
            f"Temporal window: {metadata['temporal_window']}; "
            f"horizon steps: {metadata.get('horizon_steps', 0)}; "
            f"delta_t features: {metadata.get('dt_feature_count', 0)}; "
            f"feature mode: {metadata.get('temporal_feature_mode', 'raw_window')}"
        )
    print(f"Input OOD fraction (|train z| > 3): {ood_fraction:.2%}")
    if ood_fraction > 0.20:
        print("WARNING: input EMF is strongly out of distribution for this checkpoint.")
    if np.any(outside_workspace):
        print(f"WARNING: {int(outside_workspace.sum())} predictions are outside the nominal workspace.")
    if np.any(warmup):
        print(f"Warm-up predictions: {int(warmup.sum())}; do not use them for downstream control/navigation.")


if __name__ == '__main__':
    main()

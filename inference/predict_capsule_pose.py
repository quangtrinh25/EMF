import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.temporal_windows import build_past_windows


class ResidualBlock(nn.Module):
    def __init__(self, dim=512, slope=0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(slope),
            nn.Linear(dim, dim), nn.LeakyReLU(slope),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualNet(nn.Module):
    def __init__(self, input_dim=9, n_blocks=7, hidden=512, slope=0.01):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, slope) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.Linear(128, 6),
        )
        self.register_buffer('emf_mean', torch.zeros(input_dim))
        self.register_buffer('emf_std', torch.ones(input_dim))
        self.register_buffer('pose_mean', torch.zeros(6))
        self.register_buffer('pose_std', torch.ones(6))

    def set_normalization(self, emf_mean, emf_std, pose_mean=None, pose_std=None):
        self.emf_mean.copy_(torch.as_tensor(emf_mean, dtype=torch.float32))
        self.emf_std.copy_(torch.as_tensor(emf_std, dtype=torch.float32))
        if pose_mean is not None:
            self.pose_mean.copy_(torch.as_tensor(pose_mean, dtype=torch.float32))
        if pose_std is not None:
            self.pose_std.copy_(torch.as_tensor(pose_std, dtype=torch.float32))

    def forward(self, x, return_normalized=False):
        x = (x - self.emf_mean) / (self.emf_std + 1e-8)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        if return_normalized:
            return out
        return out * (self.pose_std + 1e-8) + self.pose_mean


def position_rmse(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    diff = (pred[:, :3] - target[:, :3]) ** 2
    return float(np.sqrt(diff.sum(axis=1).mean() / 3.0))


def orientation_rmse(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    pred_cos = np.clip(pred[:, 3:], -1.0, 1.0)
    target_cos = np.clip(target[:, 3:], -1.0, 1.0)
    pred_ang = np.degrees(np.arccos(pred_cos))
    target_ang = np.degrees(np.arccos(target_cos))
    diff = (pred_ang - target_ang) ** 2
    return float(np.sqrt(diff.sum(axis=1).mean() / 3.0))


RAW_COLS = [
    'EMF1', 'EMF2', 'EMF3', 'EMF4', 'EMF5', 'EMF6', 'EMF7', 'EMF8', 'EMF9',
    'X', 'Y', 'Z', 'roll', 'pitch', 'yaw',
]


def default_checkpoint_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path('checkpoints/noleak_balanced_easy/final'),
        Path('checkpoints/custom_system_v2_noleak_balanced'),
        Path('checkpoints/custom_system_v2_noleak_norot'),
        Path('local_calib_1/models/capsule_hybrid_final'),
        script_dir / 'models' / 'capsule_hybrid_final',
        Path('models/capsule_hybrid_final'),
    ]
    for candidate in candidates:
        if (candidate / 'resnet_best.pt').exists() or (candidate / 'best.pt').exists():
            return str(candidate)
    return str(candidates[0])


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
    pred_angles = np.degrees(np.arccos(np.clip(pred[:, 3:], -1.0, 1.0)))
    features = np.column_stack([pred[:, :3], pred_angles])
    corrected_pose = features @ correction['coef'] + correction['intercept']
    corrected_pose[:, 3:] = np.clip(corrected_pose[:, 3:], 0.0, 180.0)
    corrected = np.zeros_like(pred)
    corrected[:, :3] = corrected_pose[:, :3]
    corrected[:, 3:] = np.cos(np.radians(corrected_pose[:, 3:]))
    return corrected


def load_metadata(checkpoint_dir):
    path = os.path.join(checkpoint_dir, 'model_metadata.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_model(checkpoint_dir, device):
    metadata = load_metadata(checkpoint_dir)
    mean_emf = np.load(os.path.join(checkpoint_dir, 'emf_mean.npy'))
    input_dim = int(metadata.get('input_dim', len(mean_emf)))
    model = ResidualNet(input_dim=input_dim).to(device)
    ckpt_path = os.path.join(checkpoint_dir, 'resnet_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No checkpoint found in {checkpoint_dir}')

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.set_normalization(
        mean_emf,
        np.load(os.path.join(checkpoint_dir, 'emf_std.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_std.npy')),
    )
    model.eval()
    return model, ckpt_path, metadata


def prepare_emf_features(emf, metadata):
    window = metadata.get('temporal_window')
    if window:
        if metadata.get('temporal_mode', 'past') != 'past':
            raise ValueError('Only past temporal windows are supported.')
        return build_past_windows(emf, int(window))
    return emf


def read_input(path, header):
    if header == 'auto':
        first = pd.read_csv(path, nrows=0)
        if any(str(col).startswith('EMF') for col in first.columns):
            return pd.read_csv(path)
        return pd.read_csv(path, header=None)
    if header == 'yes':
        return pd.read_csv(path)
    return pd.read_csv(path, header=None)


def extract_emf_and_target(df):
    has_named_emf = all(f'EMF{i}' in df.columns for i in range(1, 10))
    if has_named_emf:
        emf = df[[f'EMF{i}' for i in range(1, 10)]].to_numpy(dtype=np.float32)
        has_target = all(col in df.columns for col in ['X', 'Y', 'Z', 'roll', 'pitch', 'yaw'])
        target_pose = df[['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']].to_numpy(dtype=np.float32) if has_target else None
        return emf, target_pose

    if df.shape[1] >= 15:
        named = df.iloc[:, :15].copy()
        named.columns = RAW_COLS
        emf = named[[f'EMF{i}' for i in range(1, 10)]].to_numpy(dtype=np.float32)
        target_pose = named[['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']].to_numpy(dtype=np.float32)
        return emf, target_pose

    if df.shape[1] >= 9:
        emf = df.iloc[:, :9].to_numpy(dtype=np.float32)
        return emf, None

    raise ValueError('Input must contain at least 9 EMF columns.')


def target_from_pose_deg(pose_deg):
    target = np.zeros((len(pose_deg), 6), dtype=np.float32)
    target[:, :3] = pose_deg[:, :3]
    target[:, 3:] = np.cos(np.radians(pose_deg[:, 3:]))
    return target


def prediction_frame(pred, target_pose=None):
    pred_angles = np.degrees(np.arccos(np.clip(pred[:, 3:], -1.0, 1.0)))
    out = pd.DataFrame({
        'sample_index': np.arange(len(pred), dtype=int),
        'pred_X_mm': pred[:, 0],
        'pred_Y_mm': pred[:, 1],
        'pred_Z_mm': pred[:, 2],
        'pred_roll_deg': pred_angles[:, 0],
        'pred_pitch_deg': pred_angles[:, 1],
        'pred_yaw_deg': pred_angles[:, 2],
    })

    if target_pose is not None:
        target_angles = np.degrees(
            np.arccos(np.clip(np.cos(np.radians(target_pose[:, 3:])), -1.0, 1.0))
        )
        out['target_X_mm'] = target_pose[:, 0]
        out['target_Y_mm'] = target_pose[:, 1]
        out['target_Z_mm'] = target_pose[:, 2]
        out['target_roll_deg'] = target_pose[:, 3]
        out['target_pitch_deg'] = target_pose[:, 4]
        out['target_yaw_deg'] = target_pose[:, 5]
        out['target_roll_cos_angle_deg'] = target_angles[:, 0]
        out['target_pitch_cos_angle_deg'] = target_angles[:, 1]
        out['target_yaw_cos_angle_deg'] = target_angles[:, 2]
        pos_err = np.linalg.norm(pred[:, :3] - target_pose[:, :3], axis=1)
        ori_err = np.linalg.norm(pred_angles - target_angles, axis=1)
        out['position_error_mm'] = pos_err
        out['orientation_error_deg'] = ori_err

    return out


def write_metrics(path, pred, target_pose, input_path, checkpoint):
    if target_pose is None:
        return None
    target = target_from_pose_deg(target_pose)
    metrics = pd.DataFrame([{
        'input': input_path,
        'checkpoint': checkpoint,
        'num_samples': len(pred),
        'position_rmse_mm': position_rmse(pred, target),
        'orientation_rmse_deg': orientation_rmse(pred, target),
    }])
    metrics.to_csv(path, index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Predict capsule position/orientation from raw EMF CSV data.'
    )
    parser.add_argument('--input', required=True, help='CSV file with 9 EMF columns, or raw calibration CSV with 9 EMF + 6 pose columns.')
    parser.add_argument('--output', default='results/predicted_capsule_pose.csv')
    parser.add_argument('--metrics_out', default=None)
    parser.add_argument('--checkpoint_dir', default=default_checkpoint_dir())
    parser.add_argument('--input_unit', choices=['mV', 'V'], default='mV', help='Unit of EMF columns. Raw calibration files are mV.')
    parser.add_argument('--header', choices=['auto', 'yes', 'no'], default='auto')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--pose_correction', default=None, help='Optional pose output correction .npz.')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    df = read_input(args.input, args.header)
    emf, target_pose = extract_emf_and_target(df)

    if args.input_unit == 'mV':
        emf = emf * 1e-3

    model, ckpt_path, metadata = load_model(args.checkpoint_dir, device)
    features = prepare_emf_features(emf, metadata)
    with torch.no_grad():
        pred = model(torch.tensor(features, dtype=torch.float32, device=device), return_normalized=False).cpu().numpy()
    pred = apply_pose_correction(pred, load_pose_correction(args.pose_correction))

    out = prediction_frame(pred, target_pose)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out.to_csv(args.output, index=False)

    metrics_path = args.metrics_out
    if metrics_path is None and target_pose is not None:
        base, _ = os.path.splitext(args.output)
        metrics_path = f'{base}_metrics.csv'
    metrics = write_metrics(metrics_path, pred, target_pose, args.input, ckpt_path) if metrics_path else None

    print(f'Input: {args.input}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Samples: {len(pred)}')
    print(f'Saved predictions: {args.output}')
    if metrics is not None:
        print(f"Position RMSE: {metrics.iloc[0]['position_rmse_mm']:.4f} mm")
        print(f"Orientation RMSE: {metrics.iloc[0]['orientation_rmse_deg']:.4f} deg")
        print(f'Saved metrics: {metrics_path}')


if __name__ == '__main__':
    main()

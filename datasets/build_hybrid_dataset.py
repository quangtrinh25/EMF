import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import COLS, FILE_MAP, extract_emf_and_pose


def target_from_pose_deg(poses):
    targets = np.zeros((len(poses), 6), dtype=np.float32)
    targets[:, :3] = poses[:, :3]
    targets[:, 3:] = np.cos(np.radians(poses[:, 3:]))
    return targets


def augment_emf(emf_v, rng, copies, noise_std_mV, gain_drift):
    batches = []
    for _ in range(copies):
        gains = rng.uniform(1.0 - gain_drift, 1.0 + gain_drift, size=emf_v.shape)
        noise = rng.normal(0.0, noise_std_mV * 1e-3, size=emf_v.shape)
        batches.append(np.maximum(emf_v * gains + noise, 0.0))
    return np.vstack(batches).astype(np.float32)


def load_real(data_dir, file_keys):
    frames = []
    for key in file_keys:
        frames.append(pd.read_csv(os.path.join(data_dir, FILE_MAP[key]), header=None, names=COLS))
    df = pd.concat(frames, ignore_index=True)
    emf_v, poses = extract_emf_and_pose(df)
    return emf_v.astype(np.float32), target_from_pose_deg(poses)


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


def load_real_csvs(items):
    frames = []
    for _, path in parse_named_csvs(items):
        frames.append(pd.read_csv(path, header=None, names=COLS))
    df = pd.concat(frames, ignore_index=True)
    emf_v, poses = extract_emf_and_pose(df)
    return emf_v.astype(np.float32), target_from_pose_deg(poses)


def copy_split(synthetic_dir, out_dir, split):
    data = np.load(os.path.join(synthetic_dir, f'{split}.npz'))
    np.savez(os.path.join(out_dir, f'{split}.npz'), emf=data['emf'], target=data['target'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_dir', default='data/synthetic_calibrated_v2')
    parser.add_argument('--real_dir', default='data/raw_calibration')
    parser.add_argument('--out_dir', default='data/hybrid_calibrated_v2')
    parser.add_argument('--real_files', nargs='+', default=list(FILE_MAP.keys()), choices=list(FILE_MAP.keys()))
    parser.add_argument('--real_csvs', nargs='+', default=None, help='Explicit real CSVs as name=path.')
    parser.add_argument('--augment_copies', type=int, default=50)
    parser.add_argument('--noise_std_mV', type=float, default=0.5)
    parser.add_argument('--gain_drift', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train = np.load(os.path.join(args.synthetic_dir, 'train.npz'))
    if args.real_csvs is not None:
        real_emf_v, real_target = load_real_csvs(args.real_csvs)
        real_source = f"csvs={len(parse_named_csvs(args.real_csvs))}"
    else:
        real_emf_v, real_target = load_real(args.real_dir, args.real_files)
        real_source = ' '.join(args.real_files)
    real_aug = augment_emf(real_emf_v, rng, args.augment_copies, args.noise_std_mV, args.gain_drift)
    real_targets_aug = np.tile(real_target, (args.augment_copies, 1)).astype(np.float32)

    hybrid_emf = np.vstack([train['emf'], real_aug]).astype(np.float32)
    hybrid_target = np.vstack([train['target'], real_targets_aug]).astype(np.float32)
    idx = rng.permutation(len(hybrid_target))
    np.savez(
        os.path.join(args.out_dir, 'train.npz'),
        emf=hybrid_emf[idx],
        target=hybrid_target[idx],
    )
    copy_split(args.synthetic_dir, args.out_dir, 'val')
    copy_split(args.synthetic_dir, args.out_dir, 'test')

    print(f"Saved hybrid train to {args.out_dir}: synthetic={len(train['target'])}, real_aug={len(real_targets_aug)}, real_source={real_source}")
    print(f"Copied synthetic val/test from {args.synthetic_dir}")


if __name__ == '__main__':
    main()

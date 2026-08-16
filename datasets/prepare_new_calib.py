import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import COLS
from physics.rotation_repr import pose_deg_to_target_v3


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    required = {'data_dir', 'workspace', 'families', 'folds'}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    return config


def validate_protocol(config):
    families = set(config['families'])
    for fold_name, roles in config['folds'].items():
        assigned = []
        for role in ('train', 'dev', 'test'):
            values = roles.get(role, [])
            if not values:
                raise ValueError(f"Fold {fold_name} has an empty {role} role")
            unknown = set(values) - families
            if unknown:
                raise ValueError(f"Fold {fold_name}/{role} contains unknown families: {sorted(unknown)}")
            assigned.extend(values)
        if len(assigned) != len(set(assigned)):
            raise ValueError(f"Fold {fold_name} leaks a family across roles: {assigned}")
        if set(assigned) != families:
            raise ValueError(f"Fold {fold_name} must assign every family exactly once")


def workspace_bounds(config):
    workspace = config['workspace']
    center = np.asarray(workspace['center_mm'], dtype=float)
    size = np.asarray(workspace['size_mm'], dtype=float)
    if center.shape != (3,) or size.shape != (3,) or np.any(size <= 0):
        raise ValueError("workspace center_mm and size_mm must be positive XYZ triples")
    return center - size / 2.0, center + size / 2.0


def read_source(path, family, family_meta, session, emf_scale):
    frame = pd.read_csv(path, header=None, names=COLS)
    numeric = frame.apply(pd.to_numeric, errors='coerce')
    valid_mask = np.isfinite(numeric[COLS].to_numpy(dtype=float)).all(axis=1)
    invalid_rows = np.flatnonzero(~valid_mask)
    clean = numeric.loc[valid_mask].reset_index(drop=True)
    poses = clean[['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']].to_numpy(dtype=np.float32)
    emf = clean[[f'EMF{i}' for i in range(1, 10)]].to_numpy(dtype=np.float32) * emf_scale
    source_rows = np.flatnonzero(valid_mask).astype(np.int64)
    count = len(clean)
    return {
        'emf': emf,
        'target': pose_deg_to_target_v3(poses),
        'pose_deg': poses,
        'family': np.full(count, family),
        'capsule': np.full(count, family_meta['capsule']),
        'rotation_regime': np.full(count, family_meta['rotation_regime']),
        'session': np.full(count, int(session), dtype=np.int64),
        'source_file': np.full(count, path.name),
        'source_row': source_rows,
    }, invalid_rows


def concatenate(items):
    keys = items[0].keys()
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in keys}


def select_families(data, families):
    mask = np.isin(data['family'], np.asarray(families))
    return {key: value[mask] for key, value in data.items()}


def write_npz(path, arrays):
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description="Prepare leakage-safe new calibration folds.")
    parser.add_argument('--config', default='configs/new_calib_v3.yaml')
    parser.add_argument('--out_dir', default='data/new_calib_v3')
    parser.add_argument('--folds', nargs='+', default=None, help="Fold names; defaults to all configured folds.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_protocol(config)
    data_dir = Path(config['data_dir'])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    emf_scale = 1e-3 if config.get('emf_unit', 'mV') == 'mV' else 1.0
    lower, upper = workspace_bounds(config)
    tolerance = float(config['workspace'].get('tolerance_mm', 0.0))

    source_arrays = []
    quality_rows = []
    invalid_records = []
    seen_files = set()
    for family, family_meta in config['families'].items():
        for file_meta in family_meta['files']:
            source_path = data_dir / file_meta['path']
            if source_path.name in seen_files:
                raise ValueError(f"Source file assigned more than once: {source_path.name}")
            seen_files.add(source_path.name)
            arrays, invalid_rows = read_source(
                source_path, family, family_meta, file_meta['session'], emf_scale,
            )
            positions = arrays['pose_deg'][:, :3]
            outside = np.any((positions < lower - tolerance) | (positions > upper + tolerance), axis=1)
            if np.any(outside):
                raise ValueError(
                    f"{source_path.name}: {int(outside.sum())} valid rows outside the configured "
                    "100 mm workspace plus tolerance"
                )
            source_arrays.append(arrays)
            for row in invalid_rows:
                invalid_records.append({
                    'source_file': source_path.name,
                    'source_row_zero_based': int(row),
                    'reason': 'missing_or_non_numeric_value',
                })
            quality_rows.append({
                'family': family,
                'capsule': family_meta['capsule'],
                'rotation_regime': family_meta['rotation_regime'],
                'session': int(file_meta['session']),
                'source_file': source_path.name,
                'valid_rows': int(len(arrays['emf'])),
                'invalid_rows': int(len(invalid_rows)),
                'emf_min_mV': float(arrays['emf'].min() * 1000.0),
                'emf_max_mV': float(arrays['emf'].max() * 1000.0),
            })

    all_data = concatenate(source_arrays)
    requested = args.folds or list(config['folds'])
    unknown = set(requested) - set(config['folds'])
    if unknown:
        raise ValueError(f"Unknown folds: {sorted(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(quality_rows).to_csv(out_dir / 'data_quality.csv', index=False)
    pd.DataFrame(invalid_records, columns=['source_file', 'source_row_zero_based', 'reason']).to_csv(
        out_dir / 'quarantined_rows.csv', index=False,
    )
    observed_pose = all_data['pose_deg']
    orientation_span = np.ptp(observed_pose[:, 3:], axis=0)
    warnings = []
    if np.any(orientation_span < 20.0):
        warnings.append(
            'Current acquisitions do not provide broad three-axis orientation coverage; '
            'do not claim full 6-DoF orientation generalization.'
        )
    hardware = config.get('hardware_validation', {})
    physics_ready = bool(hardware) and all(bool(value) for value in hardware.values())
    if not physics_ready:
        warnings.append(
            'Physics augmentation is gated off until channel order/polarity and pose-frame transform are verified.'
        )
    with open(out_dir / 'dataset_summary.json', 'w') as stream:
        json.dump({
            'valid_rows': int(len(all_data['emf'])),
            'quarantined_rows': int(len(invalid_records)),
            'workspace_nominal': config['workspace'],
            'observed_position_min_mm': observed_pose[:, :3].min(axis=0).tolist(),
            'observed_position_max_mm': observed_pose[:, :3].max(axis=0).tolist(),
            'observed_orientation_min_deg': observed_pose[:, 3:].min(axis=0).tolist(),
            'observed_orientation_max_deg': observed_pose[:, 3:].max(axis=0).tolist(),
            'observed_orientation_span_deg': orientation_span.tolist(),
            'hardware_validation': hardware,
            'physics_augmentation_allowed': physics_ready,
            'warnings': warnings,
        }, stream, indent=2)

    split_rows = []
    for fold_name in requested:
        fold_dir = out_dir / f'fold_{fold_name}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        roles = config['folds'][fold_name]
        for role, npz_name in (('train', 'train'), ('dev', 'val'), ('test', 'test')):
            arrays = select_families(all_data, roles[role])
            write_npz(fold_dir / f'{npz_name}.npz', arrays)
            split_rows.append({
                'fold': fold_name,
                'role': role,
                'families': ' '.join(roles[role]),
                'rows': int(len(arrays['emf'])),
                'source_files': ' '.join(sorted(set(arrays['source_file'].tolist()))),
            })
        with open(fold_dir / 'protocol.json', 'w') as stream:
            json.dump({
                'fold': fold_name,
                'roles': roles,
                'target_representation': 'position_mm+rotation_6d',
                'pose_convention': config.get('pose_convention'),
                'workspace': config['workspace'],
                'hardware_validation': config.get('hardware_validation', {}),
            }, stream, indent=2)

    split_frame = pd.DataFrame(split_rows)
    split_frame.to_csv(out_dir / 'split_manifest.csv', index=False)
    print(split_frame.to_string(index=False))
    print(f"Valid rows: {len(all_data['emf'])}; quarantined rows: {len(invalid_records)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Prepared leakage-safe folds in {out_dir}")


if __name__ == '__main__':
    main()

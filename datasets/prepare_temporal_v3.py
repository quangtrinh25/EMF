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
from datasets.prepare_new_calib import workspace_bounds
from physics.rotation_repr import pose_deg_to_target_v3


TIME_FACTORS = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}


def selector_keys(selectors):
    keys = set()
    for selector in selectors:
        family = selector['family']
        for session in selector['sessions']:
            key = (family, int(session))
            if key in keys:
                raise ValueError(f"Duplicate selector {key}")
            keys.add(key)
    return keys


def validate_temporal_protocol(config):
    protocol = config.get('temporal_protocol')
    if not protocol:
        raise ValueError("Config has no temporal_protocol")
    roles = {role: selector_keys(protocol[role]) for role in ('train', 'dev', 'test')}
    for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        overlap = roles[left] & roles[right]
        if overlap:
            raise ValueError(f"Temporal protocol leaks sessions across {left}/{right}: {sorted(overlap)}")

    available = {
        (family, int(file_meta['session']))
        for family, family_meta in config['families'].items()
        for file_meta in family_meta['files']
    }
    assigned = roles['train'] | roles['dev'] | roles['test']
    if assigned != available:
        raise ValueError(
            f"Temporal protocol must assign every acquisition once; missing={sorted(available-assigned)}, "
            f"unknown={sorted(assigned-available)}"
        )
    return roles


def read_source(path, family, family_meta, file_meta, temporal_config, emf_scale):
    has_header = bool(file_meta.get('header', False))
    timestamp_column = file_meta.get('timestamp_column', temporal_config.get('timestamp_column'))
    if has_header:
        frame = pd.read_csv(path)
        missing = set(COLS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing named columns {sorted(missing)}")
        numeric = frame[COLS].apply(pd.to_numeric, errors='coerce')
        timestamp_raw = None
        if timestamp_column is not None:
            if timestamp_column not in frame.columns:
                raise ValueError(f"{path}: timestamp column {timestamp_column!r} not found")
            timestamp_raw = pd.to_numeric(frame[timestamp_column], errors='coerce').to_numpy(dtype=float)
    else:
        if timestamp_column is not None:
            raise ValueError(f"{path}: timestamp input requires header: true and a named timestamp column")
        numeric = pd.read_csv(path, header=None, names=COLS).apply(pd.to_numeric, errors='coerce')
        timestamp_raw = None

    required = numeric[COLS].to_numpy(dtype=float)
    valid = np.isfinite(required).all(axis=1)
    if timestamp_raw is not None:
        valid &= np.isfinite(timestamp_raw)
    source_rows = np.flatnonzero(valid).astype(np.int64)
    clean = numeric.loc[valid].reset_index(drop=True)
    emf = clean[[f'EMF{i}' for i in range(1, 10)]].to_numpy(dtype=np.float32) * emf_scale
    poses = clean[['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']].to_numpy(dtype=np.float32)

    timestamps_s = None
    if timestamp_raw is not None:
        unit = file_meta.get('timestamp_unit', temporal_config.get('timestamp_unit', 's'))
        if unit not in TIME_FACTORS:
            raise ValueError(f"Unsupported timestamp unit {unit!r}")
        timestamps_s = timestamp_raw[valid] * TIME_FACTORS[unit]
        if np.any(np.diff(timestamps_s) <= 0):
            raise ValueError(f"{path}: timestamps must be strictly increasing")

    return {
        'family': family,
        'capsule': family_meta['capsule'],
        'rotation_regime': family_meta['rotation_regime'],
        'session': int(file_meta['session']),
        'source_file': path.name,
        'emf': emf,
        'poses': poses,
        'source_rows': source_rows,
        'timestamps_s': timestamps_s,
        'invalid_rows': np.flatnonzero(~valid).astype(np.int64),
        'trajectory_segment_rows': file_meta.get(
            'trajectory_segment_rows', family_meta.get('trajectory_segment_rows'),
        ),
    }


def contiguous_segments(source, position_jump_threshold_mm=None):
    rows = source['source_rows']
    discontinuity = np.diff(rows) > 1
    segment_rows = source.get('trajectory_segment_rows')
    if segment_rows is not None:
        segment_rows = int(segment_rows)
        if segment_rows < 1:
            raise ValueError("trajectory_segment_rows must be positive")
        discontinuity |= (rows[1:] // segment_rows) != (rows[:-1] // segment_rows)
    if position_jump_threshold_mm is not None:
        # Diagnostic/legacy option only. Production configs should use an
        # acquisition boundary above so target positions never define inputs.
        position_step = np.linalg.norm(np.diff(source['poses'][:, :3], axis=0), axis=1)
        discontinuity |= position_step > float(position_jump_threshold_mm)
    boundaries = np.flatnonzero(discontinuity) + 1
    return np.split(np.arange(len(rows)), boundaries)


def build_source_samples(
    source, window_size, horizon_steps, include_dt,
    position_jump_threshold_mm=None, feature_mode='raw_window',
):
    single_items = []
    temporal_items = []
    for segment_id, segment in enumerate(contiguous_segments(source, position_jump_threshold_mm)):
        if len(segment) <= horizon_steps:
            continue
        for local_input in range(len(segment) - horizon_steps):
            input_index = int(segment[local_input])
            target_index = int(segment[local_input + horizon_steps])
            first_local = max(0, local_input - window_size + 1)
            indices = segment[first_local:local_input + 1]
            if len(indices) < window_size:
                indices = np.concatenate([
                    np.repeat(indices[:1], window_size - len(indices)), indices,
                ])
            window_emf = source['emf'][indices]
            if feature_mode == 'raw_window':
                temporal_features = window_emf.reshape(-1)
            elif feature_mode == 'current_plus_log_ratio':
                current = window_emf[-1]
                eps_v = 1e-4  # 0.1 mV prevents unstable ratios near zero.
                log_ratio = np.log((window_emf[:-1] + eps_v) / (current[None, :] + eps_v))
                temporal_features = np.concatenate([
                    current, np.clip(log_ratio, -5.0, 5.0).reshape(-1),
                ])
            else:
                raise ValueError(f"Unknown temporal feature mode: {feature_mode}")
            dt = np.empty(0, dtype=np.float32)
            if include_dt:
                times = source['timestamps_s'][indices]
                dt = np.diff(times).astype(np.float32)
                temporal_features = np.concatenate([temporal_features, dt])

            target = pose_deg_to_target_v3(source['poses'][target_index:target_index + 1])[0]
            common = {
                'target': target,
                'pose_deg': source['poses'][target_index],
                'current_emf': source['emf'][input_index],
                'family': source['family'],
                'capsule': source['capsule'],
                'rotation_regime': source['rotation_regime'],
                'session': source['session'],
                'source_file': source['source_file'],
                'input_source_row': source['source_rows'][input_index],
                'target_source_row': source['source_rows'][target_index],
                'segment_id': segment_id,
                'target_offset_steps': horizon_steps,
                'target_offset_seconds': (
                    float(source['timestamps_s'][target_index] - source['timestamps_s'][input_index])
                    if source['timestamps_s'] is not None else np.nan
                ),
            }
            single_items.append({'emf': source['emf'][input_index], **common})
            temporal_items.append({'emf': temporal_features.astype(np.float32), **common})
    return single_items, temporal_items


def stack_samples(samples):
    if not samples:
        raise ValueError("No temporal samples were produced")
    keys = samples[0]
    arrays = {}
    for key in keys:
        value = samples[0][key]
        if isinstance(value, np.ndarray):
            arrays[key] = np.stack([sample[key] for sample in samples])
        else:
            arrays[key] = np.asarray([sample[key] for sample in samples])
    return arrays


def write_dataset(root, role_samples, protocol_metadata):
    root.mkdir(parents=True, exist_ok=True)
    for role, filename in (('train', 'train.npz'), ('dev', 'val.npz'), ('test', 'test.npz')):
        if not role_samples[role]:
            if (root / filename).exists():
                raise ValueError(f"Refusing to leave stale {root / filename} for an empty role")
            continue
        np.savez_compressed(root / filename, **stack_samples(role_samples[role]))
    with open(root / 'protocol.json', 'w') as stream:
        json.dump(protocol_metadata, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare causal temporal v3 datasets without crossing acquisitions.")
    parser.add_argument('--config', default='configs/new_calib_v3.yaml')
    parser.add_argument('--out_dir', default='data/new_calib_v3_temporal')
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--horizon_steps', type=int, default=None)
    parser.add_argument('--include_dt', choices=['auto', 'yes', 'no'], default=None)
    parser.add_argument(
        '--feature_mode', choices=['raw_window', 'current_plus_log_ratio'], default=None,
    )
    parser.add_argument(
        '--dev_con_rot_session', type=int, choices=[1, 2, 3], default=None,
        help='Override rotating development session for leave-one-session-out validation.',
    )
    parser.add_argument(
        '--deployment_train_all_non_test', action='store_true',
        help='After CV choices are frozen, place every non-test acquisition in training and emit no validation split.',
    )
    parser.add_argument(
        '--row_as_timestamp', action='store_true',
        help='Experimental: assign each source row a unit-step time and use only causal delta-row features.',
    )
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    roles = validate_temporal_protocol(config)
    temporal_config = config.get('temporal_features', {})
    window_size = int(args.window_size or temporal_config.get('window_size', 5))
    horizon_steps = int(args.horizon_steps if args.horizon_steps is not None else temporal_config.get('horizon_steps', 0))
    include_dt_mode = args.include_dt or temporal_config.get('include_dt', 'auto')
    feature_mode = args.feature_mode or temporal_config.get('feature_mode', 'raw_window')
    position_jump_threshold_mm = temporal_config.get('segment_position_jump_mm')
    if window_size < 1 or horizon_steps < 0:
        raise ValueError("window_size must be positive and horizon_steps non-negative")

    data_dir = Path(config['data_dir'])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    emf_scale = 1e-3 if config.get('emf_unit', 'mV') == 'mV' else 1.0
    lower, upper = workspace_bounds(config)
    tolerance = float(config['workspace'].get('tolerance_mm', 0.0))

    sources = []
    for family, family_meta in config['families'].items():
        for file_meta in family_meta['files']:
            source = read_source(
                data_dir / file_meta['path'], family, family_meta, file_meta,
                temporal_config, emf_scale,
            )
            outside = np.any(
                (source['poses'][:, :3] < lower - tolerance)
                | (source['poses'][:, :3] > upper + tolerance), axis=1,
            )
            if np.any(outside):
                raise ValueError(f"{source['source_file']}: positions outside configured workspace")
            sources.append(source)

    if args.row_as_timestamp:
        if any(source['timestamps_s'] is not None for source in sources):
            raise ValueError("--row_as_timestamp cannot replace real timestamps already present in the files")
        for source in sources:
            source['timestamps_s'] = source['source_rows'].astype(float)

    has_timestamps = all(source['timestamps_s'] is not None for source in sources)
    any_timestamps = any(source['timestamps_s'] is not None for source in sources)
    if any_timestamps and not has_timestamps:
        raise ValueError("Temporal protocol cannot mix timestamped and non-timestamped acquisitions")
    if include_dt_mode == 'yes' and not has_timestamps:
        raise ValueError("--include_dt yes requested, but current sources have no timestamps")
    include_dt = has_timestamps and include_dt_mode != 'no'

    if args.dev_con_rot_session is not None and args.deployment_train_all_non_test:
        raise ValueError("Choose either leave-one-session-out CV or deployment preparation, not both")
    if args.deployment_train_all_non_test:
        available = {(source['family'], source['session']) for source in sources}
        test_keys = {key for key in available if key[0] == 'cyl_rot'}
        roles = {
            'train': available - test_keys,
            'dev': set(),
            'test': test_keys,
        }
        protocol_descriptor = {
            'name': 'cyl_rot_locked_deployment_train_all_non_test',
            'train': sorted([{'family': family, 'session': session} for family, session in roles['train']], key=lambda x: (x['family'], x['session'])),
            'dev': [],
            'test': sorted([{'family': family, 'session': session} for family, session in roles['test']], key=lambda x: (x['family'], x['session'])),
            'dev_scope': 'none_hyperparameters_and_epochs_frozen_by_leave_one_session_out_cv',
            'test_scope': 'held_out_capsule_and_trajectory_family',
        }
    elif args.dev_con_rot_session is not None:
        available = {
            (source['family'], source['session']) for source in sources
        }
        test_keys = {key for key in available if key[0] == 'cyl_rot'}
        dev_keys = {('con_rot', int(args.dev_con_rot_session))}
        roles = {
            'train': available - test_keys - dev_keys,
            'dev': dev_keys,
            'test': test_keys,
        }
        protocol_descriptor = {
            'name': f"cyl_rot_locked_dev_con_rot_session_{args.dev_con_rot_session}",
            'train': sorted([{'family': family, 'session': session} for family, session in roles['train']], key=lambda x: (x['family'], x['session'])),
            'dev': sorted([{'family': family, 'session': session} for family, session in roles['dev']], key=lambda x: (x['family'], x['session'])),
            'test': sorted([{'family': family, 'session': session} for family, session in roles['test']], key=lambda x: (x['family'], x['session'])),
            'dev_scope': 'leave_one_con_rot_acquisition_session_out',
            'test_scope': 'held_out_capsule_and_trajectory_family',
        }
    else:
        protocol_descriptor = config['temporal_protocol']

    role_by_key = {key: role for role, keys in roles.items() for key in keys}
    single_samples = {role: [] for role in roles}
    temporal_samples = {role: [] for role in roles}
    manifest_rows = []
    quarantine_rows = []
    for source in sources:
        role = role_by_key[(source['family'], source['session'])]
        single, temporal = build_source_samples(
            source, window_size, horizon_steps, include_dt,
            position_jump_threshold_mm=position_jump_threshold_mm,
            feature_mode=feature_mode,
        )
        single_samples[role].extend(single)
        temporal_samples[role].extend(temporal)
        manifest_rows.append({
            'role': role,
            'family': source['family'],
            'session': source['session'],
            'source_file': source['source_file'],
            'valid_rows': len(source['emf']),
            'invalid_rows': len(source['invalid_rows']),
            'single_samples': len(single),
            'temporal_samples': len(temporal),
            'has_timestamp': source['timestamps_s'] is not None,
            'segments': len(contiguous_segments(source, position_jump_threshold_mm)),
        })
        for row in source['invalid_rows']:
            quarantine_rows.append({
                'source_file': source['source_file'],
                'source_row_zero_based': int(row),
                'reason': 'missing_or_non_numeric_value',
            })

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(output / 'temporal_split_manifest.csv', index=False)
    pd.DataFrame(quarantine_rows, columns=['source_file', 'source_row_zero_based', 'reason']).to_csv(
        output / 'quarantined_rows.csv', index=False,
    )
    common_metadata = {
        'schema_version': 3,
        'protocol_name': protocol_descriptor['name'],
        'roles': protocol_descriptor,
        'workspace': config['workspace'],
        'hardware_validation': config.get('hardware_validation', {}),
        'window_size': window_size,
        'horizon_steps': horizon_steps,
        'has_timestamps': has_timestamps,
        'include_dt': include_dt,
        'timestamp_feature': (
            'delta_row_steps_only'
            if include_dt and args.row_as_timestamp
            else ('delta_t_seconds_only' if include_dt else None)
        ),
        'timestamp_source': 'unit_step_source_row_experiment' if args.row_as_timestamp else 'hardware_or_none',
        'timestamp_unit': 'row_step' if args.row_as_timestamp else temporal_config.get('timestamp_unit'),
        'absolute_timestamp_used': False,
        'segment_position_jump_mm': position_jump_threshold_mm,
        'segment_source': 'source_row_acquisition_period_and_invalid_gaps',
        'temporal_feature_mode': feature_mode,
        'test_opened': False,
    }
    single_metadata = {
        **common_metadata,
        'feature_layout': 'current_emf_9',
        'emf_feature_count': 9,
        'dt_feature_count': 0,
        'temporal_window': None,
    }
    temporal_metadata = {
        **common_metadata,
        'feature_layout': (
            'oldest_to_current_emf_then_delta_t'
            if feature_mode == 'raw_window'
            else 'current_emf_then_past_to_current_log_ratios_then_delta_t'
        ),
        'emf_feature_count': window_size * 9 if feature_mode == 'raw_window' else 9,
        'dt_feature_count': window_size - 1 if include_dt else 0,
        'temporal_window': window_size,
    }
    write_dataset(output / 'single', single_samples, single_metadata)
    write_dataset(output / 'temporal', temporal_samples, temporal_metadata)
    counts = pd.DataFrame(manifest_rows).groupby('role')[['single_samples', 'temporal_samples']].sum()
    print(counts.to_string())
    print(
        f"window={window_size}; horizon_steps={horizon_steps}; "
        f"include_dt={include_dt}; feature_mode={feature_mode}"
    )
    if not has_timestamps:
        print("Timestamp status: absent; causal row order is used without a time feature.")
    print("Test remains locked; preparation does not evaluate it.")


if __name__ == '__main__':
    main()

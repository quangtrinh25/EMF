import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import COLS, FILE_MAP, default_rx_params, extract_emf_and_pose
from physics.channel_correction import apply_channel_correction
from physics.forward_model import forward_model


def load_raw_frames(data_dir):
    frames = {}
    for name, filename in FILE_MAP.items():
        frames[name] = pd.read_csv(os.path.join(data_dir, filename), header=None, names=COLS)
    return frames


def parse_named_csvs(items):
    parsed = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"Expected name=path CSV argument, got: {item}")
        name, path = item.split('=', 1)
        if not name or not path:
            raise ValueError(f"Expected name=path CSV argument, got: {item}")
        parsed[name] = path
    return parsed


def load_csv_frames(items):
    frames = {}
    for name, path in parse_named_csvs(items).items():
        frame = pd.read_csv(path, header=None, names=COLS).apply(pd.to_numeric, errors='coerce')
        invalid = ~np.isfinite(frame[COLS].to_numpy(dtype=float)).all(axis=1)
        if invalid.any():
            rows = np.flatnonzero(invalid).tolist()
            raise ValueError(
                f"{path} contains invalid/missing values at zero-based rows {rows[:10]}. "
                "Clean/quarantine them before physical calibration."
            )
        frames[name] = frame
    return frames


def frame_to_arrays(df):
    return extract_emf_and_pose(df)


def unit_or_default(vec, default):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.asarray(default, dtype=float)
    return vec / norm


def unpack_theta(theta):
    theta = np.asarray(theta, dtype=float)
    positions = [
        theta[0:3].tolist(),
        theta[6:9].tolist(),
        theta[12:15].tolist(),
    ]
    orientations = [
        unit_or_default(theta[3:6], [1.0, 0.0, 0.0]).tolist(),
        unit_or_default(theta[9:12], [0.0, 1.0, 0.0]).tolist(),
        unit_or_default(theta[15:18], [0.0, 0.0, 1.0]).tolist(),
    ]
    moments = np.exp(theta[18:21]).tolist()
    gains = np.exp(theta[21:30])
    biases = theta[30:39]
    tx = {
        'positions': positions,
        'orientations': orientations,
        'moments': moments,
        'frequencies': [4000, 4500, 5000],
        'channel_correction': {
            'gains': gains.tolist(),
            'biases_v': biases.tolist(),
        },
    }
    return tx


def predict_corrected(theta, poses, rx_params):
    tx = unpack_theta(theta)
    emf = forward_model(poses, tx, rx_params)
    return apply_channel_correction(emf, tx)


def residual(theta, emf_measured_v, poses, rx_params, channel_scale):
    pred = predict_corrected(theta, poses, rx_params)
    return ((emf_measured_v - pred) / channel_scale).ravel()


def theta_from_tx(tx):
    positions = tx.get('positions')
    orientations = tx.get('orientations')
    if isinstance(positions, dict):
        positions = [positions[f'tx{i+1}'] for i in range(3)]
    if isinstance(orientations, dict):
        orientations = [orientations[f'tx{i+1}'] for i in range(3)]
    moments = np.asarray(tx.get('moments', [1.0, 1.0, 1.0]), dtype=float)
    correction = tx.get('channel_correction', {})
    gains = np.asarray(correction.get('gains', np.ones(9)), dtype=float)
    biases = np.asarray(correction.get('biases_v', np.zeros(9)), dtype=float)

    theta = []
    for pos, ori in zip(positions, orientations):
        theta.extend(np.asarray(pos, dtype=float).tolist())
        theta.extend(unit_or_default(ori, [1.0, 0.0, 0.0]).tolist())
    theta.extend(np.log(np.clip(moments, 1e-4, None)).tolist())
    theta.extend(np.log(np.clip(gains, 1e-4, None)).tolist())
    theta.extend(biases.tolist())
    return np.asarray(theta, dtype=float)


def base_theta(raw_poses):
    center = np.median(raw_poses[:, :3], axis=0)
    tx = {
        'positions': [
            [center[0], center[1] - 180.0, 10.0],
            [center[0], center[1], 10.0],
            [center[0], center[1] - 90.0, 10.0],
        ],
        'orientations': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'moments': [5.0, 5.0, 5.0],
        'channel_correction': {'gains': [1.0] * 9, 'biases_v': [0.0] * 9},
    }
    return theta_from_tx(tx)


def load_existing_theta(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return theta_from_tx(cfg['tx'])


def randomize_theta(theta, rng, first=False):
    if first:
        return theta.copy()
    out = theta.copy()
    for start in [0, 6, 12]:
        out[start:start + 3] += rng.normal(0.0, [35.0, 50.0, 30.0])
        out[start + 3:start + 6] += rng.normal(0.0, 0.45, 3)
    out[18:21] += rng.normal(0.0, 0.7, 3)
    out[21:30] += rng.normal(0.0, 0.35, 9)
    out[30:39] += rng.normal(0.0, 0.005, 9)
    return np.clip(out, lower_bounds(), upper_bounds())


def lower_bounds():
    pos_low = [-150.0, 80.0, -120.0]
    ori_low = [-3.0, -3.0, -3.0]
    bounds = []
    for _ in range(3):
        bounds.extend(pos_low)
        bounds.extend(ori_low)
    bounds.extend(np.log([0.01, 0.01, 0.01]).tolist())
    bounds.extend(np.log([0.05] * 9).tolist())
    bounds.extend([-0.08] * 9)
    return np.asarray(bounds, dtype=float)


def upper_bounds():
    pos_high = [220.0, 520.0, 220.0]
    ori_high = [3.0, 3.0, 3.0]
    bounds = []
    for _ in range(3):
        bounds.extend(pos_high)
        bounds.extend(ori_high)
    bounds.extend(np.log([200.0, 200.0, 200.0]).tolist())
    bounds.extend(np.log([20.0] * 9).tolist())
    bounds.extend([0.08] * 9)
    return np.asarray(bounds, dtype=float)


def rmse_by_channel(emf_measured_v, pred_v):
    err = emf_measured_v - pred_v
    rows = []
    for i in range(9):
        measured = emf_measured_v[:, i]
        pred = pred_v[:, i]
        rmse = float(np.sqrt(np.mean(err[:, i] ** 2)) * 1000.0)
        mae = float(np.mean(np.abs(err[:, i])) * 1000.0)
        mean_mv = float(np.mean(measured) * 1000.0)
        corr = float(np.corrcoef(measured, pred)[0, 1]) if np.std(pred) > 0 and np.std(measured) > 0 else np.nan
        rows.append({
            'channel': f'EMF {i + 1}',
            'mean_measured_mV': mean_mv,
            'rmse_mV': rmse,
            'mae_mV': mae,
            'rmse_over_mean_pct': float(rmse / (mean_mv + 1e-12) * 100.0),
            'correlation': corr,
        })
    return pd.DataFrame(rows)


def write_config(tx, rx_params, raw_poses, out_config):
    mins = raw_poses.min(axis=0)
    maxs = raw_poses.max(axis=0)
    workspace = {
        'x_mm': [float(mins[0] - 15.0), float(maxs[0] + 15.0)],
        'y_mm': [float(mins[1] - 15.0), float(maxs[1] + 15.0)],
        'z_mm': [float(mins[2] - 15.0), float(maxs[2] + 15.0)],
        'orientation_deg': {
            'roll': [float(min(-3.0, mins[3] - 2.0)), float(max(3.0, maxs[3] + 2.0))],
            'pitch': [float(min(-8.0, mins[4] - 2.0)), float(max(8.0, maxs[4] + 2.0))],
            'yaw': [float(min(-3.0, mins[5] - 2.0)), float(max(3.0, maxs[5] + 2.0))],
        },
        'n_spatial': 40,
        'n_orientation': 10,
    }
    config = {'tx': tx, 'rx': rx_params, 'workspace': workspace}
    os.makedirs(os.path.dirname(out_config), exist_ok=True)
    with open(out_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw_calibration')
    parser.add_argument('--out_config', default='configs/physics_calibrated_v2.yaml')
    parser.add_argument('--out_dir', default='results/calibration_v2')
    parser.add_argument('--starts', type=int, default=12)
    parser.add_argument('--max_nfev', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fit_mode', choices=['all', 'no_rot'], default='all')
    parser.add_argument('--fit_files', nargs='+', default=None, choices=list(FILE_MAP.keys()))
    parser.add_argument('--report_files', nargs='+', default=None, choices=list(FILE_MAP.keys()))
    parser.add_argument('--fit_csvs', nargs='+', default=None, help='Explicit calibration CSVs as name=path.')
    parser.add_argument('--report_csvs', nargs='+', default=None, help='Explicit report CSVs as name=path.')
    parser.add_argument('--workspace_from', choices=['all', 'fit'], default='all')
    parser.add_argument(
        '--warm_start_config', default=None,
        help='Optional explicit warm start. Defaults to none to avoid inheriting a contaminated fit.',
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.fit_csvs is not None:
        fit_frames = load_csv_frames(args.fit_csvs)
        frames = fit_frames.copy()
        fit_names = list(fit_frames.keys())
        if args.report_csvs is not None:
            report_frames = load_csv_frames(args.report_csvs)
        else:
            report_frames = fit_frames
    else:
        frames = load_raw_frames(args.data_dir)
        if args.fit_files is not None:
            fit_names = args.fit_files
        elif args.fit_mode == 'no_rot':
            fit_names = ['set10_cyl_no_rot']
        else:
            # Safe default: never consume the historical evaluation files.
            fit_names = ['set10_cyl_no_rot', 'set11_con_spi_rot']
        if args.report_csvs is not None:
            report_frames = load_csv_frames(args.report_csvs)
        else:
            report_frames = frames
    if args.report_csvs is not None:
        report_names = list(report_frames.keys())
    else:
        report_names = args.report_files if args.report_files is not None else list(fit_names)
    fit_df = pd.concat([frames[name] for name in fit_names], ignore_index=True)
    all_df = pd.concat([report_frames[name] for name in report_names], ignore_index=True)

    emf_fit_v, poses_fit = frame_to_arrays(fit_df)
    emf_all_v, poses_all = frame_to_arrays(all_df)
    rx_params = default_rx_params()
    channel_scale = np.maximum(np.std(emf_fit_v, axis=0), 0.004)

    warm = load_existing_theta(args.warm_start_config)
    base = warm if warm is not None else base_theta(poses_fit)

    low = lower_bounds()
    high = upper_bounds()
    best = None
    rows = []
    for idx in range(args.starts):
        x0 = randomize_theta(base, rng, first=(idx == 0))
        result = least_squares(
            residual,
            x0=x0,
            bounds=(low, high),
            args=(emf_fit_v, poses_fit, rx_params, channel_scale),
            method='trf',
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=args.max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            verbose=0,
        )
        pred_fit = predict_corrected(result.x, poses_fit, rx_params)
        fit_rmse_mv = float(np.sqrt(np.mean((emf_fit_v - pred_fit) ** 2)) * 1000.0)
        rows.append({
            'start': idx,
            'success': bool(result.success),
            'cost': float(result.cost),
            'nfev': int(result.nfev),
            'fit_rmse_mV': fit_rmse_mv,
        })
        print(f"start {idx:02d}: success={result.success} cost={result.cost:.4g} fit_rmse={fit_rmse_mv:.3f} mV")
        if best is None or fit_rmse_mv < best[0]:
            best = (fit_rmse_mv, result)

    best_rmse, best_result = best
    tx = unpack_theta(best_result.x)
    pred_all = predict_corrected(best_result.x, poses_all, rx_params)
    all_rmse_mv = float(np.sqrt(np.mean((emf_all_v - pred_all) ** 2)) * 1000.0)

    os.makedirs(args.out_dir, exist_ok=True)
    workspace_poses = poses_fit if args.workspace_from == 'fit' else poses_all
    write_config(tx, rx_params, workspace_poses, args.out_config)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'calibration_v2_starts.csv'), index=False)
    channel_df = rmse_by_channel(emf_all_v, pred_all)
    channel_df.to_csv(os.path.join(args.out_dir, 'channel_fit_metrics.csv'), index=False)
    pd.DataFrame([{
        'fit_mode': args.fit_mode,
        'fit_files': ' '.join(fit_names),
        'report_files': ' '.join(report_names),
        'workspace_from': args.workspace_from,
        'num_fit_samples': len(fit_df),
        'num_all_samples': len(all_df),
        'best_fit_rmse_mV': best_rmse,
        'all_raw_rmse_mV': all_rmse_mv,
        'best_cost': float(best_result.cost),
        'best_nfev': int(best_result.nfev),
        'out_config': args.out_config,
        'split_policy': 'explicit_fit_only_no_implicit_evaluation_files',
    }]).to_csv(os.path.join(args.out_dir, 'calibration_v2_metadata.csv'), index=False)

    print(f"Saved v2 calibrated config to {args.out_config}")
    print(f"Best fit RMSE: {best_rmse:.3f} mV")
    print(f"All raw RMSE:  {all_rmse_mv:.3f} mV")
    print(channel_df.to_string(index=False))


if __name__ == '__main__':
    main()

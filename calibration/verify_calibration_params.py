import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import (
    COLS,
    FILE_MAP,
    build_calibration_report,
    default_rx_params,
    ensure_results_dir,
    extract_emf_and_pose,
    load_measurement_frame,
    write_calibration_report,
)
from physics.channel_correction import apply_channel_correction
from physics.forward_model import forward_model


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def corrected_prediction(poses, tx_params, rx_params):
    pred = forward_model(poses, tx_params, rx_params)
    if 'channel_correction' in tx_params:
        pred = apply_channel_correction(pred, tx_params)
    return pred


def channel_metrics(data_dir, tx_params, rx_params):
    frames = []
    for name, filename in FILE_MAP.items():
        df = load_measurement_frame(data_dir, filename)
        df = df.copy()
        df['dataset'] = name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    emf_v, poses = extract_emf_and_pose(all_df)
    pred_v = corrected_prediction(poses, tx_params, rx_params)
    err_v = emf_v - pred_v

    rows = []
    for idx in range(9):
        measured = emf_v[:, idx]
        pred = pred_v[:, idx]
        rmse_mv = np.sqrt(np.mean(err_v[:, idx] ** 2)) * 1000.0
        mae_mv = np.mean(np.abs(err_v[:, idx])) * 1000.0
        mean_mv = np.mean(measured) * 1000.0
        corr = np.corrcoef(measured, pred)[0, 1] if np.std(measured) > 0 and np.std(pred) > 0 else np.nan
        rows.append({
            'channel': f'EMF {idx + 1}',
            'mean_measured_mV': mean_mv,
            'rmse_mV': rmse_mv,
            'mae_mV': mae_mv,
            'rmse_over_mean_pct': rmse_mv / (mean_mv + 1e-12) * 100.0,
            'correlation': corr,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/physics_calibrated_v2.yaml')
    parser.add_argument('--data_dir', default='data/raw_calibration')
    parser.add_argument('--out_dir', default='results/calibration_own_params')
    args = parser.parse_args()

    config = load_config(args.config)
    tx_params = config['tx']
    rx_params = config.get('rx', default_rx_params())
    out_dir = ensure_results_dir(args.out_dir)

    summary_df, details, nominal_tx, scale_factor = build_calibration_report(
        data_dir=args.data_dir,
        calibrated_tx=tx_params,
        rx_params=rx_params,
    )
    summary_path, detail_paths = write_calibration_report(summary_df, details, out_dir=out_dir)
    channel_df = channel_metrics(args.data_dir, tx_params, rx_params)
    channel_path = os.path.join(out_dir, 'channel_fit_metrics.csv')
    channel_df.to_csv(channel_path, index=False)

    combined = summary_df[summary_df['dataset'] == 'all_files'].iloc[0]
    metadata = pd.DataFrame([{
        'config': args.config,
        'data_dir': args.data_dir,
        'nominal_scale_factor': scale_factor,
        'combined_nominal_rmse_mV': combined['rmse_nominal_mV'],
        'combined_calibrated_rmse_mV': combined['rmse_calibrated_mV'],
        'combined_reduction_pct': combined['reduction_pct'],
        'has_channel_correction': 'channel_correction' in tx_params,
    }])
    metadata_path = os.path.join(out_dir, 'calibration_param_metadata.csv')
    metadata.to_csv(metadata_path, index=False)

    print(f"Config: {args.config}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved channel metrics to {channel_path}")
    print(f"Saved metadata to {metadata_path}")
    print(summary_df.to_string(index=False))
    print(channel_df.to_string(index=False))


if __name__ == '__main__':
    main()

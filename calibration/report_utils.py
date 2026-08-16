import os
import sys

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.forward_model import forward_model
from physics.channel_correction import apply_channel_correction


COLS = [
    'EMF1', 'EMF2', 'EMF3', 'EMF4', 'EMF5', 'EMF6', 'EMF7', 'EMF8', 'EMF9',
    'X', 'Y', 'Z', 'roll', 'pitch', 'yaw',
]

FILE_MAP = {
    'set10_cyl_no_rot': 'set10_2304_cyl_no_rot.csv',
    'set11_con_spi_rot': 'Set11_con_spi_rot_2.csv',
    'set12_cyl_spi_rot': 'Set12_cyl_spi_rot_2.csv',
    'set13_con_spi_no_rot': 'Set13_con_spi_no_rot.csv',
}

RESULTS_DIR = 'results/calibration'


def nominal_tx_params():
    return {
        'positions': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        'orientations': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'moments': [1.0, 1.0, 1.0],
        'frequencies': [4000, 4500, 5000],
    }


def default_rx_params():
    return {
        'areas_mm2': [9.4248, 70.0, 100.0],
        'turns': [250, 200, 200],
    }


def ensure_results_dir(out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_calibrated_tx(calib_yaml='configs/physics_calibrated.yaml'):
    with open(calib_yaml, 'r') as f:
        calib_config = yaml.safe_load(f)
    return calib_config['tx']


def load_measurement_frame(data_dir, filename):
    return pd.read_csv(os.path.join(data_dir, filename), header=None, names=COLS)


def extract_emf_and_pose(df):
    emf_measured_v = df[[f'EMF{i}' for i in range(1, 10)]].values * 1e-3
    poses = df[['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']].values
    return emf_measured_v, poses


def scaled_nominal_tx(reference_emf_v, reference_poses, rx_params):
    nominal_tx = nominal_tx_params()
    emf_nominal_unscaled = forward_model(reference_poses, nominal_tx, rx_params)
    scale_factor = np.mean(reference_emf_v) / (np.mean(emf_nominal_unscaled) + 1e-12)
    nominal_tx['moments'] = [scale_factor, scale_factor, scale_factor]
    return nominal_tx, scale_factor


def file_metrics(name, filename, data_dir, nominal_tx, calibrated_tx, rx_params):
    df = load_measurement_frame(data_dir, filename)
    emf_measured_v, poses = extract_emf_and_pose(df)

    emf_nominal_v = forward_model(poses, nominal_tx, rx_params)
    emf_calibrated_v = forward_model(poses, calibrated_tx, rx_params)
    if 'channel_correction' in calibrated_tx:
        emf_calibrated_v = apply_channel_correction(emf_calibrated_v, calibrated_tx)

    rmse_nominal_mv = np.sqrt(np.mean((emf_measured_v - emf_nominal_v) ** 2)) * 1000.0
    rmse_calibrated_mv = np.sqrt(np.mean((emf_measured_v - emf_calibrated_v) ** 2)) * 1000.0
    reduction_pct = (1.0 - rmse_calibrated_mv / (rmse_nominal_mv + 1e-12)) * 100.0

    point_rmse_nominal_mv = np.sqrt(np.mean((emf_measured_v - emf_nominal_v) ** 2, axis=1)) * 1000.0
    point_rmse_calibrated_mv = np.sqrt(np.mean((emf_measured_v - emf_calibrated_v) ** 2, axis=1)) * 1000.0

    detail_df = pd.DataFrame({
        'sample_index': np.arange(len(df), dtype=int),
        'X_mm': poses[:, 0],
        'Y_mm': poses[:, 1],
        'Z_mm': poses[:, 2],
        'roll_deg': poses[:, 3],
        'pitch_deg': poses[:, 4],
        'yaw_deg': poses[:, 5],
        'rmse_nominal_mV': point_rmse_nominal_mv,
        'rmse_calibrated_mV': point_rmse_calibrated_mv,
        'improvement_mV': point_rmse_nominal_mv - point_rmse_calibrated_mv,
    })

    summary_row = {
        'dataset': name,
        'filename': filename,
        'num_samples': len(df),
        'rmse_nominal_mV': rmse_nominal_mv,
        'rmse_calibrated_mV': rmse_calibrated_mv,
        'reduction_pct': reduction_pct,
    }
    return summary_row, detail_df


def build_calibration_report(
    data_dir='data/raw_calibration',
    calibrated_tx=None,
    rx_params=None,
):
    if rx_params is None:
        rx_params = default_rx_params()
    if calibrated_tx is None:
        calibrated_tx = load_calibrated_tx()

    ref_df = load_measurement_frame(data_dir, FILE_MAP['set11_con_spi_rot'])
    ref_emf_v, ref_poses = extract_emf_and_pose(ref_df)
    nominal_tx, scale_factor = scaled_nominal_tx(ref_emf_v, ref_poses, rx_params)

    summaries = []
    details = {}
    for name, filename in FILE_MAP.items():
        summary_row, detail_df = file_metrics(
            name=name,
            filename=filename,
            data_dir=data_dir,
            nominal_tx=nominal_tx,
            calibrated_tx=calibrated_tx,
            rx_params=rx_params,
        )
        summaries.append(summary_row)
        details[name] = detail_df

    summary_df = pd.DataFrame(summaries)
    total_samples = float(summary_df['num_samples'].sum())
    pooled_nominal = float(np.sqrt(np.sum(
        summary_df['num_samples'] * summary_df['rmse_nominal_mV'] ** 2
    ) / total_samples))
    pooled_calibrated = float(np.sqrt(np.sum(
        summary_df['num_samples'] * summary_df['rmse_calibrated_mV'] ** 2
    ) / total_samples))
    totals = {
        'dataset': 'all_files',
        'filename': 'combined',
        'num_samples': int(total_samples),
        'rmse_nominal_mV': pooled_nominal,
        'rmse_calibrated_mV': pooled_calibrated,
        'reduction_pct': (1.0 - pooled_calibrated / (pooled_nominal + 1e-12)) * 100.0,
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
    return summary_df, details, nominal_tx, scale_factor


def write_calibration_report(summary_df, details, out_dir=RESULTS_DIR):
    ensure_results_dir(out_dir)
    summary_path = os.path.join(out_dir, 'calibration_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    detail_paths = {}
    for name, detail_df in details.items():
        detail_path = os.path.join(out_dir, f'{name}_before_after.csv')
        detail_df.to_csv(detail_path, index=False)
        detail_paths[name] = detail_path
    return summary_path, detail_paths

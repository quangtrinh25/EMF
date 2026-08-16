import os
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import yaml
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.forward_model import forward_model
from calibration.report_utils import (
    COLS,
    RESULTS_DIR,
    build_calibration_report,
    default_rx_params,
    ensure_results_dir,
    write_calibration_report,
)

def unpack_params(theta):
    positions    = [theta[0:3].tolist(),  theta[6:9].tolist(),  theta[12:15].tolist()]
    orientations = [theta[3:6].tolist(),  theta[9:12].tolist(), theta[15:18].tolist()]
    moments      = [float(abs(theta[18])), float(abs(theta[19])), float(abs(theta[20]))]
    return {
        'positions':    positions,
        'orientations': orientations,
        'moments':      moments,
        'frequencies':  [4000, 4500, 5000],
    }
def initial_guess():
    cx, cy, cz_tx = 42.0, 360.0, 0.0
    return np.array([
        cx, cy, cz_tx,   1.0, 0.0, 0.0,   # TX1
        cx, cy, cz_tx,   0.0, 1.0, 0.0,   # TX2
        cx, cy, cz_tx,   0.0, 0.0, 1.0,   # TX3
        1.0, 1.0, 1.0                   # moments (scale to be fit)
    ], dtype=np.float64)

def calibration_residual(theta, emf_measured_V, poses, rx_params):
    tx_params = unpack_params(theta)
    emf_pred = forward_model(poses, tx_params, rx_params)
    return (emf_measured_V - emf_pred).ravel()

def main():
    data_dir = 'data/raw_calibration'
    df_fit = pd.concat([
        pd.read_csv(os.path.join(data_dir, 'set10_2304_cyl_no_rot.csv'), header=None, names=COLS),
        pd.read_csv(os.path.join(data_dir, 'Set13_con_spi_no_rot.csv'), header=None, names=COLS)
    ], ignore_index=True)

    df_val = pd.concat([
        pd.read_csv(os.path.join(data_dir, 'Set11_con_spi_rot_2.csv'), header=None, names=COLS),
        pd.read_csv(os.path.join(data_dir, 'Set12_cyl_spi_rot_2.csv'), header=None, names=COLS)
    ], ignore_index=True)

    print(f"Loaded {len(df_fit)} calibration samples and {len(df_val)} validation samples.")

    # EMF is in mV in CSV files -> convert to V for physics model
    emf_fit_V = df_fit[['EMF1','EMF2','EMF3','EMF4','EMF5','EMF6','EMF7','EMF8','EMF9']].values * 1e-3
    poses_fit = df_fit[['X','Y','Z','roll','pitch','yaw']].values

    emf_val_V = df_val[['EMF1','EMF2','EMF3','EMF4','EMF5','EMF6','EMF7','EMF8','EMF9']].values * 1e-3
    poses_val = df_val[['X','Y','Z','roll','pitch','yaw']].values

    rx_params = default_rx_params()

    # Run least squares calibration
    print("Running least_squares optimization to fit physical TX parameters...")
    result = least_squares(
        calibration_residual,
        x0=initial_guess(),
        args=(emf_fit_V, poses_fit, rx_params),
        method='trf',
        max_nfev=10000,
        ftol=1e-10, xtol=1e-10,
        verbose=1
    )

    print(f"Calibration converged: {result.success}")
    print(f"Final cost: {result.cost:.6e}")
    calibrated_params = unpack_params(result.x)

    # Save to yaml
    calibrated_config = {
        'tx': {
            'positions':    {f'tx{i+1}': calibrated_params['positions'][i] for i in range(3)},
            'orientations': {f'tx{i+1}': calibrated_params['orientations'][i] for i in range(3)},
            'moments':      calibrated_params['moments'],
            'frequencies':  [4000, 4500, 5000],
        },
        'rx': rx_params,
        'workspace': {
            'x_mm': [-50, 150],
            'y_mm': [250, 450],
            'z_mm': [100, 350],
            'n_spatial': 40,
            'n_orientation': 10
        }
    }

    os.makedirs('configs', exist_ok=True)
    with open('configs/physics_calibrated.yaml', 'w') as f:
        yaml.dump(calibrated_config, f, default_flow_style=False)
    print("Calibrated parameters saved to configs/physics_calibrated.yaml.")

    # Validation check
    nominal_tx = {
        'positions':    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        'orientations': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'moments':      [1.0, 1.0, 1.0],
        'frequencies':  [4000, 4500, 5000]
    }
    
    # Scale nominal moments to make comparison fair
    emf_nom_unscaled = forward_model(poses_val, nominal_tx, rx_params)
    scale_factor = np.mean(emf_val_V) / (np.mean(emf_nom_unscaled) + 1e-12)
    nominal_tx['moments'] = [scale_factor, scale_factor, scale_factor]

    emf_nom_pred = forward_model(poses_val, nominal_tx, rx_params)
    emf_cal_pred = forward_model(poses_val, calibrated_params, rx_params)

    err_nom = np.sqrt(np.mean((emf_val_V - emf_nom_pred)**2))
    err_cal = np.sqrt(np.mean((emf_val_V - emf_cal_pred)**2))

    print(f"Validation EMF RMSE before calibration (scaled nominal): {err_nom*1000:.3f} mV")
    print(f"Validation EMF RMSE after calibration: {err_cal*1000:.3f} mV")
    print(f"Error reduction: {(1.0 - err_cal/err_nom)*100:.2f}%")

    results_dir = ensure_results_dir(RESULTS_DIR)
    summary_df, details, _, scale_factor = build_calibration_report(
        data_dir=data_dir,
        calibrated_tx=calibrated_params,
        rx_params=rx_params,
    )
    summary_path, _ = write_calibration_report(summary_df, details, out_dir=results_dir)

    metadata = pd.DataFrame([{
        'optimizer_success': bool(result.success),
        'optimizer_cost': float(result.cost),
        'optimizer_nfev': int(result.nfev),
        'nominal_scale_factor': float(scale_factor),
        'validation_rmse_before_mV': float(err_nom * 1000.0),
        'validation_rmse_after_mV': float(err_cal * 1000.0),
        'validation_reduction_pct': float((1.0 - err_cal / (err_nom + 1e-12)) * 100.0),
    }])
    metadata_path = os.path.join(results_dir, 'calibration_run_metadata.csv')
    metadata.to_csv(metadata_path, index=False)

    sns.set_theme(style="whitegrid")
    plot_rows = [name for name in details.keys()]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#C44747', '#2F7DBA']

    for idx, name in enumerate(plot_rows):
        ax = axes[idx]
        detail_df = details[name]
        box = ax.boxplot(
            [detail_df['rmse_nominal_mV'], detail_df['rmse_calibrated_mV']],
            patch_artist=True,
            widths=0.45,
            boxprops=dict(color='black', alpha=0.85),
            medianprops=dict(color='black', linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.25, markersize=3),
        )
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        mean_nom = detail_df['rmse_nominal_mV'].mean()
        mean_cal = detail_df['rmse_calibrated_mV'].mean()
        reduction = (1.0 - mean_cal / (mean_nom + 1e-12)) * 100.0
        ax.set_xticklabels(['Before calib', 'After calib'])
        ax.set_ylabel('Per-pose EMF RMSE (mV)')
        ax.set_title(name.replace('_', ' ').upper())
        ax.text(
            0.04, 0.95,
            f'Before: {mean_nom:.2f} mV\nAfter: {mean_cal:.2f} mV\nReduction: {reduction:.1f}%',
            transform=ax.transAxes,
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F1F3F5', edgecolor='#ADB5BD'),
        )

    plt.suptitle('Calibration Error Before vs After Hardware Fitting', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'before_after_calibration.png')
    figure7_path = os.path.join(results_dir, 'figure7_calibration.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(figure7_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved calibration summary CSV to {summary_path}")
    print(f"Saved calibration metadata CSV to {metadata_path}")
    print(f"Saved before/after calibration plot to {plot_path}")
    print(f"Saved paper-style calibration plot to {figure7_path}")

if __name__ == '__main__':
    main()

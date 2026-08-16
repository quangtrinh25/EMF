import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import (
    RESULTS_DIR,
    build_calibration_report,
    ensure_results_dir,
    load_calibrated_tx,
    write_calibration_report,
)

def main():
    data_dir = 'data/raw_calibration'
    calib_yaml = 'configs/physics_calibrated.yaml'
    if not os.path.exists(calib_yaml):
        print(f"Calibrated config {calib_yaml} not found! Please run calibration/calibrate.py first.")
        return

    calibrated_tx = load_calibrated_tx(calib_yaml)
    summary_df, details, _, _ = build_calibration_report(
        data_dir=data_dir,
        calibrated_tx=calibrated_tx,
    )
    summary_path, _ = write_calibration_report(summary_df, details, out_dir=ensure_results_dir(RESULTS_DIR))

    print(f"{'File Name':<25} | {'Nominal RMSE (mV)':<20} | {'Calibrated RMSE (mV)':<22} | {'Reduction':<10}")
    print("-" * 88)
    for row in summary_df.itertuples(index=False):
        if row.dataset == 'all_files':
            continue
        print(f"{row.dataset:<25} | {row.rmse_nominal_mV:<20.3f} | {row.rmse_calibrated_mV:<22.3f} | {row.reduction_pct:.2f}%")
    print(f"\nSaved summary CSV to {summary_path}")

if __name__ == '__main__':
    main()

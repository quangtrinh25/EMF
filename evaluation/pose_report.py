import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import evaluate_checkpoint, write_evaluation_results


def main():
    results_dir = 'results/pose'
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    runs = [
        {
            'label': 'before_calibration',
            'data_dir': 'data/synthetic_nominal',
            'checkpoint_dir': 'checkpoints/paper_reproduction',
            'model_type': 'resnet',
        },
        {
            'label': 'after_calibration',
            'data_dir': 'data/synthetic_calibrated',
            'checkpoint_dir': 'checkpoints/custom_system',
            'model_type': 'resnet',
        },
    ]

    rows = []
    for run in runs:
        result = evaluate_checkpoint(
            run['data_dir'],
            run['checkpoint_dir'],
            run['model_type'],
            device=device,
        )
        write_evaluation_results(result, run['label'], results_dir)
        rows.append({
            'label': run['label'],
            'dataset': result['dataset'],
            'checkpoint': result['checkpoint'],
            'position_rmse_mm': result['position_rmse_mm'],
            'orientation_rmse_deg': result['orientation_rmse_deg'],
        })

    summary_path = os.path.join(results_dir, 'before_after_pose_summary.csv')
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"Saved before/after pose summary to {summary_path}")
    for row in rows:
        print(
            f"{row['label']}: Position RMSE {row['position_rmse_mm']:.4f} mm, "
            f"Orientation {row['orientation_rmse_deg']:.4f} degrees"
        )


if __name__ == '__main__':
    main()

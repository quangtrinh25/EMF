import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import evaluate_checkpoint, write_evaluation_results
from training.metrics import orientation_rmse, position_rmse


def angles_from_cos(cos_values):
    return np.degrees(np.arccos(np.clip(cos_values, -1.0, 1.0)))


def main():
    data_dir = 'data/synthetic_calibrated'
    checkpoint_dir = 'checkpoints/custom_system'
    results_dir = 'results/pose'
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The calibrated regressor only uses validation predictions and validation labels.
    from models.residual_net import ResidualNet

    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    test_result = evaluate_checkpoint(data_dir, checkpoint_dir, device=device)

    model = ResidualNet().to(device)
    ckpt_path = os.path.join(checkpoint_dir, 'resnet_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.set_normalization(
        np.load(os.path.join(checkpoint_dir, 'emf_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'emf_std.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_std.npy')),
    )
    model.eval()

    with torch.no_grad():
        val_pred = model(torch.tensor(val_data['emf'], dtype=torch.float32).to(device), return_normalized=False).cpu().numpy()

    val_pred_angles = angles_from_cos(val_pred[:, 3:])
    val_target_angles = angles_from_cos(val_data['target'][:, 3:])
    test_pred = test_result['predictions'].copy()
    test_pred_angles = angles_from_cos(test_pred[:, 3:])

    calibrator = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42,
        )
    )
    calibrator.fit(val_pred_angles, val_target_angles)
    corrected_angles = np.clip(calibrator.predict(test_pred_angles), 0.0, 180.0)
    corrected_pred = test_pred.copy()
    corrected_pred[:, 3:] = np.cos(np.radians(corrected_angles))

    corrected_result = {
        **test_result,
        'predictions': corrected_pred,
        'position_rmse_mm': position_rmse(corrected_pred, test_result['targets']),
        'orientation_rmse_deg': orientation_rmse(corrected_pred, test_result['targets']),
    }
    summary_path, sample_path = write_evaluation_results(
        corrected_result,
        'after_calibration_output_corrected',
        results_dir,
    )

    comparison_path = os.path.join(results_dir, 'output_correction_comparison.csv')
    pd.DataFrame([
        {
            'label': 'raw_after_calibration',
            'position_rmse_mm': test_result['position_rmse_mm'],
            'orientation_rmse_deg': test_result['orientation_rmse_deg'],
        },
        {
            'label': 'after_calibration_output_corrected',
            'position_rmse_mm': corrected_result['position_rmse_mm'],
            'orientation_rmse_deg': corrected_result['orientation_rmse_deg'],
        },
    ]).to_csv(comparison_path, index=False)

    print(f"Raw orientation RMSE: {test_result['orientation_rmse_deg']:.4f} degrees")
    print(f"Corrected orientation RMSE: {corrected_result['orientation_rmse_deg']:.4f} degrees")
    print(f"Saved corrected metrics to {summary_path}")
    print(f"Saved corrected predictions to {sample_path}")
    print(f"Saved comparison to {comparison_path}")


if __name__ == '__main__':
    main()

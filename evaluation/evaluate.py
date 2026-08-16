import os
import argparse
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.residual_net import ResidualNet
from models.fcn import FCN
from models.kan import KAN
from training.metrics import position_rmse, orientation_rmse

RESULTS_DIR = 'results/pose'

def build_model(model_type, input_dim=9):
    if model_type == 'resnet':
        return ResidualNet(input_dim=input_dim)
    if model_type == 'fcn':
        return FCN()
    if model_type == 'kan':
        return KAN()
    raise ValueError(f"Unknown model type: {model_type}")


def load_metadata(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "model_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def evaluate_checkpoint(data_dir, checkpoint_dir, model_type='resnet', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_npz = os.path.join(data_dir, 'test.npz')
    if not os.path.exists(test_npz):
        raise FileNotFoundError(f"No test dataset found at {test_npz}. Run datasets/generator.py first.")

    data = np.load(test_npz)
    x_test = data['emf']
    y_test = data['target']

    metadata = load_metadata(checkpoint_dir)
    input_dim = int(metadata.get("input_dim", x_test.shape[1]))
    model = build_model(model_type, input_dim=input_dim).to(device)
    ckpt_path = os.path.join(checkpoint_dir, f'{model_type}_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No trained model checkpoint found in {checkpoint_dir}. Run training/train.py first.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.set_normalization(
        np.load(os.path.join(checkpoint_dir, 'emf_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'emf_std.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_std.npy')),
    )
    model.eval()

    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(x_tensor, return_normalized=False).cpu().numpy()

    return {
        'dataset': test_npz,
        'checkpoint': ckpt_path,
        'predictions': preds,
        'targets': y_test,
        'position_rmse_mm': position_rmse(preds, y_test),
        'orientation_rmse_deg': orientation_rmse(preds, y_test),
    }

def write_evaluation_results(result, label, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, f'{label}_pose_metrics.csv')
    pd.DataFrame([{
        'label': label,
        'dataset': result['dataset'],
        'checkpoint': result['checkpoint'],
        'position_rmse_mm': result['position_rmse_mm'],
        'orientation_rmse_deg': result['orientation_rmse_deg'],
    }]).to_csv(summary_path, index=False)

    pred = result['predictions']
    target = result['targets']
    pred_angles = np.degrees(np.arccos(np.clip(pred[:, 3:], -1.0, 1.0)))
    target_angles = np.degrees(np.arccos(np.clip(target[:, 3:], -1.0, 1.0)))
    sample_df = pd.DataFrame({
        'sample_index': np.arange(len(target), dtype=int),
        'target_X_mm': target[:, 0],
        'target_Y_mm': target[:, 1],
        'target_Z_mm': target[:, 2],
        'pred_X_mm': pred[:, 0],
        'pred_Y_mm': pred[:, 1],
        'pred_Z_mm': pred[:, 2],
        'target_roll_deg': target_angles[:, 0],
        'target_pitch_deg': target_angles[:, 1],
        'target_yaw_deg': target_angles[:, 2],
        'pred_roll_deg': pred_angles[:, 0],
        'pred_pitch_deg': pred_angles[:, 1],
        'pred_yaw_deg': pred_angles[:, 2],
    })
    sample_path = os.path.join(results_dir, f'{label}_pose_predictions.csv')
    sample_df.to_csv(sample_path, index=False)
    return summary_path, sample_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/synthetic_nominal')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/paper_reproduction')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'fcn', 'kan'])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    result = evaluate_checkpoint(args.data_dir, args.checkpoint_dir, args.model_type, device=device)
    pos_rmse = result['position_rmse_mm']
    ori_rmse = result['orientation_rmse_deg']

    print("\n" + "="*40)
    print("      EVALUATION METRICS SUMMARY")
    print("="*40)
    print(f"Dataset      : {result['dataset']}")
    print(f"Position RMSE: {pos_rmse:.4f} mm")
    print(f"Orientation  : {ori_rmse:.4f} degrees")
    print("="*40)

    if args.results_dir is not None:
        label = args.label or Path(args.checkpoint_dir).name
        summary_path, sample_path = write_evaluation_results(result, label, args.results_dir)
        print(f"Saved pose metrics to {summary_path}")
        print(f"Saved pose predictions to {sample_path}")

if __name__ == '__main__':
    main()

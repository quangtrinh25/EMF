import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_real_csv import angles_from_cos, load_model, prepare_emf_features
from inference.predict_capsule_pose import extract_emf_and_target, read_input


def predict(emf_v, checkpoint_dir, device):
    model, ckpt_path, metadata = load_model(checkpoint_dir, "resnet", device)
    features = prepare_emf_features(emf_v, metadata)
    with torch.no_grad():
        pred = model(torch.tensor(features, dtype=torch.float32, device=device), return_normalized=False).cpu().numpy()
    return pred, ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Predict pose using position model + temporal orientation model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/ensemble_predicted_capsule_pose.csv")
    parser.add_argument("--position_checkpoint_dir", default="checkpoints/noleak_balanced_easy/final")
    parser.add_argument("--orientation_checkpoint_dir", default="checkpoints/temporal_balanced_easy/final")
    parser.add_argument("--input_unit", choices=["mV", "V"], default="mV")
    parser.add_argument("--header", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    df = read_input(args.input, args.header)
    emf, _ = extract_emf_and_target(df)
    if args.input_unit == "mV":
        emf = emf * 1e-3

    pos_pred, pos_ckpt = predict(emf, args.position_checkpoint_dir, device)
    ori_pred, ori_ckpt = predict(emf, args.orientation_checkpoint_dir, device)
    pred = np.zeros_like(pos_pred)
    pred[:, :3] = pos_pred[:, :3]
    pred[:, 3:] = ori_pred[:, 3:]
    pred_angles = angles_from_cos(pred[:, 3:])

    out = pd.DataFrame({
        "sample_index": np.arange(len(pred), dtype=int),
        "pred_X_mm": pred[:, 0],
        "pred_Y_mm": pred[:, 1],
        "pred_Z_mm": pred[:, 2],
        "pred_roll_deg": pred_angles[:, 0],
        "pred_pitch_deg": pred_angles[:, 1],
        "pred_yaw_deg": pred_angles[:, 2],
    })
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Input: {args.input}")
    print(f"Position checkpoint: {pos_ckpt}")
    print(f"Orientation checkpoint: {ori_ckpt}")
    print(f"Samples: {len(pred)}")
    print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main()

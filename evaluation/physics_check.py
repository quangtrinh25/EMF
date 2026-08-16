import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.residual_net import ResidualNet
from physics.forward_model import forward_model

COLS = ['EMF1','EMF2','EMF3','EMF4','EMF5','EMF6','EMF7','EMF8','EMF9',
        'X','Y','Z','roll','pitch','yaw']

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    calib_yaml = 'configs/physics_calibrated.yaml'
    if not os.path.exists(calib_yaml):
        print("Calibrated config configs/physics_calibrated.yaml not found!")
        return
    
    with open(calib_yaml, 'r') as f:
        calib_config = yaml.safe_load(f)
    tx_params = calib_config['tx']
    rx_params = calib_config['rx']

    # Load 100 helical test poses from Set11_con_spi_rot_2.csv
    df = pd.read_csv('data/raw_calibration/Set11_con_spi_rot_2.csv', header=None, names=COLS)
    df_test = df.iloc[:100]
    emf_measured_V = df_test[['EMF1','EMF2','EMF3','EMF4','EMF5','EMF6','EMF7','EMF8','EMF9']].values * 1e-3
    poses_gt = df_test[['X','Y','Z','roll','pitch','yaw']].values

    # Load ResNet model
    resnet = ResidualNet().to(device)
    
    train_npz = 'data/synthetic_calibrated/train.npz'
    if os.path.exists(train_npz):
        train_data = np.load(train_npz)
        mean_emf = train_data['emf'].mean(axis=0)
        std_emf = train_data['emf'].std(axis=0)
        mean_pose = train_data['target'].mean(axis=0)
        std_pose = train_data['target'].std(axis=0)
    else:
        mean_emf = np.zeros(9)
        std_emf = np.ones(9)
        mean_pose = np.zeros(6)
        std_pose = np.ones(6)

    resnet.set_normalization(mean_emf, std_emf, mean_pose, std_pose)

    ckpt_path = 'checkpoints/custom_system/best.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/custom_system/resnet_best.pt'
        
    if os.path.exists(ckpt_path):
        resnet.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded ResNet checkpoint successfully.")
    else:
        print("No ResNet checkpoint found at checkpoints/custom_system/best.pt!")
        return
    
    resnet.eval()

    # Predict using ResNet
    emf_tensor = torch.tensor(emf_measured_V, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_resnet = resnet(emf_tensor, return_normalized=False).cpu().numpy()

    # Postprocess ResNet predictions to poses
    poses_resnet = np.zeros((len(preds_resnet), 6))
    poses_resnet[:, :3] = preds_resnet[:, :3]
    poses_resnet[:, 3:] = np.degrees(np.arccos(np.clip(preds_resnet[:, 3:], -1.0, 1.0)))

    # Feed predicted pose back into forward physics model
    emf_reconstructed_V = forward_model(poses_resnet, tx_params, rx_params)

    # Compute percentage error: abs(recon - measured) / measured * 100
    denom = np.where(np.abs(emf_measured_V) < 1e-9, 1e-9, emf_measured_V)
    pct_errors = np.abs(emf_reconstructed_V - emf_measured_V) / denom * 100

    channels = [f'TX{l+1}RX{k+1}' for l in range(3) for k in range(3)]
    mean_errors = np.mean(pct_errors, axis=0)
    max_errors = np.max(pct_errors, axis=0)

    print("\n" + "="*50)
    print("      PHYSICS CONSISTENCY VERIFICATION SUMMARY")
    print("="*50)
    print(f"{'Channel':<10} | {'Mean Pct Error (%)':<20} | {'Max Pct Error (%)':<20}")
    print("-" * 55)
    for ch, mean_err, max_err in zip(channels, mean_errors, max_errors):
        print(f"{ch:<10} | {mean_err:<20.3f}% | {max_err:<20.3f}%")
    print("-" * 55)
    
    overall_mean = np.mean(pct_errors)
    overall_max = np.max(pct_errors)
    print(f"Overall Mean Error: {overall_mean:.4f}%")
    print(f"Overall Max Error : {overall_max:.4f}%")
    print(f"Goal (< 3.4%)     : {'MET (Success)' if overall_mean < 3.4 else 'FAILED'}")
    print("="*50)

if __name__ == '__main__':
    main()

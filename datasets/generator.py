import os
import sys
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.forward_model import forward_model
from physics.channel_correction import apply_channel_correction

def generate_dataset(n_samples, tx_params, rx_params, workspace_params=None, seed=42):
    np.random.seed(seed)

    if workspace_params is None:
        x_min, x_max = -250.0, 250.0
        y_min, y_max = -250.0, 250.0
        z_min, z_max = 150.0, 650.0
    else:
        x_min, x_max = workspace_params.get('x_mm', [-250.0, 250.0])
        y_min, y_max = workspace_params.get('y_mm', [-250.0, 250.0])
        z_min, z_max = workspace_params.get('z_mm', [150.0, 650.0])

    print(f"Sampling bounds: X [{x_min}, {x_max}], Y [{y_min}, {y_max}], Z [{z_min}, {z_max}]")

    # Sample positions in mm
    X = np.random.uniform(x_min, x_max, n_samples)
    Y = np.random.uniform(y_min, y_max, n_samples)
    Z = np.random.uniform(z_min, z_max, n_samples)

    orientation_deg = workspace_params.get('orientation_deg') if workspace_params else None
    if orientation_deg:
        a_min, a_max = orientation_deg.get('roll', [0.0, 180.0])
        b_min, b_max = orientation_deg.get('pitch', [0.0, 180.0])
        g_min, g_max = orientation_deg.get('yaw', [0.0, 180.0])
        alpha = np.random.uniform(a_min, a_max, n_samples)
        beta = np.random.uniform(b_min, b_max, n_samples)
        gamma = np.random.uniform(g_min, g_max, n_samples)
        cos_a = np.cos(np.radians(alpha))
        cos_b = np.cos(np.radians(beta))
        cos_g = np.cos(np.radians(gamma))
        print(f"Sampling orientation deg: roll [{a_min}, {a_max}], pitch [{b_min}, {b_max}], yaw [{g_min}, {g_max}]")
    else:
        # Sample orientations in cosine space, matching the paper's broad synthetic setup.
        cos_a = np.random.uniform(-1.0, 1.0, n_samples)
        cos_b = np.random.uniform(-1.0, 1.0, n_samples)
        cos_g = np.random.uniform(-1.0, 1.0, n_samples)
        alpha = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        beta = np.degrees(np.arccos(np.clip(cos_b, -1.0, 1.0)))
        gamma = np.degrees(np.arccos(np.clip(cos_g, -1.0, 1.0)))

    # Pose format for forward model: [X, Y, Z, alpha_deg, beta_deg, gamma_deg]
    poses = np.stack([X, Y, Z, alpha, beta, gamma], axis=1)

    print(f"Generating EMFs for {n_samples} samples...")
    emfs = forward_model(poses, tx_params, rx_params)
    if 'channel_correction' in tx_params:
        emfs = apply_channel_correction(emfs, tx_params)

    # Targets: [X, Y, Z, cos_a, cos_b, cos_g]
    targets = np.stack([X, Y, Z, cos_a, cos_b, cos_g], axis=1).astype(np.float32)

    # Shuffling and splitting
    idx = np.random.permutation(n_samples)
    n_train = int(0.70 * n_samples)
    n_val   = int(0.15 * n_samples)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    return {
        'train': (emfs[train_idx], targets[train_idx]),
        'val':   (emfs[val_idx],   targets[val_idx]),
        'test':  (emfs[test_idx],  targets[test_idx]),
    }

def save_dataset(split_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, (emf, tgt) in split_dict.items():
        np.savez(os.path.join(out_dir, f'{name}.npz'), emf=emf, target=tgt)
        print(f"Saved {name} to {out_dir}: emf {emf.shape}, target {tgt.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/physics_nominal.yaml')
    parser.add_argument('--out_dir', type=str, default='data/synthetic_nominal')
    parser.add_argument('--n_samples', type=int, default=640000)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    splits = generate_dataset(
        args.n_samples,
        config['tx'],
        config['rx'],
        config.get('workspace', None)
    )
    save_dataset(splits, args.out_dir)

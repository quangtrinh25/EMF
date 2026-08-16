import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.residual_net import ResidualNet
from training.losses_v3 import pose_loss_v3
from training.metrics import pose_metrics_v3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_std(values, minimum=1e-6):
    std = np.std(values, axis=0).astype(np.float32)
    return np.where(std < minimum, 1.0, std).astype(np.float32)


def load_arrays(path):
    data = np.load(path)
    emf = data['emf'].astype(np.float32)
    target = data['target'].astype(np.float32)
    if emf.ndim != 2 or emf.shape[1] < 9:
        raise ValueError(f"Expected at least nine EMF-derived features in {path}, got {emf.shape}")
    if target.ndim != 2 or target.shape[1] != 9:
        raise ValueError(f"Expected XYZ+rotation-6D target in {path}, got {target.shape}")
    return emf, target


def augment_emf_batch(
    features, gain_drift, noise_std_v, emf_feature_count,
    log_gain_std=0.0, session_offset_std_v=0.0,
):
    if emf_feature_count < 9 or emf_feature_count % 9 != 0:
        raise ValueError(f"emf_feature_count must be a positive multiple of 9, got {emf_feature_count}")
    emf = features[:, :emf_feature_count].reshape(features.shape[0], -1, 9)
    if gain_drift > 0:
        # One nine-channel gain vector is shared across every time step and
        # sample in the batch, mimicking coherent electronics/session drift.
        gains = torch.empty((1, 1, 9), device=emf.device).uniform_(
            1.0 - gain_drift, 1.0 + gain_drift,
        )
        emf = emf * gains
    if log_gain_std > 0:
        log_gain = torch.randn((1, 1, 9), device=emf.device) * log_gain_std
        emf = emf * torch.exp(log_gain).clamp(0.5, 2.0)
    if session_offset_std_v > 0:
        offset = torch.randn((1, 1, 9), device=emf.device) * session_offset_std_v
        emf = emf + offset
    if noise_std_v > 0:
        emf = emf + torch.randn_like(emf) * noise_std_v
    emf = torch.clamp(emf, min=0.0).reshape(features.shape[0], emf_feature_count)
    if emf_feature_count == features.shape[1]:
        return emf
    return torch.cat([emf, features[:, emf_feature_count:]], dim=1)


def main():
    parser = argparse.ArgumentParser(description="Train signed-orientation v3 ResNet.")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--config', default='configs/training_v3.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default=None)
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--residual_blocks', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gain_drift', type=float, default=None)
    parser.add_argument('--log_gain_std', type=float, default=None)
    parser.add_argument('--session_offset_std_mV', type=float, default=None)
    parser.add_argument('--noise_std_mV', type=float, default=None)
    parser.add_argument(
        '--fixed_epochs_no_val', action='store_true',
        help='Train all supplied training data for exactly --epochs; do not load or select on validation.',
    )
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    model_config = config.get('model', {})
    epochs = int(args.epochs if args.epochs is not None else config['epochs'])
    batch_size = int(args.batch_size if args.batch_size is not None else config['batch_size'])
    hidden = int(args.hidden if args.hidden is not None else model_config.get('hidden', 512))
    n_blocks = int(args.residual_blocks if args.residual_blocks is not None else model_config.get('residual_blocks', 7))
    slope = float(model_config.get('leaky_relu_slope', 0.01))
    seed = int(args.seed if args.seed is not None else config.get('seed', 42))
    requested_device = args.device or config.get('device', 'cuda')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA device. "
            "Check the NVIDIA driver, or explicitly use --device cpu/auto."
        )
    device = torch.device(
        'cuda'
        if requested_device in {'auto', 'cuda'} and torch.cuda.is_available()
        else 'cpu'
    )
    print(f"Training device: {device} (requested: {requested_device})", flush=True)
    set_seed(seed)

    train_emf_np, train_target_np = load_arrays(Path(args.data_dir) / 'train.npz')
    if args.fixed_epochs_no_val:
        val_emf_np = val_target_np = None
    else:
        val_emf_np, val_target_np = load_arrays(Path(args.data_dir) / 'val.npz')
    protocol_path = Path(args.data_dir) / 'protocol.json'
    protocol = {}
    if protocol_path.exists():
        with open(protocol_path) as stream:
            protocol = json.load(stream)
    input_dim = int(train_emf_np.shape[1])
    if val_emf_np is not None and val_emf_np.shape[1] != input_dim:
        raise ValueError("Train and validation feature dimensions differ")
    emf_feature_count = int(protocol.get('emf_feature_count', input_dim))
    if emf_feature_count > input_dim:
        raise ValueError("Protocol emf_feature_count exceeds input dimension")
    emf_mean = np.mean(train_emf_np, axis=0).astype(np.float32)
    emf_std = safe_std(train_emf_np)
    pose_mean = np.mean(train_target_np, axis=0).astype(np.float32)
    pose_std = safe_std(train_target_np)

    train_emf = torch.tensor(train_emf_np, device=device)
    train_target = torch.tensor(train_target_np, device=device)
    val_emf = None if val_emf_np is None else torch.tensor(val_emf_np, device=device)
    val_target = None if val_target_np is None else torch.tensor(val_target_np, device=device)

    model = ResidualNet(
        input_dim=input_dim, output_dim=9, n_blocks=n_blocks, hidden=hidden, slope=slope,
    ).to(device)
    model.set_normalization(emf_mean, emf_std, pose_mean, pose_std)
    pose_mean_t, pose_std_t = model.pose_mean, model.pose_std
    train_target_norm = (train_target - pose_mean_t) / pose_std_t
    val_target_norm = None if val_target is None else (val_target - pose_mean_t) / pose_std_t

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config.get('weight_decay', 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(config.get('scheduler_patience', 20)),
        factor=float(config.get('scheduler_factor', 0.5)),
    )
    position_weight = float(config.get('position_loss_weight', 1.0))
    orientation_weight = float(config.get('orientation_loss_weight', 1.0))
    gain_drift = float(args.gain_drift if args.gain_drift is not None else config.get('session_gain_drift', 0.0))
    log_gain_std = float(args.log_gain_std if args.log_gain_std is not None else config.get('channel_log_gain_std', 0.0))
    session_offset_std_v = float(
        args.session_offset_std_mV
        if args.session_offset_std_mV is not None
        else config.get('session_offset_std_mV', 0.0)
    ) * 1e-3
    noise_std_v = float(
        args.noise_std_mV if args.noise_std_mV is not None else config.get('noise_std_mV', 0.0)
    ) * 1e-3
    clip_norm = float(config.get('gradient_clip_norm', 0.0))
    early_patience = int(config.get('early_stopping_patience', epochs))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'schema_version': 3,
        'model_type': 'resnet',
        'input_dim': input_dim,
        'output_dim': 9,
        'target_representation': 'position_mm+rotation_6d',
        'rotation_convention': 'Rz_yaw_Ry_pitch_Rx_roll',
        'hidden': hidden,
        'residual_blocks': n_blocks,
        'leaky_relu_slope': slope,
        'train_data_dir': str(args.data_dir),
        'seed': seed,
        'checkpoint_selection': (
            'fixed_epoch_from_leave_one_session_out_cv_no_validation'
            if args.fixed_epochs_no_val else 'minimum_validation_loss_only'
        ),
        'fixed_training_epochs': epochs if args.fixed_epochs_no_val else None,
        'workspace': protocol.get('workspace'),
        'data_roles': protocol.get('roles'),
        'hardware_validation': protocol.get('hardware_validation'),
        'feature_layout': protocol.get('feature_layout', 'current_emf_9'),
        'emf_feature_count': emf_feature_count,
        'dt_feature_count': int(protocol.get('dt_feature_count', 0)),
        'temporal_window': protocol.get('temporal_window'),
        'temporal_feature_mode': protocol.get('temporal_feature_mode', 'raw_window'),
        'horizon_steps': int(protocol.get('horizon_steps', 0)),
        'timestamp_feature': protocol.get('timestamp_feature'),
        'timestamp_source': protocol.get('timestamp_source'),
        'timestamp_unit': protocol.get('timestamp_unit'),
        'absolute_timestamp_used': False,
        'dev_scope': protocol.get('roles', {}).get('dev_scope'),
        'test_scope': protocol.get('roles', {}).get('test_scope'),
        'augmentation': {
            'uniform_gain_drift': gain_drift,
            'channel_log_gain_std': log_gain_std,
            'session_offset_std_mV': session_offset_std_v * 1e3,
            'sample_noise_std_mV': noise_std_v * 1e3,
            'gain_and_offset_shared_across_batch_and_time': True,
        },
    }
    with open(checkpoint_dir / 'model_metadata.json', 'w') as stream:
        json.dump(metadata, stream, indent=2)
    np.save(checkpoint_dir / 'emf_mean.npy', emf_mean)
    np.save(checkpoint_dir / 'emf_std.npy', emf_std)
    np.save(checkpoint_dir / 'pose_mean.npy', pose_mean)
    np.save(checkpoint_dir / 'pose_std.npy', pose_std)

    history = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    print(
        f"Device={device}; train={len(train_emf_np)}; "
        f"val={0 if val_emf_np is None else len(val_emf_np)}; "
        f"input={input_dim}; model={hidden}x{n_blocks}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_emf), generator=generator, device=device)
        train_total = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x = augment_emf_batch(
                train_emf[indices], gain_drift, noise_std_v, emf_feature_count,
                log_gain_std=log_gain_std,
                session_offset_std_v=session_offset_std_v,
            )
            y = train_target_norm[indices]
            optimizer.zero_grad()
            pred_norm = model(x, return_normalized=True)
            loss, _, _ = pose_loss_v3(
                pred_norm, y, pose_mean_t, pose_std_t, position_weight, orientation_weight,
            )
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_total += float(loss.item()) * len(indices)

        row = {
            'epoch': epoch,
            'train_loss': train_total / len(train_emf),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
        }
        if not args.fixed_epochs_no_val:
            model.eval()
            with torch.no_grad():
                val_pred_norm = model(val_emf, return_normalized=True)
                val_loss, val_pos_loss, val_ori_loss = pose_loss_v3(
                    val_pred_norm, val_target_norm, pose_mean_t, pose_std_t,
                    position_weight, orientation_weight,
                )
                val_pred = model(val_emf, return_normalized=False).cpu().numpy()
            metrics = pose_metrics_v3(val_pred, val_target_np)
            scheduler.step(val_loss)
            row.update({
                'val_loss': float(val_loss.item()),
                'val_position_loss': float(val_pos_loss.item()),
                'val_orientation_loss_rad2': float(val_ori_loss.item()),
                **metrics,
            })
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            if args.fixed_epochs_no_val:
                print(f"Epoch {epoch:03d} train={row['train_loss']:.5f}")
            else:
                print(
                    f"Epoch {epoch:03d} train={row['train_loss']:.5f} val={row['val_loss']:.5f} "
                    f"pos={metrics['position_axis_rmse_mm']:.3f} mm "
                    f"SO3={metrics['orientation_geodesic_rmse_deg']:.3f} deg"
                )
        if not args.fixed_epochs_no_val:
            if row['val_loss'] < best_loss - 1e-9:
                best_loss = row['val_loss']
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_dir / 'best.pt')
                torch.save(model.state_dict(), checkpoint_dir / 'resnet_best.pt')
                with open(checkpoint_dir / 'best_validation_metrics.json', 'w') as stream:
                    json.dump(row, stream, indent=2)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if args.fixed_epochs_no_val:
        torch.save(model.state_dict(), checkpoint_dir / 'best.pt')
        torch.save(model.state_dict(), checkpoint_dir / 'resnet_best.pt')
        with open(checkpoint_dir / 'fixed_epoch_training_summary.json', 'w') as stream:
            json.dump(history[-1], stream, indent=2)

    with open(checkpoint_dir / 'training_history.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    if args.fixed_epochs_no_val:
        print(f"Saved fixed-epoch deployment checkpoint to {checkpoint_dir}")
    else:
        print(f"Saved validation-selected checkpoint to {checkpoint_dir}")


if __name__ == '__main__':
    main()

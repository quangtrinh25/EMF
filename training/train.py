import os
import argparse
import json
import random
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.residual_net import ResidualNet
from models.fcn import FCN
from models.kan import KAN
from training.losses import mse_loss, rmse_loss, weighted_pose_mse_loss
from training.metrics import position_rmse, orientation_rmse

class FastTensorDataLoader:
    def __init__(self, tensors, batch_size=1024, shuffle=True):
        self.tensors = tensors
        self.dataset_len = tensors[0].size(0)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
        else:
            self.indices = None
        self.idx = 0
        return self
        
    def __next__(self):
        if self.idx >= self.dataset_len:
            raise StopIteration
        
        if self.indices is not None:
            batch_indices = self.indices[self.idx : self.idx + self.batch_size]
            batch = tuple(t[batch_indices] for t in self.tensors)
        else:
            batch = tuple(t[self.idx : self.idx + self.batch_size] for t in self.tensors)
            
        self.idx += self.batch_size
        return batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    # Load training configs
    with open(args.config, 'r') as f:
        train_config = yaml.safe_load(f)

    # Override training config with CLI args if specified
    epochs = int(args.epochs if args.epochs is not None else train_config.get('epochs', 600))
    batch_size = int(args.batch_size if args.batch_size is not None else train_config.get('batch_size', 1024))
    lr = float(args.lr if args.lr is not None else train_config.get('learning_rate', 0.001))
    weight_decay = float(train_config.get('weight_decay', 1e-5))
    position_weight = float(args.position_weight if args.position_weight is not None else train_config.get('position_loss_weight', 1.0))
    orientation_weight = float(args.orientation_weight if args.orientation_weight is not None else train_config.get('orientation_loss_weight', 1.0))
    seed = int(train_config.get('seed', 42))
    device_name = args.device if args.device is not None else train_config.get('device', 'cuda')
    
    set_seed(seed)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_npz = os.path.join(args.data_dir, 'train.npz')
    val_npz = os.path.join(args.data_dir, 'val.npz')
    
    print(f"Loading datasets from {args.data_dir}...")
    train_data = np.load(train_npz)
    val_data = np.load(val_npz)
    
    # Pre-load entire tensors onto the training device (GPU) for maximum throughput
    train_emf = torch.tensor(train_data['emf'], dtype=torch.float32).to(device)
    train_target = torch.tensor(train_data['target'], dtype=torch.float32).to(device)
    
    val_emf = torch.tensor(val_data['emf'], dtype=torch.float32).to(device)
    val_target = torch.tensor(val_data['target'], dtype=torch.float32).to(device)
    
    mean_emf = train_data['emf'].mean(axis=0)
    std_emf = train_data['emf'].std(axis=0)
    mean_pose = train_data['target'].mean(axis=0)
    std_pose = train_data['target'].std(axis=0)
    
    print(f"Normalizing EMF with mean: {mean_emf}, std: {std_emf}")
    print(f"Normalizing Pose with mean: {mean_pose}, std: {std_pose}")

    input_dim = int(train_data['emf'].shape[1])

    # Initialize model
    if args.model_type == 'resnet':
        model = ResidualNet(input_dim=input_dim)
    elif args.model_type == 'fcn':
        if input_dim != 9:
            raise ValueError("FCN only supports 9 input features.")
        model = FCN()
    elif args.model_type == 'kan':
        if input_dim != 9:
            raise ValueError("KAN only supports 9 input features.")
        model = KAN()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.set_normalization(mean_emf, std_emf, mean_pose, std_pose)
    model = model.to(device)
    if args.init_checkpoint is not None:
        print(f"Initializing model from {args.init_checkpoint}")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        model.set_normalization(mean_emf, std_emf, mean_pose, std_pose)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=train_config.get('scheduler_patience', 20),
        factor=train_config.get('scheduler_factor', 0.5)
    )

    if args.model_type in ['fcn', 'kan']:
        loss_fn = rmse_loss
    elif position_weight != 1.0 or orientation_weight != 1.0:
        loss_fn = lambda pred, target: weighted_pose_mse_loss(
            pred,
            target,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
    else:
        loss_fn = mse_loss
    print(f"Loss weights: position={position_weight:.3f}, orientation={orientation_weight:.3f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    temporal_window = args.temporal_window
    if temporal_window is None and input_dim != 9 and input_dim % 9 == 0:
        temporal_window = input_dim // 9
    metadata = {
        "model_type": args.model_type,
        "input_dim": input_dim,
        "temporal_window": temporal_window,
        "temporal_mode": args.temporal_mode if temporal_window else None,
        "feature_layout": args.feature_layout if temporal_window else None,
    }
    with open(os.path.join(args.checkpoint_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save normalization stats as well
    np.save(os.path.join(args.checkpoint_dir, 'emf_mean.npy'), mean_emf)
    np.save(os.path.join(args.checkpoint_dir, 'emf_std.npy'), std_emf)
    np.save(os.path.join(args.checkpoint_dir, 'pose_mean.npy'), mean_pose)
    np.save(os.path.join(args.checkpoint_dir, 'pose_std.npy'), std_pose)

    # Setup normalized target tensors for loss computation
    pose_mean_t = model.pose_mean
    pose_std_t = model.pose_std
    train_target_norm = (train_target - pose_mean_t) / (pose_std_t + 1e-8)
    val_target_norm = (val_target - pose_mean_t) / (pose_std_t + 1e-8)

    train_loader = FastTensorDataLoader((train_emf, train_target_norm), batch_size=batch_size, shuffle=True)

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for emf_batch, pose_batch_norm in train_loader:
            optimizer.zero_grad()
            pred_norm = model(emf_batch, return_normalized=True)
            loss = loss_fn(pred_norm, pose_batch_norm)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * emf_batch.size(0)

        train_loss = train_loss_total / train_emf.size(0)

        # Validate
        model.eval()
        with torch.no_grad():
            pred_val_norm = model(val_emf, return_normalized=True)
            val_loss = loss_fn(pred_val_norm, val_target_norm).item()
            
            # Get raw (un-normalized) predictions for metrics
            pred_raw = model(val_emf, return_normalized=False).cpu().numpy()
            targets_val_np = val_target.cpu().numpy()
            pos_rmse = position_rmse(pred_raw, targets_val_np)
            ori_rmse = orientation_rmse(pred_raw, targets_val_np)

        scheduler.step(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Pos RMSE: {pos_rmse:.3f} mm | Ori RMSE: {ori_rmse:.3f} deg")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save standard name
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pt')
            torch.save(model.state_dict(), checkpoint_path)
            # Save model-specific name to prevent overwriting
            model_checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model_type}_best.pt')
            torch.save(model.state_dict(), model_checkpoint_path)

    print("Training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training.yaml')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_nominal')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/paper_reproduction')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'fcn', 'kan'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--position_weight', type=float, default=None)
    parser.add_argument('--orientation_weight', type=float, default=None)
    parser.add_argument('--init_checkpoint', type=str, default=None)
    parser.add_argument('--temporal_window', type=int, default=None)
    parser.add_argument('--temporal_mode', type=str, default='past')
    parser.add_argument('--feature_layout', type=str, default='oldest_to_current')
    args = parser.parse_args()
    train(args)

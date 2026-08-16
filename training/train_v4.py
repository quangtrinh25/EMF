"""Train v4 cross-generation candidates without touching sealed test arrays."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.localizer_v4 import ResidualLocalizerV4, TemporalGRULocalizerV4
from training.losses_v3 import pose_loss_v3
from training.metrics import pose_metrics_v3


PRIMARY_METRICS = (
    "position_axis_rmse_mm",
    "position_euclidean_p95_mm",
    "orientation_geodesic_rmse_deg",
    "orientation_geodesic_p95_deg",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_std(values, minimum=1e-8):
    std = np.std(values, axis=0).astype(np.float32)
    return np.where(std < minimum, 1.0, std).astype(np.float32)


def load_split(path):
    with np.load(path) as data:
        required = {"emf", "target", "generation_index", "session_key"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        return {key: data[key] for key in data.files}


def adapter_channel_std(packed, generation_index, generation_count, window_size):
    raw = packed[:, :window_size * 9].reshape(-1, window_size, 9)
    result = np.ones((generation_count, 9), dtype=np.float32)
    for generation in range(generation_count):
        values = raw[generation_index == generation].reshape(-1, 9)
        if len(values):
            result[generation] = safe_std(values)
    return result


def make_model(candidate, protocol, train_data, model_config):
    window = int(candidate["window_size"])
    generation_count = len(protocol["generation_to_index"])
    slope = float(model_config.get("leaky_relu_slope", 0.01))
    if candidate["model_type"] == "gru":
        return TemporalGRULocalizerV4(window_size=window, output_dim=9, hidden=128, slope=slope), None
    channel_std = adapter_channel_std(
        train_data["emf"], train_data["generation_index"], generation_count, window,
    )
    model = ResidualLocalizerV4(
        window_size=window,
        feature_mode=candidate["feature_mode"],
        generation_count=generation_count,
        use_calibration_adapter=bool(candidate.get("calibration_adapter", False)),
        adapter_channel_std=channel_std,
        output_dim=9,
        n_blocks=int(model_config.get("residual_blocks", 7)),
        hidden=int(model_config.get("hidden", 512)),
        slope=slope,
    )
    return model, channel_std


def identity_feature_stats(model, packed, generation_index, device):
    with torch.inference_mode():
        x = torch.as_tensor(packed, dtype=torch.float32, device=device)
        generation = torch.as_tensor(generation_index, dtype=torch.long, device=device)
        if isinstance(model, ResidualLocalizerV4):
            features = model.build_features(x, generation).cpu().numpy()
        else:
            features = packed
    return np.mean(features, axis=0).astype(np.float32), safe_std(features)


def augment_packed(packed, window_size, gain_drift, noise_std_v):
    if gain_drift <= 0 and noise_std_v <= 0:
        return packed
    raw_count = window_size * 9
    raw = packed[:, :raw_count].reshape(-1, window_size, 9)
    if gain_drift > 0:
        gain = torch.empty((1, 1, 9), device=raw.device).uniform_(
            1.0 - gain_drift, 1.0 + gain_drift,
        )
        raw = raw * gain
    if noise_std_v > 0:
        raw = raw + torch.randn_like(raw) * noise_std_v
    raw = torch.clamp(raw, min=0.0).reshape(-1, raw_count)
    return torch.cat([raw, packed[:, raw_count:]], dim=1)


def balanced_sample_weights(data, protocol, cells_per_axis=5, spatial_cap=3.0):
    generation = data["generation_index"].astype(int)
    session = data["session_key"].astype(str)
    position = data["target"][:, :3]
    workspace = protocol["workspace"]
    center = np.asarray(workspace["center_mm"], dtype=float)
    size = np.asarray(workspace["size_mm"], dtype=float)
    lower = center - size / 2.0
    cell_xyz = np.floor((position - lower) / size * cells_per_axis).astype(int)
    cell_xyz = np.clip(cell_xyz, 0, cells_per_axis - 1)
    cell = cell_xyz[:, 0] * cells_per_axis ** 2 + cell_xyz[:, 1] * cells_per_axis + cell_xyz[:, 2]

    weights = np.zeros(len(position), dtype=np.float64)
    unique_generations = np.unique(generation)
    for generation_value in unique_generations:
        generation_mask = generation == generation_value
        sessions = np.unique(session[generation_mask])
        for session_value in sessions:
            mask = generation_mask & (session == session_value)
            indices = np.flatnonzero(mask)
            session_cells, counts = np.unique(cell[indices], return_counts=True)
            count_by_cell = dict(zip(session_cells.tolist(), counts.tolist()))
            spatial = np.asarray([1.0 / np.sqrt(count_by_cell[int(value)]) for value in cell[indices]])
            spatial /= np.mean(spatial)
            spatial = np.clip(spatial, 1.0 / spatial_cap, spatial_cap)
            weights[indices] = spatial / (
                len(unique_generations) * len(sessions) * len(indices)
            )
    weights /= weights.sum()
    return weights.astype(np.float32)


def composite_score(metrics):
    values = np.asarray([max(float(metrics[key]), 1e-12) for key in PRIMARY_METRICS])
    return float(np.exp(np.mean(np.log(values))))


def main():
    parser = argparse.ArgumentParser(description="Train a generation-aware v4 candidate.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config", default="configs/training_v4.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--fixed_epochs_no_val", action="store_true")
    parser.add_argument("--init_checkpoint", default=None, help="Optional same-architecture v4 checkpoint for staged fine-tuning.")
    parser.add_argument(
        "--reset_normalization_on_init", action="store_true",
        help="Transfer learned weights but recompute input/target normalization from the current real-only train split.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "protocol.json") as stream:
        protocol = json.load(stream)
    if protocol.get("sealed_test_opened") is not False:
        raise RuntimeError("Refusing to train after the sealed test has been opened")
    if (data_dir / "sealed_test.npz").exists():
        raise RuntimeError("Refusing to train beside a sealed_test.npz artifact")
    candidate = protocol["candidate_config"]
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    epochs = int(args.epochs if args.epochs is not None else config["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else config["batch_size"])
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    requested_device = args.device or config.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu/auto or run on the host GPU")
    device = torch.device("cuda" if requested_device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu")
    set_seed(seed)
    print(f"Training v4 candidate={protocol['candidate']} on {device}", flush=True)

    train_data = load_split(data_dir / "train.npz")
    val_data = None if args.fixed_epochs_no_val else load_split(data_dir / "val.npz")
    model, channel_std = make_model(candidate, protocol, train_data, config.get("model", {}))
    model = model.to(device)
    feature_mean, feature_std = identity_feature_stats(
        model, train_data["emf"].astype(np.float32), train_data["generation_index"], device,
    )
    pose_mean = np.mean(train_data["target"], axis=0).astype(np.float32)
    pose_std = safe_std(train_data["target"])
    model.set_normalization(feature_mean, feature_std, pose_mean, pose_std)
    parent_metadata = None
    if args.init_checkpoint:
        init_dir = Path(args.init_checkpoint)
        with open(init_dir / "model_metadata.json") as stream:
            parent_metadata = json.load(stream)
        for key, expected in (
            ("candidate_config", candidate),
            ("generation_to_index", protocol["generation_to_index"]),
            ("packed_input_dim", int(protocol["packed_input_dim"])),
        ):
            if parent_metadata.get(key) != expected:
                raise ValueError(
                    f"Fine-tune checkpoint is incompatible for {key}: {parent_metadata.get(key)} != {expected}"
                )
        state = torch.load(init_dir / "best.pt", map_location=device, weights_only=True)
        if args.reset_normalization_on_init:
            buffer_suffixes = (
                "feature_mean", "feature_std", "raw_mean", "raw_std",
                "pose_mean", "pose_std", "calibration_adapter.channel_std",
            )
            state = {
                key: value for key, value in state.items()
                if not any(key.endswith(suffix) for suffix in buffer_suffixes)
            }
            incompatible = model.load_state_dict(state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            if unexpected:
                raise RuntimeError(f"Unexpected transfer keys: {unexpected}")
        else:
            model.load_state_dict(state)
        print(f"Initialized from staged checkpoint: {init_dir}", flush=True)

    train_x = torch.as_tensor(train_data["emf"], dtype=torch.float32, device=device)
    train_generation = torch.as_tensor(train_data["generation_index"], dtype=torch.long, device=device)
    train_target = torch.as_tensor(train_data["target"], dtype=torch.float32, device=device)
    target_norm = (train_target - model.pose_mean) / model.pose_std
    if val_data is not None:
        val_x = torch.as_tensor(val_data["emf"], dtype=torch.float32, device=device)
        val_generation = torch.as_tensor(val_data["generation_index"], dtype=torch.long, device=device)
        val_target = torch.as_tensor(val_data["target"], dtype=torch.float32, device=device)
        val_target_norm = (val_target - model.pose_mean) / model.pose_std

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=int(config.get("scheduler_patience", 20)), factor=float(config.get("scheduler_factor", 0.5)),
    )
    position_weight = float(config.get("position_loss_weight", 1.0))
    orientation_weight = float(config.get("orientation_loss_weight", 1.0))
    adapter_weight = float(config.get("adapter_regularization_weight", 0.001))
    gain_drift = float(config.get("session_gain_drift", 0.0))
    noise_std_v = float(config.get("noise_std_mV", 0.0)) * 1e-3
    clip_norm = float(config.get("gradient_clip_norm", 0.0))
    early_patience = int(config.get("early_stopping_patience", epochs))
    sampler_weights = (
        balanced_sample_weights(train_data, protocol)
        if candidate.get("balanced_sampler", False) else None
    )
    sampler_weights_t = None if sampler_weights is None else torch.as_tensor(sampler_weights, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 4,
        "model_type": candidate["model_type"],
        "candidate": protocol["candidate"],
        "candidate_config": candidate,
        "packed_input_dim": int(protocol["packed_input_dim"]),
        "resnet_feature_dim": int(protocol["resnet_feature_dim"]),
        "output_dim": 9,
        "target_representation": "position_mm+rotation_6d",
        "rotation_convention": "Rz_yaw_Ry_pitch_Rx_roll",
        "window_size": int(candidate["window_size"]),
        "hidden": 128 if candidate["model_type"] == "gru" else int(config.get("model", {}).get("hidden", 512)),
        "residual_blocks": 0 if candidate["model_type"] == "gru" else int(config.get("model", {}).get("residual_blocks", 7)),
        "leaky_relu_slope": float(config.get("model", {}).get("leaky_relu_slope", 0.01)),
        "generation_to_index": protocol["generation_to_index"],
        "adapter_channel_std": (
            None if not isinstance(model, ResidualLocalizerV4) or model.calibration_adapter is None
            else model.calibration_adapter.channel_std.detach().cpu().numpy().tolist()
        ),
        "train_dt_median": protocol["train_dt_median"],
        "generation_time_mode": protocol.get("generation_time_mode", {}),
        "generation_fixed_dt_seconds": protocol.get("generation_fixed_dt_seconds", {}),
        "dt_ratio_clip": protocol["dt_ratio_clip"],
        "gap_reset_multiple": protocol["gap_reset_multiple"],
        "absolute_timestamp_used": False,
        "workspace": protocol["workspace"],
        "fold": protocol["fold"],
        "seed": seed,
        "balanced_sampler": bool(candidate.get("balanced_sampler", False)),
        "balanced_sampler_policy": (
            {"workspace_cells": [5, 5, 5], "spatial_weight": "inverse_sqrt_frequency", "spatial_cap": 3.0,
             "generation_total": "equal", "session_within_generation_total": "equal"}
            if candidate.get("balanced_sampler", False) else None
        ),
        "adapter_regularization_weight": adapter_weight,
        "augmentation": {
            "session_gain_drift": gain_drift,
            "sample_noise_std_mV": noise_std_v * 1e3,
            "gain_shared_across_batch_window_and_channels_independent": True,
        },
        "sealed_test_opened_during_training": False,
        "checkpoint_selection": "fixed_epochs" if args.fixed_epochs_no_val else "minimum_balanced_four_metric_composite",
        "initialized_from_checkpoint": args.init_checkpoint,
        "initialized_from_seed": None if parent_metadata is None else parent_metadata.get("seed"),
        "normalization_reset_after_initialization": bool(args.reset_normalization_on_init),
    }
    with open(checkpoint_dir / "model_metadata.json", "w") as stream:
        json.dump(metadata, stream, indent=2)
    if isinstance(model, ResidualLocalizerV4):
        saved_feature_mean = model.feature_mean.detach().cpu().numpy()
        saved_feature_std = model.feature_std.detach().cpu().numpy()
    else:
        saved_feature_mean, saved_feature_std = feature_mean, feature_std
    saved_pose_mean = model.pose_mean.detach().cpu().numpy()
    saved_pose_std = model.pose_std.detach().cpu().numpy()
    np.save(checkpoint_dir / "feature_mean.npy", saved_feature_mean)
    np.save(checkpoint_dir / "feature_std.npy", saved_feature_std)
    np.save(checkpoint_dir / "pose_mean.npy", saved_pose_mean)
    np.save(checkpoint_dir / "pose_std.npy", saved_pose_std)

    history = []
    best_score = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        if sampler_weights_t is None:
            order = torch.randperm(len(train_x), generator=generator, device=device)
        else:
            order = torch.multinomial(
                sampler_weights_t, len(train_x), replacement=True, generator=generator,
            )
        total = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x = augment_packed(
                train_x[indices], int(candidate["window_size"]), gain_drift, noise_std_v,
            )
            optimizer.zero_grad()
            pred_norm = model(x, train_generation[indices], return_normalized=True)
            loss, _, _ = pose_loss_v3(
                pred_norm, target_norm[indices], model.pose_mean, model.pose_std,
                position_weight, orientation_weight,
            )
            loss = loss + adapter_weight * model.adapter_regularization()
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            total += float(loss.item()) * len(indices)

        row = {
            "epoch": epoch,
            "train_loss": total / len(train_x),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if val_data is not None:
            model.eval()
            with torch.inference_mode():
                pred_norm = model(val_x, val_generation, return_normalized=True)
                val_loss, pos_loss, ori_loss = pose_loss_v3(
                    pred_norm, val_target_norm, model.pose_mean, model.pose_std,
                    position_weight, orientation_weight,
                )
                pred = model(val_x, val_generation, return_normalized=False).cpu().numpy()
            metrics = pose_metrics_v3(pred, val_data["target"])
            score = composite_score(metrics)
            scheduler.step(score)
            row.update({
                "val_loss": float(val_loss.item()),
                "val_position_loss": float(pos_loss.item()),
                "val_orientation_loss_rad2": float(ori_loss.item()),
                "selection_composite": score,
                **metrics,
            })
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            if val_data is None:
                print(f"Epoch {epoch:03d} train={row['train_loss']:.6f}", flush=True)
            else:
                print(
                    f"Epoch {epoch:03d} train={row['train_loss']:.6f} "
                    f"score={row['selection_composite']:.6f} "
                    f"pos={metrics['position_axis_rmse_mm']:.3f}mm "
                    f"so3={metrics['orientation_geodesic_rmse_deg']:.3f}deg",
                    flush=True,
                )
        if val_data is not None:
            if row["selection_composite"] < best_score - 1e-12:
                best_score = row["selection_composite"]
                stale = 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
                with open(checkpoint_dir / "best_validation_metrics.json", "w") as stream:
                    json.dump(row, stream, indent=2)
            else:
                stale += 1
            if stale >= early_patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break

    if val_data is None:
        torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        with open(checkpoint_dir / "fixed_epoch_training_summary.json", "w") as stream:
            json.dump(history[-1], stream, indent=2)
    with open(checkpoint_dir / "training_history.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved v4 checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()

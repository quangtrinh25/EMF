"""Add fold-gated physics-generated causal windows to a prepared real v4 fold."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.prepare_multigeneration_v4 import sha256_file
from physics.channel_correction import apply_channel_correction
from physics.forward_model import forward_model
from physics.rotation_repr import pose_deg_to_target_v3


def synthesize_windows(count, window_size, physics_config, seed, max_position_step, max_angle_step):
    rng = np.random.default_rng(seed)
    workspace = physics_config["workspace"]
    position_low = np.asarray([workspace["x_mm"][0], workspace["y_mm"][0], workspace["z_mm"][0]])
    position_high = np.asarray([workspace["x_mm"][1], workspace["y_mm"][1], workspace["z_mm"][1]])
    orientation = workspace["orientation_deg"]
    angle_low = np.asarray([orientation["roll"][0], orientation["pitch"][0], orientation["yaw"][0]])
    angle_high = np.asarray([orientation["roll"][1], orientation["pitch"][1], orientation["yaw"][1]])
    current_position = rng.uniform(position_low, position_high, size=(count, 3))
    current_angle = rng.uniform(angle_low, angle_high, size=(count, 3))
    position_step = rng.uniform(-max_position_step, max_position_step, size=(count, 3))
    angle_step = rng.uniform(-max_angle_step, max_angle_step, size=(count, 3))
    offsets = np.arange(window_size - 1, -1, -1, dtype=float)[None, :, None]
    positions = np.clip(
        current_position[:, None, :] - offsets * position_step[:, None, :],
        position_low, position_high,
    )
    angles = np.clip(
        current_angle[:, None, :] - offsets * angle_step[:, None, :],
        angle_low, angle_high,
    )
    poses = np.concatenate([positions, angles], axis=2)
    flat_pose = poses.reshape(-1, 6)
    emf = forward_model(flat_pose, physics_config["tx"], physics_config["rx"])
    if "channel_correction" in physics_config["tx"]:
        emf = apply_channel_correction(emf, physics_config["tx"])
    emf = np.maximum(emf, 0.0).reshape(count, window_size, 9).astype(np.float32)
    dt_ratio = np.ones((count, window_size - 1), dtype=np.float32)
    packed = np.concatenate([emf.reshape(count, -1), dt_ratio], axis=1).astype(np.float32)
    current_pose = poses[:, -1].astype(np.float32)
    return packed, pose_deg_to_target_v3(current_pose), current_pose


def append_train_arrays(real, synthetic):
    count = len(synthetic["target"])
    result = {}
    for key, values in real.items():
        if key in synthetic:
            extra = synthetic[key]
        elif values.ndim == 1:
            if values.dtype.kind in {"U", "S", "O"}:
                extra = np.asarray(["physics_synthetic"] * count, dtype=values.dtype)
            elif values.dtype == bool:
                extra = np.zeros(count, dtype=bool)
            else:
                extra = np.full(count, -1, dtype=values.dtype)
        else:
            raise ValueError(f"No synthetic value rule for multidimensional array {key!r}")
        result[key] = np.concatenate([values, extra], axis=0)
    return result


def select_stage1_training_arrays(real, synthetic, train_mode):
    """Choose mixed or paper-order synthetic-only Stage-1 training arrays."""
    if train_mode == "mixed":
        return append_train_arrays(real, synthetic)
    if train_mode != "synthetic_only":
        raise ValueError(f"Unsupported train_mode {train_mode!r}")
    missing = set(real) - set(synthetic)
    if missing:
        raise ValueError(f"Synthetic-only arrays are missing keys {sorted(missing)}")
    return {key: synthetic[key] for key in real}


def main():
    parser = argparse.ArgumentParser(description="Build a real+physics-synthetic v4 training fold.")
    parser.add_argument("--real_data_dir", required=True)
    parser.add_argument("--physics_config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--synthetic_ratio", type=float, default=1.0)
    parser.add_argument(
        "--train_mode", choices=("mixed", "synthetic_only"), default="mixed",
        help=(
            "Stage-1 composition. 'mixed' preserves the existing real+synthetic "
            "protocol; 'synthetic_only' supports synthetic pretraining followed "
            "by a separately prepared real-only calibration/fine-tune stage."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_position_step_mm", type=float, default=2.0)
    parser.add_argument("--max_angle_step_deg", type=float, default=1.0)
    args = parser.parse_args()
    if args.synthetic_ratio <= 0:
        raise ValueError("synthetic_ratio must be positive")
    real_dir = Path(args.real_data_dir)
    with open(real_dir / "protocol.json") as stream:
        protocol = json.load(stream)
    if protocol.get("sealed_test_opened") is not False or (real_dir / "sealed_test.npz").exists():
        raise RuntimeError("Hybrid builder refuses any directory where test data was materialized")
    with open(args.physics_config) as stream:
        physics_config = yaml.safe_load(stream)
    gate = physics_config.get("calibration_gate", {})
    if gate.get("synthetic_allowed") is not True:
        raise RuntimeError("Fold physics config did not pass held-out calibration validation")
    with np.load(real_dir / "train.npz") as loaded:
        real_train = {key: loaded[key] for key in loaded.files}
    window_size = int(protocol["window_size"])
    synthetic_count = max(1, int(round(len(real_train["target"]) * args.synthetic_ratio)))
    packed, target, pose = synthesize_windows(
        synthetic_count, window_size, physics_config, args.seed,
        args.max_position_step_mm, args.max_angle_step_deg,
    )
    generation_to_index = dict(protocol["generation_to_index"])
    if "physics_synthetic" in generation_to_index:
        raise ValueError("Input protocol already contains physics_synthetic")
    synthetic_index = len(generation_to_index)
    generation_to_index["physics_synthetic"] = synthetic_index
    synthetic = {
        "emf": packed,
        "target": target,
        "pose_deg": pose,
        "generation_index": np.full(synthetic_count, synthetic_index, dtype=np.int64),
        "generation": np.asarray(["physics_synthetic"] * synthetic_count),
        "family": np.asarray(["physics_random_walk"] * synthetic_count),
        "session": np.zeros(synthetic_count, dtype=np.int64),
        "session_key": np.asarray(["physics_synthetic/pool/0"] * synthetic_count),
        "source_file": np.asarray([Path(args.physics_config).name] * synthetic_count),
        "source_row": np.arange(synthetic_count, dtype=np.int64),
        "segment_id": np.arange(synthetic_count, dtype=np.int64),
        "warmup": np.zeros(synthetic_count, dtype=bool),
    }
    hybrid = select_stage1_training_arrays(real_train, synthetic, args.train_mode)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(hybrid["target"]))
    hybrid = {key: values[order] for key, values in hybrid.items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "train.npz", **hybrid)
    if (real_dir / "val.npz").exists():
        shutil.copy2(real_dir / "val.npz", out_dir / "val.npz")
    protocol.update({
        "generation_to_index": generation_to_index,
        "train_dt_median": {**protocol["train_dt_median"], "physics_synthetic": 1.0},
        "generation_time_mode": {**protocol.get("generation_time_mode", {}), "physics_synthetic": "fixed_row_delta"},
        "generation_fixed_dt_seconds": {**protocol.get("generation_fixed_dt_seconds", {}), "physics_synthetic": None},
        "hybrid_training": {
            "enabled": True,
            "stage1_train_mode": args.train_mode,
            "real_samples_available": int(len(real_train["target"])),
            "real_samples_in_stage1": (
                int(len(real_train["target"])) if args.train_mode == "mixed" else 0
            ),
            "synthetic_samples": synthetic_count,
            "synthetic_ratio": args.synthetic_ratio,
            "physics_config": str(args.physics_config),
            "physics_config_sha256": sha256_file(args.physics_config),
            "calibration_gate": gate,
            "synthetic_window_policy": "random_current_pose_plus_constant_bounded_past_velocity",
            "max_position_step_mm": args.max_position_step_mm,
            "max_angle_step_deg": args.max_angle_step_deg,
            "test_labels_loaded": False,
            "deployment_train_only": not (real_dir / "val.npz").exists(),
            "paper_order_analogue": (
                "calibrated_physics_synthetic_pretrain_then_real_only_neural_calibration"
                if args.train_mode == "synthetic_only"
                else None
            ),
        },
    })
    with open(out_dir / "protocol.json", "w") as stream:
        json.dump(protocol, stream, indent=2)
    shutil.copy2(args.physics_config, out_dir / "physics_generator.yaml")
    print(json.dumps(protocol["hybrid_training"], indent=2))


if __name__ == "__main__":
    main()

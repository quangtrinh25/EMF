"""Combine the selected v3.2 prediction with a complementary hybrid v4 model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.rotation_repr import mean_rotation_6d, pose_deg_to_target_v3, target_v3_to_pose_deg
from training.metrics import pose_metrics_v3


TARGET_COLUMNS = [
    "target_X_mm", "target_Y_mm", "target_Z_mm",
    "target_roll_deg", "target_pitch_deg", "target_yaw_deg",
]
PRED_COLUMNS = [
    "pred_X_mm", "pred_Y_mm", "pred_Z_mm",
    "pred_roll_deg", "pred_pitch_deg", "pred_yaw_deg",
]


def normalized_frame(path):
    frame = pd.read_csv(path).copy()
    if "source_row" not in frame and "target_source_row" in frame:
        frame["source_row"] = frame["target_source_row"]
    required = {"source_file", "source_row", *TARGET_COLUMNS, *PRED_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def blend_targets(v32_target, hybrid_target, position_weight, orientation_weight, orientation_rule):
    position = (
        (1.0 - position_weight) * v32_target[:, :3]
        + position_weight * hybrid_target[:, :3]
    )
    if orientation_rule == "chordal":
        if abs(orientation_weight - 0.5) > 1e-12:
            raise ValueError("Weighted orientation is supported only by rotation6d_mean")
        orientation = mean_rotation_6d(np.stack([v32_target[:, 3:], hybrid_target[:, 3:]]))
    else:
        orientation = (
            (1.0 - orientation_weight) * v32_target[:, 3:]
            + orientation_weight * hybrid_target[:, 3:]
        )
    return np.column_stack([position, orientation]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen v3.2 + hybrid v4 ensemble rule.")
    parser.add_argument("--v32_predictions", required=True)
    parser.add_argument("--hybrid_predictions", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hybrid_position_weight", type=float, default=0.3)
    parser.add_argument(
        "--hybrid_orientation_weight", type=float, default=0.5,
        help=(
            "Hybrid contribution to the rotation-6D mean. Default 0.5 preserves "
            "the original equal-orientation rule."
        ),
    )
    parser.add_argument(
        "--orientation_rule", choices=["chordal", "rotation6d_mean"], default="chordal",
        help="rotation6d_mean is a fast runtime-friendly two-model blend.",
    )
    parser.add_argument("--split", choices=["development_only", "sealed_test"], default="development_only")
    parser.add_argument("--frozen_manifest", default=None)
    parser.add_argument(
        "--rule_selected_on",
        default="development_cv_screening_then_seed_7_123_robustness",
    )
    args = parser.parse_args()
    if not 0.0 <= args.hybrid_position_weight <= 1.0:
        raise ValueError("hybrid_position_weight must be in [0,1]")
    if not 0.0 <= args.hybrid_orientation_weight <= 1.0:
        raise ValueError("hybrid_orientation_weight must be in [0,1]")
    if args.orientation_rule == "chordal" and abs(args.hybrid_orientation_weight - 0.5) > 1e-12:
        raise ValueError("Weighted orientation is supported only by rotation6d_mean")
    if args.split == "sealed_test":
        if args.frozen_manifest is None:
            raise RuntimeError("Sealed ensemble requires the frozen manifest")
        with open(args.frozen_manifest) as stream:
            frozen = json.load(stream)
        bundle = frozen.get("bundle", {})
        expected_orientation = (
            "equal_chordal_SO3_mean_v3_2_hybrid_v4"
            if args.orientation_rule == "chordal"
            else "equal_rotation6d_mean_then_gram_schmidt"
        )
        if frozen.get("frozen") is not True or frozen.get("test_opened_at_freeze") is not False:
            raise RuntimeError("Invalid frozen manifest for sealed ensemble")
        if abs(float(bundle.get("hybrid_position_weight", -1)) - args.hybrid_position_weight) > 1e-12:
            raise RuntimeError("Sealed ensemble weight differs from the frozen rule")
        if bundle.get("orientation_rule") != expected_orientation:
            raise RuntimeError("Sealed ensemble orientation differs from the frozen rule")
        frozen_orientation_weight = float(bundle.get("hybrid_orientation_weight", 0.5))
        if abs(frozen_orientation_weight - args.hybrid_orientation_weight) > 1e-12:
            raise RuntimeError("Sealed ensemble orientation weight differs from the frozen rule")
        receipt = Path(args.frozen_manifest).parent / "sealed_open_receipt.json"
        if not receipt.exists():
            raise RuntimeError("Sealed-open receipt is missing")
    v32 = normalized_frame(args.v32_predictions)
    hybrid = normalized_frame(args.hybrid_predictions)
    key = ["source_file", "source_row"]
    merged = v32.merge(hybrid, on=key, suffixes=("_v32", "_hybrid"), validate="one_to_one")
    if len(merged) != len(v32) or len(merged) != len(hybrid):
        raise RuntimeError("Prediction files do not cover the same rows")
    target_pose = merged[[f"{name}_v32" for name in TARGET_COLUMNS]].to_numpy()
    hybrid_target_pose = merged[[f"{name}_hybrid" for name in TARGET_COLUMNS]].to_numpy()
    np.testing.assert_allclose(target_pose, hybrid_target_pose, atol=1e-4)
    v32_pose = merged[[f"{name}_v32" for name in PRED_COLUMNS]].to_numpy()
    hybrid_pose = merged[[f"{name}_hybrid" for name in PRED_COLUMNS]].to_numpy()
    target = pose_deg_to_target_v3(target_pose)
    v32_target = pose_deg_to_target_v3(v32_pose)
    hybrid_target = pose_deg_to_target_v3(hybrid_pose)
    alpha = args.hybrid_position_weight
    beta = args.hybrid_orientation_weight
    pred = blend_targets(
        v32_target, hybrid_target, alpha, beta, args.orientation_rule,
    )
    metrics = pose_metrics_v3(pred, target)
    pred_pose = target_v3_to_pose_deg(pred)
    generation_column = next(
        (name for name in ("generation", "generation_hybrid", "generation_v32") if name in merged),
        None,
    )
    session_column = next(
        (name for name in ("session_key", "session_key_hybrid", "session_key_v32") if name in merged),
        None,
    )
    if "session_key_hybrid" in merged and "session_key_v32" in merged:
        if not np.array_equal(
            merged["session_key_hybrid"].astype(str).to_numpy(),
            merged["session_key_v32"].astype(str).to_numpy(),
        ):
            raise RuntimeError("v3.2 and hybrid session keys differ")
    output = pd.DataFrame({
        "generation": (
            merged[generation_column].astype(str)
            if generation_column else pd.Series(["real_current"] * len(merged))
        ),
        "session_key": (
            merged[session_column].astype(str)
            if session_column else pd.Series(["unknown"] * len(merged))
        ),
        "source_file": merged["source_file"],
        "source_row": merged["source_row"],
        **{name: target_pose[:, index] for index, name in enumerate(TARGET_COLUMNS)},
        **{name: pred_pose[:, index] for index, name in enumerate(PRED_COLUMNS)},
    })
    sessions = {}
    for session in sorted(set(output["session_key"].astype(str))):
        mask = output["session_key"].astype(str).to_numpy() == session
        sessions[session] = pose_metrics_v3(pred[mask], target[mask])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = "val_predictions.csv" if args.split == "development_only" else "sealed_test_predictions.csv"
    output.to_csv(out_dir / output_name, index=False)
    summary = {
        "schema_version": 4,
        "split": args.split,
        "rule": {
            "position": f"{1-alpha:.6f}*v3_2 + {alpha:.6f}*hybrid_v4",
            "orientation": (
                "equal_chordal_SO3_mean_v3_2_hybrid_v4"
                if args.orientation_rule == "chordal"
                else (
                    f"{1-beta:.6f}*v3_2_rotation6d + "
                    f"{beta:.6f}*hybrid_v4_rotation6d_then_gram_schmidt"
                )
            ),
            "hybrid_orientation_weight": beta,
            "rule_selected_on": args.rule_selected_on,
        },
        "aggregate": metrics,
        "sessions": sessions,
        "test_opened": args.split == "sealed_test",
    }
    with open(out_dir / "summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

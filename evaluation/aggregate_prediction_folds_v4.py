"""Aggregate row-level pose predictions across development folds.

The output intentionally matches the metric portion consumed by
``quality_gate_v4.py``.  Aggregation happens from row-level errors rather than
by averaging fold summaries, which is important for pooled p95 values.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.rotation_repr import pose_deg_to_target_v3
from training.metrics import pose_metrics_v3


TARGET_COLUMNS = [
    "target_X_mm", "target_Y_mm", "target_Z_mm",
    "target_roll_deg", "target_pitch_deg", "target_yaw_deg",
]
PRED_COLUMNS = [
    "pred_X_mm", "pred_Y_mm", "pred_Z_mm",
    "pred_roll_deg", "pred_pitch_deg", "pred_yaw_deg",
]


def metrics_for(frame):
    target = pose_deg_to_target_v3(frame[TARGET_COLUMNS].to_numpy())
    prediction = pose_deg_to_target_v3(frame[PRED_COLUMNS].to_numpy())
    return pose_metrics_v3(prediction, target)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate row-level v4 development predictions across folds."
    )
    parser.add_argument("--prediction_files", nargs="+", required=True)
    parser.add_argument(
        "--session_labels", nargs="+",
        help="Optional stable labels, one per prediction file.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--model_label", default="candidate")
    args = parser.parse_args()
    if args.session_labels and len(args.session_labels) != len(args.prediction_files):
        raise ValueError("--session_labels must contain one label per prediction file")

    frames = []
    sessions = {}
    sources = []
    for index, raw_path in enumerate(args.prediction_files):
        path = Path(raw_path)
        frame = pd.read_csv(path)
        missing = set(TARGET_COLUMNS + PRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if args.session_labels:
            label = args.session_labels[index]
        elif "session_key" in frame and frame["session_key"].nunique() == 1:
            label = str(frame["session_key"].iloc[0])
        else:
            label = path.parent.name
        if label in sessions:
            raise ValueError(f"Duplicate session label: {label}")
        session_metrics = metrics_for(frame)
        sessions[label] = session_metrics
        frames.append(frame)
        sources.append({"session": label, "path": str(path), "num_samples": len(frame)})

    pooled = pd.concat(frames, ignore_index=True)
    result = {
        "schema_version": 4,
        "split": "development_cv",
        "model_label": args.model_label,
        "aggregate": metrics_for(pooled),
        "sessions": sessions,
        "sources": sources,
        "test_opened": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

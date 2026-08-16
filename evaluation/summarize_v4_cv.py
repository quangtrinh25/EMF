"""Pool disjoint v4 fold predictions into a cross-generation CV summary."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.rotation_repr import pose_deg_to_target_v3
from training.metrics import pose_metrics_v3


POSE = ["X", "Y", "Z", "roll", "pitch", "yaw"]


def main():
    parser = argparse.ArgumentParser(description="Summarize disjoint v4 CV folds.")
    parser.add_argument("--prediction_csvs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.prediction_csvs]
    all_rows = pd.concat(frames, ignore_index=True)
    identity = all_rows[["generation", "session_key", "source_file", "source_row"]]
    if identity.duplicated().any():
        raise RuntimeError("CV prediction inputs overlap; pooled summary would leak/reweight samples")
    target_pose = all_rows[[f"target_{name}_mm" if name in {"X", "Y", "Z"} else f"target_{name}_deg" for name in POSE]].to_numpy()
    pred_pose = all_rows[[f"pred_{name}_mm" if name in {"X", "Y", "Z"} else f"pred_{name}_deg" for name in POSE]].to_numpy()
    target = pose_deg_to_target_v3(target_pose)
    pred = pose_deg_to_target_v3(pred_pose)
    aggregate = pose_metrics_v3(pred, target)
    sessions = {}
    for key in sorted(all_rows["session_key"].unique()):
        mask = all_rows["session_key"].to_numpy() == key
        sessions[key] = pose_metrics_v3(pred[mask], target[mask])
    summary = {
        "schema_version": 4,
        "split": "cross_generation_disjoint_cv",
        "aggregate": aggregate,
        "sessions": sessions,
        "prediction_csvs": args.prediction_csvs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()

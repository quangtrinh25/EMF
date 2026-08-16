"""Check CPU/GPU and absolute-timestamp parity on a non-test real session."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.prepare_multigeneration_v4 import discover_sources, load_source
from inference.realtime_predictor_hybrid_v4 import RealtimeHybridPosePredictorV4


def predict_rows(predictor, emf, timestamp_offset):
    rows = []
    for index, values in enumerate(emf):
        response = predictor.predict(
            timestamp_offset + index * 1_000_000, index == 0, values,
        )
        rows.append(list(response["pose"].values()))
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser(description="Check selected hybrid runtime parity.")
    parser.add_argument("--protocol_config", default="configs/new_calib_v4.yaml")
    parser.add_argument("--source_file", default="conrot_data.csv")
    parser.add_argument("--v32_manifest", default="checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json")
    parser.add_argument("--hybrid_checkpoint_dir", required=True)
    parser.add_argument("--hybrid_position_weight", type=float, default=0.20)
    parser.add_argument("--hybrid_orientation_weight", type=float, default=0.50)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.protocol_config) as stream:
        protocol = yaml.safe_load(stream)
    matches = [
        item for item in discover_sources(protocol)
        if item["source_file"] == args.source_file
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one configured source named {args.source_file!r}")
    descriptor = matches[0]
    if descriptor["locked"] or descriptor["sealed"]:
        raise RuntimeError("Parity check refuses locked/sealed data")
    source = load_source(descriptor)
    emf = source["emf"][:args.rows]
    cpu = RealtimeHybridPosePredictorV4(
        args.v32_manifest, args.hybrid_checkpoint_dir,
        hybrid_position_weight=args.hybrid_position_weight,
        hybrid_orientation_weight=args.hybrid_orientation_weight,
        device="cpu", backend="torchscript", input_unit="V", cpu_threads=6,
    )
    cpu_pose = predict_rows(cpu, emf, 1_000_000_000)
    shifted_cpu_pose = predict_rows(cpu, emf, 10_000_000_000)
    timestamp_max_abs = float(np.max(np.abs(cpu_pose - shifted_cpu_pose)))
    gpu_pose = None
    if torch.cuda.is_available():
        gpu = RealtimeHybridPosePredictorV4(
            args.v32_manifest, args.hybrid_checkpoint_dir,
            hybrid_position_weight=args.hybrid_position_weight,
            hybrid_orientation_weight=args.hybrid_orientation_weight,
            device="cuda", backend="cuda_graph", input_unit="V",
        )
        gpu_pose = predict_rows(gpu, emf, 1_000_000_000)
    difference = None if gpu_pose is None else np.abs(cpu_pose - gpu_pose)
    result = {
        "schema_version": 4,
        "source_file": args.source_file,
        "source_is_locked_or_sealed": False,
        "rows": int(len(emf)),
        "absolute_timestamp_feature_invariance_max_abs_pose_difference": timestamp_max_abs,
        "cpu_gpu_max_abs_pose_difference": (
            None if difference is None else float(np.max(difference))
        ),
        "cpu_gpu_mean_abs_pose_difference": (
            None if difference is None else float(np.mean(difference))
        ),
        "gpu_available": bool(torch.cuda.is_available()),
        "hybrid_position_weight": args.hybrid_position_weight,
        "hybrid_orientation_weight": args.hybrid_orientation_weight,
        "orientation_rule": "weighted_rotation6d_then_gram_schmidt",
        "accepted": (
            timestamp_max_abs <= 1e-7
            and difference is not None
            and float(np.max(difference)) <= 1e-3
        ),
        "test_opened": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise RuntimeError("Hybrid runtime parity check failed")


if __name__ == "__main__":
    main()

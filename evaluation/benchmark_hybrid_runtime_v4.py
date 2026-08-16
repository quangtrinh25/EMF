"""Benchmark the complete selected v3.2.2 + hybrid-v4 realtime predictor."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.realtime_predictor_hybrid_v4 import RealtimeHybridPosePredictorV4


def benchmark(args, device, backend):
    predictor = RealtimeHybridPosePredictorV4(
        args.v32_manifest,
        args.hybrid_checkpoint_dir,
        hybrid_position_weight=args.hybrid_position_weight,
        hybrid_orientation_weight=args.hybrid_orientation_weight,
        generation=args.generation,
        device=device,
        backend=backend,
        input_unit="V",
        cpu_threads=args.cpu_threads if device == "cpu" else None,
    )
    emf = np.full(9, 0.05, dtype=np.float32)
    timestamp = 1_000_000_000
    for index in range(args.warmup + 3):
        predictor.predict(timestamp + index * 1_000_000, index == 0, emf)
    latencies = []
    start = args.warmup + 3
    for index in range(args.runs):
        t0 = time.perf_counter_ns()
        predictor.predict(timestamp + (start + index) * 1_000_000, False, emf)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "device": device,
        "backend": predictor.backend,
        "runs": args.runs,
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "scope": "feature_build_plus_six_networks_plus_blend_uncertainty_and_device_transfer",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark the complete hybrid-v4 runtime.")
    parser.add_argument("--v32_manifest", default="checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json")
    parser.add_argument("--hybrid_checkpoint_dir", required=True)
    parser.add_argument("--hybrid_position_weight", type=float, default=0.20)
    parser.add_argument("--hybrid_orientation_weight", type=float, default=0.50)
    parser.add_argument("--generation", default="real_current")
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--cpu_threads", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cpu = benchmark(args, "cpu", "torchscript")
    gpu = benchmark(args, "cuda", "cuda_graph") if torch.cuda.is_available() else None
    result = {
        "schema_version": 4,
        "cpu_p95_ms": cpu["p95_ms"],
        "gpu_p95_ms": None if gpu is None else gpu["p95_ms"],
        "cpu": cpu,
        "gpu": gpu,
        "gpu_available": bool(torch.cuda.is_available()),
        "hybrid_position_weight": args.hybrid_position_weight,
        "hybrid_orientation_weight": args.hybrid_orientation_weight,
        "orientation_rule": "weighted_rotation6d_then_gram_schmidt",
        "test_opened": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Benchmark production v4 predict latency on CPU and, when visible, CUDA."""

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

from inference.realtime_predictor_v4 import RealtimePosePredictorV4


def benchmark(checkpoints, generation, device, backend, runs, warmup):
    predictor = RealtimePosePredictorV4(
        checkpoints, generation=generation, device=device, backend=backend, input_unit="V",
    )
    median_dt = predictor.median_dt
    emf = np.full(9, 0.01, dtype=np.float32)
    timestamp = 1_000_000_000
    step_ns = max(1, int(median_dt * 1e9))
    for index in range(predictor.window_size + warmup):
        predictor.predict(timestamp + index * step_ns, index == 0, emf)
    latencies = []
    start_index = predictor.window_size + warmup
    for index in range(runs):
        t0 = time.perf_counter_ns()
        predictor.predict(timestamp + (start_index + index) * step_ns, False, emf)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "device": device,
        "backend": predictor.backend,
        "runs": runs,
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "includes_feature_build_uncertainty_and_device_transfer": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark v4 realtime CPU/GPU paths.")
    parser.add_argument("--checkpoint_dirs", nargs="+", required=True)
    parser.add_argument("--generation", default="new_2026_08_03")
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cpu = benchmark(args.checkpoint_dirs, args.generation, "cpu", "torchscript", args.runs, args.warmup)
    gpu = None
    if torch.cuda.is_available():
        gpu = benchmark(args.checkpoint_dirs, args.generation, "cuda", "cuda_graph", args.runs, args.warmup)
    result = {
        "cpu_p95_ms": cpu["p95_ms"],
        "gpu_p95_ms": None if gpu is None else gpu["p95_ms"],
        "cpu": cpu,
        "gpu": gpu,
        "gpu_available": torch.cuda.is_available(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Apply the predeclared balanced four-metric v4 selection gate."""

import argparse
import json
from pathlib import Path

import numpy as np


PRIMARY = (
    "position_axis_rmse_mm",
    "position_euclidean_p95_mm",
    "orientation_geodesic_rmse_deg",
    "orientation_geodesic_p95_deg",
)
P95 = ("position_euclidean_p95_mm", "orientation_geodesic_p95_deg")


def load_json(path):
    with open(path) as stream:
        return json.load(stream)


def latency_value(latency, key):
    value = latency.get(key)
    return float("inf") if value is None else float(value)


def main():
    parser = argparse.ArgumentParser(description="Gate a v4 candidate against a matched-fold baseline.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum_composite_improvement", type=float, default=0.03)
    parser.add_argument("--maximum_session_p95_regression", type=float, default=0.05)
    parser.add_argument("--maximum_gpu_p95_ms", type=float, default=1.5)
    parser.add_argument("--maximum_cpu_p95_ms", type=float, default=3.0)
    args = parser.parse_args()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    latency = load_json(args.latency)
    base_sessions = baseline["sessions"]
    cand_sessions = candidate["sessions"]
    if set(base_sessions) != set(cand_sessions):
        raise ValueError("Baseline and candidate do not cover the same development sessions")
    improvements = {
        key: 1.0 - float(candidate["aggregate"][key]) / float(baseline["aggregate"][key])
        for key in PRIMARY
    }
    ratios = np.asarray([
        float(candidate["aggregate"][key]) / float(baseline["aggregate"][key]) for key in PRIMARY
    ])
    composite_improvement = float(1.0 - np.exp(np.mean(np.log(ratios))))
    session_regressions = {}
    for session in base_sessions:
        for metric in P95:
            regression = float(cand_sessions[session][metric]) / float(base_sessions[session][metric]) - 1.0
            session_regressions[f"{session}:{metric}"] = regression
    checks = {
        "all_four_aggregate_metrics_improve": all(value > 0.0 for value in improvements.values()),
        "composite_improvement_at_least_threshold": composite_improvement >= args.minimum_composite_improvement,
        "no_session_p95_regression_over_threshold": max(session_regressions.values(), default=0.0) <= args.maximum_session_p95_regression,
        "cpu_latency_within_budget": latency_value(latency, "cpu_p95_ms") <= args.maximum_cpu_p95_ms,
        "gpu_latency_within_budget": latency_value(latency, "gpu_p95_ms") <= args.maximum_gpu_p95_ms,
    }
    result = {
        "schema_version": 4,
        "accepted": all(checks.values()),
        "checks": checks,
        "metric_improvement_fraction": improvements,
        "composite_improvement_fraction": composite_improvement,
        "session_p95_regression_fraction": session_regressions,
        "latency": latency,
        "decision": "select_candidate" if all(checks.values()) else "retain_v3_2_2_baseline",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Summarize the development-only synthetic/calibration ratio ablation.

This script intentionally reads only grouped ``con_rot`` development summaries
and their predeclared quality-gate decisions.  It must never read, score, or
select against the already-opened ``cyl_rot`` final labels.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emf_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("position_axis_rmse_mm", "Position axis RMSE (mm)"),
    ("position_euclidean_p95_mm", "Position Euclidean p95 (mm)"),
    ("orientation_geodesic_rmse_deg", "SO(3) RMSE (deg)"),
    ("orientation_geodesic_p95_deg", "SO(3) p95 (deg)"),
)


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(rows, columns):
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *body]) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe v4 synthetic-ratio development ablation."
    )
    parser.add_argument(
        "--baseline",
        default="results/new_calib_v4_hybrid/ensemble/cv_baseline_v32.json",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=4,
        metavar=("RATIO", "SUMMARY", "GATE", "LABEL"),
        help="May be repeated. RATIO is synthetic rows / real rows.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/synthetic_ratio_ablation_2026_08_04",
    )
    args = parser.parse_args()

    candidates = args.candidate or [
        (
            "0.25",
            "results/new_calib_v4_hybrid_ratio025/cv_blend_w20_seed42.json",
            "results/new_calib_v4_hybrid_ratio025/quality_gate.json",
            "C3 staged + frozen w20",
        ),
        (
            "0.50",
            "results/new_calib_v4_hybrid_ratio050/cv_blend_w20_seed42.json",
            "results/new_calib_v4_hybrid_ratio050/quality_gate.json",
            "C3 staged + frozen w20",
        ),
        (
            "1.00",
            "results/new_calib_v4_hybrid/ensemble/rotation6d_mean/seed42/w20/cv_summary.json",
            "results/new_calib_v4_hybrid/ensemble/rotation6d_mean/seed42/w20/quality_gate.json",
            "C3 staged + frozen w20",
        ),
    ]

    baseline_path = Path(args.baseline)
    baseline = read_json(baseline_path)
    if baseline.get("test_opened") is not False:
        raise ValueError("Baseline summary must explicitly be development-only")

    rows = []
    sources = {str(baseline_path): sha256_file(baseline_path)}
    baseline_row = {
        "model": "v3.2.2 development baseline",
        "synthetic_ratio": 0.0,
        "position_axis_rmse_mm": float(baseline["aggregate"]["position_axis_rmse_mm"]),
        "position_euclidean_p95_mm": float(
            baseline["aggregate"]["position_euclidean_p95_mm"]
        ),
        "orientation_geodesic_rmse_deg": float(
            baseline["aggregate"]["orientation_geodesic_rmse_deg"]
        ),
        "orientation_geodesic_p95_deg": float(
            baseline["aggregate"]["orientation_geodesic_p95_deg"]
        ),
        "composite_improvement_pct": 0.0,
        "worst_session_p95_regression_pct": 0.0,
        "development_gate": "reference",
        "final_use": "selected deployment",
    }
    rows.append(baseline_row)

    for ratio_text, summary_text, gate_text, label in candidates:
        summary_path = Path(summary_text)
        gate_path = Path(gate_text)
        summary = read_json(summary_path)
        gate = read_json(gate_path)
        if summary.get("test_opened") is not False:
            raise ValueError(f"{summary_path} is not explicitly development-only")
        if gate["latency"].get("test_opened") is not False:
            raise ValueError(f"{gate_path} latency is not explicitly development-only")
        ratio = float(ratio_text)
        final_use = (
            "historical final rejected; do not reopen"
            if np.isclose(ratio, 1.0)
            else "not evaluated on final; new sealed data required"
        )
        rows.append({
            "model": label,
            "synthetic_ratio": ratio,
            "position_axis_rmse_mm": float(summary["aggregate"]["position_axis_rmse_mm"]),
            "position_euclidean_p95_mm": float(
                summary["aggregate"]["position_euclidean_p95_mm"]
            ),
            "orientation_geodesic_rmse_deg": float(
                summary["aggregate"]["orientation_geodesic_rmse_deg"]
            ),
            "orientation_geodesic_p95_deg": float(
                summary["aggregate"]["orientation_geodesic_p95_deg"]
            ),
            "composite_improvement_pct": 100.0
            * float(gate["composite_improvement_fraction"]),
            "worst_session_p95_regression_pct": 100.0
            * max(gate["session_p95_regression_fraction"].values()),
            "development_gate": "accepted" if gate["accepted"] else "rejected",
            "final_use": final_use,
        })
        sources[str(summary_path)] = sha256_file(summary_path)
        sources[str(gate_path)] = sha256_file(gate_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "synthetic_ratio_ablation_seed42.csv"
    json_path = out_dir / "synthetic_ratio_ablation_seed42.json"
    md_path = out_dir / "synthetic_ratio_ablation_seed42.md"
    png_path = out_dir / "synthetic_ratio_ablation_seed42.png"
    pdf_path = out_dir / "synthetic_ratio_ablation_seed42.pdf"
    report_path = out_dir / "REPORT.md"

    columns = [
        ("model", "Model"),
        ("synthetic_ratio", "Synthetic / real"),
        ("position_axis_rmse_mm", "Position RMSE (mm)"),
        ("position_euclidean_p95_mm", "Position p95 (mm)"),
        ("orientation_geodesic_rmse_deg", "SO(3) RMSE (deg)"),
        ("orientation_geodesic_p95_deg", "SO(3) p95 (deg)"),
        ("composite_improvement_pct", "Composite gain vs v3.2.2 (%)"),
        ("worst_session_p95_regression_pct", "Worst session p95 regression (%)"),
        ("development_gate", "Development gate"),
        ("final_use", "Final status"),
    ]
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key, _ in columns])
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_table(rows, columns))

    ratios = np.asarray([row["synthetic_ratio"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.5), sharex=True)
    for ax, (metric, label) in zip(axes.flat, METRICS):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        ax.plot(ratios, values, color="#295BFF", marker="o", lw=1.5)
        ax.axhline(values[0], color="#5E6572", ls="--", lw=1.0, label="v3.2.2 baseline")
        for x, y, row in zip(ratios[1:], values[1:], rows[1:]):
            marker = "pass" if row["development_gate"] == "accepted" else "fail"
            ax.annotate(marker, (x, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("Synthetic / real training-row ratio")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Development-only ablation: physics-synthetic ratio\n"
        "C3 staged training, seed 42, frozen w20 blend (lower is better)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "schema_version": 1,
        "experiment": "v4_physics_synthetic_to_real_ratio_ablation",
        "date": "2026-08-04",
        "selection_scope": "grouped_con_rot_development_only",
        "seed": 42,
        "fixed_hybrid_position_weight": 0.2,
        "absolute_timestamp_used": False,
        "cyl_rot_read_or_scored_by_this_experiment": False,
        "rows": rows,
        "source_sha256": sources,
        "decision": (
            "Ratios 0.25 and 0.50 fail the per-session p95 gate. Ratio 1.00 "
            "remains the best balanced development setting, but its already-frozen "
            "one-time final result failed position metrics. Retain v3.2.2 and require "
            "new rotating development plus a later sealed session before reconsidering."
        ),
    }
    with open(json_path, "w") as stream:
        json.dump(payload, stream, indent=2)
    report_path.write_text(
        "# Physics-synthetic/calibration ratio ablation\n\n"
        "This is a development-only screen on the three grouped `con_rot` folds. "
        "Synthetic rows occur only in fold training; validation is real-only. "
        "The experiment does not read or score the already-opened `cyl_rot` final.\n\n"
        "![Four-metric ratio ablation](synthetic_ratio_ablation_seed42.png)\n\n"
        "## Results\n\n"
        + markdown_table(rows, columns)
        + "\n## Decision\n\n"
        "Ratios `0.25` and `0.50` improve all four aggregate metrics over the "
        "matched-fold v3.2.2 reference, but fail the predeclared per-session p95 "
        "gate: fold-s3 SO(3) p95 regresses by 8.08% and 6.13%. Ratio `1.00` "
        "remains the only balanced development pass, but its frozen one-time "
        "final result already failed position. The deployable model therefore "
        "remains v3.2.2.\n\n"
        "Latency for the two rejected screening ratios is an explicitly marked "
        "architecture-matched proxy, not a new candidate-specific benchmark.\n\n"
        "## Rebuild\n\n"
        "```bash\n"
        "# Run from the cloned repository root.\n"
        "./.venv/bin/python evaluation/summarize_synthetic_ratio_ablation_v4.py\n"
        "```\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

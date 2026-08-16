"""Build paper-style tables and figures from frozen EMF experiment artifacts.

This is a reporting-only pipeline.  It reads the already-opened final test to
reproduce frozen results, but never trains, calibrates, selects, or rewrites a
model/data artifact.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emf_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.channel_correction import apply_channel_correction
from physics.forward_model import forward_model
from physics.rotation_repr import pose_deg_to_target_v3, so3_geodesic_deg
from training.metrics import pose_metrics_v3


POSE_TARGET = [
    "target_X_mm", "target_Y_mm", "target_Z_mm",
    "target_roll_deg", "target_pitch_deg", "target_yaw_deg",
]
POSE_PRED = [
    "pred_X_mm", "pred_Y_mm", "pred_Z_mm",
    "pred_roll_deg", "pred_pitch_deg", "pred_yaw_deg",
]
COLORS = {
    "blue": "#295BFF",
    "red": "#FF5A5F",
    "green": "#159947",
    "orange": "#F39C12",
    "purple": "#7E57C2",
    "gray": "#5E6572",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def read_predictions(path):
    frame = pd.read_csv(path)
    missing = set(POSE_TARGET + POSE_PRED + ["source_file", "source_row"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)}")
    frame = frame.sort_values(["source_file", "source_row"]).reset_index(drop=True)
    target = pose_deg_to_target_v3(frame[POSE_TARGET].to_numpy())
    pred = pose_deg_to_target_v3(frame[POSE_PRED].to_numpy())
    frame["position_error_mm"] = np.linalg.norm(pred[:, :3] - target[:, :3], axis=1)
    frame["so3_error_deg"] = so3_geodesic_deg(pred[:, 3:], target[:, 3:])
    return frame


def assert_aligned(*frames):
    reference = frames[0][["source_file", "source_row", *POSE_TARGET]].to_numpy()
    for frame in frames[1:]:
        candidate = frame[["source_file", "source_row", *POSE_TARGET]].to_numpy()
        if reference.shape != candidate.shape:
            raise ValueError("Prediction artifacts cover different row counts")
        if not np.array_equal(reference[:, :2], candidate[:, :2]):
            raise ValueError("Prediction artifacts cover different source rows")
        np.testing.assert_allclose(
            reference[:, 2:].astype(float), candidate[:, 2:].astype(float), atol=1e-4,
        )


def metrics_from_frame(frame):
    return pose_metrics_v3(
        pose_deg_to_target_v3(frame[POSE_PRED].to_numpy()),
        pose_deg_to_target_v3(frame[POSE_TARGET].to_numpy()),
    )


def rolling(values, window=21):
    return pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def save_figure(fig, figures_dir, name):
    png = figures_dir / f"{name}.png"
    pdf = figures_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]


def markdown_table(frame):
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            display[column] = display[column].map(
                lambda value: str(value).replace("|", "\\|")
            )
    header = "| " + " | ".join(display.columns) + " |"
    rule = "|" + "|".join(["---"] * len(display.columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def write_table(frame, tables_dir, name):
    csv_path = tables_dir / f"{name}.csv"
    md_path = tables_dir / f"{name}.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(frame))
    return [csv_path, md_path]


def plot_ablation(hist_pretrain, hist_finetune, seed_summaries, baseline, figures_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.5))
    ax = axes[0, 0]
    ax.plot(hist_pretrain["epoch"], hist_pretrain["position_axis_rmse_mm"],
            color=COLORS["orange"], alpha=0.85, label="Real + physics synthetic")
    ax.plot(hist_finetune["epoch"], hist_finetune["position_axis_rmse_mm"],
            color=COLORS["blue"], alpha=0.9, label="Real-only fine-tune")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_title("(a) Staged calibration: position")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(hist_pretrain["epoch"], hist_pretrain["orientation_geodesic_rmse_deg"],
            color=COLORS["orange"], alpha=0.85, label="Real + physics synthetic")
    ax.plot(hist_finetune["epoch"], hist_finetune["orientation_geodesic_rmse_deg"],
            color=COLORS["blue"], alpha=0.9, label="Real-only fine-tune")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SO(3) RMSE (deg)")
    ax.set_title("(b) Staged calibration: orientation")
    ax.legend()

    ax = axes[1, 0]
    ax.semilogy(hist_finetune["epoch"], hist_finetune["train_loss"],
                color=COLORS["green"], label="Train loss")
    ax.semilogy(hist_finetune["epoch"], hist_finetune["val_loss"],
                color=COLORS["red"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized loss (log scale)")
    ax.set_title("(c) Real-only fine-tune generalization")
    ax.legend()

    ax = axes[1, 1]
    seeds = list(seed_summaries)
    position = [seed_summaries[seed]["position_axis_rmse_mm"] for seed in seeds]
    orientation = [seed_summaries[seed]["orientation_geodesic_rmse_deg"] for seed in seeds]
    x = np.arange(len(seeds))
    width = 0.34
    bars1 = ax.bar(x - width / 2, position, width, color=COLORS["blue"], label="Position RMSE")
    ax.axhline(baseline["position_axis_rmse_mm"], color=COLORS["blue"], ls="--", lw=1,
               label="v3.2 position baseline")
    ax.set_ylabel("Position RMSE (mm)", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Training seed")
    twin = ax.twinx()
    bars2 = twin.bar(x + width / 2, orientation, width, color=COLORS["green"], label="SO(3) RMSE")
    twin.axhline(baseline["orientation_geodesic_rmse_deg"], color=COLORS["green"], ls="--", lw=1,
                 label="v3.2 SO(3) baseline")
    twin.set_ylabel("SO(3) RMSE (deg)", color=COLORS["green"])
    twin.tick_params(axis="y", labelcolor=COLORS["green"])
    ax.set_title("(d) Frozen rule seed robustness")
    handles = [bars1, bars2]
    labels = ["Position RMSE", "SO(3) RMSE"]
    ax.legend(handles, labels, loc="upper center")

    fig.suptitle("Paper-style Fig. 4 analogue — Ablation and robustness", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure(fig, figures_dir, "fig4_ablation_and_seed_robustness")


def plot_calibration_effect(real_only, staged, figures_dir):
    assert_aligned(real_only, staged)
    real_metrics = metrics_from_frame(real_only)
    staged_metrics = metrics_from_frame(staged)
    x = np.arange(len(real_only))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, column, ylabel, metric in (
        (axes[0], "position_error_mm", "Position error (mm)", "position_axis_rmse_mm"),
        (axes[1], "so3_error_deg", "SO(3) error (deg)", "orientation_geodesic_rmse_deg"),
    ):
        ax.plot(x, real_only[column], color=COLORS["orange"], alpha=0.20, lw=0.6)
        ax.plot(x, rolling(real_only[column]), color=COLORS["orange"], lw=1.2,
                label=f"Real-only C3 (RMSE {real_metrics[metric]:.3f})")
        ax.plot(x, staged[column], color=COLORS["blue"], alpha=0.20, lw=0.6)
        ax.plot(x, rolling(staged[column]), color=COLORS["blue"], lw=1.2,
                label=f"Physics pretrain + real fine-tune (RMSE {staged_metrics[metric]:.3f})")
        ax.set_xlabel("Development sample index")
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[0].set_title("(a) Position")
    axes[1].set_title("(b) Orientation")
    fig.suptitle("Paper-style Fig. 7 analogue — Effect of physics-guided calibration (fold s3)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_figure(fig, figures_dir, "fig7_physics_calibration_effect")


def plot_final_benchmark(v32, v4, selected, figures_dir):
    assert_aligned(v32, v4, selected)
    # The paper displays 100 poses from one complete five-loop trajectory.
    # Use the last session because it contains exactly one clean 1,000-row
    # programmed sweep; sessions 1/2 include trailing restart rows.
    session = sorted(v32["session_key"].astype(str).unique())[-1]
    mask = v32["session_key"].astype(str) == session
    base_full = v32.loc[mask].reset_index(drop=True)
    v4_full = v4.loc[mask].reset_index(drop=True)
    selected_full = selected.loc[mask].reset_index(drop=True)
    display_indices = np.linspace(0, len(base_full) - 1, 100).round().astype(int)
    base_session = base_full.iloc[display_indices].reset_index(drop=True)
    v4_session = v4_full.iloc[display_indices].reset_index(drop=True)
    selected_session = selected_full.iloc[display_indices].reset_index(drop=True)
    center_x = 0.5 * (base_session["target_X_mm"].min() + base_session["target_X_mm"].max())
    center_y = 0.5 * (base_session["target_Y_mm"].min() + base_session["target_Y_mm"].max())
    z_origin = base_session["target_Z_mm"].min()

    def local_m(frame, prefix):
        return (
            (frame[f"{prefix}_X_mm"] - center_x) / 1000.0,
            (frame[f"{prefix}_Y_mm"] - center_y) / 1000.0,
            (frame[f"{prefix}_Z_mm"] - z_origin) / 1000.0,
        )

    fig = plt.figure(figsize=(11.2, 7.4))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    target_xyz = local_m(base_session, "target")
    pred_xyz = local_m(base_session, "pred")
    ax3d.plot(
        *target_xyz, color=COLORS["green"], lw=1.0, marker=".", ms=2.5,
        label="Ground truth",
    )
    ax3d.plot(
        *pred_xyz, color=COLORS["blue"], lw=0.9, ls="--", marker=".", ms=2.2,
        label="Predicted v3.2.2",
    )
    ax3d.set_xlabel("Local X (m)")
    ax3d.set_ylabel("Local Y (m)")
    ax3d.set_zlabel("Local Z (m)")
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=23, azim=-55)
    ax3d.set_title("(a) Current cylindrical trajectory — 100 displayed poses")
    ax3d.legend(loc="upper left")

    ax = fig.add_subplot(2, 2, 2)
    sample = np.arange(len(base_session))
    for label, target_col, pred_col, color in (
        ("Roll", "target_roll_deg", "pred_roll_deg", COLORS["green"]),
        ("Pitch", "target_pitch_deg", "pred_pitch_deg", COLORS["blue"]),
        ("Yaw", "target_yaw_deg", "pred_yaw_deg", COLORS["red"]),
    ):
        ax.plot(sample, base_session[target_col], color=color, lw=1.3, label=f"{label} target")
        ax.plot(sample, base_session[pred_col], color=color, lw=0.9, ls="--", label=f"{label} v3.2.2")
    ax.set_xlabel("Displayed pose index")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("(b) Orientation tracking")
    ax.legend(ncol=3, loc="lower center")

    x = np.arange(len(base_session))
    ax = fig.add_subplot(2, 2, 3)
    for frame, label, color in (
        (base_session, "Selected v3.2.2", COLORS["blue"]),
        (v4_session, "Hybrid-v4 component", COLORS["orange"]),
        (selected_session, "Frozen w=0.20 (rejected)", COLORS["purple"]),
    ):
        ax.plot(x, frame["position_error_mm"], color=color, lw=0.9,
                marker=".", ms=2.2, label=label)
    ax.set_xlabel("Displayed pose index")
    ax.set_ylabel("Position error (mm)")
    ax.set_title("(c) Position error — no smoothing")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    for frame, label, color in (
        (base_session, "Selected v3.2.2", COLORS["blue"]),
        (v4_session, "Hybrid-v4 component", COLORS["orange"]),
        (selected_session, "Frozen w=0.20 (rejected)", COLORS["purple"]),
    ):
        ax.plot(x, frame["so3_error_deg"], color=color, lw=0.9,
                marker=".", ms=2.2, label=label)
    ax.set_xlabel("Displayed pose index")
    ax.set_ylabel("SO(3) error (deg)")
    ax.set_title("(d) Orientation error — no smoothing")
    ax.legend()

    fig.suptitle(
        "Paper-style Fig. 8 analogue — 100 uniformly sampled poses from frozen session 3",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure(fig, figures_dir, "fig8_final_trajectory_orientation_errors")


def measured_current_emf(data_dir):
    with np.load(Path(data_dir) / "sealed_test.npz") as loaded:
        keys = list(zip(loaded["source_file"].astype(str), loaded["source_row"].astype(int)))
        packed = loaded["emf"].astype(np.float32)
    current = packed[:, 18:27]
    return {key: values for key, values in zip(keys, current)}


def reconstruct_emf(frame, physics_config, measured_map, pose_columns):
    pose = frame[pose_columns].to_numpy(dtype=float)
    reconstructed = forward_model(pose, physics_config["tx"], physics_config["rx"])
    if "channel_correction" in physics_config["tx"]:
        reconstructed = apply_channel_correction(reconstructed, physics_config["tx"])
    measured = np.stack([
        measured_map[(str(row.source_file), int(row.source_row))]
        for row in frame[["source_file", "source_row"]].itertuples(index=False)
    ])
    return measured, np.asarray(reconstructed)


def plot_physics_consistency(v32, v4, physics_config, measured_map, figures_dir):
    measured, reconstructed_target = reconstruct_emf(
        v32, physics_config, measured_map, POSE_TARGET,
    )
    measured_v32, reconstructed_v32 = reconstruct_emf(
        v32, physics_config, measured_map, POSE_PRED,
    )
    measured_v4, reconstructed_v4 = reconstruct_emf(
        v4, physics_config, measured_map, POSE_PRED,
    )
    np.testing.assert_allclose(measured, measured_v32)
    np.testing.assert_allclose(measured, measured_v4)
    session = sorted(v32["session_key"].astype(str).unique())[0]
    mask = (v32["session_key"].astype(str) == session).to_numpy()
    indices = np.flatnonzero(mask)[:300]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7))
    axes[0].plot(measured[indices, 0] * 1e3, color=COLORS["red"], lw=1.1, label="Measured EMF1")
    axes[0].plot(reconstructed_target[indices, 0] * 1e3, color=COLORS["green"], lw=1.0,
                 label="Reconstructed from ground-truth pose")
    axes[0].plot(reconstructed_v32[indices, 0] * 1e3, color=COLORS["blue"], lw=1.0,
                 label="Reconstructed from v3.2.2 pose")
    axes[0].plot(reconstructed_v4[indices, 0] * 1e3, color=COLORS["orange"], lw=0.9, ls="--",
                 label="Reconstructed from v4 pose")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Peak EMF (mV)")
    axes[0].set_title("(a) EMF1 reconstruction")
    axes[0].legend()

    denominator = np.maximum(np.abs(measured), 1e-6)
    error_target = np.mean(np.abs(reconstructed_target - measured) / denominator, axis=0) * 100.0
    error_v32 = np.mean(np.abs(reconstructed_v32 - measured) / denominator, axis=0) * 100.0
    error_v4 = np.mean(np.abs(reconstructed_v4 - measured) / denominator, axis=0) * 100.0
    x = np.arange(9)
    width = 0.27
    axes[1].bar(x - width, error_target, width, color=COLORS["green"], label="Ground-truth pose")
    axes[1].bar(x, error_v32, width, color=COLORS["blue"], label="v3.2.2 pose")
    axes[1].bar(x + width, error_v4, width, color=COLORS["orange"], label="v4 pose")
    axes[1].set_xticks(x, [f"EMF{i}" for i in range(1, 10)], rotation=30)
    axes[1].set_ylabel("Mean absolute percent error (%)")
    axes[1].set_title("(b) Per-channel reconstruction error")
    axes[1].legend()
    fig.suptitle("Paper-style Fig. 9 analogue — Physics-consistency audit", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    paths = save_figure(fig, figures_dir, "fig9_physics_consistency")
    return paths, error_target, error_v32, error_v4


def scenario_table(dev_baseline, dev_candidate, final_baseline, final_candidate):
    rows = []
    for session in ("s1", "s2", "s3"):
        for label, source, gpu, cpu in (
            ("v3.2.2", dev_baseline, 0.858054, 1.290225),
            ("Hybrid w=0.20", dev_candidate, 1.422557, 1.992323),
        ):
            metric = source["sessions"][session]
            rows.append({
                "Split": "Development CV",
                "Scenario": f"con_rot {session}",
                "Method": label,
                "Position RMSE (mm)": metric["position_axis_rmse_mm"],
                "Direction RMSE Eq.9 (deg)": metric["orientation_euler_component_rmse_deg"],
                "SO(3) RMSE (deg)": metric["orientation_geodesic_rmse_deg"],
                "Position p95 (mm)": metric["position_euclidean_p95_mm"],
                "SO(3) p95 (deg)": metric["orientation_geodesic_p95_deg"],
                "CPU p95 (ms)": cpu,
                "GPU p95 (ms)": gpu,
            })
    for label, source, gpu, cpu in (
        ("v3.2.2 (selected)", final_baseline, 0.858054, 1.290225),
        ("Hybrid w=0.20 (rejected)", final_candidate, 1.422557, 1.992323),
    ):
        metric = source["aggregate"]
        rows.append({
            "Split": "One-time final",
            "Scenario": "cyl_rot pooled",
            "Method": label,
            "Position RMSE (mm)": metric["position_axis_rmse_mm"],
            "Direction RMSE Eq.9 (deg)": metric["orientation_euler_component_rmse_deg"],
            "SO(3) RMSE (deg)": metric["orientation_geodesic_rmse_deg"],
            "Position p95 (mm)": metric["position_euclidean_p95_mm"],
            "SO(3) p95 (deg)": metric["orientation_geodesic_p95_deg"],
            "CPU p95 (ms)": cpu,
            "GPU p95 (ms)": gpu,
        })
    return pd.DataFrame(rows)


def benchmark_table(final_baseline, final_v4, final_candidate):
    paper = [
        ("Reference paper", "Optimization", 48.85, 68.89, None),
        ("Reference paper", "FCN [22]", 101.62, 40.68, None),
        ("Reference paper", "KAN [24]", 68.50, 32.17, None),
        ("Reference paper", "Paper residual model", 1.90, 3.55, 0.82),
    ]
    current = [
        ("Current 100 mm final", "v3.2.2 (selected)", final_baseline, 0.858054),
        ("Current 100 mm final", "Hybrid-v4 component", final_v4, 1.422557),
        ("Current 100 mm final", "Frozen hybrid w=0.20 (rejected)", final_candidate, 1.422557),
    ]
    rows = [{
        "Study / protocol": study,
        "Method": method,
        "Position RMSE (mm)": position,
        "RMSE / workspace edge (%)": position / 500.0 * 100.0,
        "Direction RMSE Eq.9 (deg)": direction,
        "GPU inference p95 / reported (ms)": latency,
        "Directly comparable": "No",
    } for study, method, position, direction, latency in paper]
    for study, method, summary, latency in current:
        metric = summary["aggregate"]
        rows.append({
            "Study / protocol": study,
            "Method": method,
            "Position RMSE (mm)": metric["position_axis_rmse_mm"],
            "RMSE / workspace edge (%)": metric["position_axis_rmse_mm"] / 100.0 * 100.0,
            "Direction RMSE Eq.9 (deg)": metric["orientation_euler_component_rmse_deg"],
            "GPU inference p95 / reported (ms)": latency,
            "Directly comparable": "No",
        })
    return pd.DataFrame(rows)


def feature_table():
    rows = [
        ("Algorithm", "FCN + 7 residual blocks", "5-model residual component ensemble", "C3 residual + v3.2.2 blend"),
        ("Sensor method", "Magnetic induction", "Magnetic induction", "Magnetic induction"),
        ("Tracking DOF", "6", "6", "6"),
        ("Dynamic final test", "Yes", "Yes; 3 cyl_rot sessions", "Yes; same frozen final"),
        ("Workspace", "500 x 500 x 500 mm", "100 x 100 x 100 mm", "100 x 100 x 100 mm"),
        ("Temporal input", "Single row", "Causal W3 + unit delta-row branch", "Causal W3 raw + log-ratio + delta-row"),
        ("Orientation representation", "cos(Euler)", "Rotation-6D / SO(3)", "Rotation-6D / SO(3)"),
        ("Synthetic augmentation", "64 million physics samples", "No", "1:1 fold-gated physics synthetic"),
        ("Final position RMSE", "1.90 mm", "1.1222 mm", "1.2335 mm"),
        ("Final direction RMSE Eq.9", "3.55 deg", "1.3801 deg", "1.0769 deg"),
        ("Runtime", "0.82 ms reported", "0.858 ms GPU p95; 1.290 ms CPU p95", "1.423 ms GPU p95; 1.992 ms CPU p95"),
        ("Decision", "Paper proposal", "Selected deployment", "Rejected by balanced final gate"),
    ]
    return pd.DataFrame(rows, columns=["Feature", "Reference paper", "Current v3.2.2", "Experimental v4 hybrid"])


def build_report(out_dir, artifacts, table_paths, figure_paths, physics_errors):
    v32 = artifacts["final_baseline"]["aggregate"]
    candidate = artifacts["final_candidate"]["aggregate"]
    physics_target, physics_v32, physics_v4 = physics_errors
    text = f"""# Paper-style result report — 100 mm EMF localization

Generated by `reporting/build_paper_style_report.py`.

Detailed Vietnamese pipeline comparison and honest claim audit:
[BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md](../../docs/BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md).

This report mirrors the structure of Figs. 4, 7, 8, 9 and Tables I–III in the
reference paper, but it does **not** claim a direct numerical reproduction. The
paper uses a 500 mm workspace and broader orientation motion. The current
workspace is 100 mm and the protocols differ.

The final `cyl_rot` test is used here only for reporting the already frozen
one-time evaluation. No model or blend selection occurs in this script.

## Reproduce this report

```bash
# Run from the cloned repository root.
./.venv/bin/python reporting/build_paper_style_report.py \\
  --out_dir reports/paper_style_2026_08_04
```

## Figures

![Ablation and seed robustness](figures/fig4_ablation_and_seed_robustness.png)

![Physics calibration effect](figures/fig7_physics_calibration_effect.png)

![Final trajectory and errors](figures/fig8_final_trajectory_orientation_errors.png)

Fig. 8 displays 100 uniformly sampled poses from the clean 1,000-row session 3
and does not smooth the error curves, matching the paper's display density.
It deliberately preserves the measured geometry: the current robot path is a
five-loop cylinder with an approximately constant 50 mm radius, whereas the
paper used a tapered five-loop helix. Plot styling cannot honestly create the
paper's large-bottom/small-top shape from the current data.

![Physics consistency](figures/fig9_physics_consistency.png)

## Tables

- [Table I analogue — scenarios](tables/table1_scenario_performance.md)
- [Table II analogue — real-system benchmark](tables/table2_real_system_benchmark.md)
- [Table III analogue — method/features](tables/table3_method_comparison.md)
- [Physics reconstruction error](tables/table4_physics_reconstruction_error.md)

## Frozen final result

| Model | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| v3.2.2 selected | {v32['position_axis_rmse_mm']:.4f} mm | {v32['position_euclidean_p95_mm']:.4f} mm | {v32['orientation_geodesic_rmse_deg']:.4f} deg | {v32['orientation_geodesic_p95_deg']:.4f} deg |
| Frozen hybrid w=0.20 | {candidate['position_axis_rmse_mm']:.4f} mm | {candidate['position_euclidean_p95_mm']:.4f} mm | {candidate['orientation_geodesic_rmse_deg']:.4f} deg | {candidate['orientation_geodesic_p95_deg']:.4f} deg |

The hybrid improves orientation but regresses position, so v3.2.2 remains the
selected deployment. Choosing a new rule from these plots would be leakage.

Physics-reconstruction MAPE in Fig. 9 is a diagnostic based on the fitted
current forward model. The ground-truth-pose control separates forward-model
calibration error from inverse-network pose error. Mean per-channel MAPE is
{np.mean(physics_target):.2f}% even with ground-truth pose, versus
{np.mean(physics_v32):.2f}% from v3.2.2 pose and {np.mean(physics_v4):.2f}%
from v4 pose. Therefore the dominant discrepancy is the current forward-model
calibration/channel mapping on rotating sessions, not the inverse network.
This is not the paper's reported <3.4% result and must not be presented as such.

The paper/current rows in Table II are context only. Even the workspace-edge
normalization does not make the studies directly comparable because the
trajectories, orientation ranges, sensors, splits, and error protocols differ.
"""
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Create paper-style frozen-result tables and figures.")
    parser.add_argument("--out_dir", default="reports/paper_style_2026_08_04")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    paths = {
        "history_pretrain": ROOT / "checkpoints/new_calib_v4_hybrid/C3/dev_s3_seed42/training_history.csv",
        "history_finetune": ROOT / "checkpoints/new_calib_v4_hybrid/C3/dev_s3_seed42_finetune_real_rebase/training_history.csv",
        "dev_real_only": ROOT / "results/new_calib_v4_real/C3/dev_s3_seed42/val_predictions.csv",
        "dev_staged": ROOT / "results/new_calib_v4_hybrid/C3/dev_s3_seed42_finetune_real_rebase/val_predictions.csv",
        "final_v32_predictions": ROOT / "results/new_calib_v4_final/baseline_v32/sealed_test_predictions.csv",
        "final_v4_predictions": ROOT / "results/new_calib_v4_final/hybrid_v4_component/sealed_test_predictions.csv",
        "final_candidate_predictions": ROOT / "results/new_calib_v4_final/selected_hybrid_w20/sealed_test_predictions.csv",
        "dev_baseline_summary": ROOT / "results/new_calib_v4_hybrid/ensemble/cv_baseline_v32.json",
        "dev_seed42_summary": ROOT / "results/new_calib_v4_hybrid/ensemble/rotation6d_mean/seed42/w20/cv_summary.json",
        "dev_seed7_summary": ROOT / "results/new_calib_v4_hybrid/ensemble/rotation6d_mean/seed7/w20/cv_summary.json",
        "dev_seed123_summary": ROOT / "results/new_calib_v4_hybrid/ensemble/rotation6d_mean/seed123/w20/cv_summary.json",
        "final_baseline_summary": ROOT / "results/new_calib_v4_final/baseline_v32/summary.json",
        "final_v4_summary": ROOT / "results/new_calib_v4_final/hybrid_v4_component/summary.json",
        "final_candidate_summary": ROOT / "results/new_calib_v4_final/selected_hybrid_w20/summary.json",
        "physics_config": ROOT / "configs/physics_v4_deployment.yaml",
        "sealed_data": ROOT / "data/new_calib_v4_final/C3/sealed_evaluation",
        "final_decision": ROOT / "results/new_calib_v4_final/final_decision_manifest.json",
        "reference_paper": ROOT / "Residual_Neural_Network_for_Precise_6-DoF_Capsule_Endoscope_Localization_Using_Electromagnetic_Induction.pdf",
        "detailed_vietnamese_report": ROOT / "docs/BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen report artifacts: {missing}")

    hist_pretrain = pd.read_csv(paths["history_pretrain"])
    hist_finetune = pd.read_csv(paths["history_finetune"])
    dev_real_only = read_predictions(paths["dev_real_only"])
    dev_staged = read_predictions(paths["dev_staged"])
    final_v32 = read_predictions(paths["final_v32_predictions"])
    final_v4 = read_predictions(paths["final_v4_predictions"])
    final_candidate = read_predictions(paths["final_candidate_predictions"])
    assert_aligned(final_v32, final_v4, final_candidate)

    dev_baseline = read_json(paths["dev_baseline_summary"])
    dev_seed42 = read_json(paths["dev_seed42_summary"])
    seed_summaries = {
        42: dev_seed42["aggregate"],
        7: read_json(paths["dev_seed7_summary"])["aggregate"],
        123: read_json(paths["dev_seed123_summary"])["aggregate"],
    }
    final_baseline = read_json(paths["final_baseline_summary"])
    final_v4_summary = read_json(paths["final_v4_summary"])
    final_candidate_summary = read_json(paths["final_candidate_summary"])
    decision = read_json(paths["final_decision"])
    if decision.get("candidate_final_accepted") is not False or decision.get("test_opened_once") is not True:
        raise RuntimeError("Final decision manifest is inconsistent with reporting-only status")

    figure_paths = []
    figure_paths += plot_ablation(
        hist_pretrain, hist_finetune, seed_summaries, dev_baseline["aggregate"], figures_dir,
    )
    figure_paths += plot_calibration_effect(dev_real_only, dev_staged, figures_dir)
    figure_paths += plot_final_benchmark(final_v32, final_v4, final_candidate, figures_dir)
    with open(paths["physics_config"]) as stream:
        physics_config = yaml.safe_load(stream)
    measured_map = measured_current_emf(paths["sealed_data"])
    physics_paths, physics_target, physics_v32, physics_v4 = plot_physics_consistency(
        final_v32, final_v4, physics_config, measured_map, figures_dir,
    )
    figure_paths += physics_paths

    table_paths = []
    table1 = scenario_table(dev_baseline, dev_seed42, final_baseline, final_candidate_summary)
    table_paths += write_table(table1, tables_dir, "table1_scenario_performance")
    table2 = benchmark_table(final_baseline, final_v4_summary, final_candidate_summary)
    table_paths += write_table(table2, tables_dir, "table2_real_system_benchmark")
    table3 = feature_table()
    table_paths += write_table(table3, tables_dir, "table3_method_comparison")
    table4 = pd.DataFrame({
        "Channel": [f"EMF{i}" for i in range(1, 10)],
        "Ground-truth pose reconstruction MAPE (%)": physics_target,
        "v3.2.2 pose reconstruction MAPE (%)": physics_v32,
        "v4 pose reconstruction MAPE (%)": physics_v4,
    })
    table_paths += write_table(table4, tables_dir, "table4_physics_reconstruction_error")

    artifacts = {
        "final_baseline": final_baseline,
        "final_candidate": final_candidate_summary,
    }
    report_path = build_report(
        out_dir, artifacts, table_paths, figure_paths,
        (physics_target, physics_v32, physics_v4),
    )
    generated = [*figure_paths, *table_paths, report_path]
    manifest = {
        "schema_version": 1,
        "report_type": "paper_style_analogue_not_direct_reproduction",
        "reporting_only": True,
        "test_opened_before_report": True,
        "model_selection_or_training_performed": False,
        "workspace_mm": [100.0, 100.0, 100.0],
        "reference_workspace_mm": [500.0, 500.0, 500.0],
        "generator_sha256": sha256_file(Path(__file__)),
        "source_artifacts_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for key, path in paths.items() if path.is_file()
        },
        "generated_files_sha256": {
            str(path.relative_to(out_dir)): sha256_file(path) for path in generated
        },
        "figures": [str(path.relative_to(out_dir)) for path in figure_paths],
        "tables": [str(path.relative_to(out_dir)) for path in table_paths],
        "decision": "retain_v3_2_2_baseline",
    }
    manifest_path = out_dir / "report_manifest.json"
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps({
        "out_dir": str(out_dir),
        "figures": len(figure_paths),
        "tables": len(table_paths),
        "report": str(report_path),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()

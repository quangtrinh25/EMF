"""Build a standalone, audit-style comparison with the reference EMF paper.

The script is reporting-only.  It reads frozen JSON/CSV/NPZ summaries and
copies already generated paper-style analogues.  It never trains, evaluates a
checkpoint, opens raw final labels, or performs model/blend selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emf_full_comparison_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "paper": "#3977C3",
    "ours": "#E58A2B",
    "selected": "#24935C",
    "candidate": "#7B61A8",
    "warning": "#C84A4A",
    "neutral": "#667085",
    "light": "#F4F6F8",
    "dark": "#172B4D",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig, figures_dir: Path, name: str) -> list[Path]:
    outputs = [figures_dir / f"{name}.png", figures_dir / f"{name}.pdf"]
    for output in outputs:
        kwargs = {}
        if output.suffix == ".pdf":
            kwargs["metadata"] = {"Creator": "EMF audit report", "CreationDate": None, "ModDate": None}
        fig.savefig(output, bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    return outputs


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        def format_value(value):
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{value:.4f}"
            return str(value).replace("|", "\\|").replace("\n", "<br>")
        display[column] = display[column].map(format_value)
    header = "| " + " | ".join(display.columns) + " |"
    rule = "|" + "|".join(["---"] * len(display.columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def write_table(frame: pd.DataFrame, tables_dir: Path, name: str) -> list[Path]:
    csv_path = tables_dir / f"{name}.csv"
    md_path = tables_dir / f"{name}.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(frame))
    return [csv_path, md_path]


def add_box(ax, x, y, w, h, text, color, fontsize=8.4, edge=None, linewidth=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color,
        edgecolor=edge or color,
        linewidth=linewidth,
        alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color="white" if color not in {COLORS["light"]} else COLORS["dark"], wrap=True)
    return patch


def add_arrow(ax, x1, y1, x2, y2, color=COLORS["neutral"], style="-"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.25, color=color, linestyle=style,
    )
    ax.add_patch(arrow)


def plot_pipeline(figures_dir: Path):
    fig, ax = plt.subplots(figsize=(14.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.92, "BÀI BÁO — pipeline được mô tả", fontsize=13, weight="bold", color=COLORS["paper"])
    ax.text(0.985, 0.92, "500³ mm • broad angles", ha="right", fontsize=9,
            color=COLORS["paper"], weight="bold")
    paper = [
        ("Forward model\nđiện từ giải tích", "Mô phỏng 64M\npose–EMF", "Random split\n70/15/15", "ResNet 512\n7 residual blocks", "150 pose thật\nfit tham số TX", "Retrain + test\nhelix 100 pose"),
    ][0]
    xs = np.linspace(0.015, 0.84, len(paper))
    w, h = 0.135, 0.145
    for idx, (x, label) in enumerate(zip(xs, paper)):
        add_box(ax, x, 0.70, w, h, label, COLORS["paper"], fontsize=8.1)
        if idx:
            add_arrow(ax, xs[idx - 1] + w, 0.772, x, 0.772, COLORS["paper"])
    ax.text(0.01, 0.53, "PIPELINE TA — trạng thái thực tế", fontsize=13, weight="bold", color=COLORS["ours"])
    ax.text(0.985, 0.53, "100³ mm • mainly pitch", ha="right", fontsize=9,
            color=COLORS["ours"], weight="bold")
    ours = [
        "12 robot CSV\n9 train sessions",
        "Split theo session\ntrước cửa sổ W3",
        "Fit physics chỉ\ntrên train-fold",
        "Stage 1: pure\nsynthetic C3",
        "Stage 2: real-only\ncalibration",
        "3 seed + 4 metric\n+ session gate",
        "Research-only\ncheckpoint",
    ]
    xs2 = np.linspace(0.015, 0.845, len(ours))
    w2, h2 = 0.116, 0.135
    for idx, (x, label) in enumerate(zip(xs2, ours)):
        color = COLORS["candidate"] if idx >= 3 else COLORS["ours"]
        add_box(ax, x, 0.33, w2, h2, label, color, fontsize=7.7)
        if idx:
            add_arrow(ax, xs2[idx - 1] + w2, 0.397, x, 0.397, COLORS["ours"])
    add_box(ax, 0.19, 0.085, 0.23, 0.13,
            "Deployment hiện tại\nv3.2.2 — 5-model ensemble", COLORS["selected"], fontsize=9.0)
    add_box(ax, 0.58, 0.085, 0.23, 0.13,
            "Bước bắt buộc tiếp theo\nnew rotating dev + new sealed", COLORS["warning"], fontsize=9.0)
    add_arrow(ax, 0.306, 0.33, 0.306, 0.215, COLORS["selected"])
    add_arrow(ax, 0.903, 0.33, 0.695, 0.215, COLORS["warning"], style="--")
    ax.text(0.50, 0.015,
            "Nhánh v4.4 cải thiện development nhưng chưa thay deployment: GPU gate fail và không còn sealed test mới.",
            ha="center", va="bottom", fontsize=9.2, color=COLORS["warning"], weight="bold")
    fig.suptitle("Hình 1 — Hai pipeline đặt cạnh nhau", fontsize=15, weight="bold", y=0.995)
    return save_figure(fig, figures_dir, "fig01_pipeline_side_by_side")


def plot_architecture(figures_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.8))
    specs = [
        (
            "Bài báo",
            COLORS["paper"],
            ["EMF hiện tại\n9", "Gaussian\nnormalization", "Linear 9→512", "7 × residual block\n(512→512→512)", "Head 512→128→6", "XYZ + cos(Euler)"],
            "Một mạng; single-row; MSE",
        ),
        (
            "Ta — selected v3.2.2",
            COLORS["selected"],
            ["Causal W3\n27 raw", "4 position members\n3 seed + rowtime", "Mỗi member:\n512, 7 blocks", "1 orientation member\nraw + log-ratio", "Component ensemble", "XYZ + rotation-6D\n→ SO(3)"],
            "Năm mạng; tách position/orientation",
        ),
        (
            "Ta — v4.4 research",
            COLORS["candidate"],
            ["W3 packed\n27 raw + 2 Δt", "Generation gain/bias\nadapter", "+ 18 log-ratio\n= 47 features", "512, 7 blocks", "Head 512→128→9", "0.8/0.2 XYZ\n0.7/0.3 rotation-6D"],
            "Một C3 mới + frozen v3.2.2 blend",
        ),
    ]
    for ax, (title, color, stages, note) in zip(axes, specs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, color=color, fontsize=12.5, weight="bold", pad=10)
        ys = np.linspace(0.83, 0.18, len(stages))
        for idx, (y, stage) in enumerate(zip(ys, stages)):
            add_box(ax, 0.18, y, 0.64, 0.09, stage, color, fontsize=8.0)
            if idx:
                add_arrow(ax, 0.50, ys[idx - 1], 0.50, y + 0.09, color)
        ax.text(0.50, 0.04, note, ha="center", va="center", fontsize=8.5, color=color, weight="bold")
    fig.suptitle("Hình 2 — Kiến trúc: phần giữ lại và phần đổi mới", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_figure(fig, figures_dir, "fig02_architecture_side_by_side")


def plot_leakage(figures_dir: Path):
    fig, ax = plt.subplots(figsize=(14.2, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.91, "Bài báo", fontsize=12.5, weight="bold", color=COLORS["paper"])
    add_box(ax, 0.08, 0.71, 0.22, 0.12, "64M synthetic\nrandom 70% train", COLORS["paper"])
    add_box(ax, 0.39, 0.71, 0.22, 0.12, "15% validation", COLORS["paper"])
    add_box(ax, 0.70, 0.71, 0.22, 0.12, "15% synthetic test\n+ real benchmark", COLORS["paper"])
    add_arrow(ax, 0.30, 0.77, 0.39, 0.77, COLORS["paper"])
    add_arrow(ax, 0.61, 0.77, 0.70, 0.77, COLORS["paper"])
    ax.text(0.50, 0.65,
            "Paper không công bố session-wise/nested gate hay manifest hash; không đủ thông tin để khẳng định leakage.",
            ha="center", fontsize=8.8, color=COLORS["neutral"])

    ax.text(0.02, 0.53, "Ta", fontsize=12.5, weight="bold", color=COLORS["ours"])
    add_box(ax, 0.035, 0.35, 0.17, 0.13, "TRAIN sessions\nfit scaler + physics\nbuild causal windows", COLORS["ours"], fontsize=7.8)
    add_box(ax, 0.265, 0.35, 0.17, 0.13, "DEV session\nmodel/weight/seed\nquality gate", COLORS["candidate"], fontsize=7.8)
    add_box(ax, 0.495, 0.35, 0.17, 0.13, "HASH FREEZE\nmodel + config\n+ rule", COLORS["selected"], fontsize=7.8)
    add_box(ax, 0.725, 0.35, 0.24, 0.13, "FINAL cyl_rot\nopened exactly once\n2026-08-04", COLORS["warning"], fontsize=7.8)
    for x1, x2, color in ((0.205, 0.265, COLORS["ours"]), (0.435, 0.495, COLORS["candidate"]),
                          (0.665, 0.725, COLORS["warning"])):
        add_arrow(ax, x1, 0.415, x2, 0.415, color)
    ax.plot([0.235, 0.235], [0.28, 0.52], color=COLORS["warning"], lw=2.0, ls="--")
    ax.plot([0.695, 0.695], [0.28, 0.52], color=COLORS["warning"], lw=2.0, ls="--")
    ax.text(0.235, 0.245, "session boundary", ha="center", color=COLORS["warning"], fontsize=8)
    ax.text(0.695, 0.245, "sealed boundary", ha="center", color=COLORS["warning"], fontsize=8)
    ax.text(0.50, 0.14,
            "Cửa sổ không vượt session/reset • scaler/dt median chỉ từ train • không absolute timestamp • target không vào feature",
            ha="center", fontsize=9.1, color=COLORS["dark"], weight="bold")
    ax.text(0.50, 0.065,
            "Hạn chế trung thực: raw test labels từng tồn tại local; một audit preprocessing cũ đã nhìn boundary từ target. Vì vậy cần sealed set mới cho claim publication-grade.",
            ha="center", fontsize=8.7, color=COLORS["warning"], wrap=True)
    fig.suptitle("Hình 3 — Biên chống leakage và trạng thái test", fontsize=15, weight="bold")
    return save_figure(fig, figures_dir, "fig03_validation_and_leakage_boundaries")


def plot_data_scale(pose: np.ndarray, coverage_cells: int, figures_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))

    ax = axes[0, 0]
    labels = ["Paper", "Ta"]
    edges = [500, 100]
    bars = ax.bar(labels, edges, color=[COLORS["paper"], COLORS["ours"]], width=0.58)
    ax.set_ylabel("Cạnh workspace (mm)")
    ax.set_title("(a) Workspace tuyến tính")
    for bar, value in zip(bars, edges):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 10, f"{value} mm", ha="center", weight="bold")
    ax.text(0.5, 270, "Paper có thể tích lớn hơn 125×", ha="center", color=COLORS["warning"], weight="bold")

    ax = axes[0, 1]
    values = [64_000_000, 6_019, 6_019]
    labels = ["Paper\nsynthetic", "Ta Stage 1\nsynthetic", "Ta Stage 2\nreal"]
    bars = ax.bar(labels, values, color=[COLORS["paper"], COLORS["candidate"], COLORS["ours"]])
    ax.set_yscale("log")
    ax.set_ylabel("Số mẫu (log scale)")
    ax.set_title("(b) Quy mô train được công bố/đã chạy")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.3, f"{value:,}", ha="center", fontsize=8, weight="bold")

    ax = axes[1, 0]
    current_spans = np.ptp(pose[:, 3:6], axis=0)
    x = np.arange(3)
    width = 0.36
    ax.bar(x - width / 2, [140, 140, 140], width, color=COLORS["paper"], label="Paper test range 20–160°")
    ax.bar(x + width / 2, current_spans, width, color=COLORS["ours"], label="Ta train span")
    ax.set_xticks(x, ["Roll", "Pitch", "Yaw"])
    ax.set_ylabel("Độ rộng góc (degree)")
    ax.set_title("(c) Orientation coverage")
    ax.legend()
    for idx, value in enumerate(current_spans):
        ax.text(idx + width / 2, value + 2, f"{value:.2f}°", ha="center", fontsize=8, color=COLORS["ours"], weight="bold")

    ax = axes[1, 1]
    x = np.arange(4)
    values = [12, len(pose), len(np.unique(np.round(pose[:, :3], 1), axis=0)), coverage_cells]
    labels = ["Raw CSV", "Train rows", "Unique XYZ\n(0.1 mm)", "Occupied cells\n(out of 125)"]
    bars = ax.bar(x, values, color=[COLORS["neutral"], COLORS["ours"], COLORS["candidate"], COLORS["selected"]])
    ax.set_xticks(x, labels)
    ax.set_title("(d) Dữ liệu robot hiện tại")
    ax.set_ylabel("Count")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, f"{value:,}", ha="center", fontsize=8, weight="bold")

    fig.suptitle("Hình 4 — Scale và coverage: lý do không so số tuyệt đối trực tiếp", fontsize=14.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, figures_dir, "fig04_data_scale_and_coverage")


def plot_results(baseline_dev, seeds, final_v32, final_mixed, figures_dir: Path):
    metrics = [
        ("position_axis_rmse_mm", "Position axis RMSE (mm)"),
        ("position_euclidean_p95_mm", "Position Euclidean p95 (mm)"),
        ("orientation_geodesic_rmse_deg", "SO(3) RMSE (deg)"),
        ("orientation_geodesic_p95_deg", "SO(3) p95 (deg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.3))
    labels = ["v3.2.2\nbaseline", "v4.4\nseed 42", "v4.4\nseed 7", "v4.4\nseed 123"]
    colors = [COLORS["selected"], COLORS["candidate"], COLORS["candidate"], COLORS["candidate"]]
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [baseline_dev[key], *[seeds[seed][key] for seed in (42, 7, 123)]]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(values) * 1.18)
    fig.suptitle("Hình 5 — Development CV: v4.4 robust rule cải thiện cả bốn metric ở ba seed", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outputs = save_figure(fig, figures_dir, "fig05_development_multiseed_results")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    final_metrics = [
        ("position_axis_rmse_mm", "position_euclidean_p95_mm", "Position (mm)"),
        ("orientation_geodesic_rmse_deg", "orientation_geodesic_p95_deg", "Orientation SO(3) (deg)"),
    ]
    for ax, (rmse_key, p95_key, title) in zip(axes, final_metrics):
        x = np.arange(2)
        width = 0.34
        a = [final_v32[rmse_key], final_mixed[rmse_key]]
        b = [final_v32[p95_key], final_mixed[p95_key]]
        bars1 = ax.bar(x - width / 2, a, width, color=COLORS["selected"], label="RMSE")
        bars2 = ax.bar(x + width / 2, b, width, color=COLORS["warning"], label="p95")
        ax.set_xticks(x, ["v3.2.2", "Frozen mixed v4.2"])
        ax.set_title(title)
        ax.legend()
        for bars in (bars1, bars2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.015,
                        f"{bar.get_height():.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(b) * 1.18)
    fig.suptitle("Hình 6 — Final đã mở một lần: mixed v4.2 đổi orientation lấy position và bị reject", fontsize=13.2, weight="bold")
    fig.text(0.5, 0.01, "v4.4 không xuất hiện ở đây: không được phép đánh giá lại trên cyl_rot đã mở.",
             ha="center", color=COLORS["warning"], weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    outputs += save_figure(fig, figures_dir, "fig06_one_time_final_decision")
    return outputs


def plot_paper_context(final_v32, final_mixed, figures_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.6))
    labels = ["Paper real", "Ta v3.2.2\nfinal", "Ta mixed v4.2\nfinal"]
    position = [1.90, final_v32["position_axis_rmse_mm"], final_mixed["position_axis_rmse_mm"]]
    direction = [3.55, final_v32["orientation_euler_component_rmse_deg"], final_mixed["orientation_euler_component_rmse_deg"]]
    for ax, values, title, unit in (
        (axes[0], position, "Position RMSE theo trục", "mm"),
        (axes[1], direction, "Euler-component direction RMSE", "degree"),
    ):
        bars = ax.bar(labels, values, color=[COLORS["paper"], COLORS["selected"], COLORS["warning"]])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:.3f}", ha="center", weight="bold")
    fig.suptitle("Hình 7 — Chỉ đối chiếu ngữ cảnh, KHÔNG phải head-to-head", fontsize=14.0, weight="bold")
    fig.text(0.5, 0.02,
             "Paper: 500³ mm, broad angles, 100-pose tapered helix. Ta: 100³ mm, chủ yếu pitch, 3,012-point cylinder.",
             ha="center", color=COLORS["warning"], fontsize=9.2, weight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    return save_figure(fig, figures_dir, "fig07_paper_result_context_not_head_to_head")


def plot_runtime(selected_runtime, research_runtime, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    labels = ["Paper\nRTX 3090", "v3.2.2 GPU\nRTX 5060 Ti", "v3.2.2 CPU\ni5-14600K", "v4.4 GPU\nresearch", "v4.4 CPU\nresearch"]
    values = [0.82, selected_runtime["batch_one_p95_ms"]["gpu_cuda_graph_selected_median_repetition"],
              selected_runtime["batch_one_p95_ms"]["cpu_torchscript_median_repetition"],
              research_runtime["gpu_p95_ms"], research_runtime["cpu_p95_ms"]]
    colors = [COLORS["paper"], COLORS["selected"], COLORS["selected"], COLORS["candidate"], COLORS["candidate"]]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(1.5, color=COLORS["warning"], ls="--", lw=1.5, label="v4 GPU gate 1.5 ms")
    ax.axhline(3.0, color=COLORS["neutral"], ls=":", lw=1.5, label="CPU gate 3.0 ms")
    ax.set_ylabel("Reported latency / p95 (ms)")
    ax.set_title("Runtime context — hardware và phạm vi đo khác nhau")
    ax.legend()
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f}", ha="center", weight="bold")
    ax.text(0.5, -0.25,
            "Paper statistic không được mô tả giống hệt p95; số của ta loại acquisition, feature construction và TCP/IP.",
            ha="center", transform=ax.transAxes, color=COLORS["warning"], fontsize=8.8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return save_figure(fig, figures_dir, "fig08_runtime_context")


def plot_physics_diagnostic(physics_folds, reconstruction, figures_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    labels = list(physics_folds)
    corr = [physics_folds[key]["flat_correlation"] for key in labels]
    frac = [physics_folds[key]["rmse_over_mean_fraction"] for key in labels]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, corr, width, color=COLORS["selected"], label="Flat correlation")
    ax.bar(x + width / 2, frac, width, color=COLORS["ours"], label="RMSE / mean|EMF|")
    ax.axhline(0.8, color=COLORS["selected"], ls="--", lw=1)
    ax.axhline(0.5, color=COLORS["ours"], ls="--", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Train-fold physics gates pass")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    channels = np.arange(1, 10)
    ax.plot(channels, reconstruction["Ground-truth pose reconstruction MAPE (%)"], marker="o",
            color=COLORS["warning"], label="GT pose → forward model")
    ax.plot(channels, reconstruction["v3.2.2 pose reconstruction MAPE (%)"], marker="s",
            color=COLORS["selected"], label="v3.2.2 pose")
    ax.axhline(3.4, color=COLORS["paper"], ls="--", lw=1.4, label="Paper reported <3.4%")
    ax.set_xticks(channels)
    ax.set_xlabel("EMF channel")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(b) Opened rotating final: physics mismatch")
    ax.legend(fontsize=7.2)
    fig.suptitle("Hình 9 — Calibration physics: gate cục bộ tốt nhưng chưa generalize sang rotation", fontsize=13.6, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save_figure(fig, figures_dir, "fig09_physics_calibration_audit")


def copy_existing_analogues(figures_dir: Path) -> list[Path]:
    source_dir = ROOT / "reports/paper_style_2026_08_04/figures"
    mapping = {
        "fig4_ablation_and_seed_robustness": "fig10_existing_paper_fig4_analogue",
        "fig7_physics_calibration_effect": "fig11_existing_paper_fig7_analogue",
        "fig8_final_trajectory_orientation_errors": "fig12_existing_paper_fig8_analogue",
        "fig9_physics_consistency": "fig13_existing_paper_fig9_analogue",
    }
    outputs = []
    for old, new in mapping.items():
        for suffix in (".png", ".pdf"):
            source = source_dir / f"{old}{suffix}"
            destination = figures_dir / f"{new}{suffix}"
            shutil.copy2(source, destination)
            outputs.append(destination)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Build the complete paper-vs-current pipeline report.")
    parser.add_argument("--out_dir", default="reports/full_pipeline_comparison_2026_08_05")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    source_paths = {
        "paper": ROOT / "Residual_Neural_Network_for_Precise_6-DoF_Capsule_Endoscope_Localization_Using_Electromagnetic_Induction.pdf",
        "train_npz": ROOT / "data/new_calib_v4_real/C3/deployment/train.npz",
        "split_manifest": ROOT / "data/new_calib_v4_real/C3/deployment/split_manifest.csv",
        "v32_manifest": ROOT / "checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json",
        "v32_runtime": ROOT / "results/new_calib_v3_temporal_deployment/runtime_selection_v3_2_2.json",
        "baseline_dev": ROOT / "results/new_calib_v4_hybrid/ensemble/cv_baseline_v32.json",
        "seed42": ROOT / "results/new_calib_v4_synth_then_calib/cv_blend_posw20_oriw30_seed42.json",
        "seed7": ROOT / "results/new_calib_v4_synth_then_calib/cv_blend_posw20_oriw30_seed7.json",
        "seed123": ROOT / "results/new_calib_v4_synth_then_calib/cv_blend_posw20_oriw30_seed123.json",
        "development_gate": ROOT / "results/new_calib_v4_synth_then_calib/final_development_gate_seed42.json",
        "research_runtime": ROOT / "results/new_calib_v4_synth_then_calib/research_deployment/runtime_posw20_oriw30_median.json",
        "research_manifest": ROOT / "checkpoints/new_calib_v4_synth_then_calib/C3/research_candidate_manifest.json",
        "research_report_manifest": ROOT / "reports/synth_then_calib_2026_08_05/manifest.json",
        "final_v32": ROOT / "results/new_calib_v4_final/baseline_v32/summary.json",
        "final_mixed": ROOT / "results/new_calib_v4_final/selected_hybrid_w20/summary.json",
        "final_decision": ROOT / "results/new_calib_v4_final/final_decision_manifest.json",
        "physics_s1": ROOT / "configs/physics_v4_dev_s1.yaml",
        "physics_s2": ROOT / "configs/physics_v4_dev_s2.yaml",
        "physics_s3": ROOT / "configs/physics_v4_dev_s3.yaml",
        "physics_reconstruction": ROOT / "reports/paper_style_2026_08_04/tables/table4_physics_reconstruction_error.csv",
        "stage_report": ROOT / "reports/synth_then_calib_2026_08_05/REPORT.md",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required frozen artifacts:\n" + "\n".join(missing))

    with np.load(source_paths["train_npz"]) as loaded:
        pose = loaded["pose_deg"].astype(float)
    split_manifest = pd.read_csv(source_paths["split_manifest"])
    train_manifest = split_manifest[split_manifest["role"] == "train"]
    valid_rows = int(train_manifest["valid_rows"].sum())
    invalid_rows = int(train_manifest["invalid_rows"].sum())
    if valid_rows != len(pose):
        raise RuntimeError(f"Train manifest rows {valid_rows} != train.npz rows {len(pose)}")

    workspace_min = np.asarray([-149.02, 340.255, 220.88])
    workspace_size = np.asarray([100.0, 100.0, 100.0])
    cell_index = np.floor((pose[:, :3] - workspace_min) / (workspace_size / 5.0)).astype(int)
    cell_index = np.clip(cell_index, 0, 4)
    coverage_cells = len(np.unique(cell_index, axis=0))

    baseline_dev = read_json(source_paths["baseline_dev"])["aggregate"]
    seed_data = {seed: read_json(source_paths[f"seed{seed}"])["aggregate"] for seed in (42, 7, 123)}
    final_v32 = read_json(source_paths["final_v32"])["aggregate"]
    final_mixed = read_json(source_paths["final_mixed"])["aggregate"]
    selected_runtime = read_json(source_paths["v32_runtime"])
    research_runtime = read_json(source_paths["research_runtime"])
    development_gate = read_json(source_paths["development_gate"])
    final_decision = read_json(source_paths["final_decision"])
    research_manifest = read_json(source_paths["research_report_manifest"])

    physics_folds = {}
    for fold in ("s1", "s2", "s3"):
        with source_paths[f"physics_{fold}"].open() as stream:
            cfg = yaml.safe_load(stream)
        physics_folds[fold] = cfg["calibration_gate"]["dev_metrics"]
    reconstruction = pd.read_csv(source_paths["physics_reconstruction"])

    generated = []
    generated += plot_pipeline(figures_dir)
    generated += plot_architecture(figures_dir)
    generated += plot_leakage(figures_dir)
    generated += plot_data_scale(pose, coverage_cells, figures_dir)
    generated += plot_results(baseline_dev, seed_data, final_v32, final_mixed, figures_dir)
    generated += plot_paper_context(final_v32, final_mixed, figures_dir)
    generated += plot_runtime(selected_runtime, research_runtime, figures_dir)
    generated += plot_physics_diagnostic(physics_folds, reconstruction, figures_dir)
    generated += copy_existing_analogues(figures_dir)

    status_rows = [
        ["1. Bài toán", "9 biên độ EMF → vị trí và hướng 6-DoF", "Cùng bài toán inverse localization", "Gần giống", "Lõi mục tiêu được tái tạo"],
        ["2. Số kênh", "3 TX × 3 RX = 9 EMF", "9 EMF", "Gần giống", "Cùng kích thước quan sát hiện tại"],
        ["3. Workspace", "500×500×500 mm", "100×100×100 mm", "Khác do phần cứng", "Cạnh nhỏ hơn 5×; thể tích nhỏ hơn 125×"],
        ["4. Forward physics", "Mô hình cảm ứng điện từ giải tích", "Cùng họ mô hình dipole/induction + hiệu chỉnh kênh", "Gần về nguyên lý", "Thông số và mapping phần cứng chưa tương đương"],
        ["5. Synthetic", "40³ vị trí × 10³ orientation = 64M", "~5k/fold hoặc 6,019 deployment; causal W3", "Khác lớn", "Ta chưa tái tạo scale và coverage của paper"],
        ["6. Pose synthetic", "Grid có cấu trúc phủ toàn workspace/orientation", "Current pose uniform; history bằng bounded constant velocity", "Khác", "Phù hợp W3 realtime nhưng orientation bị giới hạn bởi calib"],
        ["7. Physical calibration", "150 pose thật fit vị trí/hướng/số vòng TX", "Tối đa 1,200 row/fold; 1,800 deployment; fit 39 tham số", "Cùng mục đích, khác cách", "Đều giảm sim-to-real qua tham số vật lý"],
        ["8. Neural calibration", "Calibrated parameters đưa vào forward/inverse rồi retrain; chi tiết dataset mơ hồ", "Stage 1 synthetic-only → Stage 2 real-only, reset normalization", "Analogue có kiểm soát", "Cùng thứ tự cấp cao, không phải exact reproduction"],
        ["9. Calibration adapter", "Không báo cáo", "Gain dương + bias cho 9 kênh theo generation; identity init + regularization", "Mới", "Nhằm hấp thụ gain/bias giữa domain"],
        ["10. Input thời gian", "Một row 9 EMF", "Causal W3; v4 packed 27 EMF + 2 dt_ratio", "Mới", "Không dùng future row"],
        ["11. Timestamp", "Không dùng", "File hiện tại không có timestamp; mỗi row = 1 unit Δt", "Khác", "Không có timing biến thiên; absolute timestamp bị cấm"],
        ["12. Feature", "Raw EMF", "Raw + past/current log-ratio + relative Δt", "Mới", "Log-ratio giúp bớt nhạy scale; cần gate"],
        ["13. Backbone", "Width 512, 7 residual blocks, 2 linear/block, LeakyReLU 0.01", "Giữ nguyên width/depth/block/slope", "Rất gần", "Đây là phần neural bám sát nhất"],
        ["14. Head", "512→128→6", "v3/v4: 512→128→9", "Khác", "9 = XYZ + rotation-6D"],
        ["15. Orientation target", "cos(roll,pitch,yaw)", "Rotation-6D → Gram–Schmidt → SO(3)", "Mới", "Tránh wrap và đánh giá hình học đúng hơn"],
        ["16. Loss", "MSE", "MSE trên target chuẩn hóa + adapter regularization", "Gần + mở rộng", "Không dùng p95/GT-derived feature lúc inference"],
        ["17. Một model/ensemble", "Một ResNet đề xuất", "Selected v3.2.2 có 5 ResNet; v4.4 blend frozen baseline + C3", "Khác", "Độ chính xác đổi lấy latency/complexity"],
        ["18. Sampling", "Sampling từ grid synthetic", "Inverse-sqrt 5³ cell, cân bằng session/generation, cap 3×", "Mới", "Giảm thiên lệch vùng dày"],
        ["19. Split", "Random 70/15/15 trên synthetic", "Leave-one-session-out development; final family khóa", "Chặt hơn về leakage", "Đánh giá cross-session khó hơn random-row"],
        ["20. Cửa sổ", "Không có", "Tạo sau split; không vượt reset/session/generation", "Chặt hơn", "Ngăn adjacent-window leakage"],
        ["21. Normalization", "Gaussian normalization", "Train-fold feature/target normalization; reset khi real fine-tune", "Gần + chặt hơn", "Thống kê không lấy từ val/final"],
        ["22. Seed", "Không thấy robustness nhiều seed trong kết quả chính", "42 screening; 7 và 123 confirmation", "Chặt hơn", "v4.4 qua accuracy/session gate ở cả ba"],
        ["23. Metric", "Axis position RMSE; Euler-component direction RMSE", "Thêm Euclidean p95, SO(3) RMSE/p95 theo session", "Mở rộng", "Tail error và rotation geometry rõ hơn"],
        ["24. Model gate", "Chọn theo thí nghiệm/ablation được mô tả", "Bốn metric cùng cải thiện + composite ≥3% + per-session ≤5%", "Chặt hơn", "Không chọn vì một metric đẹp"],
        ["25. Final test", "100 pose tapered five-loop helix", "3,012 row, 3 session cylinder constant radius", "Khác", "Fig. 8 của ta không thể có taper giống paper"],
        ["26. Final lock", "Không công bố receipt/hash workflow", "Hash freeze; cyl_rot mở một lần; cấm tune lại", "Chặt hơn về protocol", "v4.2 bị reject; v3.2.2 giữ deployment"],
        ["27. Physics consistency", "EMF reconstruction <3.4%", "GT-pose reconstruction mean channel MAPE 106.19% trên final quay", "Chưa tái tạo", "Forward calibration trên rotation là thiếu sót chính"],
        ["28. Runtime", "0.82 ms, RTX 3090 + i9-10980XE", "v3.2.2: GPU p95 0.858 ms; CPU p95 1.290 ms", "Gần về latency, không direct", "Khác hardware/statistic/scope"],
        ["29. Realtime", "PC localization real time", "Stateful predictor `{timestamp_ns, reset, emf[9]}`; CPU/GPU fallback", "Mở rộng triển khai", "Acquisition/TCP/IP chưa nằm trong benchmark"],
        ["30. Robot/TCP-IP", "Không phải đóng góp chính", "Interface dự kiến model ↔ robot, chưa triển khai loop điều khiển", "Chưa hoàn tất", "Giữ ngoài training để tránh trộn scope"],
    ]
    point_table = pd.DataFrame(status_rows, columns=["Hạng mục", "Bài báo", "Ta", "Phân loại", "Kết luận kiểm toán"])

    network_table = pd.DataFrame(
        [
            ["Input", "9", "W3 raw: 27 hoặc +2 unit-row", "29 packed → 47 internal"],
            ["Backbone", "512, 7 residual blocks", "Mỗi member: 512, 7 blocks", "512, 7 blocks"],
            ["Block", "2×(512 linear), LeakyReLU 0.01", "Giống", "Giống"],
            ["Head/output", "512→128→6", "512→128→9", "512→128→9"],
            ["Orientation", "cos Euler", "rotation-6D", "rotation-6D"],
            ["Temporal", "Không", "Causal W3", "Causal W3 + dt_ratio"],
            ["Calibration adapter", "Không", "Không", "Per-generation 9 gain + 9 bias"],
            ["Feature", "Raw EMF", "Raw hoặc log-ratio branch", "27 raw + 18 log-ratio + 2 dt"],
            ["Composition", "1 model", "5-member component ensemble", "Frozen v3.2.2 + C3 weighted blend"],
            ["Status", "Published proposed", "Selected deployment", "Research-only; not deployed"],
        ],
        columns=["Thuộc tính", "Bài báo", "v3.2.2 selected", "v4.4 research"],
    )

    result_rows = [
        ["Paper simulated: helix fixed", "Paper simulation", "—", 1.85, "—", 2.81, "—", "Paper direction RMSE; not SO(3)"],
        ["Paper optimization: helix fixed", "Paper simulation", "—", 2.88, "—", 3.31, "—", "Paper Table I baseline"],
        ["Paper simulated: helix varying", "Paper simulation", "—", 1.54, "—", 4.81, "—", "Paper direction RMSE; not SO(3)"],
        ["Paper optimization: helix varying", "Paper simulation", "—", 60.91, "—", 52.71, "—", "Paper Table I baseline"],
        ["Paper simulated: random varying", "Paper simulation", "—", 1.64, "—", 5.15, "—", "Paper direction RMSE; not SO(3)"],
        ["Paper optimization: random varying", "Paper simulation", "—", 101.75, "—", 57.81, "—", "Paper Table I baseline"],
        ["Paper real", "Paper real 100 pose", "—", 1.90, "—", 3.55, "—", "Paper direction RMSE; not SO(3)"],
        ["Paper nonlinear optimization", "Paper real 100 pose", "—", 48.85, "—", 68.89, "—", "Paper Table II baseline"],
        ["Paper FCN", "Paper real 100 pose", "—", 101.62, "—", 40.68, "—", "Paper Table II baseline"],
        ["Paper KAN", "Paper real 100 pose", "—", 68.50, "—", 32.17, "—", "Paper Table II baseline"],
        ["v3.2.2 matched baseline", "Our development CV", "none", baseline_dev["position_axis_rmse_mm"], baseline_dev["position_euclidean_p95_mm"], baseline_dev["orientation_geodesic_rmse_deg"], baseline_dev["orientation_geodesic_p95_deg"], "SO(3) metrics"],
    ]
    for seed in (42, 7, 123):
        metric = seed_data[seed]
        result_rows.append([f"v4.4 synth→real", "Our development CV", seed,
                            metric["position_axis_rmse_mm"], metric["position_euclidean_p95_mm"],
                            metric["orientation_geodesic_rmse_deg"], metric["orientation_geodesic_p95_deg"],
                            "Research-only; accuracy gate pass"])
    result_rows += [
        ["v3.2.2", "Our one-time final", "deployment", final_v32["position_axis_rmse_mm"], final_v32["position_euclidean_p95_mm"], final_v32["orientation_geodesic_rmse_deg"], final_v32["orientation_geodesic_p95_deg"], "Selected remains"],
        ["Frozen mixed v4.2", "Our one-time final", 42, final_mixed["position_axis_rmse_mm"], final_mixed["position_euclidean_p95_mm"], final_mixed["orientation_geodesic_rmse_deg"], final_mixed["orientation_geodesic_p95_deg"], "Rejected final"],
        ["v4.4 synth→real", "No valid final", 42, "—", "—", "—", "—", "Must wait for new sealed test"],
    ]
    results_table = pd.DataFrame(result_rows, columns=["Model/scenario", "Protocol", "Seed", "Position RMSE (mm)", "Position p95 (mm)", "Orientation RMSE (deg)", "Orientation p95 (deg)", "Ghi chú/định nghĩa"])

    figure_table = pd.DataFrame(
        [
            ["Fig. 1", "System overview", "Fig01 pipeline side-by-side", "Khái niệm tương ứng; chưa vẽ lại hardware geometry"],
            ["Fig. 2", "TX/RX coordinates và configuration", "Chưa có exact analogue", "Cần đo/xác minh channel mapping, polarity, pose-frame trên rig"],
            ["Fig. 3", "Residual network", "Fig02 architecture side-by-side", "Backbone rất gần; input/output/composition khác"],
            ["Fig. 4", "Ablation FCN/residual, representation, activation, depth", "Fig10 analogue", "Ta có staged training + seed robustness; không tái chạy đúng ablation paper"],
            ["Fig. 5", "Ba simulation trajectories", "Fig04 data/coverage", "Ta chưa tái tạo đúng ba scenario 500 mm"],
            ["Fig. 6", "Real experimental setup", "Chưa có photo/geometry analogue", "CSV robot không thay cho bản vẽ setup đo đạc"],
            ["Fig. 7", "Before/after physical calibration", "Fig11 analogue", "Ta so neural stage trên fold s3; không phải same calibration plot"],
            ["Fig. 8", "Tapered five-loop trajectory + errors", "Fig12 analogue", "Ta vẽ 100 điểm thật, không smooth; path là cylinder constant radius nên hình khác"],
            ["Fig. 9", "EMF reconstructed consistency <3.4%", "Fig13 + Fig09 audit", "Ta không đạt: GT-pose control ~106.19% mean-channel MAPE"],
        ],
        columns=["Hình paper", "Nội dung paper", "Đối ứng của ta", "Mức tương đương trung thực"],
    )

    claim_table = pd.DataFrame(
        [
            ["Ta tái tạo cùng bài toán 9 EMF → 6-DoF", "Được", "Task/input sensor dimensionality aligned"],
            ["Backbone ResNet bám sát paper", "Được", "512 width, 7 blocks, 2 layers/block, slope 0.01"],
            ["Pipeline ta là exact reproduction", "Không", "Workspace, data scale, calibration, split, orientation và ensemble khác"],
            ["Sai số 1.122 mm tốt hơn paper 1.90 mm", "Không", "Protocol hiện tại dễ/hẹp hơn; không head-to-head"],
            ["v4.4 tốt hơn v3.2.2 trên development", "Được", "Cả 4 metric, 3 seed, per-session gate đều pass"],
            ["v4.4 là deployment mới", "Không", "GPU gate fail và không còn new sealed test"],
            ["Mixed v4.2 cải thiện final", "Chỉ orientation", "Position regressed nên balanced gate reject"],
            ["Dữ liệu hiện tại có timestamp thật", "Không", "Mỗi source row chỉ là một programmed unit step"],
            ["W3 dùng dữ liệu thời gian", "Có điều kiện", "Có causal history; unit dt không mang timing biến thiên"],
            ["Calibration physics đã tốt như paper", "Không", "Final rotating reconstruction lệch rất lớn"],
            ["Protocol leakage của ta chặt hơn mô tả paper", "Có thể nói", "Session split, fold-local preprocessing, hash/final gate rõ hơn"],
            ["Ta chứng minh paper bị leakage", "Không", "Paper thiếu chi tiết không đồng nghĩa có leakage"],
            ["Realtime TCP/IP đã hoàn tất", "Không", "Model runtime có interface; network/control integration là bước sau"],
        ],
        columns=["Mệnh đề", "Có được nói?", "Lý do"],
    )

    data_table = pd.DataFrame(
        [
            ["Raw acquisition", "12 CSV", "9 train sessions + 3 locked/final sessions"],
            ["Rows", f"{valid_rows + 3012:,} valid / {valid_rows + 3012 + invalid_rows:,} raw", f"{invalid_rows} quarantined"],
            ["Deployment non-test", f"{len(pose):,} rows", "Used for v3.2.2 and research deployment training"],
            ["Position span", ", ".join(f"{value:.2f}" for value in np.ptp(pose[:, :3], axis=0)) + " mm", "Approximately 100 mm each axis"],
            ["Orientation span", ", ".join(f"{value:.3f}" for value in np.ptp(pose[:, 3:6], axis=0)) + " deg", "Roll/pitch/yaw; primarily pitch"],
            ["Unique position", f"{len(np.unique(np.round(pose[:, :3], 1), axis=0)):,}", "XYZ rounded to 0.1 mm"],
            ["5³ spatial cells", f"{coverage_cells}/125", "Coverage exists but strongly imbalanced"],
            ["Current timestamp", "Absent", "source_row → unit Δt; absolute timestamp forbidden"],
            ["Synthetic Stage 1", f"~5k/fold; {len(pose):,} deployment", "1:1 with available real count; no real rows in Stage 1"],
            ["Real Stage 2", f"~5k/fold; {len(pose):,} deployment", "Real-only supervised calibration; normalization reset"],
        ],
        columns=["Thành phần", "Giá trị", "Ý nghĩa"],
    )

    generated += write_table(point_table, tables_dir, "table01_point_by_point_comparison")
    generated += write_table(network_table, tables_dir, "table02_architecture_comparison")
    generated += write_table(data_table, tables_dir, "table03_data_and_calibration")
    generated += write_table(results_table, tables_dir, "table04_results_by_protocol")
    generated += write_table(figure_table, tables_dir, "table05_figure_mapping")
    generated += write_table(claim_table, tables_dir, "table06_claim_audit")

    mean_seed = {key: float(np.mean([seed_data[s][key] for s in (42, 7, 123)])) for key, _ in [
        ("position_axis_rmse_mm", ""), ("position_euclidean_p95_mm", ""),
        ("orientation_geodesic_rmse_deg", ""), ("orientation_geodesic_p95_deg", "")
    ]}
    gt_mape_mean = float(reconstruction["Ground-truth pose reconstruction MAPE (%)"].mean())
    v32_pose_mape_mean = float(reconstruction["v3.2.2 pose reconstruction MAPE (%)"].mean())
    v4_pose_mape_mean = float(reconstruction["v4 pose reconstruction MAPE (%)"].mean())
    composite_gains = [
        read_json(ROOT / f"results/new_calib_v4_synth_then_calib/quality_gate_posw20_oriw30_seed{seed}.json")
        ["composite_improvement_fraction"] * 100 for seed in (42, 7, 123)
    ]

    report = f"""# Báo cáo kiểm toán: tái tạo bài báo và pipeline EMF hiện tại

Ngày dựng: **2026-08-05**. Đây là báo cáo reporting-only. Script chỉ đọc summary/artifact đã đóng băng, không chạy inference final, không mở raw label `cyl_rot`, không chọn lại model và không đổi deployment.

Bài báo tham chiếu: [Residual Neural Network for Precise 6-DoF Capsule Endoscope Localization Using Electromagnetic Induction](../../Residual_Neural_Network_for_Precise_6-DoF_Capsule_Endoscope_Localization_Using_Electromagnetic_Induction.pdf), IEEE TIM, DOI `10.1109/TIM.2026.3655917`.

## 1. Kết luận ngắn gọn nhưng không đánh tráo

Ta **đi đúng hướng**, nhưng chưa thể gọi là “tái tạo tốt toàn bộ bài báo”. Có thể chia mức độ thành bốn lớp:

1. **Đã tái tạo gần:** bài toán 9 EMF → 6-DoF; forward physics cùng họ; backbone width 512, 7 residual blocks, hai linear mỗi block, LeakyReLU 0.01; inference GPU ở mức dưới 1 ms cho deployment hiện tại.
2. **Đã thay đổi có chủ đích:** causal W3, log-ratio, rotation-6D, ensemble tách position/orientation, generation adapter, spatial/session sampler, grouped CV, multi-seed và final gate.
3. **Protocol của ta rõ/chặt hơn phần paper công bố:** split theo session trước khi tạo window; scaler/physics chỉ fit train fold; target không vào feature; absolute timestamp bị cấm; hash freeze; final test mở một lần; candidate phải thắng cả bốn metric và không làm xấu session tail quá 5%.
4. **Chưa tái tạo:** scale synthetic 64M, coverage orientation rộng, workspace 500 mm, exact 150-pose physical calibration, tapered helix và đặc biệt physics consistency <3.4%. Trên final quay của ta, ngay cả dùng ground-truth pose, mean channel MAPE vẫn **{gt_mape_mean:.2f}%**.

Quyết định hiện tại không thay đổi: **v3.2.2 vẫn là deployment**. v4.4 synthetic-only → real-only là research candidate tốt nhất trên development, nhưng GPU p95 **{research_runtime['gpu_p95_ms']:.3f} ms** vượt gate 1.5 ms và không còn sealed test mới.

![Hai pipeline](figures/fig01_pipeline_side_by_side.png)

## 2. Đối chiếu từng hạng mục

Bảng đầy đủ có bản [CSV](tables/table01_point_by_point_comparison.csv) và [Markdown](tables/table01_point_by_point_comparison.md).

{markdown_table(point_table)}

Nhãn “chặt hơn” chỉ nói về **protocol được hiện thực và ghi artifact rõ hơn**, không nói mọi mặt khoa học của ta tốt hơn paper. Paper có ưu thế rất lớn về workspace, coverage orientation, quy mô synthetic và kiểm chứng physics.

## 3. Dữ liệu synthetic và calibration: mục đích giống, cơ chế khác

### 3.1 Paper làm gì

Paper dùng forward model điện từ để tạo `40³ × 10³ = 64,000,000` cặp synthetic trong cube `500³ mm`. Mục tiêu là phủ dày không gian 6-DoF với nhãn pose chính xác, tránh phải robot-acquire hàng chục triệu điểm. Sau đó 150 pose thật được dùng để nhận dạng lại tham số phần cứng hiệu dụng; calibrated parameters được đưa vào forward/inverse model để retrain. Paper không mô tả đủ chi tiết để biết chính xác toàn bộ thành phần của retraining dataset.

### 3.2 Ta thực sự làm gì

Ta có 12 file robot CSV: 9 session non-test và 3 session `cyl_rot` đã từng khóa rồi mở đúng một lần. Trong tổng **{valid_rows + 3012 + invalid_rows:,} raw rows**, có **{valid_rows + 3012:,} valid** và **{invalid_rows} quarantined**. Deployment non-test có **{len(pose):,} rows** và phủ **{coverage_cells}/125** cell của lưới `5³`.

V4 fit forward model **riêng trong từng train fold**, dùng tối đa 1,200 train rows/fold và bốn initialization; deployment fit 1,800 rows. Bộ tham số gồm vị trí/hướng/moment của ba TX và gain/bias 9 kênh, tổng 39 tham số. Generator chỉ được bật nếu held-out physics correlation ≥0.8 và RMSE/mean(|EMF|) ≤0.5. Ba fold đạt correlation `{min(v['flat_correlation'] for v in physics_folds.values()):.3f}–{max(v['flat_correlation'] for v in physics_folds.values()):.3f}` và ratio `{min(v['rmse_over_mean_fraction'] for v in physics_folds.values()):.3f}–{max(v['rmse_over_mean_fraction'] for v in physics_folds.values()):.3f}`.

Mỗi synthetic sample là một causal W3: current pose uniform trong phạm vi calibrated workspace/orientation; hai pose quá khứ được tạo bằng vận tốc hằng ngẫu nhiên, tối đa 2 mm/step và 1°/step; EMF được forward-model hóa; `dt_ratio=1`. Stage 1 có số synthetic bằng 1× số real available nhưng **không trộn real**. Stage 2 chỉ dùng real labels và reset normalization.

Điều này gần tinh thần “synthetic trước, calibration sau”, nhưng không exact: physics generator của ta đã được fit từ train-fold real trước Stage 1; paper mô tả analytical synthetic ban đầu rồi mới physical calibration. Vì thế claim đúng là **paper-order neural analogue with fold-local calibrated physics**, không phải full reproduction.

![Data scale](figures/fig04_data_scale_and_coverage.png)

{markdown_table(data_table)}

### 3.3 Calibration data có thực sự tốt không?

Nó tốt cho supervised inverse model trong đúng domain: robot label lặp lại rất nhỏ và development error thấp. Nhưng calibration data hiện tại **chưa đủ tốt để chứng minh broad 6-DoF**:

- XYZ phủ gần 100 mm mỗi trục, nhưng density không đều; một số cell có rất nhiều điểm.
- Roll span chỉ `{np.ptp(pose[:, 3]):.3f}°`, pitch `{np.ptp(pose[:, 4]):.3f}°`, yaw `{np.ptp(pose[:, 5]):.3f}°`. Do đó “6-DoF” về output không đồng nghĩa đã stress-test đủ ba góc.
- File hiện tại không có timestamp thật. Mỗi row chỉ là một programmed unit step. W3 học thay đổi EMF theo lịch sử, còn `dt_ratio=1` gần như không thêm timing information biến thiên.
- Same-pose EMF giữa session còn drift đáng kể; learned adapter có thể giúp gain/bias nhưng không thể sửa sai channel mapping, polarity hoặc pose-frame.
- Physics gate trên held-out `con_rot` pass, nhưng final `cyl_rot` quay cho thấy forward model không generalize: GT-pose MAPE {gt_mape_mean:.2f}%, v3.2.2-pose {v32_pose_mape_mean:.2f}%, v4-pose {v4_pose_mape_mean:.2f}%.

![Physics audit](figures/fig09_physics_calibration_audit.png)

## 4. Mạng neural: giống ở backbone, khác ở bài toán biểu diễn

![Architecture](figures/fig02_architecture_side_by_side.png)

{markdown_table(network_table)}

Đổi mới chính không phải “ResNet sâu hơn”. Ta chủ ý giữ backbone paper để cô lập tác động của dữ liệu/feature/protocol. Phần mới nằm ở:

- **Causal W3:** pose hiện tại dùng row hiện tại và hai row quá khứ; không dùng tương lai.
- **Log-ratio:** `log(past+eps)-log(current+eps)` giúp mô tả biến đổi tương đối và giảm nhạy với scale chung.
- **Rotation-6D:** mạng dự đoán hai vector của rotation matrix, Gram–Schmidt về SO(3); loss/evaluation không bị wrap Euler như biểu diễn góc trực tiếp.
- **Component ensemble v3.2.2:** XYZ là trung bình ba raw-W3 seed và một rowtime-W3 member; orientation đến từ log-ratio branch. Đây là khác biệt lớn so với single model paper.
- **C3 adapter:** mỗi generation có gain dương `exp(log_gain)` và bias 9 kênh, dùng cùng correction cho mọi row trong window; identity initialization và L2-to-identity regularization.
- **Balanced sampling:** cân bằng generation/session và inverse-square-root spatial-cell frequency, cap 3×.
- **Stage-order v4.4:** 107 fixed pure-synthetic epochs → 108 fixed real-only deployment epochs; normalization được reset trước real stage.

## 5. Leakage audit

![Leakage boundaries](figures/fig03_validation_and_leakage_boundaries.png)

Các điểm đã khóa đúng:

- Split complete acquisition session trước khi fit scaler, physics model hoặc tạo temporal window.
- Window không vượt `reset`, session hoặc generation.
- Chỉ `Δt / median(Δt_train)` được phép; absolute timestamp/row index không là feature.
- Current files không có hardware timestamp nên dùng `source_row` như unit step; timestamp shift parity bằng 0 chỉ chứng minh absolute time không lọt vào predictor.
- Ground-truth position/orientation, speed/boundary từ target chỉ dùng cho train sampling/loss analysis, không dùng ở inference feature.
- Candidate selection chỉ dựa development. `cyl_rot` mở một lần ngày 2026-08-04 sau hash freeze; không được dùng lại để chỉnh v4.4.

Hạn chế phải ghi: raw final labels tồn tại local; manifest cũ ghi một preprocessing audit từng nhìn target-derived restart boundary. Sau v4, boundary được lấy từ source-row/reset. Vì lịch sử này, final cũ không nên gọi là clinical/publication-grade blind test. Cần acquisition sealed mới do một quy trình/custodian độc lập giữ.

Không có bằng chứng để nói paper bị leakage. Paper công bố random synthetic split; việc thiếu session-wise detail chỉ làm ta **không đánh giá được mức chống leakage**, không chứng minh leakage tồn tại.

## 6. Kết quả: development, final và paper phải tách protocol

### 6.1 Development CV — v4.4 thực sự tốt hơn baseline trong phạm vi này

![Development results](figures/fig05_development_multiseed_results.png)

Baseline matched-fold: position RMSE `{baseline_dev['position_axis_rmse_mm']:.4f} mm`, position p95 `{baseline_dev['position_euclidean_p95_mm']:.4f} mm`, SO(3) RMSE `{baseline_dev['orientation_geodesic_rmse_deg']:.4f}°`, SO(3) p95 `{baseline_dev['orientation_geodesic_p95_deg']:.4f}°`.

V4.4 frozen rule:

```text
XYZ = 0.8 × v3.2.2 + 0.2 × C3
rotation6D = 0.7 × v3.2.2 + 0.3 × C3, rồi Gram–Schmidt
```

Trung bình ba seed: `{mean_seed['position_axis_rmse_mm']:.4f} mm`, `{mean_seed['position_euclidean_p95_mm']:.4f} mm`, `{mean_seed['orientation_geodesic_rmse_deg']:.4f}°`, `{mean_seed['orientation_geodesic_p95_deg']:.4f}°`. Composite gains là `{composite_gains[0]:.2f}%`, `{composite_gains[1]:.2f}%`, `{composite_gains[2]:.2f}%`; cả ba qua aggregate + per-session accuracy gates.

### 6.2 Final đã mở một lần — chỉ v3.2.2 và mixed v4.2 có kết quả hợp lệ

![One-time final](figures/fig06_one_time_final_decision.png)

V3.2.2 final: position RMSE `{final_v32['position_axis_rmse_mm']:.4f} mm`, p95 `{final_v32['position_euclidean_p95_mm']:.4f} mm`, SO(3) RMSE `{final_v32['orientation_geodesic_rmse_deg']:.4f}°`, p95 `{final_v32['orientation_geodesic_p95_deg']:.4f}°`.

Frozen mixed v4.2 final: `{final_mixed['position_axis_rmse_mm']:.4f} mm`, `{final_mixed['position_euclidean_p95_mm']:.4f} mm`, `{final_mixed['orientation_geodesic_rmse_deg']:.4f}°`, `{final_mixed['orientation_geodesic_p95_deg']:.4f}°`. Orientation tốt hơn nhưng position xấu hơn; all-four gate reject. Không được lấy bài học từ final đó rồi phối lại heads — làm vậy là test leakage.

V4.4 **không có final result**. Không được suy diễn rằng development gain sẽ giữ trên new generation.

### 6.3 Đặt cạnh paper chỉ để hiểu scale

![Paper context](figures/fig07_paper_result_context_not_head_to_head.png)

Paper báo `1.90 mm` position và `3.55°` direction trên real setup. Ta có thể đặt số cạnh nhau, nhưng không được kết luận ta outperform vì:

- workspace paper có thể tích lớn hơn 125×;
- paper có roll/pitch/yaw 20–160°, ta chủ yếu pitch;
- paper dùng tapered five-loop helix 100 pose, ta dùng cylinder khoảng bán kính không đổi và 3,012 rows;
- phần cứng, calibration, training distribution và tail metric khác.

Các hàng optimization/FCN/KAN dưới đây là baseline **trên dữ liệu của paper**.
Ta chưa chạy ba phương pháp này trên current calibration data, nên chúng không
phải benchmark ngang hàng của model hiện tại.

{markdown_table(results_table)}

## 7. Runtime và mục tiêu realtime

![Runtime](figures/fig08_runtime_context.png)

Deployment v3.2.2: CPU TorchScript p95 `{selected_runtime['batch_one_p95_ms']['cpu_torchscript_median_repetition']:.3f} ms`; GPU CUDA Graph p95 `{selected_runtime['batch_one_p95_ms']['gpu_cuda_graph_selected_median_repetition']:.3f} ms`. CPU/GPU max pose difference `0.00007` trên 1,001 development rows.

V4.4 research: CPU p95 `{research_runtime['cpu_p95_ms']:.3f} ms` pass 3 ms; GPU p95 `{research_runtime['gpu_p95_ms']:.3f} ms` fail 1.5 ms. Parity max difference `{research_manifest['runtime_parity']['cpu_gpu_max_abs_pose_difference']:.8f}` và timestamp-offset difference `{research_manifest['runtime_parity']['absolute_timestamp_feature_invariance_max_abs_pose_difference']:.1f}`.

Các benchmark trên chỉ gồm network/ensemble path như manifest mô tả; chưa gồm sensor acquisition, feature construction đầy đủ, confidence, TCP/IP, robot controller hoặc scheduling jitter. Interface realtime đã định hình là:

```text
input  = {{timestamp_ns, reset, emf[9]}}
output = {{x, y, z, roll, pitch, yaw, confidence/dispersion}}
```

TCP/IP nên là lớp transport sau predictor. Model định vị capsule; robot chỉ cung cấp calibration labels lúc train và sau này nhận pose/command ở runtime. Không đưa trạng thái robot/target vào feature nếu runtime capsule không có tín hiệu tương ứng.

## 8. Đối chiếu từng hình paper

{markdown_table(figure_table)}

### Fig. 4 analogue hiện tại

![Fig4 analogue](figures/fig10_existing_paper_fig4_analogue.png)

Hình này không giả vờ tái tạo ablation FCN/cos/tanh/depth của paper. Nó mô tả staged learning và robustness theo seed của ta. Muốn gọi là reproduction Fig. 4 phải chạy đúng candidate và split paper-style, điều hiện chưa làm.

### Fig. 7 analogue hiện tại

![Fig7 analogue](figures/fig11_existing_paper_fig7_analogue.png)

So sánh real-only C3 và physics-pretrain→real trên fold s3. Đây là effect của pipeline neural calibration, không phải biểu đồ before/after 150-pose TX physical calibration giống paper.

### Fig. 8 analogue hiện tại

![Fig8 analogue](figures/fig12_existing_paper_fig8_analogue.png)

Ta lấy 100 pose uniform từ session 3 sạch và không smooth error. Hình paper “dưới to, trên nhỏ dần” vì ground-truth là tapered helix. Dữ liệu robot của ta gần cylinder bán kính 50 mm không đổi, nên hình đều/smooth hơn là đúng dữ liệu. Ép taper bằng plotting sẽ là sai khoa học.

### Fig. 9 analogue hiện tại

![Fig9 analogue](figures/fig13_existing_paper_fig9_analogue.png)

GT-pose control chứng minh phần lớn chênh lệch EMF đến từ forward calibration/channel/frame mismatch, không phải riêng inverse pose model. Đây là kết quả âm quan trọng và phải giữ nguyên.

## 9. Claim audit

{markdown_table(claim_table)}

## 10. Ta đã làm gì mới so với paper?

Các hướng có tính mới ở cấp pipeline/engineering research, nhưng chưa nên gọi là novel publication contribution nếu chưa có ablation + independent test:

1. Causal history W3 cùng reset/session-safe windowing.
2. Relative log-ratio feature và component-wise ensemble.
3. Rotation-6D + SO(3) geodesic metrics thay cos-Euler output.
4. Generation-specific positive gain/bias calibration adapter.
5. Fold-local fitted physics generator có held-out physical gate.
6. Pure-synthetic pretraining → real-only neural calibration với normalization rebase.
7. Spatial/session/generation balanced sampler.
8. Multi-objective, multi-seed, per-session tail gate và one-time sealed protocol.
9. CPU fallback + CUDA Graph runtime, parity và timestamp invariance checks.

Về logic kiểm định, kế hoạch của ta chặt hơn phần paper mô tả. Về độ bao phủ vật lý và sức nặng bằng chứng, paper vẫn mạnh hơn rõ rệt.

## 11. Bước tiếp theo trước bộ calib mới

1. Thu tối thiểu 3 rotating development sessions và 1 later sealed session, tách ngày/session rõ ràng.
2. Mở rộng roll và yaw độc lập, không chỉ pitch; ưu tiên thiết kế pose coverage thay vì thêm nhiều row lặp cùng trajectory.
3. Ghi `timestamp_ns` thật nếu tốc độ sampling có biến thiên. Nếu robot vẫn fixed-period, lưu period metadata; không kỳ vọng constant dt tự cải thiện model.
4. Xác minh channel mapping, polarity và pose-frame transform bằng hardware procedure trước khi tăng synthetic scale.
5. Fit physics chỉ trên development train sessions; yêu cầu held-out rotating physics consistency tốt hơn đáng kể trước khi sinh nhiều synthetic.
6. Freeze v4.4 architecture/rule hoặc một thay đổi được predeclare; đánh giá new sealed đúng một lần.
7. Sau khi model khóa, benchmark end-to-end acquisition → predictor → TCP/IP, không chỉ network forward.

## 12. Tái tạo báo cáo

```bash
# Chạy từ thư mục gốc của repository.
./.venv/bin/python reporting/build_full_pipeline_comparison.py \\
  --out_dir reports/full_pipeline_comparison_2026_08_05
```

Các bảng machine-readable nằm trong `tables/`; mỗi hình có PNG và PDF. `manifest.json` ghi SHA-256 của mọi source và output. Báo cáo chi tiết cũ vẫn ở [docs/BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md](../../docs/BAO_CAO_SO_SANH_PIPELINE_VOI_BAI_BAO.md).
"""
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report)
    generated.append(report_path)

    manifest = {
        "schema_version": 1,
        "date": "2026-08-05",
        "report_type": "reporting_only_full_pipeline_paper_comparison",
        "paper_doi": "10.1109/TIM.2026.3655917",
        "final_predictions_or_raw_labels_read": False,
        "training_or_model_selection_performed": False,
        "selected_deployment_changed": False,
        "selected_deployment": final_decision["selected_deployment"],
        "research_candidate": "v4.4_synthetic_only_then_real_only_pos20_ori30",
        "research_candidate_final_status": "not_evaluated_requires_new_sealed_test",
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths.values()},
        "generated_sha256": {str(path.relative_to(out_dir)): sha256_file(path) for path in generated},
        "derived_checks": {
            "non_test_rows": len(pose),
            "spatial_cells_occupied_of_125": coverage_cells,
            "v4_4_accuracy_and_session_gate_all_seeds_pass": research_manifest["accuracy_and_session_gate_all_seeds_pass"],
            "v4_4_final_development_gate_accepted": development_gate["accepted"],
            "v4_4_gpu_latency_gate": development_gate["checks"]["gpu_latency_within_budget"],
            "historical_final_test_opened_once": final_decision["test_opened_once"],
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "report": str(report_path.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "figures": len(list(figures_dir.glob("*.png"))),
        "tables": len(list(tables_dir.glob("*.csv"))),
        "selected_deployment_changed": False,
    }, indent=2))


if __name__ == "__main__":
    main()

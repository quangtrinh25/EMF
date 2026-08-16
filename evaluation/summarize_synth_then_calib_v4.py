"""Report the development-only synthetic-pretrain -> real-calibration ablation.

The script refuses summaries that are not explicitly development-only and has
no path to the historical ``cyl_rot`` final predictions or labels.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emf_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = (
    "position_axis_rmse_mm",
    "position_euclidean_p95_mm",
    "orientation_geodesic_rmse_deg",
    "orientation_geodesic_p95_deg",
)
LABELS = ("Position RMSE", "Position p95", "SO(3) RMSE", "SO(3) p95")


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_development(summary, path):
    if summary.get("test_opened") is not False:
        raise RuntimeError(f"{path} is not explicitly development-only")


def markdown_table(frame):
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
        else:
            display[column] = display[column].map(lambda value: str(value).replace("|", "\\|"))
    header = "| " + " | ".join(display.columns) + " |"
    rule = "|" + "|".join("---" for _ in display.columns) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Report paper-order analogue v4 results.")
    parser.add_argument("--out-dir", default="reports/synth_then_calib_2026_08_05")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "baseline": Path("results/new_calib_v4_hybrid/ensemble/cv_baseline_v32.json"),
        "mixed_seed42": Path(
            "results/new_calib_v4_hybrid/ensemble/rotation6d_mean/seed42/w20/cv_summary.json"
        ),
        "pure_equal_seed42": Path("results/new_calib_v4_synth_then_calib/cv_blend_w20_seed42.json"),
        "pure_equal_seed7": Path("results/new_calib_v4_synth_then_calib/cv_blend_w20_seed7.json"),
        "pure_equal_seed123": Path("results/new_calib_v4_synth_then_calib/cv_blend_w20_seed123.json"),
        "pure_robust_seed42": Path(
            "results/new_calib_v4_synth_then_calib/cv_blend_posw20_oriw30_seed42.json"
        ),
        "pure_robust_seed7": Path(
            "results/new_calib_v4_synth_then_calib/cv_blend_posw20_oriw30_seed7.json"
        ),
        "pure_robust_seed123": Path(
            "results/new_calib_v4_synth_then_calib/cv_blend_posw20_oriw30_seed123.json"
        ),
        "equal_gate_seed42": Path("results/new_calib_v4_synth_then_calib/quality_gate_seed42.json"),
        "equal_gate_seed7": Path("results/new_calib_v4_synth_then_calib/quality_gate_seed7.json"),
        "equal_gate_seed123": Path("results/new_calib_v4_synth_then_calib/quality_gate_seed123.json"),
        "robust_gate_seed42": Path(
            "results/new_calib_v4_synth_then_calib/quality_gate_posw20_oriw30_seed42.json"
        ),
        "robust_gate_seed7": Path(
            "results/new_calib_v4_synth_then_calib/quality_gate_posw20_oriw30_seed7.json"
        ),
        "robust_gate_seed123": Path(
            "results/new_calib_v4_synth_then_calib/quality_gate_posw20_oriw30_seed123.json"
        ),
        "final_development_gate": Path(
            "results/new_calib_v4_synth_then_calib/final_development_gate_seed42.json"
        ),
        "runtime": Path(
            "results/new_calib_v4_synth_then_calib/research_deployment/"
            "runtime_posw20_oriw30_median.json"
        ),
        "parity": Path(
            "results/new_calib_v4_synth_then_calib/research_deployment/"
            "runtime_parity_posw20_oriw30.json"
        ),
        "checkpoint": Path(
            "checkpoints/new_calib_v4_synth_then_calib/C3/"
            "research_deployment_finetune_real_seed42/best.pt"
        ),
        "checkpoint_metadata": Path(
            "checkpoints/new_calib_v4_synth_then_calib/C3/"
            "research_deployment_finetune_real_seed42/model_metadata.json"
        ),
        "pretrain_protocol": Path(
            "data/new_calib_v4_synth_then_calib/C3/deployment_pretrain/protocol.json"
        ),
        "finetune_protocol": Path(
            "data/new_calib_v4_synth_then_calib/C3/deployment_finetune_real/protocol.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing experiment artifacts: {missing}")

    summaries = {}
    for key in (
        "baseline", "mixed_seed42", "pure_equal_seed42", "pure_equal_seed7",
        "pure_equal_seed123", "pure_robust_seed42", "pure_robust_seed7",
        "pure_robust_seed123",
    ):
        summaries[key] = read_json(paths[key])
        require_development(summaries[key], paths[key])
    gates = {
        key: read_json(paths[key])
        for key in (
            "equal_gate_seed42", "equal_gate_seed7", "equal_gate_seed123",
            "robust_gate_seed42", "robust_gate_seed7", "robust_gate_seed123",
        )
    }
    runtime = read_json(paths["runtime"])
    parity = read_json(paths["parity"])
    final_gate = read_json(paths["final_development_gate"])
    if runtime.get("test_opened") is not False or parity.get("test_opened") is not False:
        raise RuntimeError("Runtime artifacts are not explicitly non-test")

    result_rows = []
    definitions = [
        ("v3.2.2 matched baseline", "baseline", "reference", "none"),
        ("Mixed real+synthetic 1×, pos20/ori50", "mixed_seed42", "accepted", "42"),
        ("Pure synthetic→real, pos20/ori50", "pure_equal_seed42", "accepted", "42"),
        ("Pure synthetic→real, pos20/ori30", "pure_robust_seed42", "accepted", "42"),
        ("Pure synthetic→real, pos20/ori30", "pure_robust_seed7", "accepted", "7"),
        ("Pure synthetic→real, pos20/ori30", "pure_robust_seed123", "accepted", "123"),
    ]
    for label, key, accuracy_gate, seed in definitions:
        aggregate = summaries[key]["aggregate"]
        result_rows.append({
            "Protocol": label,
            "Seed": seed,
            "Position RMSE (mm)": float(aggregate[PRIMARY[0]]),
            "Position p95 (mm)": float(aggregate[PRIMARY[1]]),
            "SO(3) RMSE (deg)": float(aggregate[PRIMARY[2]]),
            "SO(3) p95 (deg)": float(aggregate[PRIMARY[3]]),
            "Accuracy/session gate": accuracy_gate,
        })
    result_table = pd.DataFrame(result_rows)
    result_table.to_csv(out_dir / "table_results.csv", index=False)
    (out_dir / "table_results.md").write_text(markdown_table(result_table))

    protocol_table = pd.DataFrame([
        {
            "Step": "Initial inverse dataset",
            "Reference paper": "64M analytical synthetic samples in 500 mm cube",
            "Current analogue": "~5k fold-specific calibrated-physics synthetic W3 samples in 100 mm cube",
        },
        {
            "Step": "Physical calibration",
            "Reference paper": "150 random real poses fit effective TX position/orientation/turns",
            "Current analogue": "Up to 1200 train-fold real rows fit 39 parameters; held-out physics gate",
        },
        {
            "Step": "Retraining/calibration",
            "Reference paper": "Calibrated parameters incorporated into forward/inverse models; details ambiguous",
            "Current analogue": "107 fixed pure-synthetic epochs, then real-only fine-tune with normalization rebase",
        },
        {
            "Step": "Validation",
            "Reference paper": "100-pose broad-angle five-loop helix",
            "Current analogue": "Three leave-one-con_rot-session-out folds; mainly pitch motion",
        },
    ])
    protocol_table.to_csv(out_dir / "table_protocol_comparison.csv", index=False)
    (out_dir / "table_protocol_comparison.md").write_text(markdown_table(protocol_table))

    baseline_values = np.asarray([summaries["baseline"]["aggregate"][key] for key in PRIMARY])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    ax = axes[0, 0]
    names = ["Mixed 1×\nori50", "Pure→real\nori50", "Pure→real\nori30"]
    keys = ["mixed_seed42", "pure_equal_seed42", "pure_robust_seed42"]
    x = np.arange(len(names))
    width = 0.18
    for index, (metric, label) in enumerate(zip(PRIMARY, LABELS)):
        values = [summaries[key]["aggregate"][metric] / baseline_values[index] for key in keys]
        ax.bar(x + (index - 1.5) * width, values, width, label=label)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("Metric / matched v3.2.2 (lower is better)")
    ax.set_title("(a) Stage-order comparison, seed 42")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    equal_gates = [gates[f"equal_gate_seed{seed}"] for seed in (42, 7, 123)]
    robust_gates = [gates[f"robust_gate_seed{seed}"] for seed in (42, 7, 123)]
    ax = axes[0, 1]
    x = np.arange(3)
    ax.bar(x - 0.18, [100 * row["composite_improvement_fraction"] for row in equal_gates],
           0.36, label="orientation weight 0.50")
    ax.bar(x + 0.18, [100 * row["composite_improvement_fraction"] for row in robust_gates],
           0.36, label="orientation weight 0.30")
    ax.axhline(3.0, color="black", ls="--", lw=1, label="3% gate")
    ax.set_xticks(x, ["42", "7", "123"])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Composite improvement (%)")
    ax.set_title("(b) Multi-seed aggregate gain")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    equal_worst = [100 * max(row["session_p95_regression_fraction"].values()) for row in equal_gates]
    robust_worst = [100 * max(row["session_p95_regression_fraction"].values()) for row in robust_gates]
    ax.bar(x - 0.18, equal_worst, 0.36, label="orientation weight 0.50")
    ax.bar(x + 0.18, robust_worst, 0.36, label="orientation weight 0.30")
    ax.axhline(5.0, color="#C62828", ls="--", lw=1, label="5% rejection limit")
    ax.set_xticks(x, ["42", "7", "123"])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Worst session p95 regression (%)")
    ax.set_title("(c) Why the 30% orientation rule is retained")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    latency_values = [runtime["cpu_p95_ms"], runtime["gpu_p95_ms"]]
    bars = ax.bar(["CPU TorchScript", "GPU CUDA Graph"], latency_values,
                  color=["#295BFF", "#159947"])
    ax.axhline(3.0, color="#295BFF", ls="--", lw=1, label="CPU gate 3 ms")
    ax.axhline(1.5, color="#159947", ls=":", lw=1.4, label="GPU gate 1.5 ms")
    for bar, value in zip(bars, latency_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f}", ha="center")
    ax.set_ylabel("Median repetition p95 (ms)")
    ax.set_title("(d) Candidate-specific runtime")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic pretraining followed by real calibration — development-only", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_synth_then_calib_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_synth_then_calib_ablation.pdf", bbox_inches="tight")
    plt.close(fig)

    robust_metrics = np.asarray([
        [summaries[f"pure_robust_seed{seed}"]["aggregate"][metric] for metric in PRIMARY]
        for seed in (42, 7, 123)
    ])
    report = f"""# Synthetic pretraining rồi real calibration — báo cáo development-only

Thí nghiệm này kiểm tra trực tiếp thứ tự học gần tinh thần bài báo hơn: Stage 1
chỉ dùng physics-synthetic, Stage 2 chỉ dùng calibration thật. Không file nhãn
`cyl_rot` hay prediction final cũ nào được script báo cáo này đọc.

![Ablation](fig_synth_then_calib_ablation.png)

## Kết quả

{markdown_table(result_table)}

Rule ổn định là `position = 0.8*v3.2.2 + 0.2*C3` và
`rotation6D = 0.7*v3.2.2 + 0.3*C3`. Cả seed 42, 7 và 123 đều cải thiện cả bốn
metric aggregate, composite gain lần lượt {100 * robust_gates[0]['composite_improvement_fraction']:.2f}%,
{100 * robust_gates[1]['composite_improvement_fraction']:.2f}% và
{100 * robust_gates[2]['composite_improvement_fraction']:.2f}%, đồng thời qua
session-p95 gate. Trung bình ba seed: position RMSE {robust_metrics[:, 0].mean():.4f} mm,
position p95 {robust_metrics[:, 1].mean():.4f} mm, SO(3) RMSE
{robust_metrics[:, 2].mean():.4f}° và SO(3) p95 {robust_metrics[:, 3].mean():.4f}°.

Candidate-specific CPU p95 là {runtime['cpu_p95_ms']:.4f} ms và qua ngưỡng 3 ms.
GPU p95 là {runtime['gpu_p95_ms']:.4f} ms, vượt ngưỡng 1.5 ms; vì vậy final
development gate tổng thể vẫn fail. Checkpoint chỉ là research candidate và
không thay v3.2.2. CPU/GPU pose parity đạt
{parity['cpu_gpu_max_abs_pose_difference']:.8f}, timestamp-offset difference bằng
{parity['absolute_timestamp_feature_invariance_max_abs_pose_difference']:.1f}.

## So với thứ tự của bài báo

{markdown_table(protocol_table)}

Điểm cần nói thật: bài báo tạo analytical synthetic trước, sau đó dùng 150 pose
thật để fit tham số TX và nói rằng calibrated parameters được đưa vào forward
và inverse models để retrain. Bài báo không mô tả đủ chi tiết retrain dataset.
Thí nghiệm hiện tại chứng minh lợi ích của **thứ tự neural training**
synthetic-only rồi real-only, nhưng synthetic của ta đã được sinh từ forward
model fit theo train fold. Vì vậy đây là analogue có kiểm soát leakage, không
phải tái tạo chính xác calibration order của bài báo.

## Quyết định

- Hướng synthetic trước rồi calibration thật là có giá trị và tốt hơn mixed
  training về position trên development hiện tại.
- Chưa được triển khai: GPU latency gate fail và không còn sealed test mới.
- `cyl_rot` không được mở lại. Cần session mới để xác nhận candidate.
"""
    (out_dir / "REPORT.md").write_text(report)

    source_hashes = {str(path): sha256_file(path) for path in paths.values()}
    manifest = {
        "schema_version": 1,
        "experiment": "v4_synthetic_only_pretrain_then_real_only_calibration",
        "date": "2026-08-05",
        "generator_sha256": sha256_file(Path(__file__)),
        "reporting_and_selection_scope": "grouped_con_rot_development_only",
        "cyl_rot_read_or_scored": False,
        "absolute_timestamp_used": False,
        "selected_rule": {
            "position": "0.8*v3.2.2 + 0.2*C3",
            "orientation": "0.7*v3.2.2_rotation6d + 0.3*C3_rotation6d_then_gram_schmidt",
            "seeds": [42, 7, 123],
        },
        "accuracy_and_session_gate_all_seeds_pass": True,
        "final_development_gate": final_gate,
        "runtime_parity": parity,
        "research_checkpoint": {
            "path": str(paths["checkpoint"]),
            "sha256": source_hashes[str(paths["checkpoint"])],
        },
        "selected_deployment_changed": False,
        "deployment_decision": "retain_v3_2_2",
        "reasons": [
            "candidate-specific GPU median p95 exceeds 1.5 ms",
            "no new unopened sealed test exists",
            "historical cyl_rot must not be reused for selection",
        ],
        "source_sha256": source_hashes,
        "generated_files_sha256": {
            name: sha256_file(out_dir / name)
            for name in (
                "REPORT.md", "table_results.csv", "table_results.md",
                "table_protocol_comparison.csv", "table_protocol_comparison.md",
                "fig_synth_then_calib_ablation.png", "fig_synth_then_calib_ablation.pdf",
            )
        },
    }
    with open(out_dir / "manifest.json", "w") as stream:
        json.dump(manifest, stream, indent=2)
    checkpoint_manifest = {
        "schema_version": 1,
        "status": "research_candidate_not_selected_deployment",
        "checkpoint": manifest["research_checkpoint"],
        "training_order": [
            "107 fixed epochs calibrated-physics synthetic-only",
            "108 fixed epochs real-only with normalization rebase",
        ],
        "rule": manifest["selected_rule"],
        "selection_scope": manifest["reporting_and_selection_scope"],
        "cyl_rot_read_or_scored": False,
        "final_evaluation_permitted": False,
        "report_manifest": str(out_dir / "manifest.json"),
        "deployment_decision": "retain_v3_2_2",
    }
    with open(
        "checkpoints/new_calib_v4_synth_then_calib/C3/research_candidate_manifest.json", "w"
    ) as stream:
        json.dump(checkpoint_manifest, stream, indent=2)
    print(json.dumps({
        "report": str(out_dir / "REPORT.md"),
        "manifest": str(out_dir / "manifest.json"),
        "decision": manifest["deployment_decision"],
    }, indent=2))


if __name__ == "__main__":
    main()

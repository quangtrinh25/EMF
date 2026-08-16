"""Build a Vietnamese presentation deck and speaker script from frozen reports.

Reporting-only: this script consumes already generated comparison figures.  It
does not load raw calibration/final labels, run inference, train, or select a
model.  Outputs are 16:9 PNG slides, one combined PDF, a detailed speaker
script, an outline CSV, and a SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emf_presentation_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIGURES = ROOT / "reports/full_pipeline_comparison_2026_08_05/figures"

NAVY = "#102A43"
BLUE = "#3977C3"
ORANGE = "#E58A2B"
GREEN = "#24935C"
PURPLE = "#7B61A8"
RED = "#C84A4A"
GRAY = "#667085"
LIGHT = "#F5F7FA"
WHITE = "#FFFFFF"
DARK = "#172B4D"
PALE_BLUE = "#EAF2FB"
PALE_ORANGE = "#FFF0DF"
PALE_GREEN = "#E8F6EF"
PALE_RED = "#FCEBEC"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "figure.dpi": 120,
        "savefig.dpi": 144,
    })


def base_slide(title: str, number: int, section: str = "SO SÁNH PAPER ↔ PIPELINE HIỆN TẠI"):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.915), 1, 0.085, color=NAVY, zorder=0))
    ax.text(0.035, 0.957, title, color=WHITE, fontsize=22, weight="bold", va="center")
    ax.text(0.965, 0.957, f"{number:02d}", color="#B7C9DB", fontsize=14, ha="right", va="center")
    ax.plot([0.035, 0.965], [0.055, 0.055], color="#D7DEE7", lw=0.8)
    ax.text(0.035, 0.025, section, fontsize=8.5, color=GRAY, va="center")
    ax.text(0.965, 0.025, "2026-08-06", fontsize=8.5, color=GRAY, ha="right", va="center")
    return fig, ax


def rounded_box(ax, x, y, w, h, title, body="", color=BLUE, title_color=None,
                body_color=DARK, title_size=15, body_size=11, border=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color,
        edgecolor=border or color,
        linewidth=1.3,
    )
    ax.add_patch(patch)
    if body:
        ax.text(x + 0.025, y + h - 0.04, title, color=title_color or DARK,
                fontsize=title_size, weight="bold", va="top")
        ax.text(x + 0.025, y + h - 0.105, body, color=body_color,
                fontsize=body_size, va="top", linespacing=1.35, wrap=True)
    else:
        ax.text(x + w / 2, y + h / 2, title, color=title_color or WHITE,
                fontsize=title_size, weight="bold", ha="center", va="center", wrap=True)
    return patch


def arrow(ax, x1, y1, x2, y2, color=GRAY, dashed=False, width=1.8):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
        linewidth=width, color=color, linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)


def add_bullets(ax, items, x=0.07, y=0.80, size=16, color=DARK, gap=0.09,
                bullet_color=ORANGE):
    for index, item in enumerate(items):
        yy = y - index * gap
        ax.text(x, yy, "●", fontsize=size * 0.7, color=bullet_color, va="top")
        ax.text(x + 0.028, yy, item, fontsize=size, color=color, va="top",
                linespacing=1.25, wrap=True)


def add_source_figure(fig, path: Path, rect=(0.035, 0.075, 0.93, 0.82)):
    image = plt.imread(path)
    image_ax = fig.add_axes(rect)
    image_ax.imshow(image)
    image_ax.axis("off")
    return image_ax


def save_slide(fig, slides_dir: Path, number: int, slug: str) -> Path:
    output = slides_dir / f"slide{number:02d}_{slug}.png"
    fig.savefig(output, facecolor=fig.get_facecolor(), dpi=144)
    plt.close(fig)
    return output


def slide_01(slides_dir):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Circle((0.88, 0.15), 0.36, color="#173F5F", alpha=0.85))
    ax.add_patch(plt.Circle((0.08, 0.92), 0.22, color=BLUE, alpha=0.22))
    ax.text(0.07, 0.78, "TÁI TẠO BÀI BÁO EMF", fontsize=14, color="#9FC5E8", weight="bold")
    ax.text(0.07, 0.62, "Ta đã làm gần đến đâu?", fontsize=34, color=WHITE, weight="bold")
    ax.text(0.07, 0.50, "Khác gì • Kết quả ra sao • Đi tiếp thế nào", fontsize=23, color="#D9E8F5")
    ax.text(0.07, 0.34,
            "Residual Neural Network for Precise 6-DoF\nCapsule Endoscope Localization Using Electromagnetic Induction",
            fontsize=14, color="#B7C9DB", linespacing=1.35)
    rounded_box(ax, 0.66, 0.52, 0.25, 0.15, "PAPER", "500³ mm\n64M synthetic", PALE_BLUE,
                title_color=BLUE, body_color=DARK, title_size=14, body_size=12, border=BLUE)
    rounded_box(ax, 0.66, 0.29, 0.25, 0.15, "HỆ THỐNG TA", "100³ mm\nv3.2.2 selected", PALE_GREEN,
                title_color=GREEN, body_color=DARK, title_size=14, body_size=12, border=GREEN)
    ax.text(0.07, 0.10, "Báo cáo trung thực • test không mở lại • model selection không thay đổi",
            fontsize=11, color="#9FC5E8")
    return save_slide(fig, slides_dir, 1, "title")


def slide_02(slides_dir):
    fig, ax = base_slide("Kết luận trước: đúng hướng, chưa phải full reproduction", 2)
    cards = [
        (0.055, 0.55, "BÁM SÁT", "9 EMF → 6-DoF\nResNet 512 × 7 blocks\nForward physics cùng họ", PALE_BLUE, BLUE),
        (0.515, 0.55, "ĐỔI MỚI", "Causal W3 + log-ratio\nRotation-6D + SO(3)\nGrouped CV + sealed gate", PALE_GREEN, GREEN),
        (0.055, 0.20, "CHƯA TÁI TẠO", "64M synthetic / 500³ mm\nBroad roll–pitch–yaw\nPhysics consistency <3.4%", PALE_RED, RED),
        (0.515, 0.20, "QUYẾT ĐỊNH", "Deploy: v3.2.2\nv4.2: final rejected\nv4.4: research-only", PALE_ORANGE, ORANGE),
    ]
    for x, y, title, body, bg, accent in cards:
        rounded_box(ax, x, y, 0.405, 0.25, title, body, bg, title_color=accent,
                    body_color=DARK, title_size=16, body_size=13, border=accent)
    ax.text(0.5, 0.105,
            "Protocol của ta rõ hơn về leakage; bằng chứng vật lý của paper vẫn mạnh hơn.",
            fontsize=14, color=NAVY, weight="bold", ha="center")
    return save_slide(fig, slides_dir, 2, "executive_verdict")


def slide_03(slides_dir):
    fig, ax = base_slide("Bài toán chung: từ 9 EMF đến pose 6-DoF", 3)
    rounded_box(ax, 0.055, 0.43, 0.17, 0.23, "3 TX coils", "Ba tần số\ntạo trường từ", PALE_BLUE,
                title_color=BLUE, body_color=DARK, border=BLUE)
    rounded_box(ax, 0.285, 0.43, 0.17, 0.23, "Capsule + 3 RX", "3 TX × 3 RX\n= 9 biên độ EMF", PALE_ORANGE,
                title_color=ORANGE, body_color=DARK, border=ORANGE)
    rounded_box(ax, 0.515, 0.43, 0.17, 0.23, "Neural localizer", "Inverse mapping\nEMF → pose", PALE_GREEN,
                title_color=GREEN, body_color=DARK, border=GREEN)
    rounded_box(ax, 0.745, 0.43, 0.20, 0.23, "Pose hiện tại", "X, Y, Z\nroll, pitch, yaw", PALE_RED,
                title_color=RED, body_color=DARK, border=RED)
    for x1, x2, c in ((0.225, 0.285, BLUE), (0.455, 0.515, ORANGE), (0.685, 0.745, GREEN)):
        arrow(ax, x1, 0.545, x2, 0.545, c)
    ax.text(0.5, 0.77, "Đây là phần tương đồng quan trọng nhất giữa paper và hệ thống ta.",
            fontsize=17, color=NAVY, weight="bold", ha="center")
    ax.text(0.5, 0.25,
            "Khác biệt bắt đầu từ cách tạo dữ liệu, calibration, biểu diễn orientation,\nvalidation protocol và cách triển khai realtime.",
            fontsize=15, color=GRAY, ha="center", linespacing=1.35)
    return save_slide(fig, slides_dir, 3, "common_task")


def image_slide(slides_dir, number, title, source_name, slug, callout=None):
    fig, ax = base_slide(title, number)
    rect = (0.035, 0.085 if callout is None else 0.13, 0.93, 0.79 if callout is None else 0.74)
    add_source_figure(fig, SOURCE_FIGURES / source_name, rect)
    if callout:
        ax.text(0.5, 0.09, callout, fontsize=12.5, color=RED, ha="center", weight="bold")
    return save_slide(fig, slides_dir, number, slug)


def slide_06(slides_dir):
    fig, ax = base_slide("Synthetic và calibration: cùng mục đích, khác cách làm", 6)
    ax.text(0.045, 0.84, "PAPER", fontsize=17, color=BLUE, weight="bold")
    paper = [
        (0.05, "Analytical\nphysics"), (0.235, "64M\nsynthetic"), (0.42, "ResNet\ninitial train"),
        (0.605, "150 real pose\nfit TX params"), (0.79, "Calibrated\nretrain"),
    ]
    for i, (x, label) in enumerate(paper):
        rounded_box(ax, x, 0.64, 0.145, 0.12, label, color=BLUE, title_size=11.5)
        if i:
            arrow(ax, paper[i - 1][0] + 0.145, 0.70, x, 0.70, BLUE)

    ax.text(0.045, 0.53, "TA — v4.4", fontsize=17, color=ORANGE, weight="bold")
    ours = [
        (0.03, "Split real\nby session"), (0.185, "Fit 39 physics\nparams/train-fold"),
        (0.34, "Held-out\nphysics gate"), (0.495, "~5k W3\nsynthetic/fold"),
        (0.65, "107 fixed\nepochs"), (0.805, "Real-only\ncalibration"),
    ]
    for i, (x, label) in enumerate(ours):
        color = ORANGE if i < 3 else PURPLE
        rounded_box(ax, x, 0.34, 0.125, 0.12, label, color=color, title_size=9.8)
        if i:
            arrow(ax, ours[i - 1][0] + 0.125, 0.40, x, 0.40, color)
    ax.text(0.50, 0.245,
            "Điểm phải nói thật: generator của ta đã fit từ train-fold real trước Stage 1.\n"
            "Đây là paper-order neural analogue, không phải exact physical-calibration reproduction.",
            fontsize=11.5, color=RED, ha="center", linespacing=1.28, weight="bold")
    rounded_box(ax, 0.14, 0.095, 0.72, 0.095,
                "Mục tiêu chung: physics mở rộng data\n→ real calibration thu hẹp sim-to-real gap",
                color=GREEN, title_size=10.8)
    return save_slide(fig, slides_dir, 6, "synthetic_and_calibration")


def slide_14(slides_dir):
    fig, ax = base_slide("Mục tiêu cuối: định vị capsule realtime rồi giao tiếp TCP/IP", 14)
    stages = [
        (0.04, "EMF acquisition", "9 kênh mV\ntimestamp/reset", BLUE),
        (0.235, "Causal buffer", "W3, không future\nreset khi gap", ORANGE),
        (0.43, "Predictor", "v3.2.2 hiện tại\nCPU/GPU fallback", GREEN),
        (0.625, "Pose + confidence", "XYZ + RPY\ndispersion", PURPLE),
        (0.82, "TCP/IP", "gửi pose tới\nrobot/controller", RED),
    ]
    for i, (x, title, body, color) in enumerate(stages):
        rounded_box(ax, x, 0.48, 0.145, 0.22, title, body,
                    {BLUE: PALE_BLUE, ORANGE: PALE_ORANGE, GREEN: PALE_GREEN,
                     PURPLE: "#F1ECF8", RED: PALE_RED}[color],
                    title_color=color, body_color=DARK, title_size=12.5, body_size=10.5, border=color)
        if i:
            arrow(ax, stages[i - 1][0] + 0.145, 0.59, x, 0.59, color)
    ax.text(0.5, 0.79, "Model xác định vị trí/hướng của viên nang — robot không thay thế nhiệm vụ này.",
            fontsize=16, color=NAVY, weight="bold", ha="center")
    rounded_box(ax, 0.08, 0.20, 0.36, 0.16, "TRAINING", "Robot tạo pose ground truth\ncho calibration data", PALE_ORANGE,
                title_color=ORANGE, body_color=DARK, border=ORANGE)
    rounded_box(ax, 0.56, 0.20, 0.36, 0.16, "RUNTIME", "Sensor EMF → model pose\n→ TCP/IP → robot", PALE_GREEN,
                title_color=GREEN, body_color=DARK, border=GREEN)
    ax.text(0.5, 0.105,
            "Benchmark hiện tại chưa gồm acquisition, feature construction đầy đủ và network transport.",
            fontsize=12.5, color=RED, ha="center", weight="bold")
    return save_slide(fig, slides_dir, 14, "realtime_tcpip")


def slide_15(slides_dir):
    fig, ax = base_slide("Kế hoạch với bộ calib mới: đi qua từng gate", 15)
    phases = [
        (0.035, "1", "DATA", "≥3 rotating dev\n+ 1 later sealed\nbroad R/P/Y", BLUE),
        (0.195, "2", "HARDWARE", "channel mapping\npolarity\npose frame", ORANGE),
        (0.355, "3", "PHYSICS", "fit train-only\nheld-out rotation\nconsistency gate", PURPLE),
        (0.515, "4", "MODEL CV", "v3.2.2 vs v4.4\n3 seeds, 4 metrics\nsession p95", GREEN),
        (0.675, "5", "FREEZE + FINAL", "hash model/rule\nopen sealed once\nno retuning", RED),
        (0.835, "6", "REALTIME", "end-to-end latency\nconfidence\nTCP/IP soak test", NAVY),
    ]
    for i, (x, num, title, body, color) in enumerate(phases):
        ax.add_patch(plt.Circle((x + 0.055, 0.77), 0.034, color=color))
        ax.text(x + 0.055, 0.77, num, color=WHITE, fontsize=14, weight="bold", ha="center", va="center")
        rounded_box(ax, x, 0.36, 0.13, 0.32, title, body, LIGHT,
                    title_color=color, body_color=DARK, title_size=11.5, body_size=9.5, border=color)
        if i:
            arrow(ax, phases[i - 1][0] + 0.13, 0.52, x, 0.52, color)
    ax.text(0.5, 0.85, "Thu dữ liệu tốt hơn trước khi tăng model complexity", fontsize=17,
            color=NAVY, ha="center", weight="bold")
    rounded_box(ax, 0.065, 0.13, 0.27, 0.17, "ACCEPT MODEL", "4 metric cùng tốt hơn\ncomposite ≥3%; session ≤5%", PALE_GREEN,
                title_color=GREEN, body_color=DARK, title_size=11.5, body_size=9.8, border=GREEN)
    rounded_box(ax, 0.365, 0.13, 0.27, 0.17, "RUNTIME", "GPU ≤1.5 ms\nCPU ≤3.0 ms", PALE_BLUE,
                title_color=BLUE, body_color=DARK, title_size=11.5, body_size=9.8, border=BLUE)
    rounded_box(ax, 0.665, 0.13, 0.27, 0.17, "PUBLICATION CLAIM", "new sealed test\n+ broad 6-DoF coverage", PALE_RED,
                title_color=RED, body_color=DARK, title_size=11.5, body_size=9.8, border=RED)
    ax.text(0.5, 0.075, "Không dùng lại cyl_rot để chọn trọng số hoặc kiến trúc.", fontsize=12.5,
            color=RED, ha="center", weight="bold")
    return save_slide(fig, slides_dir, 15, "future_roadmap")


def slide_16(slides_dir):
    fig, ax = base_slide("Ba điều cần nhớ", 16, section="KẾT LUẬN / Q&A")
    rounded_box(ax, 0.06, 0.48, 0.27, 0.28, "01 — ĐÚNG HƯỚNG",
                "Lõi task và ResNet bám paper.\nCác mở rộng temporal/rotation/protocol có giá trị.",
                PALE_BLUE, title_color=BLUE, body_color=DARK, border=BLUE, title_size=15, body_size=12)
    rounded_box(ax, 0.365, 0.48, 0.27, 0.28, "02 — CHƯA ĐƯỢC PHÓNG ĐẠI",
                "Không claim outperform paper.\nV4.4 chưa có final hợp lệ.\nPhysics rotation còn yếu.",
                PALE_RED, title_color=RED, body_color=DARK, border=RED, title_size=14, body_size=11.5)
    rounded_box(ax, 0.67, 0.48, 0.27, 0.28, "03 — BƯỚC TIẾP THEO",
                "Calib mới broad 6-DoF.\nVerify hardware → CV → freeze\n→ sealed test → TCP/IP.",
                PALE_GREEN, title_color=GREEN, body_color=DARK, border=GREEN, title_size=15, body_size=12)
    ax.text(0.5, 0.33, "Deployment hôm nay: v3.2.2", fontsize=21, color=GREEN, weight="bold", ha="center")
    ax.text(0.5, 0.24, "Research candidate: v4.4 synthetic → real", fontsize=16, color=PURPLE, ha="center")
    ax.text(0.5, 0.13, "Q&A", fontsize=30, color=NAVY, weight="bold", ha="center")
    return save_slide(fig, slides_dir, 16, "takeaways")


SLIDES = [
    {
        "number": 1, "title": "Tái tạo bài báo EMF", "duration": "0:25",
        "message": "Đặt phạm vi: đối chiếu trung thực, không quảng cáo model.",
        "script": (
            "Hôm nay tôi trình bày ba câu hỏi: chúng ta đã bám sát bài báo đến đâu, "
            "những thay đổi của ta có giá trị gì, và cần làm gì với bộ calibration mới. "
            "Điểm quan trọng là đây không phải bài trình bày để chứng minh ta tốt hơn paper; "
            "đây là một audit để biết chính xác trạng thái trước khi đi tiếp."
        ),
        "transition": "Trước tiên, tôi đưa kết luận ngay để mọi số liệu sau có đúng ngữ cảnh.",
    },
    {
        "number": 2, "title": "Kết luận trước", "duration": "1:00",
        "message": "Đúng hướng ở lõi; protocol mới; còn ba khoảng trống lớn.",
        "script": (
            "Có bốn kết luận. Một, ta bám sát bài toán 9 EMF sang pose 6-DoF và giữ backbone "
            "ResNet gần paper. Hai, ta thêm temporal W3, log-ratio, rotation-6D và protocol chống "
            "leakage. Ba, ta chưa tái tạo quy mô 64 triệu synthetic, workspace 500 mm, broad angles "
            "và physics consistency của paper. Bốn, model triển khai vẫn là v3.2.2; v4.4 mới chỉ "
            "là research candidate. Vì vậy câu trả lời là: đúng hướng, nhưng chưa full reproduction."
        ),
        "transition": "Phần giống nhau bắt đầu từ chính bài toán vật lý.",
    },
    {
        "number": 3, "title": "Bài toán chung", "duration": "0:50",
        "message": "Cùng ánh xạ 9 EMF thành 6 thành phần pose.",
        "script": (
            "Ba cuộn phát và ba trục nhận tạo chín biên độ EMF. Mạng inverse localizer biến chín "
            "tín hiệu đó thành X, Y, Z và orientation. Đây là phần ta tái tạo đúng nhất. Robot có "
            "vai trò cung cấp ground truth khi calibration; model vẫn là thành phần xác định pose "
            "của viên nang, không phải xác định pose của robot."
        ),
        "transition": "Từ cùng một bài toán, hai pipeline bắt đầu khác nhau ở cách tạo và kiểm định dữ liệu.",
    },
    {
        "number": 4, "title": "Hai pipeline", "duration": "1:10",
        "message": "Paper ưu tiên scale synthetic; ta ưu tiên fold-local calibration và gates.",
        "script": (
            "Paper đi từ forward model giải tích, sinh 64 triệu mẫu, random split, train một ResNet, "
            "sau đó dùng 150 pose thật để hiệu chỉnh tham số TX và retrain. Ta đi từ 12 CSV robot, "
            "split theo session, fit physics chỉ trên train fold, pretrain synthetic rồi calibration "
            "real-only, xác nhận ba seed và nhiều quality gate. Nhánh màu xanh là deployment v3.2.2; "
            "nhánh v4.4 màu tím chưa được deploy vì latency gate và thiếu sealed test mới."
        ),
        "transition": "Khác biệt lớn nhất khi đọc các con số là scale dữ liệu và độ phủ pose.",
    },
    {
        "number": 5, "title": "Scale và coverage", "duration": "1:15",
        "message": "Không thể so trực tiếp 1.90 mm với 1.122 mm.",
        "script": (
            "Workspace paper có cạnh lớn hơn năm lần, tức thể tích lớn hơn 125 lần. Paper dùng 64 "
            "triệu synthetic; Stage 1 của ta chỉ khoảng năm nghìn mỗi fold và 6.019 cho deployment. "
            "Điểm quan trọng hơn là orientation: dữ liệu train của ta chỉ đổi roll 0.02 độ, pitch "
            "10 độ và yaw 0.04 độ, trong khi paper đánh giá góc rộng. Do đó sai số millimeter nhỏ hơn "
            "không chứng minh hệ thống ta tốt hơn paper."
        ),
        "transition": "Vậy synthetic và calibration của ta được dùng như thế nào?",
    },
    {
        "number": 6, "title": "Synthetic và calibration", "duration": "1:30",
        "message": "Cùng mục đích giảm sim-to-real gap; thứ tự vật lý không hoàn toàn giống.",
        "script": (
            "Mục đích chung là dùng physics để có nhiều cặp pose–EMF rồi dùng real calibration để "
            "thu hẹp sim-to-real gap. Paper sinh analytical synthetic trước, rồi dùng 150 pose để "
            "fit TX và retrain. Ta fit 39 tham số physics trên train fold trước, kiểm tra held-out gate, "
            "sau đó sinh W3 synthetic, train 107 epoch cố định và fine-tune real-only. Vì synthetic của "
            "ta đã phụ thuộc train-fold real, tôi gọi đây là paper-order neural analogue, không gọi là "
            "exact reproduction. Ưu điểm là tránh leakage; nhược điểm là synthetic bị giới hạn bởi "
            "coverage và sai số của calibration hiện tại."
        ),
        "transition": "Trên nền dữ liệu đó, backbone giống paper nhưng input và output đã thay đổi rõ rệt.",
    },
    {
        "number": 7, "title": "Kiến trúc mạng", "duration": "1:15",
        "message": "Đổi mới nằm ở feature, orientation và composition, không ở việc làm ResNet sâu hơn.",
        "script": (
            "Paper dùng một row chín EMF, chuẩn hóa Gaussian, width 512, bảy residual block và đầu ra "
            "XYZ cộng cos Euler. Ta giữ width, depth, block và activation để bám paper. v3.2.2 dùng "
            "năm model: bốn member cho position và một log-ratio member cho orientation. v4.4 thêm "
            "generation gain/bias adapter, 47 feature nội bộ và rotation-6D. Rotation-6D được chiếu "
            "về SO(3), xử lý hình học quay tốt hơn cos-Euler."
        ),
        "transition": "Những thay đổi này chỉ đáng tin nếu validation không bị lẫn session hoặc temporal window.",
    },
    {
        "number": 8, "title": "Leakage audit", "duration": "1:15",
        "message": "Protocol ta rõ hơn, nhưng không có bằng chứng paper bị leakage.",
        "script": (
            "Ta split complete session trước khi fit scaler, physics hoặc tạo W3. Window không vượt "
            "reset, session hay generation; absolute timestamp và row index không đi vào feature. "
            "Candidate được chọn trên development, hash freeze rồi cyl_rot chỉ mở một lần. Tuy vậy, "
            "raw test labels từng tồn tại local và một audit cũ từng nhìn target-derived boundary, nên "
            "claim publication-grade cần sealed set mới. Paper không công bố các biên này; điều đó "
            "không cho phép ta kết luận paper có leakage."
        ),
        "transition": "Với protocol development đó, v4.4 cho tín hiệu cải thiện khá nhất quán.",
    },
    {
        "number": 9, "title": "Development CV", "duration": "1:05",
        "message": "V4.4 thắng cả bốn metric ở cả ba seed trên development.",
        "script": (
            "Baseline matched-fold có position RMSE 0.747 mm, position p95 2.354 mm, SO3 RMSE 0.460 "
            "độ và p95 0.858 độ. Rule v4.4 dùng 20 phần trăm C3 cho position và 30 phần trăm cho "
            "orientation. Cả seed 42, 7 và 123 đều cải thiện cả bốn metric và qua session p95 gate. "
            "Đây là bằng chứng development tốt, nhưng chưa phải bằng chứng final hoặc cross-generation."
        ),
        "transition": "Lý do phải phân biệt development và final thể hiện rất rõ ở thí nghiệm v4.2 trước đó.",
    },
    {
        "number": 10, "title": "Quyết định final", "duration": "1:05",
        "message": "Mixed v4.2 cải thiện orientation nhưng làm xấu position nên bị loại.",
        "script": (
            "Trên final đã mở một lần, v3.2.2 đạt position RMSE 1.122 mm và p95 3.105 mm; SO3 RMSE "
            "2.390 độ và p95 5.058 độ. Mixed v4.2 giảm lỗi orientation, nhưng position tăng lên 1.233 "
            "mm và p95 3.473 mm. Vì gate yêu cầu cả bốn metric, v4.2 bị reject. Quan trọng: v4.4 "
            "không xuất hiện trên hình này vì ta không được đánh giá lại trên cyl_rot đã mở."
        ),
        "transition": "Khi đặt cạnh kết quả paper, ta phải giữ nguyên nguyên tắc không head-to-head.",
    },
    {
        "number": 11, "title": "So với paper", "duration": "0:55",
        "message": "Số tuyệt đối chỉ là context, không phải thứ hạng.",
        "script": (
            "Paper báo 1.90 mm position và 3.55 độ direction trên setup thật. Ta có số nhỏ hơn trên "
            "một số metric, nhưng điều kiện dễ và hẹp hơn rất nhiều. Hình cố ý ghi không head-to-head: "
            "khác workspace, quỹ đạo, orientation, sensor, split và metric tail. Câu đúng là model ta "
            "hoạt động tốt trong workspace hiện tại; câu sai là model ta đã outperform paper."
        ),
        "transition": "Sự khác biệt quỹ đạo cũng giải thích vì sao Fig. 8 của ta nhìn không giống paper.",
    },
    {
        "number": 12, "title": "Vì sao Fig. 8 khác?", "duration": "0:55",
        "message": "Paper dùng tapered helix; robot ta tạo cylinder bán kính gần như không đổi.",
        "script": (
            "Paper có đường xoắn năm vòng với bán kính thu nhỏ dần, nên hình giống một hình nón: dưới "
            "to, trên nhỏ. Dữ liệu hiện tại của ta là cylinder bán kính khoảng 50 mm gần như không đổi. "
            "Ta đã lấy 100 điểm uniform và không smooth error. Vì vậy hình đều hơn là đúng dữ liệu. "
            "Ép đường ta thành taper bằng plotting sẽ làm đẹp hình nhưng sai khoa học. Muốn giống paper, "
            "phải lập trình robot chạy tapered helix mới."
        ),
        "transition": "Khoảng trống nghiêm trọng hơn hình dáng trajectory là physics consistency.",
    },
    {
        "number": 13, "title": "Khoảng trống physics", "duration": "1:15",
        "message": "Forward model pass gate cục bộ nhưng thất bại trên rotating final.",
        "script": (
            "Ở ba fold development, correlation 0.878 đến 0.918 và RMSE trên mean EMF 0.390 đến 0.474, "
            "nên generator được phép chạy. Nhưng trên rotating final, ngay cả đưa ground-truth pose vào "
            "forward model, mean channel MAPE vẫn 106.19 phần trăm, trong khi paper báo dưới 3.4 phần trăm. "
            "Việc đường GT-pose và predicted-pose gần nhau cho thấy lỗi chính nằm ở channel mapping, "
            "polarity, pose frame hoặc physical model, không phải chỉ ở inverse neural network."
        ),
        "transition": "Sau khi sửa lớp vật lý, model mới có thể được đưa vào luồng realtime hoàn chỉnh.",
    },
    {
        "number": 14, "title": "Realtime và TCP/IP", "duration": "1:05",
        "message": "Model định vị capsule; TCP/IP chỉ là transport tới robot/controller.",
        "script": (
            "Ở runtime, sensor gửi chín EMF cùng timestamp và reset. Buffer tạo W3 nhân quả, predictor "
            "trả pose và confidence, rồi TCP/IP chuyển kết quả tới robot hoặc controller. Robot tạo "
            "ground truth khi training, nhưng không được đưa target robot vào inference nếu capsule "
            "thực tế không có tín hiệu đó. v3.2.2 hiện đạt GPU p95 0.858 ms và CPU 1.290 ms cho inference "
            "core; benchmark end-to-end vẫn phải cộng acquisition, feature, confidence và network."
        ),
        "transition": "Với bộ calib mới, ta sẽ đi theo sáu gate thay vì train ngay một model lớn hơn.",
    },
    {
        "number": 15, "title": "Roadmap bộ calib mới", "duration": "1:30",
        "message": "Dữ liệu và hardware verification đi trước model selection.",
        "script": (
            "Bước một, thu ít nhất ba rotating development session và một later sealed session, mở rộng "
            "roll, pitch, yaw độc lập. Bước hai, xác minh channel mapping, polarity và pose frame. Bước "
            "ba, fit physics train-only và kiểm tra trên held-out rotation. Bước bốn, so v3.2.2 với v4.4 "
            "bằng ba seed, bốn metric và session p95. Bước năm, hash freeze rồi mở sealed đúng một lần. "
            "Bước sáu, benchmark end-to-end và TCP/IP soak test. Gate model giữ nguyên: bốn metric cùng "
            "tốt hơn, composite ít nhất ba phần trăm, không session nào xấu quá năm phần trăm, GPU dưới "
            "1.5 ms và CPU dưới 3 ms."
        ),
        "transition": "Tôi kết thúc bằng ba điều cần nhớ.",
    },
    {
        "number": 16, "title": "Kết luận", "duration": "0:40",
        "message": "Giữ v3.2.2; v4.4 research-only; ưu tiên calib broad 6-DoF và physics verification.",
        "script": (
            "Một, ta đi đúng hướng và có nhiều mở rộng hợp lý. Hai, chưa được claim full reproduction "
            "hay outperform paper; v4.4 chưa có final hợp lệ và physics rotation còn yếu. Ba, bộ calib "
            "mới cần được thiết kế để mở rộng 6-DoF, xác minh phần cứng, rồi mới CV, freeze, sealed test "
            "và tích hợp TCP/IP. Deployment hôm nay vẫn là v3.2.2."
        ),
        "transition": "Xin mời câu hỏi.",
    },
]


def build_script(slide_paths: list[Path], out_dir: Path) -> Path:
    lines = [
        "# Kịch bản trình bày: Paper EMF và pipeline hiện tại",
        "",
        "Thời lượng mục tiêu: **15–18 phút**, 16 slide. Trạng thái model trong deck: "
        "`v3.2.2` selected deployment; mixed `v4.2` final-rejected; `v4.4` research-only.",
        "",
        "## Cách dùng",
        "",
        "- Deck PDF để trình chiếu: [SLIDE_DECK.pdf](SLIDE_DECK.pdf).",
        "- Mỗi slide PNG 16:9 nằm trong `slides/`.",
        "- Phần **Lời nói đề xuất** là kịch bản có thể đọc gần như nguyên văn.",
        "- Bản 8 phút: trình bày slide 1, 2, 4, 5, 7, 9, 10, 13, 15, 16.",
        "- Không đổi cụm từ “development-only”, “không head-to-head” và “research-only”.",
        "",
    ]
    for meta, slide_path in zip(SLIDES, slide_paths):
        relative = slide_path.relative_to(out_dir)
        lines.extend([
            f"## Slide {meta['number']} — {meta['title']}",
            "",
            f"**Thời lượng:** {meta['duration']}  ",
            f"**Thông điệp duy nhất:** {meta['message']}",
            "",
            f"![Slide {meta['number']}]({relative.as_posix()})",
            "",
            "### Lời nói đề xuất",
            "",
            f"> {meta['script']}",
            "",
            "### Chuyển slide",
            "",
            f"> {meta['transition']}",
            "",
        ])
    lines.extend([
        "## Câu hỏi có khả năng được hỏi",
        "",
        "### Vì sao không chọn v4.4 ngay khi development tốt hơn?",
        "",
        "Vì GPU p95 1.723 ms vượt gate 1.5 ms và không còn sealed test mới. Đưa vào deployment "
        "lúc này sẽ phá protocol đã định trước.",
        "",
        "### Kết quả 1.122 mm có tốt hơn 1.90 mm của paper không?",
        "",
        "Không được kết luận như vậy. Workspace, orientation coverage, trajectory, phần cứng và "
        "protocol test khác nhau.",
        "",
        "### Timestamp có giúp không?",
        "",
        "Chỉ khi khoảng lấy mẫu thực sự biến thiên hoặc có dropout/gap. Dữ liệu hiện tại mỗi row là "
        "một bước lập trình cố định, nên constant `dt_ratio=1` không mang timing information mới; W3 "
        "vẫn có ích vì chứa lịch sử EMF nhân quả.",
        "",
        "### Synthetic có nên tăng lên hàng triệu mẫu không?",
        "",
        "Chưa. Forward model hiện sai lớn trên rotation. Tăng synthetic từ một generator sai chỉ "
        "khuếch đại bias. Phải xác minh hardware/frame và held-out rotating physics trước.",
        "",
        "### Robot có phải đối tượng model định vị không?",
        "",
        "Không. Robot cung cấp pose ground truth khi calibration. Ở runtime model định vị capsule "
        "từ EMF; TCP/IP chuyển pose dự đoán tới robot/controller.",
        "",
    ])
    output = out_dir / "SLIDE_SCRIPT.md"
    output.write_text("\n".join(lines))
    return output


def make_pdf(slide_paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in slide_paths]
    fixed_time = time.gmtime(0)
    images[0].save(
        output, "PDF", resolution=144.0, save_all=True, append_images=images[1:],
        title="EMF paper comparison presentation",
        creator="EMF reporting pipeline",
        creationDate=fixed_time,
        modDate=fixed_time,
    )
    for image in images:
        image.close()


def main():
    parser = argparse.ArgumentParser(description="Build paper-comparison presentation and script.")
    parser.add_argument("--out_dir", default="reports/presentation_paper_comparison_2026_08_06")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    slides_dir = out_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    required = [
        "fig01_pipeline_side_by_side.png",
        "fig02_architecture_side_by_side.png",
        "fig03_validation_and_leakage_boundaries.png",
        "fig04_data_scale_and_coverage.png",
        "fig05_development_multiseed_results.png",
        "fig06_one_time_final_decision.png",
        "fig07_paper_result_context_not_head_to_head.png",
        "fig09_physics_calibration_audit.png",
        "fig12_existing_paper_fig8_analogue.png",
    ]
    missing = [name for name in required if not (SOURCE_FIGURES / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen source figures: {missing}")

    slide_paths = [
        slide_01(slides_dir),
        slide_02(slides_dir),
        slide_03(slides_dir),
        image_slide(slides_dir, 4, "Pipeline paper và pipeline hiện tại",
                    "fig01_pipeline_side_by_side.png", "pipeline"),
        image_slide(slides_dir, 5, "Workspace, dữ liệu và orientation coverage",
                    "fig04_data_scale_and_coverage.png", "data_coverage",
                    "Workspace paper lớn hơn 125× về thể tích; số tuyệt đối không so head-to-head."),
        slide_06(slides_dir),
        image_slide(slides_dir, 7, "Mạng neural: backbone gần, feature/output khác",
                    "fig02_architecture_side_by_side.png", "architecture"),
        image_slide(slides_dir, 8, "Leakage audit và biên test",
                    "fig03_validation_and_leakage_boundaries.png", "leakage"),
        image_slide(slides_dir, 9, "V4.4 cải thiện development ở cả ba seed",
                    "fig05_development_multiseed_results.png", "development_results",
                    "Development-only: chưa phải kết quả final hoặc cross-generation mới."),
        image_slide(slides_dir, 10, "Final đã mở một lần: vì sao vẫn giữ v3.2.2",
                    "fig06_one_time_final_decision.png", "final_decision"),
        image_slide(slides_dir, 11, "Đặt cạnh paper để hiểu context — không xếp hạng",
                    "fig07_paper_result_context_not_head_to_head.png", "paper_context"),
        image_slide(slides_dir, 12, "Fig. 8 khác vì trajectory vật lý khác",
                    "fig12_existing_paper_fig8_analogue.png", "fig8_trajectory",
                    "Paper: tapered helix. Ta: cylinder bán kính gần như không đổi; không được ép hình bằng plotting."),
        image_slide(slides_dir, 13, "Khoảng trống lớn nhất: physics consistency trên rotation",
                    "fig09_physics_calibration_audit.png", "physics_gap",
                    "GT-pose forward reconstruction vẫn ~106.19% mean-channel MAPE; paper báo <3.4%."),
        slide_14(slides_dir),
        slide_15(slides_dir),
        slide_16(slides_dir),
    ]
    if len(slide_paths) != len(SLIDES):
        raise RuntimeError("Slide image count does not match speaker metadata")

    pdf_path = out_dir / "SLIDE_DECK.pdf"
    make_pdf(slide_paths, pdf_path)
    script_path = build_script(slide_paths, out_dir)

    outline_path = out_dir / "SLIDE_OUTLINE.csv"
    with outline_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["slide", "title", "duration", "message", "image"])
        writer.writeheader()
        for meta, slide_path in zip(SLIDES, slide_paths):
            writer.writerow({
                "slide": meta["number"], "title": meta["title"], "duration": meta["duration"],
                "message": meta["message"], "image": str(slide_path.relative_to(out_dir)),
            })

    readme_path = out_dir / "README.md"
    readme_path.write_text((
        "# Paper-comparison presentation\n\n"
        "- [Slide deck PDF](SLIDE_DECK.pdf)\n"
        "- [Detailed Vietnamese speaker script](SLIDE_SCRIPT.md)\n"
        "- [Slide outline CSV](SLIDE_OUTLINE.csv)\n"
        "- Individual 16:9 PNG slides: `slides/`\n\n"
        "Rebuild:\n\n"
        "```bash\n"
        "# Run from the cloned repository root.\n"
        "./.venv/bin/python reporting/build_paper_comparison_presentation.py \\\n+  --out_dir reports/presentation_paper_comparison_2026_08_06\n"
        "```\n\n"
        "This is reporting-only. It does not read final labels/predictions, train, infer, or change deployment.\n"
    ).replace("\n+", "\n"))

    generator_path = Path(__file__).resolve()
    outputs = [*slide_paths, pdf_path, script_path, outline_path, readme_path]
    manifest = {
        "schema_version": 1,
        "date": "2026-08-06",
        "language": "vi",
        "format": "16:9_png_plus_combined_pdf_and_speaker_script",
        "slide_count": len(slide_paths),
        "target_duration_minutes": "15-18",
        "reporting_only": True,
        "raw_final_labels_or_predictions_read": False,
        "training_inference_or_model_selection": False,
        "selected_deployment_changed": False,
        "selected_deployment": "v3.2.2-cuda-graph-dual-runtime",
        "generator_sha256": sha256_file(generator_path),
        "source_figures_sha256": {name: sha256_file(SOURCE_FIGURES / name) for name in required},
        "outputs_sha256": {str(path.relative_to(out_dir)): sha256_file(path) for path in outputs},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "deck": str(pdf_path.relative_to(ROOT)),
        "script": str(script_path.relative_to(ROOT)),
        "slides": len(slide_paths),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "selected_deployment_changed": False,
    }, indent=2))


if __name__ == "__main__":
    main()

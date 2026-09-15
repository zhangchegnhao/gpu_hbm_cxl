#!/usr/bin/env python3
"""Render the runtime-v1 evaluation figures from recorded result files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/full_decode_real_pilot_runtime_v1"
DEFAULT_OUTPUT = ROOT / "outputs/qwen3_real_pilot_runtime_v1_figures"
WIDTH = 2400
HEIGHT = 1350
BG = "#F7F8FA"
FG = "#20242B"
MUTED = "#66717E"
GRID = "#D9DEE5"
BLUE = "#71849A"
RED = "#D84A3A"
GREEN = "#2F7D68"
WHITE = "#FFFFFF"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_MEDIUM = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    face: ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=face)
    return box[2] - box[0], box[3] - box[1]


def right_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    value: str,
    face: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    width, _ = text_size(draw, value, face)
    draw.text((x - width, y), value, font=face, fill=fill)


def read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((110, 68), title, font=font(FONT_BOLD, 60), fill=FG)
    draw.text((110, 150), subtitle, font=font(FONT_REGULAR, 30), fill=MUTED)


def render_latency(summary: dict[str, object], output: Path) -> None:
    end_to_end = summary["end_to_end"]
    assert isinstance(end_to_end, dict)
    rows = [
        (
            "sieve-cycle-v1 (Oracle)",
            float(end_to_end["oracle_total_latency_us"]) / 1000,
            FG,
        ),
        (
            "sieve-runtime-v1",
            float(end_to_end["runtime_total_latency_us"]) / 1000,
            RED,
        ),
        (
            "sieve-fixed-16-cycle-v1",
            float(end_to_end["fixed_16_total_latency_us"]) / 1000,
            BLUE,
        ),
        (
            "sieve (legacy)",
            float(end_to_end["legacy_sieve_total_latency_us"]) / 1000,
            BLUE,
        ),
    ]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw_title(
        draw,
        "Runtime-v1 total latency",
        "Qwen3 real pilot | 384 layer-batches | lower is better",
    )
    left, right, top = 780, 2220, 300
    bar_height, gap, x_max = 100, 56, 40.0
    label_face = font(FONT_MEDIUM, 30)
    value_face = font(FONT_BOLD, 28)
    tick_face = font(FONT_REGULAR, 24)
    chart_bottom = top + len(rows) * (bar_height + gap) - gap
    for tick in range(0, 41, 10):
        x = left + round((right - left) * tick / x_max)
        draw.line((x, top - 25, x, chart_bottom + 20), fill=GRID, width=2)
        width, _ = text_size(draw, str(tick), tick_face)
        draw.text((x - width / 2, chart_bottom + 38), str(tick), font=tick_face, fill=MUTED)
    for index, (label, latency, color) in enumerate(rows):
        y = top + index * (bar_height + gap)
        label_color = RED if color == RED else FG
        right_text(draw, left - 35, y + 32, label, label_face, label_color)
        bar_right = left + round((right - left) * latency / x_max)
        draw.rectangle((left, y, bar_right, y + bar_height), fill=color)
        value = f"{latency:.6f} ms"
        value_width, value_height = text_size(draw, value, value_face)
        value_x = max(left + 18, bar_right - value_width - 20)
        value_y = y + (bar_height - value_height) // 2 - 3
        draw.text((value_x, value_y), value, font=value_face, fill=WHITE)
    draw.text(
        (left, chart_bottom + 104),
        "Total decode latency (ms)",
        font=font(FONT_MEDIUM, 30),
        fill=FG,
    )
    draw.text(
        (110, 1250),
        "Oracle gap: 0.000019553 ms | runtime vs legacy reduction: 18.9049% | simulator result, not hardware measurement",
        font=font(FONT_REGULAR, 24),
        fill=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True, dpi=(200, 200))


def render_prefix(
    evaluation: dict[str, object],
    evaluation_dir: Path,
    output: Path,
) -> None:
    rows = list(
        csv.DictReader(
            (evaluation_dir / "layers.csv").open(encoding="utf-8", newline="")
        )
    )
    counts: dict[int, list[int]] = {}
    for row in rows:
        oracle = int(row["oracle_gpu_prefix"])
        runtime = int(row["runtime_gpu_prefix"])
        counts.setdefault(oracle, [0, 0])[0] += 1
        counts.setdefault(runtime, [0, 0])[1] += 1
    prefixes = list(range(min(counts), max(counts) + 1))
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw_title(draw, "GPU/PIM boundary selection", "Runtime prediction versus exact cycle-v1 Oracle")
    left, right, bottom, top = 220, 2220, 1090, 300
    chart_height = bottom - top
    max_count = max(max(pair) for pair in counts.values())
    group_width = (right - left) / len(prefixes)
    tick_face = font(FONT_REGULAR, 23)
    label_face = font(FONT_MEDIUM, 28)
    for tick in range(0, max_count + 1, max(1, max_count // 4)):
        y = bottom - round(chart_height * tick / max_count)
        draw.line((left, y, right, y), fill=GRID, width=2)
        right_text(draw, left - 18, y - 14, str(tick), tick_face, MUTED)
    bar_width = min(42, group_width * 0.31)
    for index, prefix in enumerate(prefixes):
        x_center = left + group_width * (index + 0.5)
        oracle_count, runtime_count = counts.get(prefix, [0, 0])
        bars = (
            (-bar_width * 0.56, oracle_count, FG),
            (bar_width * 0.56, runtime_count, RED),
        )
        for offset, count, color in bars:
            x0 = round(x_center + offset - bar_width / 2)
            x1 = round(x_center + offset + bar_width / 2)
            y = bottom - round(chart_height * count / max_count)
            draw.rectangle((x0, y, x1, bottom), fill=color)
        width, _ = text_size(draw, str(prefix), tick_face)
        draw.text((x_center - width / 2, bottom + 22), str(prefix), font=tick_face, fill=FG)
    draw.text((left, 1160), "GPU expert prefix length", font=label_face, fill=FG)
    draw.text((left, 1215), "Dark: Oracle | Red: runtime", font=font(FONT_REGULAR, 24), fill=MUTED)
    metrics = evaluation["decision_metrics"]
    assert isinstance(metrics, dict)
    draw.text(
        (1280, 1160),
        (
            f"Prefix match: {metrics['prefix_matches']}/{metrics['layer_batches']} "
            f"({float(metrics['prefix_match_rate']) * 100:.3f}%)"
        ),
        font=font(FONT_BOLD, 28),
        fill=GREEN,
    )
    draw.text(
        (1280, 1215),
        "No selected layer extrapolated beyond the calibration range",
        font=font(FONT_REGULAR, 24),
        fill=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True, dpi=(200, 200))


def render_errors(evaluation: dict[str, object], output: Path) -> None:
    metrics = evaluation["prediction_metrics"]
    assert isinstance(metrics, dict)
    values = [
        (
            "GPU READ",
            float(metrics["gpu_expert_read"]["mae_us"]),
            float(metrics["gpu_expert_read"]["max_absolute_error_us"]),
            BLUE,
        ),
        (
            "PIM pipeline",
            float(metrics["pim_expert_pipeline"]["mae_us"]),
            float(metrics["pim_expert_pipeline"]["max_absolute_error_us"]),
            RED,
        ),
    ]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw_title(
        draw,
        "Prediction error on non-anchor counts",
        "Within-pilot holdout points | absolute timing error in microseconds",
    )
    left, right, top = 650, 1650, 360
    row_height, gap, x_max = 120, 100, 0.04
    label_face = font(FONT_MEDIUM, 31)
    value_face = font(FONT_BOLD, 27)
    tick_face = font(FONT_REGULAR, 23)
    chart_bottom = top + len(values) * (row_height + gap) - gap
    for tick in (0.0, 0.01, 0.02, 0.03, 0.04):
        x = left + round((right - left) * tick / x_max)
        draw.line((x, top - 20, x, chart_bottom + 20), fill=GRID, width=2)
        label = f"{tick:.2f}"
        width, _ = text_size(draw, label, tick_face)
        draw.text((x - width / 2, chart_bottom + 35), label, font=tick_face, fill=MUTED)
    for index, (label, mae, maximum, color) in enumerate(values):
        y = top + index * (row_height + gap)
        right_text(draw, left - 35, y + 40, label, label_face, FG)
        draw.rectangle((left, y, left + round((right - left) * mae / x_max), y + row_height), fill=color)
        maximum_x = left + round((right - left) * maximum / x_max)
        draw.line(
            (maximum_x, y - 10, maximum_x, y + row_height + 10),
            fill=FG,
            width=5,
        )
        draw.text(
            (right + 30, y + 40),
            f"MAE {mae:.6f} | max {maximum:.6f}",
            font=value_face,
            fill=FG,
        )
    draw.text(
        (left, chart_bottom + 102),
        "Absolute error (us)",
        font=font(FONT_MEDIUM, 30),
        fill=FG,
    )
    draw.text(
        (110, 1250),
        "42 GPU and 54 PIM non-anchor counts; anchors are excluded from these metrics.",
        font=font(FONT_REGULAR, 24),
        fill=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True, dpi=(200, 200))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evaluation = read_summary(args.results / "runtime_evaluation/summary.json")
    render_latency(evaluation, args.output / "runtime_v1_total_latency.png")
    render_prefix(
        evaluation,
        args.results / "runtime_evaluation",
        args.output / "runtime_v1_prefix_selection.png",
    )
    render_errors(evaluation, args.output / "runtime_v1_prediction_error.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

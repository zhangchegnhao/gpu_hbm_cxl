#!/usr/bin/env python3
"""Render the runtime-v2 signature-holdout budget comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results/full_decode_real_pilot_runtime_v2_holdout/summary.json"
DEFAULT_OUTPUT = ROOT / "outputs/qwen3_real_pilot_runtime_v2_holdout_figures"
WIDTH = 2400
HEIGHT = 1350
BG = "#F7F8FA"
FG = "#20242B"
MUTED = "#66717E"
GRID = "#D9DEE5"
BLUE = "#71849A"
RED = "#D84A3A"
GREEN = "#2F7D68"
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


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, object]],
    *,
    left: int,
    right: int,
    top: int,
    title: str,
    value_key: str,
    maximum: float,
    value_label: str,
    digits: int,
) -> None:
    draw.text((left, top), title, font=font(FONT_BOLD, 34), fill=FG)
    chart_top = top + 90
    bar_height = 76
    gap = 50
    label_face = font(FONT_MEDIUM, 28)
    value_face = font(FONT_BOLD, 25)
    axis_left = left + 155
    for index, row in enumerate(rows):
        budget = int(row["budget"])
        value = float(row[value_key])
        y = chart_top + index * (bar_height + gap)
        draw.text((left, y + 20), f"{budget} pts", font=label_face, fill=FG)
        draw.line((axis_left, y + bar_height + 15, right, y + bar_height + 15), fill=GRID, width=2)
        bar_right = axis_left + round((right - axis_left) * value / maximum)
        color = GREEN if budget == 25 else (RED if budget == 5 else BLUE)
        draw.rectangle((axis_left, y, bar_right, y + bar_height), fill=color)
        value_text = f"{value:.{digits}f} {value_label}"
        draw.text((bar_right + 16, y + 20), value_text, font=value_face, fill=FG)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = list(summary["budgets"])
    chart_rows: list[dict[str, object]] = []
    for row in rows:
        chart_rows.append(
            {
                **row,
                "mismatches": int(row["holdout_layer_batches"])
                - int(row["prefix_matches"]),
            }
        )
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((100, 70), "Runtime-v2 signature holdout", font=font(FONT_BOLD, 60), fill=FG)
    draw.text(
        (100, 150),
        "5-fold cross-validation | 340 signatures | 384 held-out layer-batches per budget",
        font=font(FONT_REGULAR, 30),
        fill=MUTED,
    )
    draw_panel(
        draw,
        chart_rows,
        left=100,
        right=1120,
        top=275,
        title="Oracle prefix mismatches (lower is better)",
        value_key="mismatches",
        maximum=10,
        value_label="layers",
        digits=0,
    )
    draw_panel(
        draw,
        chart_rows,
        left=1280,
        right=2280,
        top=275,
        title="Maximum placement regret (lower is better)",
        value_key="max_regret_us",
        maximum=0.11,
        value_label="us",
        digits=6,
    )
    best = min(chart_rows, key=lambda row: (int(row["mismatches"]), float(row["mean_regret_us"])))
    draw.text(
        (100, 1100),
        (
            f"Best tested budget: {best['budget']} nonzero points | "
            f"prefix match {int(best['prefix_matches'])}/384 | "
            f"mean regret {float(best['mean_regret_us']):.8f} us"
        ),
        font=font(FONT_BOLD, 31),
        fill=GREEN,
    )
    draw.text(
        (100, 1170),
        "One fold contains a GPU count outside its training range, but no selected placement uses extrapolation.",
        font=font(FONT_REGULAR, 25),
        fill=MUTED,
    )
    draw.text(
        (100, 1240),
        "Same-pilot signature holdout only; this is not evidence of cross-prompt or cross-batch generalization.",
        font=font(FONT_REGULAR, 25),
        fill=MUTED,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "runtime_v2_holdout_budget_comparison.png"
    image.save(output, format="PNG", optimize=True, dpi=(200, 200))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

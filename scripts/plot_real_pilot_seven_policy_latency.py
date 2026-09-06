#!/usr/bin/env python3
"""Render the real-pilot cycle-v1 seven-policy latency comparison."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


POLICY_LATENCY_MS = (
    ("sieve-cycle-v1", 30.551096),
    ("sieve-fixed-16-cycle-v1", 34.003663),
    ("pimoe", 35.457817),
    ("sieve", 37.673216),
    ("allexp", 44.893344),
    ("gpu-only", 49.714591),
    ("noexp", 50.254423),
)

WIDTH = 2400
HEIGHT = 1350
CHART_LEFT = 690
CHART_RIGHT = 2220
CHART_TOP = 270
BAR_HEIGHT = 86
BAR_GAP = 38
X_MAX_MS = 55.0

BACKGROUND = "#F7F8FA"
FOREGROUND = "#20242B"
MUTED = "#66717E"
GRID = "#D9DEE5"
BASE_BAR = "#71849A"
HIGHLIGHT = "#D84A3A"
WHITE = "#FFFFFF"

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_MEDIUM = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot total decode latency for the seven real-pilot cycle-v1 policies"
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path(
            "outputs/qwen3_real_pilot_cycle_v1_figures/"
            "seven_policy_total_latency_horizontal_bar.png"
        ),
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path(
            "outputs/qwen3_real_pilot_cycle_v1_figures/"
            "seven_policy_total_latency_horizontal_bar.svg"
        ),
    )
    return parser.parse_args()


def x_for_latency(latency_ms: float) -> int:
    return CHART_LEFT + round((CHART_RIGHT - CHART_LEFT) * latency_ms / X_MAX_MS)


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.exists():
        raise FileNotFoundError(f"required font does not exist: {path}")
    return ImageFont.truetype(str(path), size=size)


def draw_right_aligned(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]), xy[1]), text, font=font, fill=fill)


def render_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = load_font(FONT_BOLD, 62)
    subtitle_font = load_font(FONT_REGULAR, 31)
    label_font = load_font(FONT_MEDIUM, 31)
    label_bold_font = load_font(FONT_BOLD, 31)
    value_font = load_font(FONT_BOLD, 29)
    tick_font = load_font(FONT_REGULAR, 25)
    note_font = load_font(FONT_REGULAR, 24)

    draw.text(
        (110, 70),
        "七策略总时延",
        font=title_font,
        fill=FOREGROUND,
    )
    draw.text(
        (110, 151),
        "真实 Qwen3 pilot · cycle-v1 · 数值越低越好",
        font=subtitle_font,
        fill=MUTED,
    )

    chart_bottom = CHART_TOP + len(POLICY_LATENCY_MS) * (BAR_HEIGHT + BAR_GAP) - BAR_GAP
    for tick in range(0, 51, 10):
        x = x_for_latency(float(tick))
        draw.line((x, CHART_TOP - 18, x, chart_bottom + 20), fill=GRID, width=2)
        tick_text = str(tick)
        box = draw.textbbox((0, 0), tick_text, font=tick_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, chart_bottom + 37),
            tick_text,
            font=tick_font,
            fill=MUTED,
        )

    for index, (policy, latency_ms) in enumerate(POLICY_LATENCY_MS):
        y = CHART_TOP + index * (BAR_HEIGHT + BAR_GAP)
        is_best = policy == "sieve-cycle-v1"
        color = HIGHLIGHT if is_best else BASE_BAR
        current_label_font = label_bold_font if is_best else label_font
        label_color = HIGHLIGHT if is_best else FOREGROUND
        label_box = draw.textbbox((0, 0), policy, font=current_label_font)
        label_y = y + (BAR_HEIGHT - (label_box[3] - label_box[1])) / 2 - label_box[1]
        draw_right_aligned(
            draw,
            (CHART_LEFT - 34, round(label_y)),
            policy,
            font=current_label_font,
            fill=label_color,
        )

        bar_right = x_for_latency(latency_ms)
        draw.rectangle((CHART_LEFT, y, bar_right, y + BAR_HEIGHT), fill=color)
        value = f"{latency_ms:.3f} ms"
        value_box = draw.textbbox((0, 0), value, font=value_font)
        value_y = y + (BAR_HEIGHT - (value_box[3] - value_box[1])) / 2 - value_box[1]
        value_x = bar_right - (value_box[2] - value_box[0]) - 22
        draw.text((value_x, round(value_y)), value, font=value_font, fill=WHITE)

    axis_title = "总时延（ms）"
    axis_box = draw.textbbox((0, 0), axis_title, font=subtitle_font)
    draw.text(
        ((CHART_LEFT + CHART_RIGHT - (axis_box[2] - axis_box[0])) / 2, chart_bottom + 88),
        axis_title,
        font=subtitle_font,
        fill=FOREGROUND,
    )
    draw.text(
        (110, 1260),
        "Batch=8，8 个 Decode step，共 64 个生成 token；结果为模拟总时延，并非真实 B200 实测。",
        font=note_font,
        fill=MUTED,
    )

    image.save(path, format="PNG", optimize=True, dpi=(200, 200))


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    fill: str,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-family="Noto Sans CJK SC, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def render_svg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chart_bottom = CHART_TOP + len(POLICY_LATENCY_MS) * (BAR_HEIGHT + BAR_GAP) - BAR_GAP
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        svg_text(110, 126, "七策略总时延", size=62, fill=FOREGROUND, weight=700),
        svg_text(
            110,
            188,
            "真实 Qwen3 pilot · cycle-v1 · 数值越低越好",
            size=31,
            fill=MUTED,
        ),
    ]

    for tick in range(0, 51, 10):
        x = x_for_latency(float(tick))
        elements.append(
            f'<line x1="{x}" y1="{CHART_TOP - 18}" x2="{x}" y2="{chart_bottom + 20}" '
            f'stroke="{GRID}" stroke-width="2"/>'
        )
        elements.append(svg_text(x, chart_bottom + 66, str(tick), size=25, fill=MUTED, anchor="middle"))

    for index, (policy, latency_ms) in enumerate(POLICY_LATENCY_MS):
        y = CHART_TOP + index * (BAR_HEIGHT + BAR_GAP)
        is_best = policy == "sieve-cycle-v1"
        color = HIGHLIGHT if is_best else BASE_BAR
        label_color = HIGHLIGHT if is_best else FOREGROUND
        label_weight = 700 if is_best else 500
        bar_right = x_for_latency(latency_ms)
        bar_width = bar_right - CHART_LEFT
        middle_y = y + BAR_HEIGHT / 2 + 11
        elements.append(
            svg_text(
                CHART_LEFT - 34,
                middle_y,
                policy,
                size=31,
                fill=label_color,
                weight=label_weight,
                anchor="end",
            )
        )
        elements.append(
            f'<rect x="{CHART_LEFT}" y="{y}" width="{bar_width}" height="{BAR_HEIGHT}" fill="{color}"/>'
        )
        elements.append(
            svg_text(
                bar_right - 22,
                middle_y,
                f"{latency_ms:.3f} ms",
                size=29,
                fill=WHITE,
                weight=700,
                anchor="end",
            )
        )

    elements.extend(
        [
            svg_text(
                (CHART_LEFT + CHART_RIGHT) / 2,
                chart_bottom + 119,
                "总时延（ms）",
                size=31,
                fill=FOREGROUND,
                anchor="middle",
            ),
            svg_text(
                110,
                1298,
                "Batch=8，8 个 Decode step，共 64 个生成 token；结果为模拟总时延，并非真实 B200 实测。",
                size=24,
                fill=MUTED,
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    render_png(args.png)
    render_svg(args.svg)
    print(f"PNG: {args.png}")
    print(f"SVG: {args.svg}")


if __name__ == "__main__":
    main()

"""Balloon drawing and automatic placement.

Auto-placement scores candidate windows by edge energy and drops the balloon
into the quietest region, biased toward the top of the panel because vertical
strips are read top-down and dialogue should not sit below the action it
belongs to.
"""
from __future__ import annotations

import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..models import Balloon

# Per-kind visual treatment: (fill, outline, dash, corner radius as % of height)
_STYLES = {
    "speech":    {"fill": "#FFFFFF", "outline": "#2A1F2E", "radius": 0.42, "dash": False},
    "thought":   {"fill": "#FFFFFF", "outline": "#2A1F2E", "radius": 0.50, "dash": True},
    "narration": {"fill": "#FFF9EC", "outline": "#5A4632", "radius": 0.06, "dash": False},
    "shout":     {"fill": "#FFFFFF", "outline": "#1A1016", "radius": 0.18, "dash": False},
    "whisper":   {"fill": "#F7F5FA", "outline": "#6A5F70", "radius": 0.42, "dash": True},
    # Cultivation-genre system window.
    "system":    {"fill": "#0E1A2BE0", "outline": "#5FC7FF", "radius": 0.04, "dash": False},
}


def edge_energy(img: Image.Image, downscale: int = 8) -> np.ndarray:
    """Low-resolution map of visual busyness. High values = detailed regions."""
    small = img.convert("L").resize(
        (max(1, img.width // downscale), max(1, img.height // downscale)),
        Image.BILINEAR,
    )
    return np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32)


def auto_anchor(
    img: Image.Image,
    box_w: int,
    box_h: int,
    taken: list[tuple[int, int, int, int]],
    top_bias: float = 0.5,
) -> tuple[int, int]:
    """Find the top-left corner of the quietest free window for a balloon.

    `taken` holds pixel boxes already occupied by other balloons in this panel
    so multiple balloons do not stack on top of each other.
    """
    energy = edge_energy(img)
    scale = img.width / max(1, energy.shape[1])
    eh, ew = energy.shape

    win_w = max(1, int(box_w / scale))
    win_h = max(1, int(box_h / scale))
    if win_w >= ew or win_h >= eh:
        return (img.width - box_w) // 2, 24

    # Summed-area table for O(1) window means.
    integral = energy.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))

    best, best_xy = float("inf"), (0, 0)
    step = max(1, min(win_w, win_h) // 4)
    for y in range(0, eh - win_h + 1, step):
        for x in range(0, ew - win_w + 1, step):
            total = (
                integral[y + win_h, x + win_w]
                - integral[y, x + win_w]
                - integral[y + win_h, x]
                + integral[y, x]
            )
            score = total / (win_w * win_h)
            # Penalise lower placements so dialogue reads before the beat lands.
            score += (y / max(1, eh - win_h)) * 255.0 * top_bias

            px, py = int(x * scale), int(y * scale)
            if any(
                px < tx + tw and px + box_w > tx and py < ty + th and py + box_h > ty
                for tx, ty, tw, th in taken
            ):
                continue
            if score < best:
                best, best_xy = score, (px, py)

    return best_xy


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Greedy wrap using real glyph metrics rather than character counts."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.getbbox(trial)[2] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_balloon(
    img: Image.Image,
    balloon: Balloon,
    font: ImageFont.FreeTypeFont,
    *,
    padding: int = 22,
    line_spacing: float = 1.25,
    max_width_frac: float = 0.62,
    taken: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int]:
    """Render one balloon onto `img` in place. Returns its pixel box."""
    taken = taken if taken is not None else []
    style = _STYLES.get(balloon.kind, _STYLES["speech"])

    max_text_w = int(img.width * max_width_frac) - padding * 2
    lines = _wrap(balloon.text.strip(), font, max_text_w)

    line_h = int((font.getbbox("Ay")[3] - font.getbbox("Ay")[1]) * line_spacing)
    text_w = max((font.getbbox(l)[2] for l in lines), default=0)
    text_h = line_h * len(lines)

    box_w = text_w + padding * 2
    box_h = text_h + padding * 2

    if balloon.x is not None and balloon.y is not None:
        x = int(balloon.x * img.width - box_w / 2)
        y = int(balloon.y * img.height - box_h / 2)
    else:
        x, y = auto_anchor(img, box_w, box_h, taken)

    x = max(12, min(x, img.width - box_w - 12))
    y = max(12, min(y, img.height - box_h - 12))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    radius = int(box_h * style["radius"])

    od.rounded_rectangle(
        [x, y, x + box_w, y + box_h],
        radius=radius,
        fill=style["fill"],
        outline=style["outline"],
        width=3,
    )
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (0, 0))

    d = ImageDraw.Draw(img)
    text_fill = "#DCF2FF" if balloon.kind == "system" else "#1A1016"
    ty = y + padding
    for line in lines:
        lw = font.getbbox(line)[2]
        d.text((x + (box_w - lw) // 2, ty), line, font=font, fill=text_fill)
        ty += line_h

    taken.append((x, y, box_w, box_h))
    return x, y, box_w, box_h

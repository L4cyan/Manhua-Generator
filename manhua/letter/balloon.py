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
# `tail` is what separates a balloon from a caption box: speech comes out of a
# mouth and points at it, narration does not belong to anyone in the panel.
_STYLES = {
    "speech":    {"fill": "#FFFFFF", "outline": "#2A1F2E", "radius": 0.42, "dash": False, "tail": True},
    "thought":   {"fill": "#FFFFFF", "outline": "#2A1F2E", "radius": 0.50, "dash": True,  "tail": False},
    "narration": {"fill": "#FFF9EC", "outline": "#5A4632", "radius": 0.06, "dash": False, "tail": False},
    "shout":     {"fill": "#FFFFFF", "outline": "#1A1016", "radius": 0.18, "dash": False, "tail": True},
    "whisper":   {"fill": "#F7F5FA", "outline": "#6A5F70", "radius": 0.42, "dash": True,  "tail": True},
    # Cultivation-genre system window.
    "system":    {"fill": "#0E1A2BE0", "outline": "#5FC7FF", "radius": 0.04, "dash": False, "tail": False},
}


def _tail_points(img: Image.Image, x: int, y: int, w: int, h: int, taken=None):
    """Triangle from the balloon edge toward whoever is probably speaking.

    There is no face detector here, so the target is the busiest region of the
    panel that is not under the balloon: in a portrait or two-shot that is the
    face, which is what a tail should point at. It only has to be roughly
    right -- a tail aimed at the correct half of the panel reads as lettering,
    and no tail at all reads as a placeholder.
    """
    energy = edge_energy(img)
    eh, ew = energy.shape
    if eh < 4 or ew < 4:
        return None
    scale = img.width / max(1, ew)

    e = energy.copy()
    bx0, by0 = int(x / scale), int(y / scale)
    bx1, by1 = int((x + w) / scale), int((y + h) / scale)
    e[max(0, by0):by1 + 1, max(0, bx0):bx1 + 1] = 0.0
    if e.max() <= 0:
        return None

    # Centroid of the top-decile energy, which clusters on faces and linework.
    thresh = np.quantile(e[e > 0], 0.90) if (e > 0).any() else 0.0
    ys, xs = np.nonzero(e >= max(thresh, 1e-6))
    if len(xs) == 0:
        return None
    tx, ty = float(xs.mean()) * scale, float(ys.mean()) * scale

    cx, cy = x + w / 2, y + h / 2
    base = max(14, min(w, h) // 5)

    # Leave from the edge facing the target, so the tail never crosses the body.
    if abs(tx - cx) > abs(ty - cy):
        ex = x + w if tx > cx else x
        p0 = (ex, int(cy - base / 2))
        p2 = (ex, int(cy + base / 2))
    else:
        ey = y + h if ty > cy else y
        p0 = (int(cx - base / 2), ey)
        p2 = (int(cx + base / 2), ey)

    # A tail is a short spur, not a leader line. Left uncapped it stretches
    # most of the way across the panel and reads as a stray stroke, so clamp
    # it to a fraction of the balloon it belongs to.
    import math

    dx, dy = tx - cx, ty - cy
    dist = math.hypot(dx, dy) or 1.0
    reach = min(dist * 0.55, max(h * 0.75, 46))
    tipx = int(cx + dx / dist * reach)
    tipy = int(cy + dy / dist * reach)
    tipx = max(4, min(tipx, img.width - 4))
    tipy = max(4, min(tipy, img.height - 4))

    # Never let a tail run through another balloon.
    for tx0, ty0, tw0, th0 in (taken or []):
        if tx0 <= tipx <= tx0 + tw0 and ty0 <= tipy <= ty0 + th0:
            return None
    return [p0, (tipx, tipy), p2]


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
    band: tuple[int, int] | None = None,
    column: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Find the top-left corner of the quietest free window for a balloon.

    `taken` holds pixel boxes already occupied by other balloons in this panel
    so multiple balloons do not stack on top of each other. `band` pins the
    search to a horizontal strip (pixel y, height) and `column` to a vertical
    one, which is how a partially anchored balloon keeps its reading position
    on one axis while still dodging detail on the other.
    """
    energy = edge_energy(img)
    scale = img.width / max(1, energy.shape[1])
    eh, ew = energy.shape

    win_w = max(1, int(box_w / scale))
    win_h = max(1, int(box_h / scale))
    if win_w >= ew or win_h >= eh:
        return (img.width - box_w) // 2, 24

    y_lo, y_hi = 0, eh - win_h
    x_lo, x_hi = 0, ew - win_w
    if band is not None:
        by = int(band[0] / scale)
        y_lo = y_hi = max(0, min(by, eh - win_h))
    if column is not None:
        bx = int(column[0] / scale)
        x_lo = x_hi = max(0, min(bx, ew - win_w))

    # Summed-area table for O(1) window means.
    integral = energy.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))

    best, best_xy = float("inf"), (max(0, x_lo) * int(scale), max(0, y_lo) * int(scale))
    step = max(1, min(win_w, win_h) // 4)
    for y in range(y_lo, y_hi + 1, step):
        for x in range(x_lo, x_hi + 1, step):
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

    # A long line inside a narrow box becomes a tower that swallows the panel.
    # Real lettering widens the balloon instead, so the aspect stays readable:
    # give long text more width and re-wrap until it is no taller than it is
    # wide, up to the hard ceiling.
    text = balloon.text.strip()
    line_h = int((font.getbbox("Ay")[3] - font.getbbox("Ay")[1]) * line_spacing)
    frac = max_width_frac
    while True:
        max_text_w = int(img.width * frac) - padding * 2
        lines = _wrap(text, font, max_text_w)
        text_w = max((font.getbbox(l)[2] for l in lines), default=0)
        text_h = line_h * len(lines)
        if text_h <= text_w * 1.15 or frac >= 0.88:
            break
        frac = min(0.88, frac + 0.08)

    box_w = text_w + padding * 2
    box_h = text_h + padding * 2

    # Partial anchors matter for panels with several balloons: pinning y fixes
    # the reading order top-to-bottom, while x is still free to dodge the face.
    if balloon.x is not None and balloon.y is not None:
        x = int(balloon.x * img.width - box_w / 2)
        y = int(balloon.y * img.height - box_h / 2)
    elif balloon.y is not None:
        y = int(balloon.y * img.height - box_h / 2)
        x, _ = auto_anchor(img, box_w, box_h, taken, band=(y, box_h))
    elif balloon.x is not None:
        x = int(balloon.x * img.width - box_w / 2)
        _, y = auto_anchor(img, box_w, box_h, taken, column=(x, box_w))
    else:
        x, y = auto_anchor(img, box_w, box_h, taken)

    x = max(12, min(x, img.width - box_w - 12))
    y = max(12, min(y, img.height - box_h - 12))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # Radius off the SHORT side, not the height: scaling it by height alone
    # turned every tall balloon into a stadium and left short ones nearly
    # square, so no two balloons on a page shared a silhouette.
    radius = int(min(box_w, box_h) * style["radius"])
    radius = max(6, min(radius, min(box_w, box_h) // 2))

    # Tail first, so the balloon body draws over its base and hides the seam.
    if style["tail"]:
        tail = _tail_points(img, x, y, box_w, box_h, taken)
        if tail:
            od.polygon(tail, fill=style["fill"], outline=style["outline"])

    od.rounded_rectangle(
        [x, y, x + box_w, y + box_h],
        radius=radius,
        fill=style["fill"],
        outline=style["outline"],
        width=3,
    )
    if style["tail"] and tail:
        # Redraw the two tail edges on top; the rounded rect just covered them.
        od.line([tail[0], tail[1]], fill=style["outline"], width=3)
        od.line([tail[1], tail[2]], fill=style["outline"], width=3)
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

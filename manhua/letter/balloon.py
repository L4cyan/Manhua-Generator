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
    "speech":    {"fill": "#FFFFFF", "outline": "#2A1F2E", "radius": 0.42, "dash": False, "tail": True,  "shape": "oval"},
    "thought":   {"fill": "#FFFFFF", "outline": "#2A1F2E", "radius": 0.50, "dash": True,  "tail": False, "shape": "oval"},
    "narration": {"fill": "#FFF9EC", "outline": "#5A4632", "radius": 0.06, "dash": False, "tail": False, "shape": "rect"},
    "shout":     {"fill": "#FFFFFF", "outline": "#1A1016", "radius": 0.18, "dash": False, "tail": True,  "shape": "rect"},
    "whisper":   {"fill": "#F7F5FA", "outline": "#6A5F70", "radius": 0.42, "dash": True,  "tail": True,  "shape": "oval"},
    # Cultivation-genre system window.
    "system":    {"fill": "#0E1A2BE0", "outline": "#5FC7FF", "radius": 0.04, "dash": False, "tail": False, "shape": "rect"},
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


def _tail_to(x: int, y: int, w: int, h: int, tx: int, ty: int):
    """Triangle from the balloon edge toward an explicit target point.

    Replaces the old guess-where-the-speaker-is version. The caller says where
    the tail should point; this only works out which edge it leaves from and
    keeps the spur short enough to read as lettering rather than a stray line.
    """
    import math

    cx, cy = x + w / 2, y + h / 2
    base = max(12, min(w, h) // 5)
    if abs(tx - cx) > abs(ty - cy):
        ex = x + w if tx > cx else x
        p0, p2 = (ex, int(cy - base / 2)), (ex, int(cy + base / 2))
    else:
        ey = y + h if ty > cy else y
        p0, p2 = (int(cx - base / 2), ey), (int(cx + base / 2), ey)

    dx, dy = tx - cx, ty - cy
    dist = math.hypot(dx, dy) or 1.0
    reach = min(dist, max(h * 0.9, 40))
    return [p0, (int(cx + dx / dist * reach), int(cy + dy / dist * reach)), p2]


def edge_energy(img: Image.Image, downscale: int = 8) -> np.ndarray:
    """Low-resolution map of visual busyness. High values = detailed regions."""
    small = img.convert("L").resize(
        (max(1, img.width // downscale), max(1, img.height // downscale)),
        Image.BILINEAR,
    )
    return np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32)


def skin_map(img: Image.Image, downscale: int = 8) -> np.ndarray:
    """Where the skin is, at the same resolution as the energy map.

    Balloons must not cover faces, and edge energy alone does not protect them:
    a face is busy, but so is foliage and so is architecture, and the placement
    search happily treats a cheek as just another textured region. Skin tone is
    a crude proxy for "a face is here" and it is the one thing in a panel that
    must stay visible.
    """
    small = img.convert("RGB").resize(
        (max(1, img.width // downscale), max(1, img.height // downscale)),
        Image.BILINEAR,
    )
    a = np.asarray(small).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    skin = ((r > 95) & (g > 40) & (b > 20) & (r > g + 12) & (r > b + 12)
            & (np.abs(r - g) > 8))
    return skin.astype(np.float32)


def auto_anchor(
    img: Image.Image,
    box_w: int,
    box_h: int,
    taken: list[tuple[int, int, int, int]],
    top_bias: float = 0.5,
    band: tuple[int, int] | None = None,
    column: tuple[int, int] | None = None,
    centre_bias: float = 0.40,
    skin_penalty: float = 6.0,
) -> tuple[int, int]:
    """Find the top-left corner of the quietest free window for a balloon.

    `taken` holds pixel boxes already occupied by other balloons in this panel
    so multiple balloons do not stack on top of each other. `band` pins the
    search to a horizontal strip (pixel y, height) and `column` to a vertical
    one, which is how a partially anchored balloon keeps its reading position
    on one axis while still dodging detail on the other.
    """
    energy = edge_energy(img)
    skin = skin_map(img)
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

    # Summed-area tables for O(1) window means.
    integral = energy.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    sk = skin.cumsum(0).cumsum(1)
    sk = np.pad(sk, ((1, 0), (1, 0)))

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
            # Pull toward the horizontal centre. Purely energy-driven placement
            # wanders to whichever margin happens to be emptiest, so balloons
            # land hard against alternating edges and the page reads as noise.
            # Lettering wants a stable column, not the mathematically quietest
            # pixel.
            cx_off = abs((x + win_w / 2) / max(1, ew) - 0.5) * 2.0
            score += cx_off * 255.0 * centre_bias

            # Covering a face is the one placement error a reader always
            # notices, so it outweighs everything else here. This penalty also
            # counteracts the centre pull above, which on a portrait panel aims
            # a balloon straight at the head.
            face = (sk[y + win_h, x + win_w] - sk[y, x + win_w]
                    - sk[y + win_h, x] + sk[y, x]) / (win_w * win_h)
            score += face * 255.0 * skin_penalty

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
        if (font.getbbox(trial)[2] - font.getbbox(trial)[0]) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _line_extent(font, line: str) -> tuple[float, float]:
    """True ink width and height of one line, not the advance box.

    getbbox()[2] is the right edge measured from the origin and ignores a
    negative left bearing, so a line that starts with an overhanging glyph
    measures narrower than it draws. Every balloon in the chapter overflowed
    by a few percent because of exactly that.
    """
    bb = font.getbbox(line)
    return (bb[2] - bb[0]), (bb[3] - bb[1])


def _fits_oval(font, lines: list[str], a: float, b: float, line_h: int,
               margin: float) -> bool:
    """Does every line fit inside the ellipse at the y it will be DRAWN at?

    The wrap decides widths for one set of row positions; the renderer centres
    the block, which puts the lines somewhere else. Checking against the drawn
    positions is what closes that gap.
    """
    n = len(lines)
    if n == 0:
        return True
    total = line_h * n
    for i, line in enumerate(lines):
        w, h = _line_extent(font, line)
        # Centre of this line, relative to the ellipse centre.
        y = -total / 2 + (i + 0.5) * line_h
        # Use the line's REAL ink height, not the nominal line height. A short
        # balloon is sized so tightly that the difference between the two is
        # the whole margin, which is why one-line balloons were the ones that
        # overflowed while long ones were fine.
        yy = abs(y) + max(h, line_h) / 2 + margin
        if yy >= b:
            return False
        avail = a * (1.0 - (yy / b) ** 2) ** 0.5
        if w / 2 + margin > avail:
            return False
    return True


def _wrap_oval(text: str, font: ImageFont.FreeTypeFont, a: float, b: float,
               line_h: int) -> list[str] | None:
    """Wrap text to fit inside the ellipse with semi-axes `a` and `b`.

    A speech balloon is an oval, so the usable width is not constant: it is
    widest across the middle and narrows toward the top and bottom. Wrapping
    to a rectangle and then drawing an ellipse around it is what forces the
    ellipse to be ~1.4x bigger than it needs to be in both directions, which
    is why naive oval balloons swallow the panel. Tapering the lines instead
    keeps the balloon close to the size of the text.

    Returns None if the text cannot be made to fit, so the caller can grow the
    ellipse and try again.
    """
    words = text.split()
    if not words:
        return []
    n_max = max(1, int((2 * b) // line_h))
    lines: list[str] = []
    i = 0
    for row in range(n_max):
        if i >= len(words):
            break
        # Vertical centre of this line, measured from the ellipse centre, for
        # a block of `n_max` lines centred vertically.
        y = -(n_max * line_h) / 2 + (row + 0.5) * line_h
        inner = 1.0 - (y / b) ** 2
        if inner <= 0:
            continue
        avail = 2 * a * (inner ** 0.5) - 12      # small side bearing
        if avail <= 0:
            continue
        cur = ""
        while i < len(words):
            trial = f"{cur} {words[i]}".strip()
            if (font.getbbox(trial)[2] - font.getbbox(trial)[0]) <= avail or not cur:
                cur = trial
                i += 1
            else:
                break
        if not cur:
            return None
        lines.append(cur)
    return lines if i >= len(words) else None


def _layout_balloon(img, balloon, font, padding, line_spacing, max_width_frac, style):
    """Size a balloon's box for a given font, without drawing anything.

    Split out so draw_balloon can try several type sizes and keep the first
    that respects the panel-area cap.
    """
    text = balloon.text.strip()
    line_h = int((font.getbbox("Ay")[3] - font.getbbox("Ay")[1]) * line_spacing)
    if style["shape"] == "oval":
        flat = _wrap(text, font, int(img.width * max_width_frac) - padding * 2)
        tw = max((font.getbbox(l)[2] for l in flat), default=10)
        th = line_h * len(flat)
        a = tw * 0.62 + padding
        b = max(th * 0.66, th * 0.5 + line_h * 0.55) + padding
        for _ in range(28):
            lines = _wrap_oval(text, font, a, b, line_h)
            if lines is not None and a * 2 <= img.width * 0.92:
                return int(a * 2), int(b * 2)
            a *= 1.07
            b *= 1.05
        return tw + padding * 2, th + padding * 2
    frac = max_width_frac
    while True:
        lines = _wrap(text, font, int(img.width * frac) - padding * 2)
        text_w = max((font.getbbox(l)[2] for l in lines), default=0)
        text_h = line_h * len(lines)
        if text_h <= text_w * 1.15 or frac >= 0.88:
            return text_w + padding * 2, text_h + padding * 2
        frac = min(0.88, frac + 0.08)


def draw_balloon(
    img: Image.Image,
    balloon: Balloon,
    font: ImageFont.FreeTypeFont,
    *,
    padding: int = 22,
    line_spacing: float = 1.25,
    max_width_frac: float = 0.52,
    taken: list[tuple[int, int, int, int]] | None = None,
    max_area_frac: float = 0.16,
) -> tuple[int, int, int, int]:
    """Render one balloon onto `img` in place. Returns its pixel box.

    `max_area_frac` is the important limit. Sizing a balloon purely from its
    text means a long line produces a balloon that eats a third of the panel,
    and a reader loses the art to the lettering. Real lettering shrinks the
    type instead. If the balloon would exceed this share of the panel, the font
    is stepped down and the whole thing re-laid until it fits.
    """
    taken = taken if taken is not None else []
    style = _STYLES.get(balloon.kind, _STYLES["speech"])

    # A dragged width overrides the style default. Clamped, because a balloon
    # wrapped to 5% of the panel is one letter per line and one wrapped to 100%
    # runs off both edges.
    if getattr(balloon, "width", None):
        max_width_frac = min(0.92, max(0.12, float(balloon.width)))
        # The area cap exists to stop AUTO-sized balloons swallowing the art.
        # A width that was dragged there is a decision, so give it room before
        # the type starts shrinking under it.
        max_area_frac = max(max_area_frac, 0.30)

    # Try progressively smaller type until the balloon respects the area cap.
    base_size = getattr(font, "size", 32)
    for attempt in range(6):
        size = max(13, int(base_size * (0.88 ** attempt)))
        try:
            f2 = font.font_variant(size=size) if attempt else font
        except Exception:
            f2 = font
        box = _layout_balloon(img, balloon, f2, padding, line_spacing,
                              max_width_frac, style)
        if box is None:
            continue
        bw, bh = box
        if bw * bh <= img.width * img.height * max_area_frac or attempt == 5:
            font = f2
            break

    # A long line inside a narrow box becomes a tower that swallows the panel.
    # Real lettering widens the balloon instead, so the aspect stays readable:
    # give long text more width and re-wrap until it is no taller than it is
    # wide, up to the hard ceiling.
    text = balloon.text.strip()
    line_h = int((font.getbbox("Ay")[3] - font.getbbox("Ay")[1]) * line_spacing)
    oval = style["shape"] == "oval"

    if oval:
        # Grow an ellipse until the tapered wrap fits inside it. Starting from
        # the rectangular text size and inflating is what keeps the balloon
        # close to the size of its text instead of ~1.4x in both directions.
        flat = _wrap(text, font, int(img.width * max_width_frac) - padding * 2)
        tw = max((font.getbbox(l)[2] for l in flat), default=10)
        th = line_h * len(flat)
        a = tw * 0.62 + padding
        b = max(th * 0.66, th * 0.5 + line_h * 0.55) + padding
        lines = None
        margin = max(6.0, padding * 0.45)
        for _ in range(40):
            cand = _wrap_oval(text, font, a, b, line_h)
            # Not enough to wrap successfully: the lines must still fit at the
            # positions the renderer will actually put them.
            if cand is not None and _fits_oval(font, cand, a, b, line_h, margin):
                lines = cand
                break
            a *= 1.05
            b *= 1.04
            if a * 2 > img.width * 0.95:
                lines = cand
                break
        if lines is None:                      # give up gracefully
            lines, oval = flat, False
            box_w, box_h = tw + padding * 2, th + padding * 2
        else:
            # Floor the size. Very short text ("...", "No.") produced tall thin
            # slivers, because the ellipse was sized from the text alone and a
            # three-character line is barely wider than one glyph. A speech
            # balloon has a minimum readable shape regardless of its contents.
            min_w = line_h * 3.4
            min_h = line_h * 2.1
            box_w = int(max(a * 2, min_w))
            box_h = int(max(b * 2, min_h))
            # Keep it from going taller than it is wide on short lines.
            if box_h > box_w * 0.95:
                box_h = int(box_w * 0.95)
    else:
        # A long line inside a narrow box becomes a tower that swallows the
        # panel. Real lettering widens the box instead.
        frac = max_width_frac
        while True:
            max_text_w = int(img.width * frac) - padding * 2
            lines = _wrap(text, font, max_text_w)
            # True ink width, not the advance edge -- see _line_extent.
            text_w = max((_line_extent(font, l)[0] for l in lines), default=0)
            text_h = line_h * len(lines)
            if text_h <= text_w * 1.15 or frac >= 0.88:
                break
            frac = min(0.88, frac + 0.08)
        box_w = int(text_w) + padding * 2
        box_h = int(text_h) + padding * 2

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

    edge = max(18, int(img.width * 0.035))
    x = max(edge, min(x, img.width - box_w - edge))
    y = max(edge, min(y, img.height - box_h - edge))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # Radius off the SHORT side, not the height: scaling it by height alone
    # turned every tall balloon into a stadium and left short ones nearly
    # square, so no two balloons on a page shared a silhouette.
    radius = int(min(box_w, box_h) * style["radius"])
    radius = max(6, min(radius, min(box_w, box_h) // 2))

    tail = None
    # Tail first, so the balloon body draws over its base and hides the seam.
    # A speaker who is not in the panel gets no tail at all -- a voice from
    # offscreen has no mouth in frame, and a tail pointing at whoever IS there
    # silently reattributes the line to them.
    # A tail is drawn ONLY where one was asked for. Automatic placement aimed
    # at the busiest region of the panel, which produced triangles pointing at
    # trees, and worse, silently assigned a line to whoever was in frame. The
    # editor places these by clicking; nothing guesses.
    tx_n = getattr(balloon, "tail_x", None)
    ty_n = getattr(balloon, "tail_y", None)
    if style["tail"] and tx_n is not None and ty_n is not None:
        tail = _tail_to(x, y, box_w, box_h,
                        int(tx_n * img.width), int(ty_n * img.height))
        if tail:
            od.polygon(tail, fill=style["fill"], outline=style["outline"])

    if oval:
        od.ellipse([x, y, x + box_w, y + box_h],
                   fill=style["fill"], outline=style["outline"], width=3)
    else:
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

    # Centre the text block in BOTH axes, using PIL's middle anchor.
    #
    # It was top-aligned from `y + padding` while _wrap_oval lays its lines out
    # assuming a vertically centred block, so text floated high in every oval.
    # Horizontally it measured width as getbbox()[2] -- the right edge, not the
    # width -- which ignores left side bearing and nudged every line off centre.
    # Letting PIL anchor each line at its own middle removes both errors and the
    # font-metric guesswork with them.
    total_h = line_h * len(lines)
    cx = x + box_w / 2
    ty = y + (box_h - total_h) / 2 + line_h / 2
    for line in lines:
        d.text((cx, ty), line, font=font, fill=text_fill, anchor="mm")
        ty += line_h

    taken.append((x, y, box_w, box_h))
    return x, y, box_w, box_h

"""Vertical long-strip assembly.

Panels are individually generated, so this module owns everything that makes
them read as one continuous scroll: uniform width, consistent gutters,
full-bleed handling, and export slicing that never cuts through artwork.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass
class CanvasCfg:
    width: int = 800
    gutter: int = 28
    margin: int = 0
    background: str = "#FFFFFF"
    slice_max_height: int = 1280
    border: str = "#141018"
    border_width: int = 3


# The gutter is the pacing control of a vertical scroll, and a single fixed
# value for every transition is what makes a chapter read as a slideshow
# instead of a comic. These are the sizes vertical-scroll comics actually use:
# a tight gap keeps the thumb moving through fast action, and a large one
# forces the reader to stop before a reveal lands.
PAUSE_GUTTER: dict[str, int] = {
    "tight": 40,     # rapid dialogue, blow-by-blow action
    "normal": 110,
    "beat": 240,     # a reaction, a held glance
    "scene": 480,    # somewhere else, or some other time
    "cliff": 760,    # make them wait for it
}


@dataclass
class PlacedPanel:
    """A panel's position in the assembled strip, for hit-testing in the UI."""

    panel_id: str
    top: int
    height: int


def _fit(img: Image.Image, target_w: int) -> Image.Image:
    if img.width == target_w:
        return img
    h = round(img.height * target_w / img.width)
    return img.resize((target_w, h), Image.LANCZOS)


def compose(
    panels: list[tuple],
    cfg: CanvasCfg,
) -> tuple[Image.Image, list[PlacedPanel]]:
    """Stack panels into one tall image.

    Each entry is `(panel_id, image, full_bleed)` and may carry two more
    fields, `pause` and `inset`:

      pause  a key of PAUSE_GUTTER controlling the gap BEFORE this panel.
             This is how the strip gets a rhythm rather than a constant beat.
      inset  0.0-0.45, the fraction of the canvas width to pull the panel in
             by. Varying panel width is what stops a scroll reading as a
             stack of identical rectangles; a narrow panel also reads as a
             quieter, smaller moment, which is a storytelling tool.

    Full-bleed panels ignore the margin and the border and butt against their
    neighbours, which is how a vertical scroll signals scale or impact.
    """
    if not panels:
        return Image.new("RGB", (cfg.width, 1), cfg.background), []

    norm = []
    for entry in panels:
        pid, img, bleed = entry[0], entry[1], entry[2]
        pause = entry[3] if len(entry) > 3 else "normal"
        inset = entry[4] if len(entry) > 4 else 0.0
        norm.append((pid, img, bleed, pause, float(inset or 0.0)))

    prepared = []
    for i, (pid, img, bleed, pause, inset) in enumerate(norm):
        if bleed:
            w = cfg.width
        else:
            side = cfg.margin + int(cfg.width * min(max(inset, 0.0), 0.45) / 2)
            w = cfg.width - side * 2
        prepared.append((pid, _fit(img, w), bleed, pause, w))

    total_h = 0
    placements: list[PlacedPanel] = []
    for i, (pid, img, bleed, pause, w) in enumerate(prepared):
        if i > 0:
            # A full-bleed panel butts against what came before it; everything
            # else gets the gap its own pacing asks for.
            total_h += 0 if bleed else PAUSE_GUTTER.get(pause, cfg.gutter)
        placements.append(PlacedPanel(pid, total_h, img.height))
        total_h += img.height

    strip = Image.new("RGB", (cfg.width, total_h), cfg.background)
    draw = ImageDraw.Draw(strip)
    for (pid, img, bleed, pause, w), place in zip(prepared, placements):
        x = 0 if bleed else (cfg.width - w) // 2
        strip.paste(img, (x, place.top))
        if not bleed and cfg.border_width > 0:
            # A drawn edge is what turns an image into a panel.
            draw.rectangle(
                [x, place.top, x + img.width - 1, place.top + img.height - 1],
                outline=cfg.border, width=cfg.border_width,
            )

    return strip, placements


def slice_for_upload(
    strip: Image.Image,
    placements: list[PlacedPanel],
    cfg: CanvasCfg,
) -> list[Image.Image]:
    """Cut the strip into upload-sized chunks along gutters.

    Hosts cap image height (WEBTOON's practical limit is ~1280px), but a naive
    cut lands mid-face. Cuts are snapped to the nearest gutter, and a panel
    taller than the cap is emitted whole rather than damaged.
    """
    # Any panel boundary is a safe cut: no artwork spans it. Where a gutter
    # exists we cut through its middle; where a full-bleed panel removed the
    # gutter, the shared edge itself is still clean.
    safe: list[int] = [0]
    for prev, nxt in zip(placements, placements[1:]):
        gap_start, gap_end = prev.top + prev.height, nxt.top
        safe.append((gap_start + gap_end) // 2 if gap_end > gap_start else gap_start)
    safe.append(strip.height)

    chunks: list[Image.Image] = []
    start = 0
    while start < strip.height:
        limit = start + cfg.slice_max_height
        candidates = [c for c in safe if start < c <= limit]
        # No gutter inside the cap means one oversized panel; take it whole.
        end = max(candidates) if candidates else next(
            (c for c in safe if c > start), strip.height
        )
        chunks.append(strip.crop((0, start, strip.width, end)))
        start = end

    return chunks


_FORMATS = {
    "png": ("png", {}),
    "jpg": ("jpg", {"format": "JPEG", "subsampling": 0, "optimize": True}),
    "jpeg": ("jpg", {"format": "JPEG", "subsampling": 0, "optimize": True}),
    "webp": ("webp", {"format": "WEBP", "method": 5}),
}


def add_credit(strip: Image.Image, text: str, cfg: CanvasCfg) -> Image.Image:
    """Add a thin credit line under the last panel.

    Promotion, not protection: it is a line of pixels in the reader's copy and
    anyone can crop it. It is here because a strip that travels well should say
    where it came from, not because it enforces anything. Off in one click.
    """
    from PIL import ImageDraw, ImageFont

    if not text:
        return strip
    band = max(34, strip.width // 34)
    out = Image.new("RGB", (strip.width, strip.height + band), cfg.background)
    out.paste(strip, (0, 0))
    d = ImageDraw.Draw(out)
    size = max(11, band // 3)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default(size)
    d.text((strip.width // 2, strip.height + band // 2), text,
           font=font, fill="#9a9aa1", anchor="mm")
    return out


def export(
    strip: Image.Image,
    placements: list[PlacedPanel],
    cfg: CanvasCfg,
    out_dir: str | Path,
    stem: str = "episode",
    fmt: str = "png",
    quality: int = 92,
) -> list[Path]:
    """Write the strip whole, plus upload-sized slices.

    PNG is lossless and enormous; a 40,000px strip lands around 60MB, which
    most hosts reject. JPEG at quality 92 with no chroma subsampling is what
    scanlation sites actually run, and keeps the lettering crisp.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext, opts = _FORMATS.get(fmt.lower(), _FORMATS["png"])
    if opts.get("format") in ("JPEG", "WEBP"):
        opts = {**opts, "quality": max(1, min(100, quality))}
    if opts.get("format") == "JPEG" and strip.mode != "RGB":
        strip = strip.convert("RGB")

    def save(img: Image.Image, path: Path) -> Path:
        try:
            img.save(path, **opts)
            return path
        except OSError:
            # The previous file is open in a viewer. Write beside it rather
            # than throwing away a finished render over a file handle.
            alt = path.with_name(f"{path.stem}_new{path.suffix}")
            img.save(alt, **opts)
            print(f"  (previous file was locked; wrote {alt.name})")
            return alt

    written = [save(strip, out / f"{stem}_full.{ext}")]
    for i, chunk in enumerate(slice_for_upload(strip, placements, cfg), start=1):
        if opts.get("format") == "JPEG" and chunk.mode != "RGB":
            chunk = chunk.convert("RGB")
        written.append(save(chunk, out / f"{stem}_{i:03d}.{ext}"))
    return written

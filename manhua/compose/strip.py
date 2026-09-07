"""Vertical long-strip assembly.

Panels are individually generated, so this module owns everything that makes
them read as one continuous scroll: uniform width, consistent gutters,
full-bleed handling, and export slicing that never cuts through artwork.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class CanvasCfg:
    width: int = 800
    gutter: int = 28
    margin: int = 0
    background: str = "#FFFFFF"
    slice_max_height: int = 1280


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
    panels: list[tuple[str, Image.Image, bool]],
    cfg: CanvasCfg,
) -> tuple[Image.Image, list[PlacedPanel]]:
    """Stack panels into one tall image.

    `panels` is (panel_id, image, full_bleed). Full-bleed panels ignore the
    side margin and butt against their neighbours with no gutter above, which
    is how webtoons signal a scale or impact moment.
    """
    if not panels:
        return Image.new("RGB", (cfg.width, 1), cfg.background), []

    inner_w = cfg.width - cfg.margin * 2
    prepared: list[tuple[str, Image.Image, bool]] = [
        (pid, _fit(img, cfg.width if bleed else inner_w), bleed) for pid, img, bleed in panels
    ]

    total_h = 0
    placements: list[PlacedPanel] = []
    for i, (pid, img, bleed) in enumerate(prepared):
        if i > 0 and not bleed:
            total_h += cfg.gutter
        placements.append(PlacedPanel(pid, total_h, img.height))
        total_h += img.height

    strip = Image.new("RGB", (cfg.width, total_h), cfg.background)
    for (pid, img, bleed), place in zip(prepared, placements):
        strip.paste(img, (0 if bleed else cfg.margin, place.top))

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


def export(
    strip: Image.Image,
    placements: list[PlacedPanel],
    cfg: CanvasCfg,
    out_dir: str | Path,
    stem: str = "episode",
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = out / f"{stem}_full.png"
    strip.save(full)

    written = [full]
    for i, chunk in enumerate(slice_for_upload(strip, placements, cfg), start=1):
        p = out / f"{stem}_{i:03d}.png"
        chunk.save(p)
        written.append(p)
    return written

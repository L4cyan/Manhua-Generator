"""Composite panel assembly: background plates + cut-out characters.

This exists because single-pass generation cannot hold a shot. A character's
identity lock is a facial description, and against four words of framing the
face wins, so "wide shot of a man standing in a clearing" reliably returns a
portrait with a tiny figure pasted into it. Worse, every panel re-invents its
environment, so the same lecture hall is a different room in every panel.

The fix is the one webtoon studios already use: build the background once as a
reusable asset, draw the character separately, and composite.

    plate      one wide background per LOCATION, generated once and cached.
               Rendered with no character in the prompt at all, which is why
               plates come out as real places instead of hazy portraits.
    crop       the panel's framing is a CROP of the plate, not a new render.
               An establishing shot takes the whole plate; a medium shot takes
               a third of it. Framing becomes geometry instead of a negotiation
               with the sampler.
    upscale    the crop is enlarged with RealESRGAN rather than re-diffused,
               so a close crop stays the same room.
    character  rendered alone on flat grey, matted with BRIA RMBG, scaled to
               the shot and composited.

Because every panel in a location crops the same plate, the room is literally
the same room, which text prompting cannot guarantee.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..models import Panel, Shot

# Plates are rendered wide and large so that even a tight crop has pixels.
PLATE_W, PLATE_H = 1536, 864

# How much of the plate a shot's crop covers, as a fraction of plate width.
# This is the whole framing mechanism: distance is a crop, not a prompt.
SHOT_CROP: dict[Shot, float] = {
    Shot.establishing: 1.00,
    Shot.wide: 0.74,
    Shot.full: 0.66,
    Shot.medium: 0.46,
    Shot.over_shoulder: 0.46,
    Shot.pov: 0.60,
    Shot.close: 0.30,
    Shot.extreme_close: 0.20,
    Shot.reaction: 0.30,
    Shot.insert: 0.26,
}

# Character height as a fraction of the finished panel height. Over 1.0 means
# the figure is cropped by the panel edge, which is what a medium or close
# shot actually is.
CHAR_HEIGHT: dict[Shot, float] = {
    Shot.establishing: 0.30,
    Shot.wide: 0.62,
    Shot.full: 0.88,
    Shot.medium: 1.45,
    Shot.over_shoulder: 1.45,
    Shot.pov: 1.20,
    Shot.close: 3.10,
    Shot.extreme_close: 5.00,
    Shot.reaction: 3.10,
    Shot.insert: 2.00,
}

# A plate must be an empty stage. Naming what must NOT be there matters more
# than naming what should: a plate with a character already in it cannot have
# one composited into it.
PLATE_CLAUSE = (
    "empty scene, no people, no characters, nobody, unobstructed foreground, "
    "clear ground plane, background plate, environment art, "
    "detailed architecture, correct perspective, clean straight lines"
)

# Atmosphere clauses that turn a place into a haze. A plate wants geometry.
PLATE_STRIP = [
    "ethereal atmospheric lighting",
    "prominent rim lighting",
    "glowing magical qi aura",
    "floating light particles",
    "dramatic and ethereal vibe",
    "flowing gravity-defying movement",
    "highly stylized sharp angular V-shaped jawlines",
    "narrow piercing eyes with highly detailed vibrant irises",
    "prominent stylized eyelashes",
    "glossy banded specular highlights on hair",
    "sharp structural folds on clothing fabric",
]

# The character is rendered alone so the matte is clean. Flat, even, and on a
# colour that does not appear in hair or skin.
# `simple background` and `white background` are the Danbooru tags that
# actually produce a flat backdrop. Prose like "plain studio backdrop" gives a
# studio -- with a floor, a horizon line and a shadow -- and the matte then
# keeps the floor along with the character.
CHAR_PLATE = (
    "simple background, white background, plain background, "
    "no background, no scenery, no floor, no shadow on the ground, "
    "isolated on white, flat even lighting"
)


@dataclass
class PanelPlan:
    """How one panel is assembled."""

    panel_id: str
    location: str
    # 0-1 position of the crop centre within the plate. Panels in the same
    # location pan across one plate instead of re-rendering the room.
    crop_x: float = 0.5
    crop_y: float = 0.5
    # 0-1 position of the character's feet within the panel.
    char_x: float = 0.5
    char_y: float = 0.94
    char_flip: bool = False
    # Explicit asset slot; otherwise one is chosen from the shot and
    # the character's expression.
    asset: str = ""


def plate_style(style):
    """Style lock for a background plate: no anatomy, no atmosphere haze."""
    s = copy.deepcopy(style)
    body = s.style_body
    for clause in PLATE_STRIP:
        body = body.replace(clause + ", ", "").replace(", " + clause, "").replace(clause, "")
    s.style_body = ", ".join(p.strip() for p in body.split(",") if p.strip())
    return s


def crop_plate(plate: Image.Image, panel: Panel, plan: PanelPlan,
               out_w: int, out_h: int) -> Image.Image:
    """Take the panel's framing out of the plate as a crop."""
    frac = SHOT_CROP.get(panel.shot, 0.6)
    target_ar = out_w / out_h

    cw = plate.width * frac
    chh = cw / target_ar
    if chh > plate.height:
        chh = plate.height
        cw = chh * target_ar

    cx = plan.crop_x * plate.width
    cy = plan.crop_y * plate.height
    left = max(0, min(cx - cw / 2, plate.width - cw))
    top = max(0, min(cy - chh / 2, plate.height - chh))
    box = (int(left), int(top), int(left + cw), int(top + chh))
    return plate.crop(box)


def place_character(bg: Image.Image, cut: Image.Image, panel: Panel,
                    plan: PanelPlan) -> Image.Image:
    """Scale the matted character to the shot and composite onto the crop."""
    target_h = CHAR_HEIGHT.get(panel.shot, 0.9) * bg.height
    scale = target_h / max(1, cut.height)
    new = cut.resize((max(1, int(cut.width * scale)),
                      max(1, int(cut.height * scale))), Image.LANCZOS)
    if plan.char_flip:
        new = new.transpose(Image.FLIP_LEFT_RIGHT)

    out = bg.convert("RGBA")
    # char_y is where the character's FEET sit, so the paste is anchored at the
    # bottom of the figure. Anchoring at the top makes every shot float.
    x = int(plan.char_x * out.width - new.width / 2)
    y = int(plan.char_y * out.height - new.height)
    out.alpha_composite(new, (max(-new.width // 2, x), y) if x > -new.width else (0, y))
    return out.convert("RGB")


def char_style(style):
    """Style lock for a character plate: no atmosphere, no drama.

    The same clauses that make a panel look good make a cut-out impossible.
    "ethereal atmospheric lighting", "prominent rim lighting" and "dynamic
    composition" all imply a scene with depth and a light source somewhere in
    it, so the backdrop never comes out flat and the matte keeps half the
    frame.
    """
    s = copy.deepcopy(style)
    body = s.style_body
    for clause in ("ethereal atmospheric lighting", "prominent rim lighting",
                   "glowing magical qi aura", "floating light particles",
                   "dramatic and ethereal vibe", "dynamic composition",
                   "flowing gravity-defying movement"):
        body = body.replace(clause + ", ", "").replace(", " + clause, "").replace(clause, "")
    s.style_body = ", ".join(p.strip() for p in body.split(",") if p.strip())
    return s


def character_request(character, ref, style, world: str, seed: int):
    """A render request for ONE character alone on a flat backdrop.

    Built from scratch rather than from the panel, because the panel's `action`
    and `setting` describe the whole scene -- other people, the weather, the
    room. Feed those to a character render and it paints the scene again, the
    backdrop is never flat, and the matte has nothing clean to cut against.

    The backdrop tags go FIRST and weighted. Placed after the identity block
    they sit ~200 tokens deep, behind everything the style lock says about
    lighting and depth, and lose.
    """
    from ..render.base import RenderRequest

    framing = ("(simple background:1.5), (white background:1.5), isolated on white, "
               "(full body:1.5), (head to toe:1.4), full figure, entire body visible, "
               "feet visible, shoes visible, standing, camera far back, solo, one person, "
               "no scenery, no floor, no cast shadow")
    bits = [framing, "1girl" if character.sex == "female" else "1boy"]
    if character.framing_hint:
        bits.append(character.framing_hint)
    bits.append(character.appearance_prompt(ref, world=world, distance="near"))
    # `pose` only -- gaze and expression are already inside appearance_prompt,
    # and repeating them was emitting "back to the viewer" twice.
    if ref is not None and ref.pose:
        bits.append(ref.pose)

    neg = ", ".join(x for x in (
        style.negative, character.negative,
        "portrait, close-up, face focus, bust, upper body only, cowboy shot, cropped legs, "
        "scenery, background detail, buildings, trees, sky, furniture, other people, crowd, "
        "detailed background, gradient background, floor, ground, horizon",
    ) if x)

    loras = []
    if character.lora:
        loras.append((character.lora, character.lora_weight))

    cs = char_style(style)
    return RenderRequest(
        positive=cs.positive(", ".join(b for b in bits if b), register="neutral"),
        negative=neg, width=832, height=1216, seed=seed, loras=loras, style=cs,
    )


def matte_on_white(img: Image.Image, tol: int = 34) -> Image.Image:
    """Cut a figure off a white backdrop deterministically.

    Preferred over a matting model when the backdrop is white BY CONSTRUCTION,
    which is the case for character plates. A learned matte has to guess what
    the subject is, and here it kept guessing that the ground-shadow band the
    model insists on painting across the bottom was part of the character.

    Two steps:
      1. flood the near-white background inward from all four edges, so white
         enclosed BY the figure (a shirt, a highlight) is never punched out;
      2. keep only the largest remaining blob, which drops the ground band and
         any stray marks in a corner.
    """
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w, _ = a.shape
    near_white = ((a > 255 - tol).all(axis=2))

    # Flood fill from the border through near-white pixels.
    from collections import deque
    bg = np.zeros((h, w), dtype=bool)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if near_white[y, x] and not bg[y, x]:
                bg[y, x] = True; dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near_white[y, x] and not bg[y, x]:
                bg[y, x] = True; dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and near_white[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                dq.append((ny, nx))

    fg = ~bg
    # Largest connected component of the foreground.
    try:
        from scipy import ndimage
        lab, n = ndimage.label(fg)
        if n > 1:
            sizes = ndimage.sum(fg, lab, range(1, n + 1))
            fg = lab == (int(np.argmax(sizes)) + 1)
    except Exception:
        pass

    alpha = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    out = img.convert("RGBA")
    out.putalpha(alpha)
    return out


def harmonise(cut: Image.Image, bg: Image.Image, strength: float = 0.38) -> Image.Image:
    """Pull a cut-out character's colour toward the scene it is standing in.

    A figure matted off a white studio backdrop and dropped into a night void
    still looks studio-lit, and that mismatch is the single loudest tell that a
    panel was composited. Real compositors colour-match; this is the cheap
    version of the same idea: shift the subject's per-channel mean and spread
    partway toward the background's.

    `strength` is deliberately well below 1.0 -- matching completely would
    drain the character's own palette and, in a strongly tinted scene, turn him
    the colour of the wall.
    """
    a = np.asarray(cut.convert("RGBA")).astype(np.float32)
    rgb, alpha = a[..., :3], a[..., 3:]
    solid = alpha[..., 0] > 200
    if not solid.any():
        return cut

    b = np.asarray(bg.convert("RGB")).astype(np.float32)
    for c in range(3):
        src = rgb[..., c][solid]
        s_mu, s_sd = float(src.mean()), float(src.std()) or 1.0
        t_mu, t_sd = float(b[..., c].mean()), float(b[..., c].std()) or 1.0
        # Blend the target statistics rather than adopting them outright.
        mu = s_mu + (t_mu - s_mu) * strength
        sd = s_sd + (t_sd - s_sd) * strength * 0.6
        rgb[..., c] = (rgb[..., c] - s_mu) * (sd / s_sd) + mu

    out = np.concatenate([np.clip(rgb, 0, 255), alpha], axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def ground_shadow(bg: Image.Image, cx: float, cy: float, width: int,
                  opacity: int = 90) -> Image.Image:
    """Soft contact shadow under the feet.

    Without one a composited figure hovers. This is the cheapest possible fix
    and it does more for "is this character in the scene" than the colour
    match does.
    """
    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = max(8, int(width * 0.85))
    h = max(4, int(w * 0.17))
    x, y = int(cx * bg.width), int(cy * bg.height)
    d.ellipse([x - w // 2, y - h // 2, x + w // 2, y + h // 2], fill=(0, 0, 0, opacity))
    layer = layer.filter(ImageFilter.GaussianBlur(max(3, h // 2)))
    out = bg.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")

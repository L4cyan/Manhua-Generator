"""Zero-dependency backend that renders prompt cards instead of art.

Exists so the whole pipeline -- breakdown, composition, lettering, slicing,
and the studio UI -- can be exercised without a GPU or any model downloads.
Useful for CI and for contributors who only want to touch the layout code.
"""
from __future__ import annotations

import colorsys
import hashlib
import textwrap

from PIL import Image, ImageDraw

from .base import Backend, RenderRequest


class MockBackend(Backend):
    def render(self, req: RenderRequest) -> Image.Image:
        # Deterministic hue from the seed so panels are visually distinguishable
        # and a reroll visibly changes something.
        h = int(hashlib.sha1(str(req.seed).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        r, g, b = colorsys.hsv_to_rgb(h, 0.28, 0.92)
        bg = (int(r * 255), int(g * 255), int(b * 255))

        img = Image.new("RGB", (req.width, req.height), bg)
        draw = ImageDraw.Draw(img)
        draw.rectangle([8, 8, req.width - 8, req.height - 8], outline=(60, 45, 70), width=4)

        # Strip the style prefix; only the panel-specific tail is informative here.
        content = req.positive
        marker = "professional colouring"
        if marker in content:
            content = content.split(marker)[0]
        content = content[-700:]

        y = 40
        for line in textwrap.wrap(content, width=max(20, req.width // 11))[:26]:
            draw.text((28, y), line, fill=(40, 30, 50))
            y += 20

        draw.text((28, req.height - 40), f"seed {req.seed} · {req.width}x{req.height}", fill=(90, 70, 100))
        return img

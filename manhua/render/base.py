"""Render backend interface.

Every backend takes a fully-assembled prompt pair plus locked sampler settings
and returns a PIL image. Swapping backends must never change the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from ..config import StyleLock
from ..models import Character, Panel


@dataclass
class RenderRequest:
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    # (lora_filename, weight) pairs: style lora first, then character loras.
    loras: list[tuple[str, float]]
    style: StyleLock


class Backend(ABC):
    @abstractmethod
    def render(self, req: RenderRequest) -> Image.Image:
        """Produce one panel image. Must be deterministic for a fixed seed."""

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass


def build_request(
    panel: Panel,
    style: StyleLock,
    bible: dict[str, Character],
    seed: int,
) -> RenderRequest:
    """Turn a Panel into a backend-agnostic render request.

    This is the only place the style lock and the panel content are joined,
    which is what guarantees no panel can accidentally skip the style.
    """
    width, height = style.size(panel.aspect)

    loras: list[tuple[str, float]] = []
    if style.render.style_lora:
        loras.append((style.render.style_lora, style.render.style_lora_weight))
    for ref in panel.characters:
        char = bible.get(ref.id)
        if char and char.lora:
            loras.append((char.lora, char.lora_weight))

    # Per-shot negatives fight the portrait bias baked into anime checkpoints:
    # without them an "establishing" panel still returns a face filling frame.
    from ..models import SHOT_NEGATIVES

    shot_neg = SHOT_NEGATIVES.get(panel.shot, "")
    negative = f"{style.negative}, {shot_neg}" if shot_neg else style.negative

    return RenderRequest(
        positive=style.positive(
            panel.content_prompt(bible),
            has_fx=bool(panel.fx),
            register=panel.world,
        ),
        negative=negative,
        width=width,
        height=height,
        seed=seed,
        loras=loras,
        style=style,
    )

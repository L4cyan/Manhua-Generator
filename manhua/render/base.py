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


# Style clauses that describe a PERSON. Harmless when someone is in frame,
# actively destructive when nobody is: see the note in build_request.
SCENERY_STYLE_STRIP = [
    "highly stylized sharp angular V-shaped jawlines",
    "narrow piercing eyes with highly detailed vibrant irises",
    "prominent stylized eyelashes",
    "glossy banded specular highlights on hair",
    "sharp structural folds on clothing fabric",
]


def _scenery_style(style: StyleLock) -> StyleLock:
    """A copy of the style lock with the anatomy clauses removed."""
    import copy

    quiet = copy.deepcopy(style)
    body = quiet.style_body
    for clause in SCENERY_STYLE_STRIP:
        body = body.replace(clause + ", ", "").replace(", " + clause, "").replace(clause, "")
    quiet.style_body = ", ".join(p.strip() for p in body.split(",") if p.strip())
    return quiet


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
    char_neg = ", ".join(
        c.negative for ref in panel.characters
        if (c := bible.get(ref.id)) and c.negative
    )
    negative = ", ".join(x for x in (style.negative, shot_neg, char_neg) if x)

    # A panel with nobody in it still inherited the style lock's anatomy
    # clauses, and "no humans" plus "narrow piercing eyes with highly detailed
    # vibrant irises" is a contradiction the model resolves by painting a pair
    # of giant disembodied eyes over the scenery. Empty-cast panels drop the
    # clauses that describe a body; they keep every clause about line, colour
    # and light, which is what actually carries the series look.
    if not panel.characters and not panel.extras:
        style = _scenery_style(style)

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

"""Typed schema for an episode. This is the contract between every stage:
script breakdown emits it, the renderer consumes it, the compositor lays it out.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Shot(str, Enum):
    """Camera distance. Drives framing tokens injected into the prompt."""

    establishing = "establishing"
    wide = "wide"
    full = "full"
    medium = "medium"
    close = "close"
    extreme_close = "extreme_close"
    over_shoulder = "over_shoulder"
    pov = "pov"
    insert = "insert"
    reaction = "reaction"


# Framing per shot type, written in Danbooru tag vocabulary.
#
# Illustrious/NoobAI checkpoints are trained on Danbooru tags, so `scenery,
# no humans, wide shot` steers them far harder than a prose description like
# "extreme wide establishing shot of a vast landscape". Prose framing is the
# main reason these models collapse every panel into a character portrait.
SHOT_TOKENS: dict[Shot, str] = {
    Shot.establishing: "scenery, wide shot, landscape, panorama, extremely distant view",
    Shot.wide: "wide shot, full body, environment visible, distant view",
    Shot.full: "full body, standing, head to toe",
    Shot.medium: "cowboy shot, upper body",
    Shot.close: "portrait, close-up, face focus, head and shoulders",
    Shot.extreme_close: "extreme close-up, eye focus, macro detail",
    Shot.over_shoulder: "over the shoulder, from behind, depth of field, blurry foreground",
    Shot.pov: "pov, first-person view",
    Shot.insert: "object focus, still life, close-up, depth of field",
    Shot.reaction: "upper body, face focus, expressive",
}

# Tags pushed into the NEGATIVE prompt per shot, to fight the portrait bias.
# Without these an "establishing" panel still returns a face filling the frame.
SHOT_NEGATIVES: dict[Shot, str] = {
    Shot.establishing: "portrait, close-up, face focus, upper body, 1boy, 1girl",
    Shot.wide: "portrait, close-up, face focus",
    Shot.full: "portrait, close-up, cropped legs",
    Shot.medium: "full body, scenery",
    Shot.close: "full body, wide shot, scenery",
    Shot.extreme_close: "full body, wide shot, scenery",
    Shot.over_shoulder: "",
    Shot.pov: "",
    Shot.insert: "1boy, 1girl, face, portrait",
    Shot.reaction: "full body, wide shot",
}

Aspect = Literal["tall", "square", "wide", "banner", "full_bleed"]
# Which world a panel is set in. Genre words are strong enough to override
# clothing and architecture, so an isekai has to be able to say "this scene is
# the modern world" per panel. Keys must exist in style.yaml -> registers.
Register = Literal["cultivation", "modern", "neutral"]
BalloonKind = Literal["speech", "thought", "narration", "shout", "whisper", "system"]


class CharacterRef(BaseModel):
    """A character appearing in one panel."""

    id: str = Field(description="Key into the character bible, e.g. 'lin_yao'")
    expression: str = Field(default="neutral", description="e.g. 'cold smirk', 'wide-eyed shock'")
    pose: str = Field(default="", description="e.g. 'arms crossed, standing on cliff edge'")
    gaze: str = Field(default="", description="e.g. 'looking at viewer', 'eyes downcast'")
    # Overrides the bible default when a scene changes their outfit.
    outfit: str | None = None


class Balloon(BaseModel):
    """One lettered text element."""

    kind: BalloonKind = "speech"
    speaker: str | None = None
    text: str
    # 0.0-1.0 normalised anchor inside the panel. None = auto-place into the
    # lowest-detail region (see letter.balloon.auto_anchor).
    x: float | None = None
    y: float | None = None


class SFX(BaseModel):
    """Onomatopoeia / impact text drawn as art rather than in a balloon."""

    text: str
    x: float = 0.5
    y: float = 0.5
    scale: float = 1.0
    rotation: float = 0.0


class Panel(BaseModel):
    id: str
    beat: int = Field(description="Index of the story beat this panel belongs to")
    shot: Shot = Shot.medium
    aspect: Aspect = "tall"

    characters: list[CharacterRef] = Field(default_factory=list)
    action: str = Field(description="What is physically happening, visual terms only")
    setting: str = Field(default="", description="Location and background")
    lighting: str = Field(default="", description="Light source, colour, time of day")
    camera: str = Field(default="", description="e.g. 'low angle', 'dutch tilt', 'bird's eye'")
    world: Register = Field(
        default="cultivation",
        description="Which world this panel is set in. Use 'modern' for "
                    "real-world / pre-transmigration scenes, otherwise the genre "
                    "clause puts modern characters in cultivation robes.",
    )
    fx: list[str] = Field(default_factory=list, description="e.g. 'golden qi aura', 'sword glow'")

    dialogue: list[Balloon] = Field(default_factory=list)
    sfx: list[SFX] = Field(default_factory=list)

    # Fixed at breakdown time so a re-render of the episode is reproducible.
    seed: int | None = None
    # Set by the QA gate after a reroll, for audit.
    reroll_count: int = 0

    def content_prompt(self, bible: dict[str, "Character"]) -> str:
        """Panel-specific half of the prompt. The style lock supplies the rest."""
        parts: list[str] = [SHOT_TOKENS[self.shot]]

        if self.camera:
            parts.append(self.camera)

        n = len(self.characters)
        if n == 1:
            parts.append("1girl" if bible.get(self.characters[0].id, _UNKNOWN).sex == "female" else "1boy")
        elif n > 1:
            parts.append(f"{n} people")
        else:
            # An empty cast means a scenery or object panel. Anime checkpoints
            # will happily invent a character anyway, so say so explicitly --
            # `no humans` is a strong Danbooru tag.
            parts.append("no humans")

        for ref in self.characters:
            char = bible.get(ref.id)
            if char is None:
                continue
            parts.append(char.appearance_prompt(ref))

        if self.action:
            parts.append(self.action)
        if self.setting:
            parts.append(self.setting)
        if self.lighting:
            parts.append(self.lighting)
        parts.extend(self.fx)

        return ", ".join(p.strip() for p in parts if p and p.strip())


class Character(BaseModel):
    """An entry in the character bible. `appearance` is the identity lock:
    it is emitted into every prompt the character appears in, unchanged.
    """

    id: str
    name: str
    sex: Literal["male", "female", "other"] = "other"

    # Immutable physical description. Written once, never varied per panel.
    appearance: str
    default_outfit: str = ""

    # Trained identity LoRA. This is what actually holds a face together
    # across hundreds of panels; the text description alone will not.
    lora: str | None = None
    lora_weight: float = 0.8
    trigger: str | None = Field(default=None, description="LoRA activation token")

    # Path to the turnaround sheet used as the QA drift reference.
    sheet_dir: str | None = None

    def appearance_prompt(self, ref: CharacterRef | None = None) -> str:
        parts: list[str] = []
        if self.trigger:
            parts.append(self.trigger)
        parts.append(self.appearance)
        outfit = (ref.outfit if ref and ref.outfit else self.default_outfit)
        if outfit:
            parts.append(outfit)
        if ref:
            if ref.expression:
                parts.append(f"{ref.expression} expression")
            if ref.pose:
                parts.append(ref.pose)
            if ref.gaze:
                parts.append(ref.gaze)
        return ", ".join(p for p in parts if p)


_UNKNOWN = Character(id="_unknown", name="Unknown", appearance="")


class Episode(BaseModel):
    series: str
    number: int
    title: str = ""
    panels: list[Panel] = Field(default_factory=list)

    def by_beat(self) -> dict[int, list[Panel]]:
        out: dict[int, list[Panel]] = {}
        for p in self.panels:
            out.setdefault(p.beat, []).append(p)
        return out

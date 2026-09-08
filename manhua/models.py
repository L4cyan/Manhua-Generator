"""Typed schema for an episode. This is the contract between every stage:
script breakdown emits it, the renderer consumes it, the compositor lays it out.
"""
from __future__ import annotations

import re
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
    Shot.establishing: "(extreme wide shot:1.4), scenery, landscape, panorama, (distant view:1.3), tiny figures, environment fills the frame",
    Shot.wide: "(wide shot:1.4), (full body:1.3), whole figure visible head to toe, environment visible around them, camera far back",
    Shot.full: "(full body:1.4), head to toe, whole figure in frame, feet visible, standing, camera far enough back to see all of them",
    Shot.medium: "cowboy shot, upper body",
    Shot.close: "portrait, close-up, face focus, head and shoulders",
    Shot.extreme_close: "extreme close-up, eye focus, macro detail",
    Shot.over_shoulder: "over the shoulder, from behind, depth of field, blurry foreground",
    Shot.pov: "pov, first-person view",
    Shot.insert: "object focus, still life, close-up, depth of field",
    Shot.reaction: "upper body, face focus, expressive",
}

# How big the character should be IN THE FRAME, said in plain words.
#
# `Shot.wide` is a label; it tells the model nothing about scale. Anima's text
# encoder is a language model, not CLIP, so it reads a sentence about how much
# of the picture a person occupies far better than it reads a tag. Stating the
# fraction explicitly is the difference between "wide shot" (which came back as
# a portrait) and "the figure occupies about half the frame height".
#
# Say what the environment is doing too: "the environment fills most of the
# image" gives the model something to draw INSTEAD of a face.
SHOT_SCALE: dict[Shot, str] = {
    Shot.establishing:
        "the character is a tiny distant figure occupying roughly one tenth of the "
        "frame height, the environment fills almost the entire image, seen from very "
        "far away",
    Shot.wide:
        "the whole figure is visible from head to toe and occupies about half the "
        "frame height, with the environment visible all around them, camera well back",
    Shot.full:
        "the figure stands head to toe filling about four fifths of the frame height, "
        "with clear space above the head and below the feet",
    Shot.medium:
        "the figure is framed from the knees upward and fills the frame vertically, "
        "the background still clearly visible behind them",
    Shot.over_shoulder:
        "seen past the shoulder of a foreground figure, the subject at conversational "
        "distance, framed from the waist up",
    Shot.pov:
        "a first-person view of the scene, no subject in the near foreground",
    Shot.close:
        "the head and shoulders fill the frame",
    Shot.extreme_close:
        "a single facial feature fills the entire frame",
    Shot.reaction:
        "head and shoulders fill the frame, the expression is the subject",
    Shot.insert:
        "a single object fills the frame, no person present",
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

# How far the camera is, which decides how much of a character to describe.
#
# This is the single biggest cause of "every panel is a face". An identity lock
# is a FACE description -- jawline, irises, lashes, brows, nose, mouth -- and
# against four words of framing the face wins, so a wide shot comes back as a
# portrait with a tiny figure pasted into it. At thirty metres none of that is
# visible anyway. Far shots get the silhouette: build, hair, wardrobe.
SHOT_DISTANCE: dict[Shot, str] = {
    Shot.establishing: "far",
    Shot.wide: "far",
    Shot.full: "far",
    Shot.medium: "far",
    Shot.over_shoulder: "far",
    Shot.pov: "far",
    Shot.close: "near",
    Shot.extreme_close: "near",
    Shot.reaction: "near",
    Shot.insert: "near",
}

Aspect = Literal["tall", "square", "wide", "banner", "full_bleed"]
# Which world a panel is set in. Genre words are strong enough to override
# clothing and architecture, so an isekai has to be able to say "this scene is
# the modern world" per panel. Keys must exist in style.yaml -> registers.
Register = Literal["cultivation", "modern", "neutral"]
BalloonKind = Literal["speech", "thought", "narration", "shout", "whisper", "system"]


# Words that only exist on a face. A clause containing one of these is dropped
# when the camera is far enough away that it could not be seen.
_FACE_WORDS = (
    "eye", "iris", "lash", "eyebrow", "brow", "jaw", "cheek", "nose", "mouth",
    "lip", "chin", "freckle", "expression", "gaze", "face", "facial",
)


# Clauses worth weighting inside an appearance string. Hair is the single
# strongest recognition cue in this art style and the first thing to drift, so
# it gets the emphasis while eye colour and build stay at their natural weight.
_HAIR_WORDS = ("hair", "bangs", "fringe", "ponytail", "braid", "topknot",
               "sidelock", "sideburn", "hairline", "curls", "queue", "bun")


def _emphasise(text: str, weight: float, hair_only: bool = False) -> str:
    """Wrap text in ComfyUI attention weighting.

    ComfyUI reads `(text:1.2)`, so any literal parenthesis already in the
    description has to be escaped or it silently becomes a weight group and
    eats the rest of the clause.
    """
    if weight == 1.0 or not text.strip():
        return text

    def wrap(s: str) -> str:
        s = s.replace("(", r"\(").replace(")", r"\)")
        return f"({s}:{weight:g})"

    if not hair_only:
        return wrap(text)

    out = []
    for clause in text.split(","):
        c = clause.strip()
        if not c:
            continue
        out.append(wrap(c) if any(w in c.lower() for w in _HAIR_WORDS) else c)
    return ", ".join(out)


def _silhouette(appearance: str) -> str:
    """Keep only what is readable at a distance: build, hair, colouring."""
    kept = [c.strip() for c in appearance.split(",")
            if c.strip() and not any(w in c.lower() for w in _FACE_WORDS)]
    return ", ".join(kept)


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
    # False when the speaker is not visible in this panel -- a voice from
    # offscreen, a god speaking out of a light. Such a balloon must not grow a
    # tail: a tail points at a mouth, and pointing it at whoever IS in frame
    # silently reassigns the line to them. This is exactly how "I've found you,
    # my inheritor" ended up reading as the protagonist's own line.
    in_panel: bool = True
    # Manual tail target, normalised 0-1 within the panel. None means NO tail.
    # Tails are not placed automatically any more: the auto-placer aimed at the
    # busiest region of the panel, which is a decent guess and a bad default --
    # it produced stray triangles pointing at scenery, and it silently
    # attributed lines to whoever happened to be in frame. The editor sets
    # these by clicking.
    tail_x: float | None = None
    tail_y: float | None = None
    text: str
    # 0.0-1.0 normalised anchor inside the panel. None = auto-place into the
    # lowest-detail region (see letter.balloon.auto_anchor).
    x: float | None = None
    y: float | None = None
    # How wide to wrap this one, as a fraction of the panel. None = the style
    # default. A balloon is always sized from its text, so this is the handle
    # on the SHAPE: the same line reads as a tall column or a wide bar
    # depending on where it wraps, and only a person can say which suits the
    # art underneath it.
    width: float | None = None


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
    # Vertical-scroll pacing. `pause` sizes the gap BEFORE this panel
    # (tight | normal | beat | scene | cliff) and `inset` pulls it in from
    # the canvas edges. Together they are what stop a chapter reading as a
    # stack of identical rectangles.
    pause: str = "normal"
    # Overrides SHOT_SCALE when a panel needs a scale its shot type does not
    # imply -- "he is a speck at the bottom of the frame", say.
    figure_scale: str = ""
    inset: float = 0.0
    # Unnamed people who are not in the bible (a lecture hall, a crowd).
    # Keeps `no humans` off panels that clearly contain people.
    extras: int = 0
    extras_sex: str = "male"
    # How much to draw the strangers. "faceless" is what manhua actually
    # does for mob characters; "detailed" is for the one student who
    # speaks, who needs a face because the reader looks at them.
    extras_style: str = "faceless"   # faceless | silhouette | detailed
    # Set by the QA gate after a reroll, for audit.
    reroll_count: int = 0

    def content_prompt(self, bible: dict[str, "Character"]) -> str:
        """Panel-specific half of the prompt. The style lock supplies the rest."""
        shot_tokens = SHOT_TOKENS[self.shot]
        if self.extras:
            # "scenery" and "extremely distant view" read as an empty place.
            # A lecture hall of students under an establishing shot came back
            # as rows of empty desks every time. Drop the emptiness words and
            # say there is a crowd, or the count tag below is arguing with the
            # framing and the framing wins.
            for empty in ("scenery, ", "extremely distant view", "no humans"):
                shot_tokens = shot_tokens.replace(empty, "")
            shot_tokens = ", ".join(p.strip() for p in shot_tokens.split(",") if p.strip())
            shot_tokens = f"{shot_tokens}, crowd, many people, group of people"
            if self.extras_style == "faceless":
                shot_tokens += (", seen from behind, backs of heads, rows of people, "
                                "a full room of people, background characters, "
                                "faces turned away, indistinct distant faces")
            elif self.extras_style == "silhouette":
                shot_tokens += (", silhouette, dark silhouettes, backlit figures, "
                                "featureless shapes, no facial features")
        parts: list[str] = [shot_tokens]
        # Say the figure scale in plain words, not just the shot tag.
        scale = self.figure_scale or SHOT_SCALE.get(self.shot, "")
        if scale:
            parts.append(scale)

        if self.camera:
            parts.append(self.camera)

        # Composition hints belong here, beside the shot tokens, NOT inside the
        # character's appearance. Editing the identity lock to fix a hairline
        # changed the character's whole face; the same words placed in the
        # framing zone fix it and leave the face alone.
        for ref in self.characters:
            char = bible.get(ref.id)
            if char is not None and char.framing_hint:
                parts.append(char.framing_hint)

        n = len(self.characters)
        if n == 1:
            parts.append("1girl" if bible.get(self.characters[0].id, _UNKNOWN).sex == "female" else "1boy")
        elif n > 1:
            # Danbooru counts by sex: 2boys, 1boy 1girl, 2girls. "2 people" is
            # not a tag the model knows, and each character's appearance string
            # begins with its own "1boy", so a two-hander used to assert one
            # male twice while also claiming two people. The count tags are
            # stripped from the individual descriptions below.
            males = sum(1 for r in self.characters
                        if bible.get(r.id, _UNKNOWN).sex != "female")
            females = n - males
            bits = []
            if males:
                bits.append("1boy" if males == 1 else f"{males}boys")
            if females:
                bits.append("1girl" if females == 1 else f"{females}girls")
            parts.append(", ".join(bits))
        elif self.extras:
            # People who are not in the bible: a lecture hall of students, a
            # crowd in a square. They still need a count tag, and they must NOT
            # get `no humans` -- naming a person in the action while asserting
            # `no humans` is a contradiction the model resolves by rendering
            # something garish and half-formed.
            parts.append("1girl" if self.extras == 1 and self.extras_sex == "female"
                         else "1boy" if self.extras == 1
                         else f"{self.extras} people, crowd")
        else:
            # An empty cast means a scenery or object panel. Anime checkpoints
            # will happily invent a character anyway, so say so explicitly --
            # `no humans` is a strong Danbooru tag.
            parts.append("no humans")

        # Action and setting come BEFORE the character description. A full
        # identity block runs ~80 tokens, and anything after it lands in a
        # later CLIP chunk where it carries much less weight -- which is how
        # "falling backwards through a white void" turned into a man standing
        # in a street. Identity still lands early because these are short.
        if self.action:
            parts.append(self.action)
        if self.setting:
            parts.append(self.setting)

        for ref in self.characters:
            char = bible.get(ref.id)
            if char is None:
                continue
            desc = char.appearance_prompt(
                ref, world=self.world, distance=SHOT_DISTANCE.get(self.shot, "near"))
            if n > 1:
                # Drop the per-character count tag; the scene-level one above
                # is the only one that should speak for the whole panel.
                desc = re.sub(r"(?:(?<=^)|(?<=, ))(1boy|1girl)(?:, |$)", "", desc)
            parts.append(desc)

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
    # Per-world wardrobe, keyed by Panel.world. An isekai protagonist wears a
    # blazer and glasses in act one and robes after transmigrating, and the
    # panel already knows which world it is in -- so the outfit follows from
    # that rather than needing a manual override on every single panel.
    outfits: dict[str, str] = Field(default_factory=dict)
    # Appended to the negative prompt whenever this character appears.
    # Needed because the style lock pushes traits onto everyone -- e.g.
    # "flowing gravity-defying movement" gives every character long hair,
    # and only the character can say that is wrong for them.
    negative: str = ""

    # Composition hints emitted in the FRAMING zone of the prompt, beside the
    # shot tokens, never inside `appearance`. This is where to put things like
    # "forehead visible, parted bangs": a self-contradicting clause in the
    # identity lock makes the model flip a coin per seed, and rewriting the
    # identity lock to settle it visibly changed the character's face. The
    # same words placed here fix the composition and leave the face alone.
    framing_hint: str = ""

    # Optional explicit description for far shots. Leave empty and one is
    # derived from `appearance` by dropping every clause about the face.
    appearance_far: str = ""

    # Trained identity LoRA. This is what actually holds a face together
    # across hundreds of panels; the text description alone will not.
    lora: str | None = None
    lora_weight: float = 0.8
    trigger: str | None = Field(default=None, description="LoRA activation token")

    # "cast" is a real character the storyboard may reference by id.
    # "extra" is a reusable asset -- a faceless mob, a crowd filler -- that
    # lives in the bible for its art but must never be offered to the
    # storyboard model as someone who can appear in a scene. Leaving the
    # crowd asset in the roster got it written into panel actions as a
    # character standing opposite the protagonist.
    role: str = "cast"

    # Path to the turnaround sheet used as the QA drift reference.
    sheet_dir: str | None = None

    # Attention weight on the two traits that carry recognition and drift the
    # fastest. ComfyUI applies this by pushing each token's embedding away from
    # the empty-prompt embedding (sd1_clip.encode_token_weights), so it works
    # on the Anima/Qwen encoder exactly as it does on CLIP.
    #
    # Deliberately 1.0 by default: turning it on changes the prompt of every
    # panel, and a chapter already rendered and approved must not silently
    # drift because a default moved. New characters get it from the extractor.
    emphasis: float = 1.0

    def appearance_prompt(self, ref: CharacterRef | None = None,
                          world: str = "", include_outfit: bool = True,
                          distance: str = "near") -> str:
        parts: list[str] = []
        if self.trigger:
            parts.append(self.trigger)
        # A far shot gets the silhouette, not the face. Emitting a full facial
        # description into a wide shot is what makes the model ignore the
        # framing and render a portrait instead.
        if distance == "far" and self.appearance_far:
            look = self.appearance_far
        elif distance == "far":
            look = _silhouette(self.appearance)
        else:
            look = self.appearance
        parts.append(_emphasise(look, self.emphasis, hair_only=True))
        # Explicit per-panel override wins, then the world's wardrobe entry,
        # then the default. Callers that supply their own outfit (the sheet
        # generator cycles the whole wardrobe) pass include_outfit=False,
        # otherwise the default is emitted alongside theirs and the character
        # ends up described in two outfits at once.
        if include_outfit:
            outfit = (
                (ref.outfit if ref and ref.outfit else None)
                or self.outfits.get(world)
                or self.default_outfit
            )
            if outfit:
                # The whole outfit is weighted, not just a clause of it: a
                # character in the wrong clothes reads as a different person
                # just as fast as one with the wrong face.
                parts.append(_emphasise(outfit, self.emphasis))
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

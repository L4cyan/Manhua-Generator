"""Prose -> panels, via Claude.

The output is deliberately reviewed as text before anything renders: fixing a
wrong shot type here costs nothing, and 30 seconds of GPU time later.

Schema-validated with `messages.parse`, so a malformed breakdown fails loudly
at the API boundary instead of producing half-broken panels downstream.
"""
from __future__ import annotations

import os
import re
import textwrap

from pydantic import BaseModel, Field

from ..models import Balloon, Character, CharacterRef, Panel, Shot

MODEL = os.environ.get("MANHUA_MODEL", "claude-opus-5")


# ---------- what the model returns ----------
# Kept flat and loose (plain strings, not enums) because constraining the model
# too hard here produces worse panelling than validating afterwards.


class DraftCharacter(BaseModel):
    id: str = Field(description="Character bible id. Must be one of the supplied ids.")
    expression: str = Field(default="", description="Facial expression, e.g. 'cold smirk'")
    pose: str = Field(default="", description="Body pose, e.g. 'arms crossed, back turned'")
    gaze: str = Field(default="", description="e.g. 'looking at viewer', 'eyes downcast'")


class DraftBalloon(BaseModel):
    kind: str = Field(description="speech | thought | narration | shout | whisper | system")
    speaker: str = Field(default="", description="Character id, or empty for narration")
    text: str


class DraftPanel(BaseModel):
    shot: str = Field(description="One of: " + ", ".join(s.value for s in Shot))
    aspect: str = Field(description="tall | square | wide | banner | full_bleed")
    characters: list[DraftCharacter] = Field(default_factory=list)
    action: str = Field(description="What is physically visible. Purely visual, no interiority.")
    setting: str = Field(default="", description="Location and background")
    lighting: str = Field(default="", description="Light source, colour, time of day")
    camera: str = Field(default="", description="e.g. 'low angle', 'dutch tilt'")
    register: str = Field(
        default="cultivation",
        description="'modern' for real-world scenes (a city, a classroom, a phone), "
                    "'cultivation' for the xianxia world, 'neutral' if neither fits.",
    )
    fx: list[str] = Field(default_factory=list, description="Visual effects, e.g. 'golden qi aura'")
    dialogue: list[DraftBalloon] = Field(default_factory=list)


class Breakdown(BaseModel):
    panels: list[DraftPanel]


# ---------- prompt ----------

SYSTEM = textwrap.dedent(
    """
    You are a webtoon storyboard artist adapting prose into a vertical
    long-strip manhua (Chinese cultivation/xianxia comic, full colour).

    Break the passage into panels for a top-to-bottom scroll. Vertical strips
    read differently from printed pages:

    - Pacing is controlled by panel HEIGHT, not by panel count per row. A
      `tall` panel takes longer to scroll past, so it lands harder.
    - Vary the shot rhythm. Never place three consecutive panels at the same
      shot distance; alternate wide/medium/close so the eye keeps moving.
    - Open a new location with an `establishing` or `wide` shot so the reader
      is oriented before you cut in close.
    - Reserve `full_bleed` for one genuine peak per scene - a reveal, an
      impact, a power manifesting. It runs edge to edge with no gutter, so
      overusing it flattens the whole strip.
    - A silent panel is a real beat. Reaction shots with no dialogue give
      weight to the line that follows.

    Rules for the fields:

    - `action` describes only what a reader can SEE. Never write interiority
      ("he realises", "she remembers") - convert it to a visible expression
      or gesture, or move it into a `narration` balloon.
    - NEVER describe a character's permanent appearance in `action`: no hair,
      no eye colour, no skin, no build, no clothing. All of that is supplied
      automatically from the character bible, and a second, looser copy of it
      in the action FIGHTS the bible and wins about half the time, which is
      what makes a character change face between panels.

      Wrong: "He stumbles to the mirror, his reflection showing a young man
      with long black hair, pale skin and sharp eyebrows."
      Right: "He stumbles to the mirror and stares at his reflection, eyes
      wide."

      Expression, pose, gaze and what they are physically doing: yes. What
      they permanently look like: never.
    - Name the character in `action` whenever it is about them. "He stumbles"
      tells the renderer nothing about who is in the panel; "Lin Mo stumbles"
      does.
    - EVERY character in EVERY panel gets an `expression` and a `pose`. This is
      not optional and it is not decoration: a character with neither is drawn
      standing still with a blank face, and a page of those is a page of
      nothing happening.

      `expression` is the face doing something specific. "eyes wide, mouth
      slightly open", "one brow raised, mouth flat", "jaw tight, staring past
      him". Not "calm", not "neutral", not "shocked" on its own.

      `pose` is the whole body. Say what the weight is doing and what the hands
      are doing. "half risen from the floor, one hand braced on the boards, the
      other clutching his head", "leaning back against the doorframe, arms
      folded, one ankle crossed over the other". Not "standing".

      Vary them. If the last panel had someone standing and staring, this one
      does not.

    - `fx` is for visible effects: qi auras, sword glow, shattering stone,
      drifting petals.
    - Keep dialogue short. Long lines need large balloons that cover the art.
      Split a long speech across consecutive panels instead.
    - `system` balloons are cultivation status windows, and the window itself is
      DRAWN FOR YOU from the balloon. Put the message in a `system` balloon and
      NEVER describe the window in `action`. Writing "a blue notification screen
      appears showing a message" makes the artist draw their own screen full of
      unreadable scribble on top of the real one.

    Return 4-8 panels unless the passage clearly needs more.
    """
).strip()


def story_to_panels(
    story: str,
    *,
    bible: dict[str, Character],
    episode_no: int = 1,
    beat_start: int = 0,
    start_index: int = 1,
    target_panels: int = 6,
    provider: str | None = None,
    model: str | None = None,
) -> list[Panel]:
    """Break a passage of prose into validated Panel objects.

    Provider is auto-detected: local Ollama if it is running, otherwise Claude
    if an API key is set. Override with `provider=` or MANHUA_PROVIDER.
    """
    from . import providers

    if provider is None or model is None:
        detected_provider, detected_model = providers.detect()
        provider = provider or detected_provider
        model = model or detected_model

    # Only real cast goes in the roster. Reusable crowd assets live in the
    # bible for their art, but offering one to the storyboard model as a person
    # who can appear in a scene got "student_mob" written into panel actions as
    # a character standing opposite the protagonist.
    cast = {cid: c for cid, c in bible.items() if getattr(c, "role", "cast") == "cast"}
    roster = "\n".join(
        f"- {cid}: {c.name} ({c.sex}) - {c.appearance[:90]}" for cid, c in cast.items()
    ) or "- (no characters defined)"

    user = textwrap.dedent(
        f"""
        Characters available (use these exact ids, and only these):
        {roster}

        Aim for roughly {target_panels} panels.

        Passage:
        ---
        {story.strip()}
        ---
        """
    ).strip()

    if provider == "ollama":
        raw = providers.call_ollama(SYSTEM, user, Breakdown.model_json_schema(), model)
        # A small model can satisfy the schema while still emitting junk values,
        # so validate leniently and let _to_panels sanitise the rest.
        draft = Breakdown.model_validate(raw)
    else:
        draft = providers.call_claude(SYSTEM, user, Breakdown, model)

    return _to_panels(
        draft,
        bible=bible,
        episode_no=episode_no,
        beat=beat_start,
        start_index=start_index,
    )


# A cultivation status window, not a caption. The genre's system messages arrive
# in brackets, and drawn as a plain narration box they read as a stray label --
# "Ding!" in a white rectangle over a robe.
_SYSTEM_STARTS = (
    "ding", "system", "status", "quest", "level up", "warning", "host",
    "ability", "gift", "skill", "notice", "congratulations", "recommended",
    "task", "mission", "detected", "activated",
)

# "Lin Tian sneers, 'You are still alive?'" is a stage direction with a line
# inside it. Lettered whole, the balloon says the character's own name and then
# describes them in the third person.
#
# Two patterns, and double quotes are tried first: a straight apostrophe is far
# more often a contraction than a quote mark, and treating it as one cut
# "You're still alive?" down to "You". The single-quote form only matches when
# the marks are not touching a letter on the outside.
_QUOTED = re.compile(r"[\"“]([^\"”]{2,})[\"”]")
_QUOTED_SINGLE = re.compile(r"(?<!\w)['‘]([^'’]{2,})['’](?!\w)")


def _clean_balloon(text: str, bible: dict[str, Character]) -> tuple[str, str, str | None]:
    """Return (text, kind_override, speaker) for one drafted line."""
    t = re.sub(r"\*{1,3}|_{2,}", "", text).strip()      # markdown from the prose
    speaker: str | None = None

    bracketed = bool(re.fullmatch(r"\[.*\]", t, re.S))
    t = t.strip("[]").strip()

    m = _QUOTED.search(t) or _QUOTED_SINGLE.search(t)
    if m and m.start() <= 1 and m.end() >= len(t) - 1:
        # The whole line is the quote. Comic balloons carry no quote marks.
        t = m.group(1).strip()
    elif m and len(m.group(1)) < len(t) - 2:
        # There is prose around the quote. Keep the quote, and if a character is
        # named in the part outside it, that is who is speaking.
        outside = (t[:m.start()] + " " + t[m.end():]).lower()
        for cid, ch_ in bible.items():
            names = {ch_.name.lower(), cid.replace("_", " ")}
            if any(n and n in outside for n in names):
                speaker = cid
                break
        t = m.group(1).strip()

    kind = ""
    if bracketed or t.lower().startswith(_SYSTEM_STARTS):
        kind = "system"
    return t, kind, speaker


# Permanent traits. If one of these turns up in an `action` it came from the
# model describing the character instead of what they are doing, and it fights
# the identity lock: the bible says "long black hair, centre-parted, low tail",
# the action says "long black hair", and the loose version wins as often as not.
_LOOK_NOUNS = (
    "hair", "skin", "complexion", "eyebrow", "brow", "freckle", "build",
    "physique", "stature", "jawline", "cheekbone", "beard", "moustache",
)
# "his brown eyes" is identity. "eyes wide" is an expression and must survive,
# and so must "a golden eye in the heavens" -- an early version matched any
# colour next to "eye" and deleted the SUBJECT of a panel, leaving "revealing a
# in the heavens". A possessive is required: it is a character trait only when
# it belongs to somebody, and the whole phrase goes, not half of it.
_LOOK_EYES = re.compile(
    r"[,;]?\s*\b(?:his|her|their|its|with)\s+"
    r"(?:dark|pale|light|deep|bright|black|brown|blue|green|gr[ea]y|amber|"
    r"hazel|golden|violet|red|silver)(?:[\w-]+\s+){0,2}eyes\b"
    r"(?:\s+(?:\w+\s+){0,3}?(?=[,.;]|$))?", re.I)

# A status window is LETTERED by us, from a `system` balloon, with real text in
# a real font. Describing one in the action makes the image model draw its own:
# a blue rectangle full of glyph-shaped noise, over the art.
# Only the window's own clause is removed, not the sentence around it: "Lin Mo
# steps back as a blue system window opens in front of him, one hand raised"
# has to keep the stepping back and the raised hand.
_UI_SENTENCE = re.compile(
    r"\b(?:a|an|the)?\s*(?:[\w-]+\s+){0,3}?"
    r"(?:notification|status|system)\s+"
    r"(?:screen|window|panel|box|message|prompt|display|interface)\b"
    r"[^,.;!?]*", re.I)
_UI_TRAILING = re.compile(
    r"[^.!?]*\bthe (?:screen|window|panel)\s+(?:shows|displays|reads|says)\b"
    r"[^.!?]*[.!?]?", re.I)
# What is left after cutting a clause out of the middle of a sentence.
_DANGLE = re.compile(
    r"\s*\b(?:as|while|when|and|with|showing|then|at|to|toward|towards|"
    r"back at|down at|up at|over at|into|onto)\b\s*(?=[,.;!?]|$)", re.I)

_WITH_LOOK = re.compile(
    r"[,;]?\s*\b(?:with|having|showing|revealing)\s+[^.;]*?"
    r"(?:" + "|".join(_LOOK_NOUNS) + r")[^.;]*", re.I)

# Pronouns that mean "the person already established", which is exactly what a
# name matcher cannot see.
_PRONOUN = re.compile(
    r"\b(?:he|him|his|she|her|hers|they|them|their|himself|herself|themselves)\b"
    # "A handsome face is seen in a mirror" is a person. Adjectives are allowed
    # between the article and the noun, or the commonest phrasing never matches.
    r"|\b(?:the|a|an|his|her)\s+(?:[\w-]+\s+){0,2}"
    r"(?:face|figure|reflection|silhouette|hands?)\b",
    re.I)


def strip_appearance(action: str) -> str:
    """Take permanent appearance back out of an action line.

    The prompt forbids it and the model does it anyway, roughly one panel in
    eight. Deleting it deterministically is better than asking again: the
    bible's description is the one that has to win, and it only wins if it is
    the only one in the prompt.
    """
    if not action.strip():
        return action

    text = _UI_TRAILING.sub("", action)
    text = _UI_SENTENCE.sub("", text)
    had_ui = text.strip() != action.strip()
    if had_ui:
        text = _DANGLE.sub("", text)
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"\s+([,.;])", r"\1", text)
    text = _WITH_LOOK.sub("", text)
    text = _LOOK_EYES.sub("", text)

    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        clauses = [c.strip() for c in sentence.split(",")]
        kept = []
        for c in clauses:
            low = c.lower()
            has_look = any(n in low for n in _LOOK_NOUNS)
            # A clause with no verb that names a permanent trait is a
            # description, not an action. "pale skin" goes; "he pushes the hair
            # out of his eyes" stays.
            if has_look and not re.search(
                    r"\b(is|are|was|were|has|have|stands?|sits?|turns?|looks?|"
                    r"pushes?|brushes?|falls?|whips?|streams?|clutch\w*|grips?|"
                    r"holds?|moves?|steps?|walks?|runs?|leans?)\b", low):
                continue
            if c:
                kept.append(c)
        s = ", ".join(kept).strip(" ,")
        if s:
            out.append(s if s.endswith((".", "!", "?")) else s + ".")

    cleaned = " ".join(out)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")
    if cleaned:
        return cleaned + "."
    # Nothing survived. When the whole action was a description of a status
    # window, handing the original back would put the drawn screen straight
    # into the prompt again, so give the panel the only thing that is actually
    # meant to be visible: the person reacting to it. The window itself is
    # lettered from the balloon.
    if had_ui:
        return ("eyes fixed on something bright hanging in the air in front of "
                "them, head tilted slightly back")
    # Otherwise a stray adjective is better than a panel with nothing to draw.
    return action.strip()


def inherit_cast(panels: list[Panel]) -> list[Panel]:
    """Give pronoun-only panels the cast of the panel before them.

    "He stumbles towards the mirror" names nobody, so name matching finds
    nobody, so the panel renders with no identity lock at all and the model
    draws a stranger. Nine panels of forty-one in one chapter. Whoever the
    scene was just about is the right answer, and it is the answer a reader
    assumes too.
    """
    for i, p in enumerate(panels):
        if p.characters or not _PRONOUN.search(p.action):
            continue
        # Backwards first, then forwards, and never across a beat boundary: a
        # new scene may be about a different person entirely. Forwards matters
        # because a scene often OPENS on a face -- "a handsome face is seen in
        # a mirror" -- and there is nothing behind it to inherit from.
        for pool in (reversed(panels[:i]), panels[i + 1:]):
            found = None
            for other in pool:
                if other.beat != p.beat:
                    break
                if other.characters:
                    found = other
                    break
            if found:
                p.characters = [CharacterRef(id=r.id) for r in found.characters]
                break
    return panels


def pace(panels: list[Panel]) -> list[Panel]:
    """Give the strip a rhythm: gap sizes and panel widths.

    Without this every gap is the same 110px and every panel is the same full
    width, which reads as a slideshow of identical rectangles no matter how
    good the art is. Pacing in a vertical scroll IS the panel design: the gap
    before a panel is how long the reader waits for it, and a narrower panel
    reads as a smaller, quieter moment.

    Derived from the panels rather than asked of the model, because it follows
    mechanically from what is already decided (whose beat, what shot, is anyone
    speaking) and a small model given one more field to fill just fills it
    randomly.
    """
    quiet = {Shot.insert, Shot.extreme_close}
    close = {Shot.close, Shot.extreme_close, Shot.reaction}
    wide = {Shot.establishing, Shot.wide}

    for i, p in enumerate(panels):
        prev = panels[i - 1] if i else None

        if prev is None:
            p.pause = "normal"
        elif p.beat != prev.beat:
            p.pause = "scene"                      # somewhere else, or later
        elif prev.aspect == "full_bleed":
            p.pause = "scene"                      # let the big one land first
        elif p.shot in wide and prev.shot in close:
            p.pause = "beat"                       # pulling out from a face
        elif not p.dialogue and prev.dialogue:
            p.pause = "beat"                       # a silent reaction to a line
        elif p.dialogue and prev.dialogue:
            p.pause = "tight"                      # keep a back-and-forth moving
        else:
            p.pause = "normal"

        if p.aspect == "full_bleed":
            p.inset = 0.0
        elif p.shot in quiet:
            p.inset = 0.18                         # a detail, not a scene
        elif p.shot is Shot.reaction and not p.dialogue:
            p.inset = 0.12
        else:
            p.inset = 0.0

    return panels


def _to_panels(
    draft: Breakdown,
    *,
    bible: dict[str, Character],
    episode_no: int,
    beat: int,
    start_index: int,
) -> list[Panel]:
    """Coerce the loose draft into the strict Panel model.

    Unknown shots, aspects and character ids are dropped rather than raised on:
    one hallucinated field should not throw away an otherwise good breakdown.
    """
    valid_aspects = {"tall", "square", "wide", "banner", "full_bleed"}
    panels: list[Panel] = []

    for i, d in enumerate(draft.panels):
        try:
            shot = Shot(d.shot.strip().lower())
        except ValueError:
            shot = Shot.medium

        aspect = d.aspect.strip().lower()
        if aspect not in valid_aspects:
            aspect = "tall"

        chars = [
            CharacterRef(
                id=c.id,
                expression=c.expression,
                pose=c.pose,
                gaze=c.gaze,
            )
            for c in d.characters
            if c.id in bible
        ]

        # Backfill the cast by reading the action text.
        #
        # Small local models reliably WRITE "Ling Yan steps onto the road" and
        # then leave the characters array empty, which strips the identity lock
        # out of the panel and lets the art drift. Name matching is
        # deterministic, instant and does not care how good the model is, so it
        # runs regardless rather than as a fallback.
        named = {r.id for r in chars}
        haystack = f"{d.action} {d.setting}".lower()
        for cid, ch_ in bible.items():
            if cid in named or getattr(ch_, "role", "cast") != "cast":
                continue
            # Match on the full name and the id, plus a first name only when
            # it is distinctive. Taking the first token blindly matched "The"
            # in "The Psionic Immortal" and put her in every panel containing
            # the word "the".
            stop = {"the", "a", "an", "lord", "lady", "master", "elder", "sir"}
            aliases = {cid.replace("_", " "), ch_.name.lower()}
            for tok in ch_.name.lower().split():
                if len(tok) >= 4 and tok not in stop:
                    aliases.add(tok)
            # Whole-word matching, so "tam" does not fire inside "tamper".
            if any(a and re.search(r"\b" + re.escape(a) + r"\b", haystack)
                   for a in aliases):
                chars.append(CharacterRef(id=cid))
                named.add(cid)

        balloons = []
        for b in d.dialogue:
            if not b.text.strip():
                continue
            text, forced, found = _clean_balloon(b.text, bible)
            if not text:
                continue
            kind = b.kind if b.kind in {
                "speech", "thought", "narration", "shout", "whisper", "system"
            } else "speech"
            speaker = b.speaker if b.speaker in bible else found
            balloons.append(Balloon(
                kind=forced or kind,
                speaker=speaker or None,
                # A line whose speaker is not in this panel must not grow a
                # tail: a tail points at a mouth, and pointing it at whoever IS
                # in frame silently reassigns the line to them.
                in_panel=speaker is None or any(r.id == speaker for r in chars),
                text=text,
            ))

        panels.append(
            Panel(
                id=f"ep{episode_no:02d}_p{start_index + i:03d}",
                beat=beat,
                shot=shot,
                aspect=aspect,  # type: ignore[arg-type]
                characters=chars,
                action=strip_appearance(d.action),
                setting=d.setting,
                lighting=d.lighting,
                camera=d.camera,
                world=(d.register if d.register in
                       ("cultivation", "modern", "neutral") else "cultivation"),
                fx=[f for f in d.fx if f.strip()],
                dialogue=balloons,
            )
        )

    return pace(inherit_cast(panels))

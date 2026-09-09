"""Read a chapter and propose the character bible for it.

Without this a new project starts with an empty bible, every panel breaks down
with an empty cast, and nothing holds a character together between panels. It
is the step that has to happen BEFORE the storyboard, because the storyboard
references characters by id.

The output is a proposal, not a commit. Appearance strings are the identity
lock -- emitted verbatim into every prompt the character appears in -- so they
are worth a human glance before hundreds of panels inherit them.
"""
from __future__ import annotations

import json
import re
import textwrap

from pydantic import BaseModel, Field

from ..models import Character
from . import providers

SYSTEM = textwrap.dedent(
    """
    You read a chapter of a web novel and list the characters in it, writing a
    visual description precise enough to draw the same person twice.

    For each character give:

    - `id`: lowercase snake_case, from their name. "Ling Yan" -> ling_yan.
    - `name`: as written in the prose.
    - `sex`: "male" or "female".
    - `role`: "main" for the viewpoint character, "side" for a named character
      who speaks or acts, "extra" for unnamed background people (a crowd, "the
      guards"). Give extras a generic id like `guard` or `student`.
    - `appearance`: PERMANENT physical traits only. Never clothing, never mood,
      never anything that changes between scenes.

      THE HAIR IS THE MOST IMPORTANT PART OF THIS FIELD. It is what a reader
      recognises a character by at a glance, and it is the first thing to go
      wrong. Give all five of: its length, its colour, its texture, how it is
      worn or tied, and what the front does. Five words is not enough; a line
      is about right.

      Then age, build, height, skin tone, eye colour and shape, and one
      distinguishing mark that is theirs alone.

      Write it as ONE flowing list of comma-separated clauses, the way an
      artist briefs another artist. Never use "Label: value" form: a form is a
      thing to fill in, and this is a thing to draw from.

    - `outfit`: BUILD A COSTUME, do not summarise one. Work outward: the inner
      garment, then the outer, then the belt or sash, then the footwear, then
      one accessory. For each, name the garment, its cut, its colour, and how
      it fastens. Add one detail nobody else in the story has.

      A vague outfit is the single biggest cause of a character appearing in a
      suit in one panel and a robe in the next.
    - `notes`: one line on who they are, for the human reading this.

    Rules:

    - If the prose does not describe someone, INVENT a specific appearance that
      fits the setting and their role. Vagueness is worse than invention: the
      image model will invent anyway, and differently every time.
    - NEVER hedge. Do not write "not mentioned", "unknown", "likely wears",
      "presumably", "typical of the setting", or any phrase describing what the
      text does or does not say. You are writing a costume department's notes,
      not a report on the source. If you do not know, DECIDE, and write the
      decision as plain fact.
    - EVERY CHARACTER MUST LOOK DIFFERENT FROM EVERY OTHER CHARACTER. No two
      of them share a hair colour, a hairstyle, a garment, a colour scheme or
      an age. Write each description while looking at the ones you have already
      written, and if two would read alike on the page, change one of them. A
      cast where everybody wears the same robe is a failed answer, and it is
      the single most common way this goes wrong.
    - Rank matters and shows. A clan heir, a servant and a beaten outcast do
      not wear the same cloth: silk and embroidery at the top, coarse undyed
      hemp at the bottom.
    - Match the setting. A cultivation story gets robes, not blazers.
    - Do not list the narrator unless they are a character in the scene.
    - Do not invent characters who are not in the passage.
    - Characters listed as ALREADY ESTABLISHED keep their id. Copy their
      appearance and outfit back to me unchanged: they have already been drawn
      that way, and changing the words changes the face.
    """
).strip()

# Phrases that mean the model described the SOURCE instead of the character.
# Any of these in an identity lock and the image model gets a sentence about
# what a text file does not say, which it then tries to draw.
_HEDGES = (
    "not mentioned", "not specified", "not described", "no specific",
    "unspecified", "unknown", "likely", "presumably", "probably", "typical of",
    "given the setting", "not stated", "n/a", "none given", "no description",
    "assumed", "implied", "unclear",
)

# Fallbacks when the model hedges anyway. Varied so an invented cast does not
# come out in one uniform, and keyed by id so a character keeps the same one.
_INVENTED_OUTFITS = [
    "long cross-collar hanfu robe in muted indigo, wide dark sash knotted at the "
    "waist, plain cloth boots",
    "layered grey travelling robe with a high collar, leather belt with a brass "
    "buckle, worn black boots",
    "dark green short-sleeved tunic over narrow trousers, cloth wrappings at the "
    "forearms, straw sandals",
    "pale cream inner robe under a slate outer coat, fastened with three cord "
    "toggles down the chest, soft boots",
    "russet padded jacket closed with a diagonal fastening, black trousers "
    "tucked into calf-high boots",
]
_INVENTED_LOOKS = [
    "early twenties, lean build, straight black hair tied back at the nape, "
    "dark brown eyes, pale skin",
    "late twenties, broad shouldered, short cropped black hair, heavy brows, "
    "brown eyes, tanned skin",
    "mid teens, slight build, shoulder-length dark hair worn loose, wide black "
    "eyes, fair skin",
    "middle aged, wiry, greying hair pulled into a short topknot, narrow eyes, "
    "weathered skin, a scar through one eyebrow",
    "early thirties, average build, chin-length dark hair parted at the side, "
    "grey eyes, olive skin",
]


def _hedged(text: str) -> bool:
    t = (text or "").strip().lower()
    return not t or any(h in t for h in _HEDGES)


def _invent(cid: str, kind: str) -> str:
    """A specific description, chosen deterministically from the id."""
    pool = _INVENTED_LOOKS if kind == "appearance" else _INVENTED_OUTFITS
    return pool[sum(map(ord, cid)) % len(pool)]


# "Age: 16. Build: Slim. Eye colour: Dark brown." is a form, and a form is a
# thing to fill in rather than a thing to draw from. It also breaks the two
# places that read an appearance as a list of clauses: the far-shot silhouette,
# which drops clauses about the face, and the hair weighting.
_LABELS: tuple[tuple[str, str], ...] = (
    (r"age", "{} years old"),
    (r"build|physique", "{} build"),
    (r"height", "{} tall"),
    (r"eye colou?r", "{} eyes"),
    (r"eye shape", "{} eyes"),
    (r"hair colou?r", "{} hair"),
    # Non-capturing throughout: a capture group here shifts the value out of
    # group 1, and "One distinguishing mark: a small scar" came back as "one".
    (r"skin(?: tone)?", "{} skin"),
    (r"(?:one )?distinguishing (?:mark|feature)s?", "{}"),
    (r"gender|sex", ""),
)


def tidy_labels(text: str) -> str:
    """Turn "Label: value" writing into ordinary comma-separated clauses."""
    if ":" not in text:
        return text
    out = text
    for pattern, shape in _LABELS:
        out = re.sub(
            rf"\b(?:{pattern})\s*:\s*([^.;,]+)[.;,]?",
            lambda m, sh=shape: (sh.format(m.group(1).strip().rstrip(".").lower()) + ", ")
            if sh else "",
            out, flags=re.I)
    # Sentences become clauses, so the whole thing reads as one list.
    out = re.sub(r"\.\s+(?=[a-z0-9])", ", ", out)
    out = re.sub(r"\.\s+(?=[A-Z])", ", ", out)
    out = re.sub(r",\s*,", ", ", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.")
    return out


def _overlap(a: str, b: str) -> float:
    """How much two descriptions share, ignoring filler."""
    stop = {"a", "an", "the", "and", "with", "of", "in", "on", "at", "over",
            "under", "his", "her", "their", "it", "is", "to", "from", "by"}
    wa = {w for w in re.findall(r"[a-z]{3,}", a.lower())} - stop
    wb = {w for w in re.findall(r"[a-z]{3,}", b.lower())} - stop
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def find_clones(chars: list[tuple[Character, dict]], threshold: float = 0.6
                ) -> list[tuple[str, str, str]]:
    """Characters whose descriptions are near-copies of each other.

    The prompt used to carry a worked example of a good costume, and the model
    dressed the entire cast in it: a clan heir, a beaten outcast and a servant
    all in the same charcoal robe with silver cloud scrollwork. The example is
    gone, but a model that copies its first answer onto the rest is a failure
    mode worth detecting rather than trusting a prompt about.
    """
    out = []
    for i, (a, _) in enumerate(chars):
        for b, _ in chars[i + 1:]:
            for field, label in (("appearance", "look"), ("default_outfit", "outfit")):
                if _overlap(getattr(a, field), getattr(b, field)) >= threshold:
                    out.append((a.id, b.id, label))
    return out


def clean_lock(text: str, cid: str, kind: str) -> tuple[str, str]:
    """Strip hedging from an identity lock. Returns (text, flag).

    "white robes, likely a symbol of his status within the clan" is half fact
    and half commentary on the source. Cutting the whole line throws away the
    white robes, so cut at the hedge and keep what came before it. Only when
    nothing concrete survives is a description invented outright.

    The flag is "" if the text was already fine, "trimmed" if a hedge was cut
    off, "thin" if what remained is too little to draw from, or "invented".
    """
    t = tidy_labels((text or "").strip())
    lowered = t.lower()
    cut = min((i for i in (lowered.find(h) for h in _HEDGES) if i != -1), default=-1)
    flag = ""
    if cut != -1:
        head = t[:cut].rstrip()
        # Drop the connective that introduced the hedge, so "white robes, but"
        # does not become "white robes, but".
        head = re.sub(r"[,;:]?\s*\b(but|though|although|and|however|while)\s*$",
                      "", head, flags=re.I)
        t = head.rstrip(" ,;:-")
        flag = "trimmed"

    if not t:
        return _invent(cid, kind), "invented"
    # Two or three words is true but not enough for a model to hold steady on.
    if len(t) < 16 or len(t.split()) < 3:
        return t, "thin"
    return t, flag


class DraftCharacter(BaseModel):
    id: str
    name: str
    sex: str = "male"
    role: str = "side"
    appearance: str = ""
    outfit: str = ""
    notes: str = ""


class CastList(BaseModel):
    characters: list[DraftCharacter] = Field(default_factory=list)


def cast_from_story(story: str, *, setting: str = "",
                    existing: dict[str, Character] | None = None,
                    provider: str | None = None,
                    model: str | None = None) -> list[tuple[Character, dict]]:
    """Propose bible entries for a chapter.

    Returns (Character, info) pairs, where info carries `notes` plus what the
    editor needs to warn about: whether this entry came from an earlier
    chapter, and which fields had to be invented because the prose never said.

    `existing` is the project's current bible. Passing it is what stops chapter
    four giving a character a different face from chapter one: an established
    character keeps their identity lock verbatim rather than being described
    afresh from whatever this chapter happens to mention.
    """
    if provider is None or model is None:
        p, m = providers.detect()
        provider, model = provider or p, model or m

    established = ""
    if existing:
        lines = "\n".join(
            f"- {cid}: {c.name} ({c.sex}) | appearance: {c.appearance} | "
            f"outfit: {c.default_outfit}"
            for cid, c in existing.items()
        )
        established = (
            "\nALREADY ESTABLISHED from earlier chapters. If one of these people "
            "is in this passage, use their exact id and copy their appearance "
            "and outfit back unchanged:\n" + lines + "\n"
        )

    user = textwrap.dedent(
        f"""
        Setting: {setting or "infer it from the passage"}
        {established}
        Passage:
        ---
        {story.strip()}
        ---
        """
    ).strip()

    if provider == "ollama":
        raw = providers.call_ollama(SYSTEM, user, CastList.model_json_schema(), model)
    else:
        raw = providers.call_claude(SYSTEM, user, CastList.model_json_schema(), model)

    data = CastList(**(raw if isinstance(raw, dict) else json.loads(raw)))

    out: list[tuple[Character, dict]] = []
    for d in data.characters:
        cid = "".join(ch if ch.isalnum() else "_" for ch in d.id.lower()).strip("_")
        if not cid:
            continue
        # `role` in the bible is a rendering distinction -- cast get an identity
        # lock and a LoRA, extras are crowd assets -- so main and side collapse
        # to "cast" here while the finer label stays in the notes.
        role = "extra" if d.role.lower().startswith("extra") else "cast"
        appearance, outfit = d.appearance.strip(), d.outfit.strip()
        invented: list[str] = []

        prior = (existing or {}).get(cid)
        if prior:
            # An established character is not re-described. The words in the
            # bible are what has already been drawn.
            appearance = prior.appearance
            outfit = prior.default_outfit
            role = getattr(prior, "role", role)
        else:
            # The prompt forbids hedging, but a small model does it anyway, and
            # "no specific outfit mentioned, but he likely wears traditional
            # clothing" is not something an image model can draw. Decide instead.
            appearance, fa = clean_lock(appearance, cid, "appearance")
            outfit, fo = clean_lock(outfit, cid, "outfit")
            if fa:
                invented.append("appearance")
            if fo:
                invented.append("outfit")

        char = Character(
            id=cid,
            name=d.name or cid,
            sex="female" if d.sex.lower().startswith("f") else "male",
            appearance=appearance,
            default_outfit=outfit,
            role=role,
            sheet_dir=None,
            # Weight the hair and the outfit in every prompt this character
            # appears in. They are what recognition rests on and what drifts
            # first, and the emphasis costs nothing at render time.
            emphasis=1.0 if role == "extra" else 1.2,
        )
        out.append((char, {
            "notes": f"[{d.role}] {d.notes}".strip(),
            "reused": bool(prior),
            "invented": invented,
        }))
    return out

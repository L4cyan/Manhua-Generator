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
    - `appearance`: PERMANENT physical traits only. Age, build, height, hair
      colour AND cut AND length, skin tone, eye colour and shape, and any
      distinguishing mark. Be specific: "short cropped black hair, unkempt" not
      "dark hair". Never include clothing here, never include mood or emotion,
      never include anything that changes between scenes.
    - `outfit`: what they wear, in the same detail. NAME THE GARMENT, ITS CUT,
      ITS COLOUR AND ITS FASTENING: "dust-coloured hanfu travelling robe, long
      sleeved cross-collar, wide dark sash tied at the waist, worn cloth boots".
      A vague outfit is the single biggest cause of a character appearing in a
      suit in one panel and a robe in the next.
    - `notes`: one line on who they are, for the human reading this.

    Rules:

    - If the prose does not describe someone, INVENT a specific appearance that
      fits the setting and their role. Vagueness is worse than invention: the
      image model will invent anyway, and differently every time.
    - Match the setting. A cultivation story gets robes, not blazers.
    - Do not list the narrator unless they are a character in the scene.
    - Do not invent characters who are not in the passage.
    """
).strip()


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


def cast_from_story(story: str, *, setting: str = "", provider: str | None = None,
                    model: str | None = None) -> list[tuple[Character, str]]:
    """Propose bible entries for a chapter. Returns (Character, notes) pairs."""
    if provider is None or model is None:
        p, m = providers.detect()
        provider, model = provider or p, model or m

    user = textwrap.dedent(
        f"""
        Setting: {setting or "infer it from the passage"}

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

    out: list[tuple[Character, str]] = []
    for d in data.characters:
        cid = "".join(ch if ch.isalnum() else "_" for ch in d.id.lower()).strip("_")
        if not cid:
            continue
        # `role` in the bible is a rendering distinction -- cast get an identity
        # lock and a LoRA, extras are crowd assets -- so main and side collapse
        # to "cast" here while the finer label stays in the notes.
        role = "extra" if d.role.lower().startswith("extra") else "cast"
        char = Character(
            id=cid,
            name=d.name or cid,
            sex="female" if d.sex.lower().startswith("f") else "male",
            appearance=d.appearance.strip(),
            default_outfit=d.outfit.strip(),
            role=role,
            sheet_dir=None,
        )
        out.append((char, f"[{d.role}] {d.notes}".strip()))
    return out

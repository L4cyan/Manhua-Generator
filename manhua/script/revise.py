"""Rewrite one panel from a plain-language note.

The field editor is exact and slow. Most edits are one sentence -- "make it
night", "pull the camera back", "he should look afraid" -- and a small model
turns that into the two or three fields it actually touches perfectly well.

Deliberately narrow: it may only change what a panel LOOKS like. It cannot add
or remove characters, rewrite dialogue, or renumber anything, because those are
decisions with consequences elsewhere in the chapter and belong to a person.
"""
from __future__ import annotations

import textwrap

from pydantic import BaseModel, Field

from ..models import Character, Panel, Shot
from . import providers

SYSTEM = textwrap.dedent(
    """
    You edit a single storyboard panel of a comic. You are given the panel's
    current fields and one instruction from the artist. Return the panel's
    fields with the instruction applied.

    Rules:

    - Change ONLY what the instruction asks for. Every other field comes back
      exactly as it was given, character for character. Do not tidy, improve or
      rephrase anything you were not asked to touch.
    - `action` is what is physically visible. Never interiority ("he realises"),
      never permanent appearance (hair, eyes, build, clothing) -- that comes
      from the character bible and repeating it makes the art drift.
    - `expression`, `pose` and `gaze` describe the characters listed, in the
      same order. Leave a character's entry alone unless the instruction is
      about them.
    - `camera` is the angle and framing: "low angle", "over the shoulder",
      "bird's eye", "dutch tilt".
    - `lighting` is the light source, its colour and the time of day.
    - `fx` is visible effects only: qi auras, sword glow, drifting petals.
    - If the instruction is about how close the shot is, change `shot`.

    Return every field, changed or not.
    """
).strip()


class CharBit(BaseModel):
    id: str
    expression: str = ""
    pose: str = ""
    gaze: str = ""


class Revised(BaseModel):
    shot: str = Field(description="One of: " + ", ".join(s.value for s in Shot))
    action: str
    setting: str = ""
    lighting: str = ""
    camera: str = ""
    fx: list[str] = Field(default_factory=list)
    characters: list[CharBit] = Field(default_factory=list)


def revise(panel: Panel, instruction: str, *, bible: dict[str, Character],
           provider: str | None = None, model: str | None = None) -> dict:
    """Apply `instruction` to `panel` in place. Returns the new field values."""
    if provider is None or model is None:
        p, m = providers.detect()
        provider, model = provider or p, model or m

    who = "\n".join(
        f"- {r.id} ({bible[r.id].name if r.id in bible else r.id}): "
        f"expression={r.expression!r} pose={r.pose!r} gaze={r.gaze!r}"
        for r in panel.characters
    ) or "- (nobody in this panel)"

    user = textwrap.dedent(
        f"""
        Current panel:
          shot: {panel.shot.value}
          action: {panel.action}
          setting: {panel.setting}
          lighting: {panel.lighting}
          camera: {panel.camera}
          fx: {", ".join(panel.fx) or "(none)"}

        Characters in it (keep exactly these, in this order):
        {who}

        Instruction from the artist:
        {instruction.strip()}
        """
    ).strip()

    if provider == "ollama":
        raw = providers.call_ollama(SYSTEM, user, Revised.model_json_schema(), model)
        out = Revised.model_validate(raw)
    else:
        out = providers.call_claude(SYSTEM, user, Revised, model)

    try:
        panel.shot = Shot(out.shot.strip().lower())
    except ValueError:
        pass                                    # a bad shot name is not worth losing the edit
    panel.action = out.action.strip() or panel.action
    panel.setting = out.setting.strip()
    panel.lighting = out.lighting.strip()
    panel.camera = out.camera.strip()
    panel.fx = [f.strip() for f in out.fx if f.strip()]

    # Characters are matched by id, never added or dropped: the model is not
    # allowed to decide who is in the scene.
    by_id = {c.id: c for c in out.characters}
    for ref in panel.characters:
        got = by_id.get(ref.id)
        if not got:
            continue
        ref.expression = got.expression.strip() or ref.expression
        ref.pose = got.pose.strip() or ref.pose
        ref.gaze = got.gaze.strip() or ref.gaze

    return {
        "shot": panel.shot.value, "action": panel.action, "setting": panel.setting,
        "lighting": panel.lighting, "camera": panel.camera, "fx": panel.fx,
    }

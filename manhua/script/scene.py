"""Second pass: fill in what the storyboard left blank.

A small model asked to do everything at once does the easy fields and drops the
rest. A real breakdown came back with the action written and `setting`,
`lighting` and `camera` all empty on every panel, which is the worst possible
outcome: an empty setting does not mean "no background", it means the image
model invents one, differently, for every panel in the scene.

So the work is split. The breakdown decides WHAT HAPPENS. This decides WHERE,
and it decides it ONCE PER SCENE rather than once per panel, which is what
makes ten consecutive panels share a room instead of drifting through ten
rooms that happen to have similar descriptions.
"""
from __future__ import annotations

import textwrap

from pydantic import BaseModel, Field

from ..models import Panel, Shot
from . import providers

SYSTEM = textwrap.dedent(
    """
    You are the location designer for a comic. You are given a scene of prose
    and the panels already broken out of it. Describe the ONE place this scene
    happens in, so that every panel can be drawn in the same place.

    - `location`: what the camera sees behind the characters. Give the kind of
      place, then the material it is built of, then two or three fixed things
      that are physically in it. Roughly twenty to forty words.
    - `lighting`: the source of the light, its colour, the direction it comes
      from, and the time of day.
    - `weather`: rain, snow, wind, haze, or "clear". One or two words.
    - `props`: three to six things visible somewhere in this scene, each a
      short noun phrase. These get spread across the panels so the background
      has something in it other than a wall.

    Rules:

    - READ THE PASSAGE AND USE WHAT IT SAYS. Your `location` must contain at
      least two things that are actually named in the prose above. If it
      mentions lanterns, beams, a courtyard, a mirror, those are the things to
      build the set out of. A description that could belong to any scene in any
      chapter is a failed answer.
    - Be concrete. Every vague word is a decision handed to a machine that will
      make it differently every time.
    - Where the prose is silent, DECIDE. Never write "unspecified" or "not
      mentioned": you are dressing a set, not reporting on a script.
    - Describe the PLACE, never the people in it.
    """
).strip()


class SceneBrief(BaseModel):
    location: str = ""
    lighting: str = ""
    weather: str = ""
    props: list[str] = Field(default_factory=list)


# Camera angles, cycled so consecutive panels are not all shot from eye level.
# A storyboard that leaves `camera` empty gets a flat, staged scene where every
# panel is the same height off the ground.
_ANGLES: dict[str, list[str]] = {
    "far": [
        "eye level, camera well back",
        "low angle looking up, sky filling the top of the frame",
        "high angle looking down over the scene",
        "eye level, slight dutch tilt",
    ],
    "near": [
        "eye level, straight on",
        "slightly low angle, looking up at the face",
        "three-quarter view, slightly high angle",
        "over the shoulder, shallow depth of field",
    ],
}

_NEAR = {Shot.close, Shot.extreme_close, Shot.reaction, Shot.insert}


def brief_for(prose: str, panels: list[Panel], *, provider: str | None = None,
              model: str | None = None) -> SceneBrief:
    """Ask for one location description covering a whole scene."""
    if provider is None or model is None:
        p, m = providers.detect()
        provider, model = provider or p, model or m

    beats = "\n".join(f"- {p.action}" for p in panels if p.action)[:1500]
    said = "\n".join(f'- "{b.text}"' for p in panels for b in p.dialogue)[:600]
    user = textwrap.dedent(
        f"""
        Panels in this scene:
        {beats or "- (none)"}

        Lines spoken:
        {said or "- (none)"}

        The prose:
        ---
        {prose.strip()[:4000]}
        ---
        """
    ).strip()

    if provider == "ollama":
        raw = providers.call_ollama(SYSTEM, user, SceneBrief.model_json_schema(), model)
        return SceneBrief.model_validate(raw)
    return providers.call_claude(SYSTEM, user, SceneBrief, model)


def apply_brief(panels: list[Panel], brief: SceneBrief) -> list[Panel]:
    """Fill the blanks, without overwriting anything the storyboard decided."""
    place = ", ".join(x for x in (brief.location.strip(),
                                  brief.weather.strip().lower()) if x
                      and x.lower() not in ("clear", "none", "n/a"))
    props = [p.strip() for p in brief.props if p.strip()]

    for i, p in enumerate(panels):
        if not p.setting.strip() and place:
            # One prop per panel, rotating, so the background varies inside a
            # consistent location instead of being the same wall every time.
            extra = props[i % len(props)] if props else ""
            p.setting = f"{place}, {extra}" if extra else place
        if not p.lighting.strip() and brief.lighting.strip():
            p.lighting = brief.lighting.strip()
        if not p.camera.strip():
            pool = _ANGLES["near" if p.shot in _NEAR else "far"]
            p.camera = pool[i % len(pool)]
    return panels


def dress(prose: str, panels: list[Panel], *, provider: str | None = None,
          model: str | None = None) -> list[Panel]:
    """Fill blank setting/lighting/camera across one scene. Never raises.

    A failure here must not lose a storyboard that already worked, so a bad
    response leaves the panels as they were rather than throwing away the
    minute the breakdown took.
    """
    if not panels or all(p.setting.strip() and p.lighting.strip() and p.camera.strip()
                         for p in panels):
        return panels
    try:
        brief = brief_for(prose, panels, provider=provider, model=model)
    except Exception:
        brief = SceneBrief()
    return apply_brief(panels, brief)

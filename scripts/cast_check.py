"""Cast extraction: reuse established characters, never hedge.

Two failures this guards against. A character described in chapter one coming
back with a different face in chapter four, and the model answering "no
specific outfit mentioned" -- a sentence about the source text, which the image
model then tries to draw.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manhua.models import Character
from manhua.script.cast import _HEDGES, cast_from_story

# Deliberately silent about what anyone looks like.
STORY = """
Bao Qing waited at the well until the lanterns went out.

"You came," said Shen Yu, from the dark under the eaves. "I did not think you
would."

"You asked."

"I asked a lot of people." Shen Yu stepped into the open. "You are the only one
who came."

Behind them, somewhere past the wall, a bell rang once and stopped.
"""

ESTABLISHED = {
    "shen_yu": Character(
        id="shen_yu", name="Shen Yu", sex="female", role="cast",
        appearance="late twenties, tall, silver-white hair in a high tail, "
                   "pale grey eyes, a thin scar across the left cheek",
        default_outfit="black cross-collar robe with silver embroidery at the "
                       "cuffs, wide grey sash, tall riding boots",
    )
}


def main() -> int:
    print("extracting (established: shen_yu)...")
    pairs = cast_from_story(STORY, setting="a cultivation-era walled town",
                            existing=ESTABLISHED)
    if not pairs:
        print("FAILED: nothing came back")
        return 1

    bad = 0
    for c, info in pairs:
        tags = []
        if info["reused"]:
            tags.append("REUSED")
        if info["invented"]:
            tags.append("INVENTED " + "+".join(info["invented"]))
        print(f"\n  {c.id} ({c.name}) {' '.join(tags)}")
        print(f"    appearance: {c.appearance}")
        print(f"    outfit    : {c.default_outfit}")
        for field, text in (("appearance", c.appearance), ("outfit", c.default_outfit)):
            hit = next((h for h in _HEDGES if h in text.lower()), None)
            if hit:
                print(f"    !! {field} still hedges: {hit!r}")
                bad += 1
            if not text.strip():
                print(f"    !! {field} is empty")
                bad += 1

    prior = ESTABLISHED["shen_yu"]
    got = next((c for c, _ in pairs if c.id == "shen_yu"), None)
    if got is None:
        print("\n!! shen_yu was not recognised as the established character")
        bad += 1
    elif got.appearance != prior.appearance or got.default_outfit != prior.default_outfit:
        print("\n!! shen_yu's identity lock was rewritten")
        print(f"   was: {prior.appearance}")
        print(f"   now: {got.appearance}")
        bad += 1
    else:
        print("\n  shen_yu's identity lock survived unchanged")

    print("\nRESULT:", "OK" if not bad else f"FAILED ({bad} problems)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

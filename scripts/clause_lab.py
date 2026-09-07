"""Fast clause lab: try several prompt fixes at once, at 4 steps.

Serial A/Bs at 8 steps cost ~6 minutes each and only answer one question. This
renders every candidate against the SAME seeds at 4 steps (~10s), one row per
variant, so several ideas get answered in one batch. Composition -- where the
hairline sits, whether the fringe covers the forehead -- is settled in the
first steps of denoising, so 4 steps is representative for framing and layout
questions. It is NOT representative for fine rendering, so confirm the winner
at 8 steps before adopting it.

Two rules, both learned the hard way:

  * Variants that touch the identity lock are flagged. Rewriting it fixed the
    fringe and wrecked the face, and a win on the clause under test means
    nothing if the character changed.
  * Score every row against the approved keepers, not against each other.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from manhua.bible.sheet import sheet_requests
from manhua.workspace import Workspace, make_backend

SLOT = "angle_front_close__w0"
SEEDS = [4242, 2104242, 4204242, 6304242, 8404242, 1104242]
STEPS, CFG = 4, 1.0

OLD_BANGS = "curtain bangs parted over the forehead"

# framing_hint: inserted at the tail of the per-slot framing text, before the
#   "1boy" tag. Identity lock untouched.
# appearance_add: appended to the bangs clause inside the identity lock. A
#   two-token addition, not the 20-token rewrite that broke the face.
VARIANTS: list[tuple[str, str, str]] = [
    ("A_baseline",     "framing_hint",   ""),
    ("B_forehead",     "framing_hint",   "forehead, forehead visible, "),
    ("C_partedbangs",  "framing_hint",   "parted bangs, forehead visible, "),
    ("D_sweptaway",    "framing_hint",   "hair swept away from the face, forehead fully visible, "),
    ("E_appendfore",   "appearance_add", "forehead visible"),
    ("F_appendparted", "appearance_add", "parted bangs, forehead visible"),
]

ws = Workspace()
proj = ws.load_project("psionic-cultivation")
proj.reload()
be = make_backend("auto")

base_char = proj.bible["ling_yan"]


def build(kind: str, text: str):
    if kind == "appearance_add":
        char = copy.deepcopy(base_char)
        char.appearance = char.appearance.replace(OLD_BANGS, f"{OLD_BANGS}, {text}")
        assert text in char.appearance
        req = dict(sheet_requests(char, proj.style, base_seed=4242))[SLOT]
    else:
        req = copy.deepcopy(dict(sheet_requests(base_char, proj.style, base_seed=4242))[SLOT])
        if text:
            i = req.positive.index("1boy, ")
            req.positive = req.positive[:i] + text + req.positive[i:]
    req.style = copy.deepcopy(req.style)
    req.style.render.steps = STEPS
    req.style.render.cfg = CFG
    return req


def ident(t: str) -> str:
    return t[t.index("1boy, mature male"):t.index("calm intelligent expression")]


built = [(name, kind, build(kind, text)) for name, kind, text in VARIANTS]
ref = ident(built[0][2].positive)
for name, kind, req in built:
    same = ident(req.positive) == ref
    print(f"{name:16} {kind:15} identity lock unchanged: {'yes' if same else 'NO - face may move'}")
print()

out = Path("out/clause_lab")
out.mkdir(parents=True, exist_ok=True)
paths: dict[tuple[str, int], Path] = {}
t_start = time.time()
for name, _, req in built:
    for seed in SEEDS:
        req.seed = seed
        img = be.render(req)
        p = out / f"{name}_s{seed}.png"
        img.save(p)
        paths[(name, seed)] = p
    print(f"  {name} done  ({time.time() - t_start:.0f}s elapsed)", flush=True)

TW, TH, LAB = 300, 438, 26
sheet = Image.new("RGB", (len(SEEDS) * TW, len(built) * (TH + LAB)), "white")
d = ImageDraw.Draw(sheet)
for row, (name, _, _) in enumerate(built):
    for col, seed in enumerate(SEEDS):
        x, y = col * TW, row * (TH + LAB)
        sheet.paste(Image.open(paths[(name, seed)]).resize((TW, TH)), (x, y))
        d.text((x + 5, y + TH + 6), f"{name}  {seed}", fill="black")
sheet.save("out/clause_lab.png")
print(f"\n{len(built)} variants x {len(SEEDS)} seeds at {STEPS} steps "
      f"in {time.time() - t_start:.0f}s -> out/clause_lab.png")

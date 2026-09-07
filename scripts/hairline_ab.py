"""A/B the hairline hint WITHOUT touching the identity lock.

Rewriting the appearance string to fix the fringe worked on the fringe and
wrecked the face: softer, longer, weaker jaw. The identity lock is what holds
the face, so this test leaves it byte-identical and puts the hint in the
per-slot FRAMING text instead, which already varies from slot to slot and which
every approved render saw a different version of.

Scored on two criteria, not one. A clause that fixes the fringe and moves the
face is a failure -- that is the mistake this script exists to prevent.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from manhua.bible.sheet import sheet_requests
from manhua.workspace import Workspace, make_backend

SLOT = "angle_front_close__w0"
SEEDS = [4242, 2104242, 4204242, 6304242, 8404242, 1104242]

# Goes at the tail of the framing section, immediately before the "1boy" tag
# that starts the identity block. The identity block itself is untouched.
HINT = "forehead visible, parted bangs, hair swept away from the centre of the forehead, "

ws = Workspace()
proj = ws.load_project("psionic-cultivation")
proj.reload()
be = make_backend("auto")

base = dict(sheet_requests(proj.bible["ling_yan"], proj.style, base_seed=4242))[SLOT]
anchor = "1boy, "
assert base.positive.count(anchor) >= 1

import copy

variants = {}
variants["A_plain"] = base
b = copy.deepcopy(base)
i = b.positive.index(anchor)
b.positive = b.positive[:i] + HINT + b.positive[i:]
variants["B_hint"] = b

# Prove the identity block did not move.
def ident(t: str) -> str:
    return t[t.index("1boy, mature male"):t.index("calm intelligent expression")]

assert ident(variants["A_plain"].positive) == ident(variants["B_hint"].positive), \
    "identity block changed -- the whole point of this variant is that it does not"
print("identity block identical in A and B: yes\n")

out = Path("out/hairline_ab")
out.mkdir(parents=True, exist_ok=True)
paths = {}
for label, req in variants.items():
    for seed in SEEDS:
        req.seed = seed
        t0 = time.time()
        img = be.render(req)
        p = out / f"{label}_s{seed}.png"
        img.save(p)
        paths[(label, seed)] = p
        print(f"  {label}  seed {seed}  {time.time() - t0:.0f}s", flush=True)

TW, TH, LAB = 330, 482, 26
sheet = Image.new("RGB", (len(SEEDS) * TW, 2 * (TH + LAB)), "white")
d = ImageDraw.Draw(sheet)
for row, label in enumerate(("A_plain", "B_hint")):
    for col, seed in enumerate(SEEDS):
        x, y = col * TW, row * (TH + LAB)
        sheet.paste(Image.open(paths[(label, seed)]).resize((TW, TH)), (x, y))
        d.text((x + 5, y + TH + 6), f"{label}  {seed}", fill="black")
sheet.save("out/hairline_ab.png")
print("\ntop = no hint, bottom = framing hint -> out/hairline_ab.png")

"""Judge a freshly trained character LoRA, and decide whether to retrain.

Two questions decide that, and one render answers neither:

  1. Does it hold the face?  Same prompt and seed with and without the LoRA,
     across several framings and both wardrobes. If the LoRA column does not
     look more like the reference sheet than the base column, it did not
     learn him.
  2. Is the strength right?  A weight sweep on one prompt. Too low and the
     face drifts back to the base model; too high and it burns in the sheet's
     grey background, its lighting and its poses, and every panel starts
     looking like a reference photo.

Output is two contact sheets: `out/lora_qa_pairs.png` (base vs LoRA) and
`out/lora_qa_weights.png` (the sweep).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from manhua.models import CharacterRef, Panel, Shot
from manhua.render.base import build_request
from manhua.workspace import Workspace, make_backend

TRIGGER = "lingyan1n"

# Deliberately not the sheet's own framings: a LoRA that only works on the
# poses it was trained on is not usable in a chapter.
CASES = [
    ("portrait_modern", Shot.close, "modern", "turning to look at the viewer",
     "university lecture hall", "warm window light"),
    ("full_modern", Shot.full, "modern", "standing with hands in pockets",
     "empty corridor", "cool overhead light"),
    ("angry_robe", Shot.close, "cultivation", "glaring, brows drawn down",
     "night forest", "cold moonlight"),
    ("action_robe", Shot.wide, "cultivation", "walking forward, coat moving",
     "mountain path at dusk", "low warm sun behind him"),
]
WEIGHTS = [0.4, 0.6, 0.8, 1.0]
SEED = 31337


def panel_for(name, shot, world, action, setting, light) -> Panel:
    return Panel(id=name, beat=0, shot=shot, aspect="tall", world=world,
                 characters=[CharacterRef(id="ling_yan",
                                          expression="calm, half-lidded eyes")],
                 action=action, setting=setting, lighting=light, fx=[])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="ling_yan_v1.safetensors")
    ap.add_argument("--weight", type=float, default=0.8)
    args = ap.parse_args()

    ws = Workspace()
    proj = ws.load_project("psionic-cultivation")
    proj.reload()
    be = make_backend("auto")
    out = Path("out/lora_qa")
    out.mkdir(parents=True, exist_ok=True)

    def render(panel, loras, tag):
        req = build_request(panel, proj.style, proj.bible, seed=SEED)
        if loras:
            # The trigger has to appear in the prompt or the LoRA is inert.
            req.positive = f"{TRIGGER}, {req.positive}"
            req.loras = list(loras)
        t0 = time.time()
        img = be.render(req)
        p = out / f"{tag}.png"
        img.save(p)
        print(f"  {tag}  {time.time() - t0:.0f}s", flush=True)
        return p

    # 1. base vs LoRA, same seed
    pairs = []
    for name, shot, world, action, setting, light in CASES:
        panel = panel_for(name, shot, world, action, setting, light)
        a = render(panel, None, f"{name}_base")
        b = render(copy.deepcopy(panel), [(args.lora, args.weight)], f"{name}_lora")
        pairs.append((name, a, b))

    TW, TH, LAB = 330, 482, 26
    sheet = Image.new("RGB", (len(pairs) * TW, 2 * (TH + LAB)), "white")
    d = ImageDraw.Draw(sheet)
    for col, (name, a, b) in enumerate(pairs):
        for row, p in ((0, a), (1, b)):
            sheet.paste(Image.open(p).resize((TW, TH)), (col * TW, row * (TH + LAB)))
            d.text((col * TW + 5, row * (TH + LAB) + TH + 6),
                   f"{'base' if row == 0 else f'LoRA {args.weight}'}  {name}", fill="black")
    sheet.save("out/lora_qa_pairs.png")

    # 2. weight sweep on the portrait
    name, shot, world, action, setting, light = CASES[0]
    panel = panel_for(name, shot, world, action, setting, light)
    sweep = [render(copy.deepcopy(panel), [(args.lora, w)], f"sweep_{w}") for w in WEIGHTS]
    sheet = Image.new("RGB", (len(sweep) * TW, TH + LAB), "white")
    d = ImageDraw.Draw(sheet)
    for i, (w, p) in enumerate(zip(WEIGHTS, sweep)):
        sheet.paste(Image.open(p).resize((TW, TH)), (i * TW, 0))
        d.text((i * TW + 5, TH + 6), f"weight {w}", fill="black")
    sheet.save("out/lora_qa_weights.png")

    print("\nout/lora_qa_pairs.png    top row = base, bottom = LoRA")
    print("out/lora_qa_weights.png  strength sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

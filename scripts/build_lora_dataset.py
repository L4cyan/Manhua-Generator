"""Turn a culled reference sheet into an sd-scripts training dataset.

Captioning rule, and it is the one that decides whether the LoRA is usable:
caption what should stay EDITABLE, omit what should be baked in. So the
captions name the trigger, the framing and the expression, and never describe
his face, hair or build. Anything described in the caption becomes a thing the
model can be talked out of later; anything left unsaid gets absorbed into the
trigger word, which is exactly where a character's identity belongs.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manhua.bible.sheet import ANGLES, EXPRESSIONS

SHEET = Path("workspace/projects/psionic-cultivation/bible/ling_yan")
OUT = Path("workspace/projects/psionic-cultivation/bible/_training/ling_yan")
TRIGGER = "lingyan1n"

# Wardrobe IS captioned: he has two outfits and both must stay promptable.
WARDROBE = {"w0": "wearing a dark grey suit jacket and white dress shirt",
            "w1": "wearing a black high-collared robe"}

angles = dict(ANGLES)
exprs = dict(EXPRESSIONS)


def _plain(s: str) -> str:
    """Strip prompt weight syntax. `(full body:1.4)` is an instruction to the
    sampler, not something a caption should teach the model to expect."""
    return re.sub(r"\((.*?):[\d.]+\)", r"\1", s)


def caption_for(stem: str) -> str:
    name = stem.replace("ling_yan_", "")
    slot, _, w = name.partition("__")
    bits = [TRIGGER]
    if slot.startswith("angle_"):
        bits.append(_plain(angles.get(slot[6:], "portrait")))
    elif slot.startswith("expr_"):
        bits.append("close-up portrait, front view")
        bits.append(_plain(exprs.get(slot[5:], "neutral expression")))
    bits.append(WARDROBE.get(w, ""))
    bits.append("plain grey background")
    return ", ".join(b for b in bits if b)


def main() -> int:
    imgs = sorted(p for p in SHEET.glob("*.png"))
    if not imgs:
        print(f"no images in {SHEET}")
        return 1

    img_dir = OUT / "4_lingyan"     # sd-scripts convention: <repeats>_<name>
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    for p in imgs:
        shutil.copy2(p, img_dir / p.name)
        (img_dir / f"{p.stem}.txt").write_text(caption_for(p.stem), encoding="utf-8")

    toml = f"""# Anima character LoRA dataset for Ling Yan.
# Bucketing is on because the sheet is 832x1216 portrait, not square, and
# squashing a face to 1:1 teaches the LoRA a squashed face.
[general]
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
resolution = 768
batch_size = 1
caption_extension = ".txt"

  [[datasets.subsets]]
  image_dir = "{img_dir.resolve().as_posix()}"
  num_repeats = 4
"""
    cfg = OUT / "dataset.toml"
    cfg.write_text(toml, encoding="utf-8")

    print(f"{len(imgs)} images -> {img_dir}")
    print(f"dataset config -> {cfg}")
    print(f"trigger word: {TRIGGER}\n")
    for p in imgs[:4]:
        print(f"  {p.stem}")
        print(f"    {caption_for(p.stem)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

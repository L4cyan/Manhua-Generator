"""Tile a reference sheet into one image so it can be culled at a glance."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

src = Path(sys.argv[1] if len(sys.argv) > 1
           else "workspace/projects/psionic-cultivation/bible/ling_yan")
imgs = sorted(p for p in src.glob("*.png") if not p.name.startswith("_"))
if not imgs:
    sys.exit("no images")

COLS, TW, TH, LABEL = 6, 260, 380, 22
rows = -(-len(imgs) // COLS)
sheet = Image.new("RGB", (COLS * TW, rows * (TH + LABEL)), "white")
d = ImageDraw.Draw(sheet)

for i, p in enumerate(imgs):
    x, y = (i % COLS) * TW, (i // COLS) * (TH + LABEL)
    sheet.paste(Image.open(p).resize((TW, TH)), (x, y))
    # Strip the character prefix; keep the pose and wardrobe marker.
    tag = p.stem.replace("ling_yan_", "")
    d.rectangle([x, y + TH, x + TW, y + TH + LABEL], fill="white")
    d.text((x + 4, y + TH + 4), tag[:38], fill="black")

out = Path("out/contact_sheet.png")
sheet.save(out)
print(f"{len(imgs)} images -> {out}  ({sheet.width}x{sheet.height})")

"""Letter and compose one panel set into a finished vertical strip.

    python scripts/build_strip.py --panels v2

Reads panels from chapters/001/panels_<set> (or panels/ for the single-pass
set), draws the balloons, stacks them with per-panel pacing gutters and panel
borders, and writes the strip plus upload slices to export_<set>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageFont

Image.MAX_IMAGE_PIXELS = None

from manhua.compose.strip import compose, export
from manhua.letter.balloon import draw_balloon
from manhua.workspace import Workspace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="v2", help="panel set: v1 uses panels/, others panels_<set>/")
    ap.add_argument("--project", default="psionic-cultivation")
    ap.add_argument("--chapter", type=int, default=1)
    args = ap.parse_args()

    ws = Workspace()
    proj = ws.load_project(args.project)
    proj.reload()
    ch = proj.chapter(args.chapter)

    base = proj.dir / "chapters" / f"{args.chapter:03d}"
    pdir = base / ("panels" if args.panels == "v1" else f"panels_{args.panels}")
    if not pdir.is_dir():
        print(f"no panel directory {pdir}")
        return 1

    cfg = proj.lettering
    size = int(cfg.get("font_size", 44))
    try:
        font = ImageFont.truetype(cfg.get("font", ""), size)
    except Exception:
        font = ImageFont.load_default(size)

    items = []
    missing = 0
    for p in ch.panels:
        path = pdir / f"{p.id}.png"
        if not path.exists():
            missing += 1
            continue
        img = Image.open(path).convert("RGB")
        if p.dialogue:
            taken: list = []
            for balloon in p.dialogue:
                draw_balloon(img, balloon, font,
                             padding=int(cfg.get("padding", 26)),
                             line_spacing=float(cfg.get("line_spacing", 1.2)),
                             taken=taken)
        items.append((p.id, img, p.aspect == "full_bleed", p.pause, p.inset))

    if not items:
        print("no rendered panels")
        return 1

    strip, places = compose(items, proj.canvas)
    out_dir = base / f"export_{args.panels}"
    files = export(strip, places, proj.canvas, out_dir, stem=f"ch{args.chapter:03d}")
    print(f"{args.panels}: {strip.width}x{strip.height} · {len(items)} panels · "
          f"{len(files) - 1} slices" + (f" · {missing} missing" if missing else ""))
    print(f"  {files[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

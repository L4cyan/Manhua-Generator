"""Render a chapter under a DIFFERENT style lock, without touching the project.

    python scripts/render_style.py psionic-cultivation 1 watercolour --out v9

The style lock is the whole look of a series, so swapping it on the project
would restyle every chapter already rendered. This loads an alternative lock,
renders into its own panel directory, and leaves the project's own style alone
-- which is what makes "show me this chapter in another style" a safe thing to
try rather than a decision.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manhua.render.base import build_request
from manhua.workspace import Workspace, make_backend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("chapter", type=int)
    ap.add_argument("style", help="style id under workspace/styles/")
    ap.add_argument("--out", default="", help="panel dir suffix; defaults to the style id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--draft", action="store_true",
                    help="fast preview tier: 4 steps at 0.7 scale")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.0)
    args = ap.parse_args()

    ws = Workspace()
    proj = ws.load_project(args.project)
    proj.reload()
    ch = proj.chapter(args.chapter)
    style = ws.load_style(args.style)
    be = make_backend(args.backend)

    suffix = args.out or args.style
    out = proj.dir / "chapters" / f"{args.chapter:03d}" / f"panels_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"style '{style.name}' -> {out}")

    panels = ch.panels[:args.limit] if args.limit else ch.panels
    t0 = time.time()
    # Draft tier. On a 6GB card the fixed cost of paging the model dominates,
    # so RESOLUTION buys more than step count: 8->4 steps saves ~9s, halving the
    # frame saves another ~9s on top. Full quality is for panels that survive
    # the edit, not for the first pass.
    steps = args.steps or (4 if args.draft else 0)
    scale = args.scale or (0.7 if args.draft else 0.0)

    for i, p in enumerate(panels, 1):
        req = build_request(p, style, proj.bible, seed=p.seed or 7)
        if steps or scale:
            import copy as _copy
            req.style = _copy.deepcopy(req.style)
            if steps:
                req.style.render.steps = steps
            if scale:
                # Snap to a multiple of 64; latent sizes must divide cleanly.
                req.width = max(320, int(req.width * scale) // 64 * 64)
                req.height = max(320, int(req.height * scale) // 64 * 64)
        img = be.render(req)
        img.save(out / f"{p.id}.png")
        print(f"  [{i}/{len(panels)}] {p.id} {p.shot.value}", flush=True)

    dt = time.time() - t0
    print(f"\n{len(panels)} panels in {dt/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

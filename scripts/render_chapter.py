"""Render every unrendered panel of a chapter, through the running studio.

    python scripts/render_chapter.py fishshhshs 1

Goes beat by beat so a failure costs one scene rather than the chapter, and
prints a line per panel so an overnight run leaves a readable log.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

HOST = "http://127.0.0.1:7860"


def get(path: str) -> dict:
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.load(r)


def post(path: str) -> dict:
    req = urllib.request.Request(HOST + path, method="POST", data=b"",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> int:
    project = sys.argv[1] if len(sys.argv) > 1 else "fishshhshs"
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    base = f"/api/projects/{project}/chapters/{number}"

    ch = get(base)
    todo = [p for p in ch["panels"] if not p["rendered"] and not p["locked"]]
    beats = sorted({p["beat"] for p in todo})
    print(f"{ch['title'] or 'chapter ' + str(number)}: {len(todo)} panels to render "
          f"across {len(beats)} beats", flush=True)

    t0 = time.time()
    for beat in beats:
        try:
            job = post(f"{base}/beats/{beat}/render")
        except Exception as exc:
            print(f"  beat {beat}: could not start ({exc})", flush=True)
            continue
        seen = None
        while True:
            st = get(f"/api/jobs/{job['id']}")
            if st["detail"] and st["detail"] != seen:
                seen = st["detail"]
                print(f"  [{time.time() - t0:6.0f}s] beat {beat}  {seen}", flush=True)
            if st["state"] == "error":
                print(f"  beat {beat} FAILED: {st['error'][:200]}", flush=True)
                break
            if st["state"] == "ok":
                print(f"  [{time.time() - t0:6.0f}s] beat {beat} done "
                      f"({st['done']}/{st['total']})", flush=True)
                break
            time.sleep(3)

    ch = get(base)
    done = sum(1 for p in ch["panels"] if p["rendered"])
    mins = (time.time() - t0) / 60
    print(f"\n{done}/{len(ch['panels'])} panels rendered in {mins:.1f} min", flush=True)

    if done:
        try:
            r = post(f"{base}/export")
            print(f"exported {r['panels']} panels, {r['height']}px tall -> {r['dir']}",
                  flush=True)
        except Exception as exc:
            print(f"export failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Hammer a chapter with concurrent saves and reads.

This is the regression test for the bug that made panels vanish from the
editor: two threads writing the same shared `chapter.tmp`, so the tail of the
longer write survived past the end of the shorter one and every later read of
the chapter raised JSONDecodeError.
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manhua.models import Balloon, Panel
from manhua.workspace import Workspace

ROOT = Path("out/_conc_ws")
WRITERS, READERS, ROUNDS = 8, 4, 60


def main() -> int:
    shutil.rmtree(ROOT, ignore_errors=True)
    ws = Workspace(root=ROOT)
    proj = ws.create_project("Concurrency")
    ch = proj.chapter(1)

    # Long panels first, so a later short write would leave a visible tail.
    ch.panels = [
        Panel(id=f"c001_p{i:03d}", beat=0, action="x" * 400,
              dialogue=[Balloon(kind="speech", text="y" * 200)])
        for i in range(1, 41)
    ]
    ch.save()

    errors: list[str] = []

    def writer(n: int) -> None:
        for r in range(ROUNDS):
            c = proj.chapter(1)
            # Alternate long and short payloads: identical sizes would hide the
            # exact failure this is testing for.
            keep = 40 if r % 2 else 4
            c.panels = c.panels[:keep]
            try:
                c.save()
            except Exception as exc:                  # pragma: no cover
                errors.append(f"writer {n}: {exc}")

    def reader(n: int) -> None:
        for _ in range(ROUNDS * 3):
            try:
                json.loads((ROOT / "projects" / proj.id / "chapters" / "001" /
                            "chapter.json").read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"reader {n}: {type(exc).__name__}: {exc}")

    threads = ([threading.Thread(target=writer, args=(i,)) for i in range(WRITERS)] +
               [threading.Thread(target=reader, args=(i,)) for i in range(READERS)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stray = list((ROOT / "projects" / proj.id / "chapters" / "001").glob("*.tmp"))
    print(f"{WRITERS} writers x {ROUNDS} saves, {READERS} readers")
    print("errors     :", len(errors))
    for e in errors[:5]:
        print("  ", e)
    print("stray tmp  :", [p.name for p in stray] or "none")
    ok = not errors and not stray
    print("RESULT     :", "OK" if ok else "FAILED")
    shutil.rmtree(ROOT, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

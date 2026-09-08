"""Edit a panel while a render batch is running, and keep the edit.

The reported symptom was "I bugged the editor by modifying scene 1 while it was
generating panels 1-6". Two writers each held their own copy of the whole
chapter and each save wrote all of it, so whichever finished last silently
reverted the other: an edit made during a batch disappeared, and a render
finishing after an edit lost the `rendered` flag.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from manhua.models import Panel
from manhua.studio.server import create_app
from manhua.workspace import Workspace

ROOT = Path(f"out/_edit_race_{os.getpid()}")
N = 12


def main() -> int:
    shutil.rmtree(ROOT, ignore_errors=True)
    c = TestClient(create_app(workspace_root=str(ROOT), backend="mock"))
    pid = c.post("/api/projects", json={"name": "Race"}).json()["id"]

    ws = Workspace(root=ROOT)
    ch = ws.load_project(pid).chapter(1)
    ch.panels = [Panel(id=f"c001_p{i:03d}", beat=0, action=f"original action {i}")
                 for i in range(1, N + 1)]
    ch.save()

    base = f"/api/projects/{pid}/chapters/1"
    job = c.post(f"{base}/beats/0/render").json()

    # Edit panel 1 while the batch is in flight, the way a person would.
    edits, done = [], threading.Event()

    def edit() -> None:
        for i in range(6):
            if done.is_set():
                break
            text = f"EDITED action {i}"
            r = c.patch(f"{base}/panels/c001_p001", json={"action": text})
            if r.status_code == 200:
                edits.append(text)
            time.sleep(0.05)

    t = threading.Thread(target=edit)
    t.start()
    for _ in range(400):
        st = c.get(f"/api/jobs/{job['id']}").json()
        if st["state"] in ("ok", "error"):
            break
        time.sleep(0.05)
    done.set()
    t.join()

    after = c.get(base).json()
    p1 = next(p for p in after["panels"] if p["id"] == "c001_p001")
    rendered = sum(1 for p in after["panels"] if p["rendered"])
    last = edits[-1] if edits else "(no edit landed)"

    print(f"batch          : {st['state']}, {rendered}/{N} panels rendered")
    print(f"edits sent     : {len(edits)}")
    print(f"last edit sent : {last}")
    print(f"panel 1 action : {p1['action']}")
    print(f"panel 1 render : {p1['rendered']}")

    ok = p1["action"] == last and rendered == N and p1["rendered"]
    if p1["action"] != last:
        print("  !! the edit was reverted by the render job's stale copy")
    if rendered != N:
        print(f"  !! {N - rendered} panel(s) lost their rendered flag")
    print("RESULT         :", "OK" if ok else "FAILED")
    shutil.rmtree(ROOT, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

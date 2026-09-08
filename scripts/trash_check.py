"""Delete, restore, expire and empty the bin, through the API.

Deleting a chapter throws away hours of GPU time, so the recovery path gets a
test rather than a hope.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from manhua.studio.server import create_app
from manhua.workspace import TRASH_DAYS

ROOT = Path("out/_trash_ws")


def main() -> int:
    shutil.rmtree(ROOT, ignore_errors=True)
    c = TestClient(create_app(workspace_root=str(ROOT), backend="mock"))

    pid = c.post("/api/projects", json={"name": "Doomed"}).json()["id"]
    c.post(f"/api/projects/{pid}/chapters", json={"title": "Chapter 2"})
    art = ROOT / "projects" / pid / "chapters" / "001" / "panels"
    art.mkdir(parents=True, exist_ok=True)
    (art / "c001_p001.png").write_bytes(b"pretend art")

    # --- delete a chapter, then get it back ---
    assert c.delete(f"/api/projects/{pid}/chapters/1").status_code == 200
    assert not (ROOT / "projects" / pid / "chapters" / "001").exists(), "not moved"
    items = c.get("/api/trash").json()["items"]
    print(f"after chapter delete : {len(items)} in bin "
          f"({items[0]['kind']} '{items[0]['name']}', purges in "
          f"{items[0]['purges_in']} days)")

    assert c.post(f"/api/trash/{items[0]['id']}/restore").status_code == 200
    assert (art / "c001_p001.png").read_bytes() == b"pretend art", "art lost"
    print("restored chapter     : art intact, bin now",
          len(c.get("/api/trash").json()["items"]))

    # --- restoring onto something that is back again must refuse ---
    c.delete(f"/api/projects/{pid}/chapters/1")
    tid = c.get("/api/trash").json()["items"][0]["id"]
    (ROOT / "projects" / pid / "chapters" / "001").mkdir(parents=True)
    r = c.post(f"/api/trash/{tid}/restore")
    print("restore over live    :", r.status_code, r.json().get("detail", "")[:60])
    assert r.status_code == 409
    shutil.rmtree(ROOT / "projects" / pid / "chapters" / "001")
    assert c.post(f"/api/trash/{tid}/restore").status_code == 200

    # --- delete the whole project ---
    assert c.delete(f"/api/projects/{pid}").status_code == 200
    assert c.get("/api/projects").json() == [], "project still listed"
    items = c.get("/api/trash").json()["items"]
    print(f"after project delete : {len(items)} in bin, projects listed 0")
    assert c.post(f"/api/trash/{items[0]['id']}/restore").status_code == 200
    assert len(c.get("/api/projects").json()) == 1, "project did not come back"
    print("restored project     : listed again")

    # --- 30 days later ---
    c.delete(f"/api/projects/{pid}")
    d = next(p for p in (ROOT / ".trash").iterdir() if (p / "meta.json").is_file())
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["deleted"] = (datetime.now(timezone.utc)
                       - timedelta(days=TRASH_DAYS + 1)).isoformat(timespec="seconds")
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    left = c.get("/api/trash").json()["items"]
    print(f"aged past {TRASH_DAYS} days  : {len(left)} left (auto-purged)")
    assert left == []

    # --- empty it by hand ---
    pid2 = c.post("/api/projects", json={"name": "Also Doomed"}).json()["id"]
    c.delete(f"/api/projects/{pid2}")
    r = c.delete("/api/trash").json()
    print("empty bin            : removed", r["removed"],
          "| left", len(c.get("/api/trash").json()["items"]))
    assert c.get("/api/trash").json()["items"] == []

    print("RESULT               : OK")
    shutil.rmtree(ROOT, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

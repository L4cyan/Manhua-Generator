
"""Smoke test the studio API end to end, without a GPU or a model."""
import shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from manhua.studio.server import create_app

root = Path("out/_smoke_ws")
shutil.rmtree(root, ignore_errors=True)
app = create_app(workspace_root=str(root), backend="mock")
c = TestClient(app)

print("styles :", [(s["id"], s["name"], s["is_default"]) for s in c.get("/api/styles").json()])

p = c.post("/api/projects", json={"name": "Smoke Test"}).json()
pid = p["id"]
print("project:", pid, "style:", p["style_id"])

r = c.post(f"/api/projects/{pid}/cast/commit", json={"characters": [
    {"id": "ling_yan", "name": "Ling Yan", "sex": "male", "role": "cast",
     "appearance": "lean angular face, black hair with a violet streak",
     "default_outfit": "dark grey wool coat"},
    {"id": "student", "name": "Student", "sex": "female", "role": "extra",
     "appearance": "generic student"},
]})
print("commit :", r.status_code, r.json())
print("bible  :", (root / "projects" / pid / "characters.yaml").read_text(encoding="utf-8")[-260:])

ch = c.get(f"/api/projects/{pid}/chapters/1").json()
print("cast in chapter:", ch["characters"])

# A panel patched from the editor: characters must survive as models.
from manhua.models import Panel
from manhua.workspace import Workspace
ws = Workspace(root=root)
proj = ws.load_project(pid)
chap = proj.chapter(1)
chap.panels.append(Panel(id="c001_p001", beat=0, action="stands on a road"))
chap.save()

r = c.patch(f"/api/projects/{pid}/chapters/1/panels/c001_p001", json={
    "characters": [{"id": "ling_yan", "expression": "cold", "pose": "arms crossed"}],
    "dialogue": [{"kind": "speech", "text": "Interesting."}]})
print("patch  :", r.status_code)
q = c.get(f"/api/projects/{pid}/chapters/1/panels/c001_p001/prompt").json()
print("prompt :", q["positive"][:170], "...")
assert "violet streak" in q["positive"], "identity lock missing from the prompt"

j = c.post(f"/api/projects/{pid}/chapters/1/panels/c001_p001/render").json()
for _ in range(50):
    st = c.get(f"/api/jobs/{j['id']}").json()
    if st["state"] in ("ok", "error"):
        break
    time.sleep(0.1)
print("render :", st["state"], st["error"] or "")

print("index  :", c.get("/").status_code, len(c.get("/").text), "bytes")
print("OK")

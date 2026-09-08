"""Studio HTTP server: an AI manhua workspace.

Renders are serialised through one worker thread. A 6GB card cannot run two
SDXL jobs at once, and queueing them beats an OOM mid-chapter. The UI polls
job state instead of holding a connection open for 30 seconds.
"""
from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from ..compose.strip import compose, export
from ..models import Panel
from ..workspace import Chapter, Project, Workspace, make_backend

STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    id: str
    kind: str
    total: int
    done: int = 0
    state: str = "queued"          # queued | running | ok | error
    error: str | None = None
    detail: str = ""


class Worker:
    """Single-threaded render queue: one GPU, one job at a time."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue[tuple[Job, Callable[[Job], None]]] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, kind: str, total: int, fn: Callable[[Job], None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, total=total)
        self.jobs[job.id] = job
        self._q.put((job, fn))
        return job

    def _loop(self) -> None:
        while True:
            job, fn = self._q.get()
            job.state = "running"
            try:
                fn(job)
                job.state = "ok"
            except Exception as exc:
                job.state, job.error = "error", str(exc)[:500]
            finally:
                self._q.task_done()


# ---------------------------------------------------------------- payloads


class NewProject(BaseModel):
    name: str
    description: str = ""
    style_id: str = "xianxia-premium-webtoon"
    style_addendum: str = ""


class ProjectPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    style_id: str | None = None
    style_addendum: str | None = None


class NewChapter(BaseModel):
    title: str = ""


class ChapterPatch(BaseModel):
    title: str | None = None
    source_story: str | None = None


class PanelPatch(BaseModel):
    shot: str | None = None
    aspect: str | None = None
    action: str | None = None
    setting: str | None = None
    lighting: str | None = None
    camera: str | None = None
    fx: list[str] | None = None
    seed: int | None = None
    characters: list[dict] | None = None
    dialogue: list[dict] | None = None


class BreakdownReq(BaseModel):
    story: str
    target_panels: int = 6


class ReorderReq(BaseModel):
    order: list[str]


# ---------------------------------------------------------------- app


def create_app(workspace_root: str = "workspace", backend: str = "comfy",
               host: str = "127.0.0.1:8188") -> FastAPI:
    ws = Workspace(root=Path(workspace_root))
    ws.seed_default_style()
    render_backend = make_backend(backend, host)
    worker = Worker()
    app = FastAPI(title="Manhua Studio")

    # ---------- resolvers ----------

    def get_project(pid: str) -> Project:
        try:
            return ws.load_project(pid)
        except FileNotFoundError:
            raise HTTPException(404, f"no project '{pid}'")

    def get_chapter(pid: str, n: int) -> Chapter:
        proj = get_project(pid)
        if n not in proj.chapter_numbers():
            raise HTTPException(404, f"no chapter {n} in '{pid}'")
        return proj.chapter(n)

    def serialise_chapter(ch: Chapter) -> dict:
        return {
            "number": ch.number,
            "title": ch.title,
            "source_story": ch.source_story,
            "canvas": vars(ch.project.canvas),
            "characters": {
                cid: {"name": c.name, "has_lora": bool(c.lora)}
                for cid, c in ch.project.bible.items()
            },
            "panels": [
                {
                    **p.model_dump(mode="json"),
                    "rendered": ch.stat(p.id).rendered,
                    "locked": ch.stat(p.id).locked,
                    "error": ch.stat(p.id).error,
                }
                for p in ch.panels
            ],
        }

    # ---------- shell ----------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        ok = render_backend.ping() if hasattr(render_backend, "ping") else True
        return {"backend": type(render_backend).__name__, "reachable": ok}

    @app.get("/api/styles")
    def styles() -> list[dict]:
        return ws.list_styles()

    # ---------- projects ----------

    @app.get("/api/projects")
    def list_projects() -> list[dict]:
        return [p.summary() for p in ws.list_projects()]

    @app.post("/api/projects")
    def create_project(req: NewProject) -> dict:
        if not req.name.strip():
            raise HTTPException(400, "a project needs a name")
        proj = ws.create_project(
            req.name.strip(), req.description, req.style_id, req.style_addendum
        )
        return proj.summary()

    @app.get("/api/projects/{pid}")
    def read_project(proj: Project = Depends(get_project)) -> dict:
        return proj.summary()

    @app.patch("/api/projects/{pid}")
    def patch_project(patch: ProjectPatch, proj: Project = Depends(get_project)) -> dict:
        for k, v in patch.model_dump(exclude_none=True).items():
            setattr(proj, k, v)
        proj.save()
        proj.reload()
        return proj.summary()

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str) -> dict:
        ws.delete_project(pid)
        return {"ok": True}

    @app.get("/api/projects/{pid}/bible")
    def read_bible(proj: Project = Depends(get_project)) -> dict:
        text = proj.bible_file.read_text(encoding="utf-8") if proj.bible_file.exists() else ""
        return {"yaml": text}

    @app.put("/api/projects/{pid}/bible")
    def write_bible(body: dict, proj: Project = Depends(get_project)) -> dict:
        import yaml as _yaml

        text = body.get("yaml", "")
        try:
            _yaml.safe_load(text)
        except Exception as exc:
            raise HTTPException(400, f"invalid YAML: {exc}")
        proj.bible_file.write_text(text, encoding="utf-8")
        proj.reload()
        return {"ok": True, "characters": len(proj.bible)}

    # ---------- chapters ----------

    @app.post("/api/projects/{pid}/chapters")
    def new_chapter(req: NewChapter, proj: Project = Depends(get_project)) -> dict:
        return serialise_chapter(proj.new_chapter(req.title))

    @app.get("/api/projects/{pid}/chapters/{n}")
    def read_chapter(ch: Chapter = Depends(get_chapter)) -> dict:
        return serialise_chapter(ch)

    @app.patch("/api/projects/{pid}/chapters/{n}")
    def patch_chapter(patch: ChapterPatch, ch: Chapter = Depends(get_chapter)) -> dict:
        for k, v in patch.model_dump(exclude_none=True).items():
            setattr(ch, k, v)
        ch.save()
        return serialise_chapter(ch)

    @app.delete("/api/projects/{pid}/chapters/{n}")
    def delete_chapter(pid: str, n: int) -> dict:
        get_project(pid).delete_chapter(n)
        return {"ok": True}

    # ---------- panels ----------

    @app.get("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/image")
    def panel_image(panel_id: str, ch: Chapter = Depends(get_chapter)):
        path = ch.panel_path(panel_id)
        if not path.exists():
            raise HTTPException(404, "not rendered")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.patch("/api/projects/{pid}/chapters/{n}/panels/{panel_id}")
    def patch_panel(panel_id: str, patch: PanelPatch,
                    ch: Chapter = Depends(get_chapter)) -> dict:
        p = ch.get(panel_id)
        if p is None:
            raise HTTPException(404, f"no panel {panel_id}")
        for k, v in patch.model_dump(exclude_none=True).items():
            setattr(p, k, v)
        ch.save()
        return p.model_dump(mode="json")

    @app.delete("/api/projects/{pid}/chapters/{n}/panels/{panel_id}")
    def delete_panel(panel_id: str, ch: Chapter = Depends(get_chapter)) -> dict:
        ch.panels = [p for p in ch.panels if p.id != panel_id]
        ch.status.pop(panel_id, None)
        ch.panel_path(panel_id).unlink(missing_ok=True)
        ch.save()
        return {"ok": True}

    @app.post("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/lock")
    def lock_panel(panel_id: str, locked: bool = True,
                   ch: Chapter = Depends(get_chapter)) -> dict:
        ch.stat(panel_id).locked = locked
        ch.save()
        return {"id": panel_id, "locked": locked}

    @app.post("/api/projects/{pid}/chapters/{n}/panels/reorder")
    def reorder(req: ReorderReq, ch: Chapter = Depends(get_chapter)) -> dict:
        index = {pid_: i for i, pid_ in enumerate(req.order)}
        ch.panels.sort(key=lambda p: index.get(p.id, 10**6))
        ch.save()
        return serialise_chapter(ch)

    # ---------- render ----------

    @app.post("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/render")
    def render_one(panel_id: str, reroll: bool = False,
                   ch: Chapter = Depends(get_chapter)) -> dict:
        p = ch.get(panel_id)
        if p is None:
            raise HTTPException(404, f"no panel {panel_id}")

        def run(job: Job) -> None:
            job.detail = panel_id
            ch.render_panel(p, render_backend, new_seed=reroll)
            job.done = 1

        return vars(worker.submit("render", 1, run))

    @app.post("/api/projects/{pid}/chapters/{n}/beats/{beat}/render")
    def render_beat(beat: int, only_missing: bool = True,
                    ch: Chapter = Depends(get_chapter)) -> dict:
        targets = [
            p for p in ch.panels
            if p.beat == beat
            and not ch.stat(p.id).locked
            and not (only_missing and ch.stat(p.id).rendered)
        ]
        if not targets:
            raise HTTPException(400, "nothing to render in this beat")

        def run(job: Job) -> None:
            for p in targets:
                job.detail = p.id
                ch.render_panel(p, render_backend)
                job.done += 1

        return vars(worker.submit("beat", len(targets), run))

    @app.get("/api/jobs/{jid}")
    def job_state(jid: str) -> dict:
        job = worker.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return vars(job)

    # ---------- script ----------

    @app.post("/api/projects/{pid}/chapters/{n}/breakdown")
    def breakdown(req: BreakdownReq, ch: Chapter = Depends(get_chapter)) -> dict:
        from ..script.breakdown import story_to_panels

        if not ch.project.bible:
            raise HTTPException(
                400, "this project has no characters yet - add them in the bible first"
            )

        beat = ch.next_beat()
        panels = story_to_panels(
            req.story,
            bible=ch.project.bible,
            episode_no=ch.number,
            beat_start=beat,
            start_index=len(ch.panels) + 1,
            target_panels=req.target_panels,
        )
        # Panel ids are chapter-scoped; the breakdown numbers them per episode.
        for i, p in enumerate(panels, start=len(ch.panels) + 1):
            p.id = f"c{ch.number:03d}_p{i:03d}"

        ch.panels.extend(panels)
        ch.source_story = (ch.source_story + "\n\n" + req.story).strip()
        ch.save()
        return {"added": len(panels), "beat": beat, "chapter": serialise_chapter(ch)}

    # ---------- export ----------

    @app.post("/api/projects/{pid}/chapters/{n}/export")
    def do_export(letter: bool = True, ch: Chapter = Depends(get_chapter)) -> dict:
        items: list[tuple[str, Image.Image, bool]] = []
        for p in ch.panels:
            path = ch.panel_path(p.id)
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            if letter and p.dialogue:
                img = letter_panel(img, p, ch.project)
            items.append((p.id, img, p.aspect == "full_bleed", p.pause, p.inset))

        if not items:
            raise HTTPException(400, "no rendered panels to export")

        strip, places = compose(items, ch.project.canvas)
        written = export(
            strip, places, ch.project.canvas, ch.dir / "export", stem=f"ch{ch.number:03d}"
        )
        return {
            "files": [str(w) for w in written],
            "height": strip.height,
            "panels": len(items),
        }

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    return app


def letter_panel(img: Image.Image, panel: Panel, project: Project) -> Image.Image:
    """Draw a panel's balloons onto a copy of its render."""
    from PIL import ImageFont

    from ..letter.balloon import draw_balloon

    cfg = project.lettering
    size = int(cfg.get("font_size", 26))
    try:
        font = ImageFont.truetype(cfg.get("font", ""), size)
    except Exception:
        font = ImageFont.load_default(size)

    out = img.copy()
    taken: list[tuple[int, int, int, int]] = []
    for balloon in panel.dialogue:
        draw_balloon(
            out, balloon, font,
            padding=int(cfg.get("padding", 22)),
            line_spacing=float(cfg.get("line_spacing", 1.25)),
            taken=taken,
        )
    return out

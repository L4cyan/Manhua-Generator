"""Studio HTTP server: an AI manhua workspace.

Renders are serialised through one worker thread. A 6GB card cannot run two
SDXL jobs at once, and queueing them beats an OOM mid-chapter. The UI polls
job state instead of holding a connection open for 30 seconds.
"""
from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from ..compose.strip import add_credit, compose, export
from ..models import Balloon, Character, CharacterRef, Panel
from ..workspace import (DEFAULT_STYLE, TRASH_DAYS, Chapter, Project, Workspace,
                         make_backend)

STATIC = Path(__file__).parent / "static"

# Printed under the last panel on export, unless switched off. See the Licence
# section of the README for why this is a courtesy and not a control.
CREDIT = "made with Manhua Generator  ·  lacyan.me"


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
    # Anything the caller needs back. Renders write files and need nothing
    # here; the language jobs return their proposal for review.
    result: dict | None = None


class Worker:
    """One serialised job lane.

    Two lanes exist, sharing a job registry. Renders share a lane because a
    6GB card cannot run two diffusion jobs at once and queueing beats an OOM
    mid-chapter. Language jobs get their own, so a two-minute cast extraction
    does not sit behind a forty-panel render.
    """

    def __init__(self, jobs: dict[str, Job]) -> None:
        self.jobs = jobs
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
    style_id: str = DEFAULT_STYLE
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
    # Fields the editor needs but the original patch model predates.
    world: str | None = None
    extras: int | None = None
    extras_sex: str | None = None
    extras_style: str | None = None
    pause: str | None = None
    inset: float | None = None
    figure_scale: str | None = None


class StylePatch(BaseModel):
    style_id: str


class BreakdownReq(BaseModel):
    story: str
    target_panels: int = 6
    # True when the prose box holds the whole chapter and this run should
    # supersede the storyboard, rather than adding a scene to the end of it.
    replace: bool = False


class CastReq(BaseModel):
    story: str
    setting: str = ""


class CastCommit(BaseModel):
    characters: list[dict]
    # True when the payload is the whole bible, so anyone missing from it was
    # deleted. The extraction flow sends False: it is proposing additions, not
    # claiming to know about everyone already there.
    replace: bool = False


class ReorderReq(BaseModel):
    order: list[str]


class ReviseReq(BaseModel):
    instruction: str


class RerenderReq(BaseModel):
    # A new seed changes the composition as well as the identity. Off by
    # default: after a cast edit you want to see the SAME shot with the right
    # character, not a different shot you now have to judge from scratch.
    new_seeds: bool = False
    cast_only: bool = True


class ExportReq(BaseModel):
    letter: bool = True
    fmt: str = "png"        # png | jpg | webp
    quality: int = 92
    credit: bool = True


# ---------------------------------------------------------------- chunking


def scene_chunks(story: str, want: int) -> list[str]:
    """Split prose into roughly `want` scene-sized pieces.

    Explicit scene breaks win: an author who wrote `<<----------->>` has already
    said where the cuts are, and splitting anywhere else there merges two
    locations into one beat. Otherwise fall back to paragraph boundaries,
    packing them to an even length so no chunk is a single line.
    """
    marked = [s.strip() for s in re.split(r"^\s*<<-+>>\s*$", story, flags=re.M) if s.strip()]
    if len(marked) >= want:
        return marked

    paras = [p.strip() for p in re.split(r"\n\s*\n", story) if p.strip()]
    if want <= 1 or len(paras) <= 1:
        return [story.strip()]

    want = min(want, len(paras))
    total = sum(len(p) for p in paras)
    out: list[str] = []
    buf: list[str] = []
    used = 0
    for i, p in enumerate(paras):
        buf.append(p)
        used += len(p)
        left_paras = len(paras) - i - 1
        left_chunks = want - len(out) - 1
        # Close a chunk once it has its share, or early if there are only just
        # enough paragraphs left to fill the chunks still owed.
        if left_chunks > 0 and (used >= total / want or left_paras <= left_chunks):
            out.append("\n\n".join(buf))
            buf, used = [], 0
    if buf:
        out.append("\n\n".join(buf))
    return out


# ---------------------------------------------------------------- app


def create_app(workspace_root: str = "workspace", backend: str = "comfy",
               host: str = "127.0.0.1:8188") -> FastAPI:
    ws = Workspace(root=Path(workspace_root))
    ws.seed_default_style()
    render_backend = make_backend(backend, host)
    jobs: dict[str, Job] = {}
    worker = Worker(jobs)          # the GPU lane
    agent = Worker(jobs)           # the language-model lane
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
                cid: {
                    "name": c.name,
                    "sex": c.sex,
                    "role": getattr(c, "role", "cast"),
                    "appearance": c.appearance,
                    "has_lora": bool(c.lora),
                }
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
        try:
            return {"ok": True, "trash_id": ws.delete_project(pid)}
        except FileNotFoundError:
            raise HTTPException(404, f"no project '{pid}'")

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
        try:
            return {"ok": True, "trash_id": ws.delete_chapter(pid, n)}
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    # ---------- trash ----------

    @app.get("/api/trash")
    def read_trash() -> dict:
        return {"items": ws.list_trash(), "days": TRASH_DAYS}

    @app.post("/api/trash/{tid}/restore")
    def restore(tid: str) -> dict:
        try:
            return {"ok": True, "item": ws.restore_trash(tid)}
        except FileNotFoundError:
            raise HTTPException(404, "not in the bin")
        except FileExistsError as exc:
            raise HTTPException(409, str(exc))

    @app.delete("/api/trash/{tid}")
    def purge(tid: str) -> dict:
        ws.purge_trash(tid)
        return {"ok": True}

    @app.delete("/api/trash")
    def empty() -> dict:
        return {"ok": True, "removed": ws.empty_trash()}

    # ---------- panels ----------

    def letter_cache(ch: Chapter, panel_id: str, p: Panel, src: Path) -> Path:
        """Path for this panel's lettered composite, keyed by what went into it.

        Lettering a panel is not free: laying out an oval balloon is an
        iterative fit, and doing 48 of them on every page load left half the
        strip blank for eight seconds. The key covers the art, the dialogue and
        the lettering settings, so any change misses the cache and nothing
        stale can survive.
        """
        seed = json.dumps(
            [src.stat().st_mtime_ns, [b.model_dump(mode="json") for b in p.dialogue],
             ch.project.lettering],
            sort_keys=True, default=str,
        )
        key = hashlib.sha1(seed.encode()).hexdigest()[:16]
        return ch.dir / ".letter" / f"{panel_id}-{key}.png"

    def build_letter(ch: Chapter, panel_id: str, p: Panel, src: Path
                     ) -> tuple[Path, list[dict]]:
        cache = letter_cache(ch, panel_id, p, src)
        meta = cache.with_suffix(".json")
        if cache.exists() and meta.exists():
            try:
                return cache, json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                pass                                  # rebuild rather than serve junk

        img, boxes = letter(Image.open(src).convert("RGB"), p, ch.project)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Drop this panel's older entries: only the current one is ever wanted,
        # and a balloon dragged fifty times should not leave fifty files.
        for old in cache.parent.glob(f"{panel_id}-*"):
            if old.stem != cache.stem:
                old.unlink(missing_ok=True)
        img.save(cache)
        meta.write_text(json.dumps({"w": img.width, "h": img.height,
                                    "balloons": boxes}), encoding="utf-8")
        return cache, {"w": img.width, "h": img.height, "balloons": boxes}

    @app.get("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/image")
    def panel_image(panel_id: str, letter_it: bool = Query(False, alias="letter"),
                    ch: Chapter = Depends(get_chapter)):
        path = ch.panel_path(panel_id)
        if not path.exists():
            raise HTTPException(404, "not rendered")
        p = ch.get(panel_id)
        if not letter_it or p is None or not p.dialogue:
            return FileResponse(path, headers={"Cache-Control": "no-cache"})
        cache, _ = build_letter(ch, panel_id, p, path)
        return FileResponse(cache, headers={"Cache-Control": "no-cache"})

    @app.get("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/balloons")
    def panel_balloons(panel_id: str, ch: Chapter = Depends(get_chapter)) -> dict:
        """Where every balloon landed, in panel pixels, for the drag layer."""
        p = ch.get(panel_id)
        if p is None:
            raise HTTPException(404, f"no panel {panel_id}")
        path = ch.panel_path(panel_id)
        if not path.exists() or not p.dialogue:
            return {"w": 0, "h": 0, "balloons": []}
        return build_letter(ch, panel_id, p, path)[1]

    @app.patch("/api/projects/{pid}/chapters/{n}/panels/{panel_id}")
    def patch_panel(panel_id: str, patch: PanelPatch,
                    ch: Chapter = Depends(get_chapter)) -> dict:
        if ch.get(panel_id) is None:
            raise HTTPException(404, f"no panel {panel_id}")
        # Under the chapter lock, on freshly-read state: a render batch running
        # in the other lane holds its own copy of the whole chapter, and
        # without this the later of the two saves reverts the other.
        with ch.project.edit_chapter(ch.number) as live:
            p = live.get(panel_id)
            for k, v in patch.model_dump(exclude_none=True).items():
            # Pydantic does not validate on assignment, so a patched list would
            # stay as raw dicts and the very next render would die reaching for
            # `.id` on one. Rebuild the models explicitly.
                if k == "characters":
                    v = [CharacterRef(**c) for c in v]
                elif k == "dialogue":
                    v = [Balloon(**b) for b in v]
                setattr(p, k, v)
            out = p.model_dump(mode="json")
        return out


    @app.get("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/prompt")
    def panel_prompt(panel_id: str, ch: Chapter = Depends(get_chapter)) -> dict:
        """The exact strings that will be sent to the sampler.

        Surfaced because almost every panel problem is a prompt problem, and
        guessing what the pipeline assembled from a dozen sources is the
        slowest way to debug one.
        """
        from ..render.base import build_request

        p = ch.get(panel_id)
        if p is None:
            raise HTTPException(404, f"no panel {panel_id}")
        req = build_request(p, ch.project.style, ch.project.bible, seed=p.seed or 0)
        return {"positive": req.positive, "negative": req.negative,
                "width": req.width, "height": req.height,
                "loras": [{"name": n, "weight": w} for n, w in req.loras]}

    @app.put("/api/projects/{pid}/style")
    def set_style(patch: StylePatch, pid: str) -> dict:
        """Swap the whole series look. Panels already rendered keep their art
        until they are re-rendered, which is what makes trying a style cheap."""
        proj = ws.load_project(pid)
        try:
            ws.load_style(patch.style_id)
        except Exception:
            raise HTTPException(404, f"no style {patch.style_id}")
        proj.style_id = patch.style_id
        proj.save()
        proj.reload()
        return {"style_id": proj.style_id}

    @app.delete("/api/projects/{pid}/chapters/{n}/panels/{panel_id}")
    def delete_panel(panel_id: str, ch: Chapter = Depends(get_chapter)) -> dict:
        with ch.project.edit_chapter(ch.number) as live:
            live.panels = [p for p in live.panels if p.id != panel_id]
            live.status.pop(panel_id, None)
        ch.panel_path(panel_id).unlink(missing_ok=True)
        return {"ok": True}

    @app.post("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/lock")
    def lock_panel(panel_id: str, locked: bool = True,
                   ch: Chapter = Depends(get_chapter)) -> dict:
        with ch.project.edit_chapter(ch.number) as live:
            live.stat(panel_id).locked = locked
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

    @app.post("/api/projects/{pid}/chapters/{n}/rerender")
    def rerender(req: RerenderReq, ch: Chapter = Depends(get_chapter)) -> dict:
        """Re-render panels that were made with an out-of-date character bible.

        The identity lock is read from the bible at render time, so panels
        rendered AFTER a cast edit are already correct and only the earlier
        ones are stale. Locked panels are never touched: that is what the lock
        is for, and a re-render is exactly when you are glad you set it.
        """
        targets = [
            p for p in ch.panels
            if not ch.stat(p.id).locked
            and ch.stat(p.id).rendered
            and (p.characters or not req.cast_only)
        ]
        if not targets:
            raise HTTPException(400, "nothing to re-render (all locked, unrendered, "
                                     "or without a cast)")

        def run(job: Job) -> None:
            for i, p in enumerate(targets, 1):
                job.detail = f"{p.id} ({i} of {len(targets)})"
                ch.render_panel(p, render_backend, new_seed=req.new_seeds)
                job.done = i

        return vars(worker.submit("rerender", len(targets), run))

    @app.post("/api/projects/{pid}/chapters/{n}/scrap")
    def scrap(keep_storyboard: bool = True, keep_locked: bool = True,
              ch: Chapter = Depends(get_chapter)) -> dict:
        """Throw the rendered art away so the chapter renders again from the top.

        The usual reason is a cast change: the storyboard is still right, the
        pictures are of the wrong person. So by default the panels, the
        dialogue, the pacing and the cast all stay exactly as they are and only
        the images go, which puts every panel back to "not rendered".

        `keep_storyboard=false` also drops the panels, for when the breakdown
        itself is what needs redoing. The prose survives either way: it is the
        part that was typed by hand.

        Nothing is deleted. A chapter of art is hours of GPU time and this is a
        button someone will press by accident, so it moves to a dated folder
        beside the chapter.
        """
        import shutil
        from datetime import datetime, timezone

        locked = {p.id for p in ch.panels if ch.stat(p.id).locked} if keep_locked else set()
        src = ch.dir / "panels"
        moved, n_art = "", 0

        if src.exists() and any(src.glob("*.png")):
            stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
            dest = ch.dir / f"panels_scrapped_{stamp}"
            dest.mkdir(parents=True, exist_ok=True)
            for img in src.glob("*.png"):
                if img.stem in locked:
                    continue           # a locked panel is finished; leave it be
                shutil.move(str(img), str(dest / img.name))
                n_art += 1
            moved = str(dest.resolve()) if n_art else ""
            if not n_art:
                dest.rmdir()

        n_panels = len(ch.panels)
        with ch.project.edit_chapter(ch.number) as live:
            if keep_storyboard:
                for p in live.panels:
                    if p.id in locked:
                        continue
                    st = live.stat(p.id)
                    st.rendered, st.error = False, None
            else:
                live.panels = [p for p in live.panels if p.id in locked]
                live.status = {k: v for k, v in live.status.items() if k in locked}

        return {"ok": True, "panels": n_panels, "images": n_art,
                "kept_locked": len(locked), "moved_to": moved,
                "storyboard_kept": keep_storyboard}

    @app.get("/api/projects/{pid}/chapters/{n}/stale")
    def stale(cast_only: bool = True, ch: Chapter = Depends(get_chapter)) -> dict:
        """How many panels a re-render would touch, for the confirmation."""
        n_locked = sum(1 for p in ch.panels if ch.stat(p.id).locked)
        n = sum(1 for p in ch.panels
                if not ch.stat(p.id).locked and ch.stat(p.id).rendered
                and (p.characters or not cast_only))
        return {"count": n, "locked": n_locked, "total": len(ch.panels)}

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
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return vars(job)

    @app.post("/api/projects/{pid}/chapters/{n}/panels/{panel_id}/revise")
    def revise_panel(panel_id: str, req: ReviseReq,
                     ch: Chapter = Depends(get_chapter)) -> dict:
        """Rewrite a panel from a plain-language note.

        The field editor is precise and slow to use. Most of the time the note
        is one sentence ("make it night, and put him further away"), and a
        model is perfectly good at turning that into the four fields it
        touches. Applied straight away because everything it changes is
        reversible and visible.
        """
        p = ch.get(panel_id)
        if p is None:
            raise HTTPException(404, f"no panel {panel_id}")
        if not req.instruction.strip():
            raise HTTPException(400, "say what should change")

        def run(job: Job) -> None:
            from ..script.revise import revise

            job.detail = "rewriting the panel"
            before = p.model_dump(mode="json")
            changed = revise(p, req.instruction, bible=ch.project.bible)
            ch.save()
            job.result = {
                "panel": p.model_dump(mode="json"),
                "changed": [k for k in changed if before.get(k) != changed[k]],
            }
            job.done = 1

        return vars(agent.submit("revise", 1, run))

    # ---------- cast ----------

    @app.get("/api/projects/{pid}/cast")
    def read_cast(proj: Project = Depends(get_project)) -> dict:
        return {"characters": [c.model_dump(mode="json") for c in proj.bible.values()]}

    @app.post("/api/projects/{pid}/cast/extract")
    def extract_cast(req: CastReq, proj: Project = Depends(get_project)) -> dict:
        """Read a chapter and propose bible entries for the people in it.

        A proposal, not a commit. `appearance` becomes the identity lock that
        every prompt this character appears in inherits verbatim, so it is
        worth a human glance before hundreds of panels are built on it.
        """
        from ..script.cast import cast_from_story

        if not req.story.strip():
            raise HTTPException(400, "paste the chapter first")

        def run(job: Job) -> None:
            from ..script import providers

            _, model = providers.detect()
            job.detail = f"{model or 'the local model'} is reading "                          f"{len(req.story.split())} words"
            pairs = cast_from_story(
                req.story,
                setting=req.setting or proj.description,
                existing=proj.bible,
            )
            job.detail = f"writing {len(pairs)} identity locks"
            job.result = {
                "characters": [
                    {**c.model_dump(mode="json"), **info,
                     "existing": c.id in proj.bible}
                    for c, info in pairs
                ]
            }
            job.done = 1

        return vars(agent.submit("cast", 1, run))

    @app.post("/api/projects/{pid}/cast/commit")
    def commit_cast(req: CastCommit, proj: Project = Depends(get_project)) -> dict:
        """Merge approved characters into the bible, keeping its header."""
        import yaml as _yaml

        old = proj.bible_file.read_text(encoding="utf-8") if proj.bible_file.exists() else ""
        raw = _yaml.safe_load(old) or {}
        entries = {} if req.replace else (raw.get("characters") or {})

        for item in req.characters:
            fields = {k: v for k, v in item.items() if k in Character.model_fields}
            try:
                char = Character(**fields)
            except Exception as exc:
                raise HTTPException(400, f"{item.get('id', '?')}: {exc}")
            entries[char.id] = char.model_dump(
                mode="json", exclude={"id"}, exclude_defaults=True
            )

        # Keep the explanatory header: it is the only place the identity-lock
        # rule is written down where someone editing the file will see it.
        head: list[str] = []
        for line in old.splitlines(keepends=True):
            if line.strip() and not line.lstrip().startswith("#"):
                break
            head.append(line)

        body = _yaml.safe_dump(
            {"characters": entries}, sort_keys=False, allow_unicode=True, width=100
        )
        proj.bible_file.write_text("".join(head) + body, encoding="utf-8")
        proj.reload()
        return {"ok": True, "characters": len(proj.bible)}

    # ---------- script ----------

    @app.post("/api/projects/{pid}/chapters/{n}/breakdown")
    def breakdown(req: BreakdownReq, ch: Chapter = Depends(get_chapter)) -> dict:
        from ..script.breakdown import story_to_panels
        from ..script.scene import dress

        if not ch.project.bible:
            raise HTTPException(
                400, "no characters yet - extract the cast from this chapter first"
            )
        if not req.story.strip():
            raise HTTPException(400, "paste the chapter first")

        if req.replace:
            # The panels are being rebuilt from the same prose, so their art no
            # longer matches anything. Clearing the images too keeps `rendered`
            # honest instead of leaving a strip of pictures from a storyboard
            # that no longer exists.
            for old in ch.panels:
                ch.panel_path(old.id).unlink(missing_ok=True)
            ch.panels, ch.status = [], {}
            ch.source_story = ""
            ch.save()

        beat = ch.next_beat()
        # A whole chapter does not fit in one breakdown call: asked for forty
        # panels at once, a small model returns eight and drops the second half
        # of the passage. Split into scene-sized chunks and let each become a
        # beat, which is also the unit "Render beat" works in.
        chunks = scene_chunks(req.story, max(1, round(req.target_panels / 6)))
        per = max(3, round(req.target_panels / len(chunks)))

        def run(job: Job) -> None:
            added = 0
            for k, chunk in enumerate(chunks):
                job.detail = (f"scene {k + 1} of {len(chunks)}: "
                              f"{len(chunk.split())} words into about {per} panels")
                panels = story_to_panels(
                    chunk,
                    bible=ch.project.bible,
                    episode_no=ch.number,
                    beat_start=beat + k,
                    start_index=len(ch.panels) + 1,
                    target_panels=per,
                )
                # Second pass: the breakdown decides what happens, this decides
                # where. Once per scene, so ten panels share one room instead of
                # drifting through ten rooms that sound alike.
                job.detail = f"scene {k + 1} of {len(chunks)}: dressing the set"
                dress(chunk, panels)

                # Panel ids are chapter-scoped; the breakdown numbers per episode.
                for i, p in enumerate(panels, start=len(ch.panels) + 1):
                    p.id = f"c{ch.number:03d}_p{i:03d}"
                ch.panels.extend(panels)
                added += len(panels)
                # Saved per chunk so a failure halfway keeps the scenes that
                # already worked instead of throwing away ten minutes.
                ch.save()
                job.done = k + 1

            ch.source_story = (req.story if req.replace
                               else ch.source_story + "\n\n" + req.story).strip()
            ch.save()
            job.result = {"added": added, "beat": beat, "scenes": len(chunks)}

        return vars(agent.submit("breakdown", len(chunks), run))

    # ---------- export ----------

    @app.post("/api/projects/{pid}/chapters/{n}/export")
    def do_export(req: ExportReq | None = None,
                  ch: Chapter = Depends(get_chapter)) -> dict:
        req = req or ExportReq()
        items: list[tuple[str, Image.Image, bool]] = []
        for p in ch.panels:
            path = ch.panel_path(p.id)
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            if req.letter and p.dialogue:
                img = letter_panel(img, p, ch.project)
            items.append((p.id, img, p.aspect == "full_bleed", p.pause, p.inset))

        if not items:
            raise HTTPException(400, "no rendered panels to export")

        strip, places = compose(items, ch.project.canvas)
        if req.credit:
            strip = add_credit(strip, CREDIT, ch.project.canvas)
        written = export(
            strip, places, ch.project.canvas, ch.dir / "export",
            stem=f"ch{ch.number:03d}", fmt=req.fmt, quality=req.quality,
        )
        return {
            "files": [str(w) for w in written],
            "dir": str((ch.dir / "export").resolve()),
            "height": strip.height,
            "panels": len(items),
            "fmt": req.fmt,
        }

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    return app


def letter(img: Image.Image, panel: Panel, project: Project
           ) -> tuple[Image.Image, list[dict]]:
    """Draw a panel's balloons onto a copy, and report where they landed.

    The boxes are what makes the editor's dialogue layer honest: the drag
    handles sit exactly where the letterer actually drew, so what is dragged
    is what gets exported, rather than a CSS approximation of it.
    """
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
    boxes: list[dict] = []
    for i, balloon in enumerate(panel.dialogue):
        x, y, w, h = draw_balloon(
            out, balloon, font,
            padding=int(cfg.get("padding", 22)),
            line_spacing=float(cfg.get("line_spacing", 1.25)),
            taken=taken,
        )
        boxes.append({
            "i": i, "x": x, "y": y, "w": w, "h": h,
            "kind": balloon.kind, "text": balloon.text,
            "speaker": balloon.speaker,
            "tail_x": balloon.tail_x, "tail_y": balloon.tail_y,
            "placed": balloon.x is not None,
            "width": balloon.width,
        })
    return out, boxes


def letter_panel(img: Image.Image, panel: Panel, project: Project) -> Image.Image:
    """Draw a panel's balloons onto a copy of its render."""
    return letter(img, panel, project)[0]

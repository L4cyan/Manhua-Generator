"""Workspace: projects, chapters, and the style library.

Disk layout (everything is plain files, so a project is portable and
diffable, and nothing is trapped in a database):

    workspace/
      styles/
        xianxia-premium-webtoon.yaml     shared style locks
      projects/
        psionic-cultivation/
          project.json                   name, description, chosen style
          characters.yaml                per-project character bible
          story/ch01.md                  source prose
          chapters/
            001/
              chapter.json               title + panels + per-panel status
              panels/*.png               rendered art
              export/                    composed strip and upload slices
"""
from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .compose.strip import CanvasCfg
from .config import StyleLock, load_bible
from .models import Character, Panel
from .render.base import Backend, build_request

WORKSPACE = Path("workspace")
DEFAULT_STYLE = "xianxia-premium-webtoon"


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "untitled"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- chapter


@dataclass
class PanelStatus:
    rendered: bool = False
    locked: bool = False
    error: str | None = None


@dataclass
class Chapter:
    project: "Project"
    number: int
    title: str = ""
    panels: list[Panel] = field(default_factory=list)
    status: dict[str, PanelStatus] = field(default_factory=dict)
    source_story: str = ""

    # ---------- paths ----------

    @property
    def dir(self) -> Path:
        return self.project.dir / "chapters" / f"{self.number:03d}"

    @property
    def file(self) -> Path:
        return self.dir / "chapter.json"

    @property
    def panel_dir(self) -> Path:
        d = self.dir / "panels"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def panel_path(self, panel_id: str) -> Path:
        return self.panel_dir / f"{panel_id}.png"

    # ---------- persistence ----------

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "number": self.number,
            "title": self.title,
            "source_story": self.source_story,
            "updated": now(),
            "panels": [p.model_dump(mode="json") for p in self.panels],
            "status": {k: vars(v) for k, v in self.status.items()},
        }
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.file)
        self.project.touch()

    @classmethod
    def load(cls, project: "Project", number: int) -> "Chapter":
        ch = cls(project=project, number=number)
        if not ch.file.exists():
            return ch
        raw = json.loads(ch.file.read_text(encoding="utf-8"))
        ch.title = raw.get("title", "")
        ch.source_story = raw.get("source_story", "")
        ch.panels = [Panel.model_validate(p) for p in raw.get("panels", [])]
        ch.status = {k: PanelStatus(**v) for k, v in raw.get("status", {}).items()}
        return ch

    # ---------- panels ----------

    def get(self, panel_id: str) -> Panel | None:
        return next((p for p in self.panels if p.id == panel_id), None)

    def stat(self, panel_id: str) -> PanelStatus:
        return self.status.setdefault(panel_id, PanelStatus())

    def next_panel_id(self) -> str:
        return f"c{self.number:03d}_p{len(self.panels) + 1:03d}"

    def next_beat(self) -> int:
        return max((p.beat for p in self.panels), default=-1) + 1

    def render_panel(self, panel: Panel, backend: Backend, *, new_seed: bool = False) -> Path:
        """Render one panel. Locked panels are never touched."""
        st = self.stat(panel.id)
        if st.locked:
            return self.panel_path(panel.id)

        if new_seed or panel.seed is None:
            panel.seed = random.randint(0, 2**31 - 1)
            if new_seed:
                panel.reroll_count += 1

        req = build_request(panel, self.project.style, self.project.bible, panel.seed)
        req.positive = self.project.apply_addendum(req.positive)
        try:
            img: Image.Image = backend.render(req)
        except Exception as exc:
            st.error, st.rendered = str(exc)[:400], False
            self.save()
            raise

        img.save(self.panel_path(panel.id))
        st.rendered, st.error = True, None
        self.save()
        return self.panel_path(panel.id)


# ---------------------------------------------------------------- project


@dataclass
class Project:
    ws: "Workspace"
    id: str
    name: str
    description: str = ""
    style_id: str = DEFAULT_STYLE
    style_addendum: str = ""
    created: str = field(default_factory=now)
    updated: str = field(default_factory=now)
    canvas: CanvasCfg = field(default_factory=CanvasCfg)
    lettering: dict[str, Any] = field(default_factory=dict)

    _style: StyleLock | None = None
    _bible: dict[str, Character] | None = None

    # ---------- paths ----------

    @property
    def dir(self) -> Path:
        return self.ws.root / "projects" / self.id

    @property
    def file(self) -> Path:
        return self.dir / "project.json"

    @property
    def bible_file(self) -> Path:
        return self.dir / "characters.yaml"

    @property
    def story_dir(self) -> Path:
        d = self.dir / "story"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------- lazily resolved config ----------

    @property
    def style(self) -> StyleLock:
        if self._style is None:
            style = self.ws.load_style(self.style_id)
            # The shipped style file names a placeholder checkpoint. Point it at
            # whatever was actually detected, so a fresh clone renders without
            # the user having to edit YAML first.
            try:
                from .bootstrap import apply_to_style, autoconfigure

                apply_to_style(style, autoconfigure(self.ws.root))
            except Exception:
                pass
            self._style = style
        return self._style

    @property
    def bible(self) -> dict[str, Character]:
        if self._bible is None:
            self._bible = load_bible(self.bible_file) if self.bible_file.exists() else {}
        return self._bible

    def reload(self) -> None:
        self._style = None
        self._bible = None

    def apply_addendum(self, positive: str) -> str:
        """Append the project's extra style prompt.

        It trails the assembled prompt so a long addendum can never push the
        character identity out of CLIP's first chunk.
        """
        extra = self.style_addendum.strip().strip(",")
        return f"{positive}, {extra}" if extra else positive

    # ---------- persistence ----------

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.updated = now()
        self.file.write_text(
            json.dumps(
                {
                    "id": self.id,
                    "name": self.name,
                    "description": self.description,
                    "style_id": self.style_id,
                    "style_addendum": self.style_addendum,
                    "created": self.created,
                    "updated": self.updated,
                    "canvas": vars(self.canvas),
                    "lettering": self.lettering,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def touch(self) -> None:
        self.updated = now()
        if self.file.exists():
            self.save()

    # ---------- chapters ----------

    def chapter_numbers(self) -> list[int]:
        root = self.dir / "chapters"
        if not root.exists():
            return []
        return sorted(int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit())

    def chapters(self) -> list[Chapter]:
        return [Chapter.load(self, n) for n in self.chapter_numbers()]

    def chapter(self, number: int) -> Chapter:
        return Chapter.load(self, number)

    def new_chapter(self, title: str = "") -> Chapter:
        n = max(self.chapter_numbers(), default=0) + 1
        ch = Chapter(project=self, number=n, title=title or f"Chapter {n}")
        ch.save()
        return ch

    def delete_chapter(self, number: int) -> None:
        shutil.rmtree(self.dir / "chapters" / f"{number:03d}", ignore_errors=True)
        self.touch()

    def summary(self) -> dict:
        chapters = []
        for n in self.chapter_numbers():
            ch = Chapter.load(self, n)
            chapters.append(
                {
                    "number": n,
                    "title": ch.title,
                    "panels": len(ch.panels),
                    "rendered": sum(1 for p in ch.panels if ch.stat(p.id).rendered),
                }
            )
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "style_id": self.style_id,
            "style_addendum": self.style_addendum,
            "characters": len(self.bible),
            "updated": self.updated,
            "chapters": chapters,
        }


# ---------------------------------------------------------------- workspace


DEFAULT_BIBLE = """\
# Character bible for this project.
#
# `appearance` is the identity lock: it is emitted verbatim into every prompt
# the character appears in. Write it once, then do not edit it mid-series --
# changing it desyncs new panels from the ones already rendered.
#
# The text gets you roughly 70% consistency. The rest is `lora`, a trained
# identity LoRA. Without one, expect visible drift past ~40 panels.

characters: {}
"""


@dataclass
class Workspace:
    root: Path = WORKSPACE

    # ---------- styles ----------

    @property
    def style_dir(self) -> Path:
        d = self.root / "styles"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def seed_default_style(self, source: Path = Path("config/style.yaml")) -> None:
        """Copy the shipped style lock into the workspace on first run."""
        target = self.style_dir / f"{DEFAULT_STYLE}.yaml"
        if not target.exists() and source.exists():
            shutil.copy(source, target)

    def list_styles(self) -> list[dict]:
        self.seed_default_style()
        out = []
        for f in sorted(self.style_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            out.append(
                {
                    "id": f.stem,
                    "name": raw.get("name", f.stem),
                    "lead": " ".join(str(raw.get("style_lead", "")).split())[:180],
                    "is_default": f.stem == DEFAULT_STYLE,
                }
            )
        return out

    def load_style(self, style_id: str) -> StyleLock:
        self.seed_default_style()
        path = self.style_dir / f"{style_id}.yaml"
        if not path.exists():
            path = self.style_dir / f"{DEFAULT_STYLE}.yaml"
        return StyleLock.load(path)

    # ---------- projects ----------

    @property
    def project_dir(self) -> Path:
        d = self.root / "projects"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_projects(self) -> list[Project]:
        out = []
        for d in sorted(self.project_dir.iterdir()) if self.project_dir.exists() else []:
            if d.is_dir() and (d / "project.json").exists():
                out.append(self.load_project(d.name))
        return sorted(out, key=lambda p: p.updated, reverse=True)

    def load_project(self, pid: str) -> Project:
        raw = json.loads((self.project_dir / pid / "project.json").read_text(encoding="utf-8"))
        return Project(
            ws=self,
            id=raw["id"],
            name=raw["name"],
            description=raw.get("description", ""),
            style_id=raw.get("style_id", DEFAULT_STYLE),
            style_addendum=raw.get("style_addendum", ""),
            created=raw.get("created", now()),
            updated=raw.get("updated", now()),
            canvas=CanvasCfg(**raw.get("canvas", {})),
            lettering=raw.get("lettering", {}),
        )

    def create_project(
        self,
        name: str,
        description: str = "",
        style_id: str = DEFAULT_STYLE,
        style_addendum: str = "",
    ) -> Project:
        pid = slug(name)
        # Never silently merge into an existing project's folder.
        if (self.project_dir / pid).exists():
            n = 2
            while (self.project_dir / f"{pid}-{n}").exists():
                n += 1
            pid = f"{pid}-{n}"

        proj = Project(
            ws=self,
            id=pid,
            name=name,
            description=description,
            style_id=style_id,
            style_addendum=style_addendum,
            lettering={
                "font": "assets/fonts/dialogue.ttf",
                "font_size": 26,
                "line_spacing": 1.25,
                "padding": 22,
            },
        )
        proj.save()
        proj.bible_file.write_text(DEFAULT_BIBLE, encoding="utf-8")
        proj.new_chapter("Chapter 1")
        return proj

    def delete_project(self, pid: str) -> None:
        shutil.rmtree(self.project_dir / pid, ignore_errors=True)


def make_backend(kind: str = "auto", host: str = "127.0.0.1:8188") -> Backend:
    """Build a render backend.

    "auto" resolves through bootstrap detection, which is what makes a fresh
    clone work with no configuration: it finds an existing checkpoint, decides
    whether torch is usable, and falls back to mock rather than erroring.
    """
    if kind == "auto":
        from .bootstrap import autoconfigure, build_backend

        return build_backend(autoconfigure())

    if kind == "mock":
        from .render.mock import MockBackend

        return MockBackend()

    if kind == "fallback":
        from .bootstrap import autoconfigure, build_backend

        cfg = autoconfigure()
        cfg.backend = "fallback"
        return build_backend(cfg)

    if kind == "remote":
        from .bootstrap import autoconfigure, build_backend

        cfg = autoconfigure()
        cfg.backend = "remote"
        return build_backend(cfg)

    if kind == "native":
        from .bootstrap import autoconfigure
        from .render.native import NativeBackend

        s = autoconfigure()
        return NativeBackend(s.checkpoint, lora_dir=s.lora_dir or None, low_vram=s.low_vram)

    from .render.comfy import ComfyBackend

    return ComfyBackend(host=host)

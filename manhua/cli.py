"""Command line entry point."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .net import enable_system_certs
from .workspace import Workspace, make_backend

app = typer.Typer(add_completion=False, help="Long-strip manhua generation workspace.")
console = Console()

WS = typer.Option("workspace", "--workspace", "-w", help="Workspace directory")
BACKEND = typer.Option("auto", "--backend", "-b",
                       help="auto (detect) | native (no ComfyUI) | comfy | mock")
HOST = typer.Option("127.0.0.1:8188", "--comfy-host", help="ComfyUI address")


def _resolve(ws: Workspace, project: str, chapter: int):
    try:
        proj = ws.load_project(project)
    except FileNotFoundError:
        console.print(f"[red]no project '{project}'[/red]")
        console.print("run [cyan]manhua projects[/cyan] to list them")
        raise typer.Exit(1)
    if chapter not in proj.chapter_numbers():
        console.print(f"[red]no chapter {chapter}[/red] in '{project}' "
                      f"(has {proj.chapter_numbers() or 'none'})")
        raise typer.Exit(1)
    return proj, proj.chapter(chapter)


@app.command()
def studio(
    workspace: str = WS,
    backend: str = BACKEND,
    comfy_host: str = HOST,
    port: int = typer.Option(7860),
    host: str = typer.Option("127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Launch the studio: projects, chapters, breakdown, render, reroll, export."""
    import threading
    import webbrowser

    import uvicorn

    from .studio.server import create_app

    url = f"http://{host}:{port}"
    console.print(f"[bold]Manhua Studio[/bold] -> [cyan]{url}[/cyan]  ([dim]{backend}[/dim])")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        create_app(workspace, backend, comfy_host),
        host=host, port=port, log_level="warning",
    )


@app.command("projects")
def list_projects(workspace: str = WS) -> None:
    """List every project in the workspace."""
    ws = Workspace(root=Path(workspace))
    projects = ws.list_projects()
    if not projects:
        console.print("[yellow]no projects yet[/yellow] - create one with "
                      "[cyan]manhua new \"My Story\"[/cyan]")
        raise typer.Exit()

    table = Table("id", "name", "style", "cast", "chapters", "panels")
    for p in projects:
        s = p.summary()
        table.add_row(
            p.id, p.name, p.style_id, str(s["characters"]),
            str(len(s["chapters"])),
            str(sum(c["panels"] for c in s["chapters"])),
        )
    console.print(table)


@app.command("new")
def new_project(
    name: str = typer.Argument(..., help="Project title"),
    description: str = typer.Option("", "--description", "-d"),
    style: str = typer.Option("xianxia-premium-webtoon", "--style", "-s"),
    workspace: str = WS,
) -> None:
    """Create a project (with Chapter 1) in the workspace."""
    ws = Workspace(root=Path(workspace))
    proj = ws.create_project(name, description, style)
    console.print(f"[green]created[/green] {proj.id}")
    console.print(f"  bible   {proj.bible_file}")
    console.print(f"  chapter {proj.chapter(1).file}")
    console.print("\nAdd your cast to the bible, then "
                  f"[cyan]manhua script {proj.id} 1 <story.md>[/cyan]")


@app.command()
def script(
    project: str = typer.Argument(...),
    chapter: int = typer.Argument(1),
    story_file: Path = typer.Argument(..., help="Prose file to break down"),
    panels: int = typer.Option(6, "--panels", "-n", help="Target panels for this beat"),
    workspace: str = WS,
) -> None:
    """Break a prose file into panels and append them as a new beat."""
    from .script.breakdown import story_to_panels

    ws = Workspace(root=Path(workspace))
    proj, ch = _resolve(ws, project, chapter)

    if not proj.bible:
        console.print(f"[red]no characters defined[/red] - edit {proj.bible_file} first")
        raise typer.Exit(1)

    beat = ch.next_beat()
    console.print(f"breaking down [cyan]{story_file}[/cyan] into beat {beat}…")
    new = story_to_panels(
        story_file.read_text(encoding="utf-8"),
        bible=proj.bible, episode_no=ch.number, beat_start=beat,
        start_index=len(ch.panels) + 1, target_panels=panels,
    )
    for i, p in enumerate(new, start=len(ch.panels) + 1):
        p.id = f"c{ch.number:03d}_p{i:03d}"

    ch.panels.extend(new)
    ch.save()

    table = Table("id", "shot", "aspect", "cast", "action")
    for p in new:
        table.add_row(p.id, p.shot.value, p.aspect,
                      ",".join(c.id for c in p.characters) or "-",
                      (p.action[:46] + "…") if len(p.action) > 46 else p.action)
    console.print(table)
    console.print(f"[green]+{len(new)} panels[/green] -> {ch.file}")


@app.command()
def render(
    project: str = typer.Argument(...),
    chapter: int = typer.Argument(1),
    beat: int = typer.Option(-1, help="One beat only; -1 renders every unrendered panel"),
    force: bool = typer.Option(False, help="Re-render panels that already exist"),
    workspace: str = WS,
    backend: str = BACKEND,
    comfy_host: str = HOST,
) -> None:
    """Headless render. Locked panels are always skipped."""
    from rich.progress import Progress

    ws = Workspace(root=Path(workspace))
    _, ch = _resolve(ws, project, chapter)
    be = make_backend(backend, comfy_host)

    targets = [p for p in ch.panels
               if (beat < 0 or p.beat == beat)
               and not ch.stat(p.id).locked
               and (force or not ch.stat(p.id).rendered)]
    if not targets:
        console.print("[yellow]nothing to render[/yellow]")
        raise typer.Exit()

    failed = 0
    with Progress(console=console) as bar:
        task = bar.add_task("rendering", total=len(targets))
        for p in targets:
            bar.update(task, description=f"[cyan]{p.id}[/cyan] {p.shot.value}")
            try:
                ch.render_panel(p, be)
            except Exception as exc:
                failed += 1
                console.print(f"[red]{p.id} failed:[/red] {exc}")
            bar.advance(task)

    ok = len(targets) - failed
    console.print(f"[green]{ok} rendered[/green]" + (f" · [red]{failed} failed[/red]" if failed else ""))


@app.command()
def export(
    project: str = typer.Argument(...),
    chapter: int = typer.Argument(1),
    letter: bool = typer.Option(True, "--letter/--no-letter", help="Draw dialogue balloons"),
    workspace: str = WS,
) -> None:
    """Compose rendered panels into a long strip and slice it for upload."""
    from PIL import Image

    from .compose.strip import compose, export as write
    from .studio.server import letter_panel

    ws = Workspace(root=Path(workspace))
    proj, ch = _resolve(ws, project, chapter)

    items = []
    for p in ch.panels:
        path = ch.panel_path(p.id)
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        if letter and p.dialogue:
            img = letter_panel(img, p, proj)
        items.append((p.id, img, p.aspect == "full_bleed"))

    if not items:
        console.print("[red]no rendered panels[/red]")
        raise typer.Exit(1)

    strip, places = compose(items, proj.canvas)
    files = write(strip, places, proj.canvas, ch.dir / "export", stem=f"ch{ch.number:03d}")
    console.print(f"[green]{strip.width}x{strip.height}[/green] · {len(items)} panels "
                  f"· {len(files) - 1} upload slices")
    for f in files:
        console.print(f"  {f}")


@app.command()
def sheet(
    project: str = typer.Argument(...),
    character: str = typer.Argument(..., help="Character id from the bible"),
    seed: int = typer.Option(-1, help="Base seed; -1 picks one at random"),
    workspace: str = WS,
    backend: str = BACKEND,
    comfy_host: str = HOST,
) -> None:
    """Render a character turnaround sheet and emit a LoRA training config.

    This is the consistency workflow: sheet -> cull by hand -> train -> set the
    character's `lora` field. Text description alone drifts past ~40 panels.
    """
    from rich.progress import Progress

    from .bible.sheet import render_sheet, sheet_requests, write_training_config

    ws = Workspace(root=Path(workspace))
    try:
        proj = ws.load_project(project)
    except FileNotFoundError:
        console.print(f"[red]no project '{project}'[/red]")
        raise typer.Exit(1)

    char = proj.bible.get(character)
    if char is None:
        console.print(f"[red]no character '{character}'[/red] - "
                      f"bible has: {', '.join(proj.bible) or 'nobody'}")
        raise typer.Exit(1)

    out = proj.dir / "bible" / character
    be = make_backend(backend, comfy_host)
    total = len(sheet_requests(char, proj.style))

    console.print(f"rendering {total} reference images for [cyan]{char.name}[/cyan] -> {out}")
    with Progress(console=console) as bar:
        task = bar.add_task("sheet", total=total)

        def step(name: str) -> None:
            bar.update(task, description=f"[cyan]{name}[/cyan]")
            bar.advance(task)

        files = render_sheet(
            char, proj.style, be, out,
            base_seed=None if seed < 0 else seed, on_progress=step,
        )

    cfg = write_training_config(char, proj.style, out, proj.dir / "bible" / "_training")
    console.print(f"\n[green]{len(files)} images[/green] -> {out}")
    console.print(f"[green]training config[/green] -> {cfg}")
    console.print(
        "\n[bold]Next:[/bold]\n"
        "  1. Cull the sheet by hand. Delete anything that is not clearly the SAME person.\n"
        "     This matters more than any training setting.\n"
        "  2. Train with kohya_ss / sd-scripts using the config above.\n"
        f"  3. Set [cyan]lora:[/cyan] on '{character}' in {proj.bible_file}"
    )


@app.command()
def doctor(
    workspace: str = WS,
    backend: str = BACKEND,
    comfy_host: str = HOST,
) -> None:
    """Check the setup: workspace, styles, backend, characters, fonts."""
    ws = Workspace(root=Path(workspace))
    ws.seed_default_style()
    warn = 0

    styles = ws.list_styles()
    console.print(f"[green]v[/green] workspace {ws.root.resolve()}")
    console.print(f"[green]v[/green] {len(styles)} style(s): "
                  + ", ".join(s["id"] for s in styles))

    be = make_backend(backend, comfy_host)
    if hasattr(be, "ping"):
        if be.ping():
            console.print(f"[green]v[/green] ComfyUI reachable at {comfy_host}")
        else:
            console.print(f"[red]x[/red] ComfyUI unreachable at {comfy_host} - is it running?")
            warn += 1
    else:
        console.print(f"[yellow]![/yellow] {type(be).__name__}: placeholder art, no real rendering")

    projects = ws.list_projects()
    if not projects:
        console.print("[yellow]![/yellow] no projects yet")
    for p in projects:
        s = p.summary()
        console.print(f"\n[bold]{p.name}[/bold] ({p.id}) · style {p.style_id}")
        if not p.bible:
            console.print("  [yellow]![/yellow] no characters - breakdown will refuse to run")
            warn += 1
        for cid, c in p.bible.items():
            if c.lora:
                console.print(f"  [green]v[/green] {cid}: {c.lora}")
            else:
                console.print(f"  [yellow]![/yellow] {cid}: no identity LoRA "
                              "- expect drift past ~40 panels")
                warn += 1
        for c in s["chapters"]:
            console.print(f"  ch{c['number']:>3} {c['title'] or '-':<24} "
                          f"{c['rendered']}/{c['panels']} rendered")

    font = Path("assets/fonts/dialogue.ttf")
    if font.exists():
        console.print(f"\n[green]v[/green] lettering font {font}")
    else:
        console.print(f"\n[yellow]![/yellow] no {font} - balloons fall back to a default font")
        warn += 1

    console.print("\n[bold green]ready[/bold green]" if not warn
                  else f"\n[bold yellow]{warn} warning(s) above[/bold yellow]")


@app.command()
def prompt(
    project: str = typer.Argument(...),
    chapter: int = typer.Argument(...),
    panel_id: str = typer.Argument(...),
    workspace: str = WS,
) -> None:
    """Print the exact prompt a panel will be rendered with."""
    from .render.base import build_request

    ws = Workspace(root=Path(workspace))
    proj, ch = _resolve(ws, project, chapter)
    p = ch.get(panel_id)
    if p is None:
        console.print(f"[red]no panel {panel_id}[/red]")
        raise typer.Exit(1)

    req = build_request(p, proj.style, proj.bible, p.seed or 0)
    positive = proj.apply_addendum(req.positive)

    console.print(f"[bold]{panel_id}[/bold]  {req.width}x{req.height}  seed {req.seed}")
    console.print(f"[dim]loras:[/dim] {req.loras or 'none'}")
    console.print(f"\n[green]positive[/green] ({proj.style.est_tokens(positive)} tok)\n{positive}")
    console.print(f"\n[red]negative[/red]\n{req.negative}")

    chunk = proj.style.identity_chunk(p.content_prompt(proj.bible))
    if chunk > 0:
        console.print(f"\n[yellow]![/yellow] character identity sits in CLIP chunk {chunk}; "
                      "shorten style_lead so it lands in chunk 0")


def main() -> None:
    # Antivirus/corporate TLS interception breaks certifi-based
    # verification; use the OS trust store instead.
    enable_system_certs()
    app()


if __name__ == "__main__":
    main()

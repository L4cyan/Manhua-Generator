"""One command from a cold machine to a working studio.

The studio is three processes that have to be up in the right order: ComfyUI
holds the GPU, Ollama writes the script, and only then is the editor useful.
Starting them by hand means three windows and remembering that ComfyUI needs
`--lowvram` or Anima crawls on a 6GB card.

So this does it. It finds ComfyUI, starts it if it is not already answering,
waits for it, reports on Ollama, and then runs the editor. Everything it
cannot fix it says plainly rather than failing silently later: a red dot in
the corner of the editor is a much worse way to learn ComfyUI never started.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "workspace" / "settings.json"

COMFY_PORT = 8188
OLLAMA_PORT = 11434
STUDIO_PORT = 7860

# Where ComfyUI usually ends up on Windows. Checked in order.
COMFY_GUESSES = [
    r"C:\ComfyUI\ComfyUI_windows_portable",
    r"C:\ComfyUI",
    r"C:\ComfyUI_windows_portable",
    r"D:\ComfyUI\ComfyUI_windows_portable",
    r"D:\ComfyUI_windows_portable",
]

RELEASE = "https://github.com/comfyanonymous/ComfyUI/releases/latest"


# ---------------------------------------------------------------- output


if os.name == "nt":
    # Turns on VT processing in cmd.exe on Windows 10+. Without it the escape
    # codes below print as literal garbage.
    os.system("")


class C:
    on = sys.stdout.isatty()
    dim = "\033[2m" if on else ""
    red = "\033[31m" if on else ""
    green = "\033[32m" if on else ""
    yellow = "\033[33m" if on else ""
    cyan = "\033[36m" if on else ""
    bold = "\033[1m" if on else ""
    off = "\033[0m" if on else ""


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def ok(msg: str) -> None:
    say(f"{C.green}OK{C.off}   {msg}")


def warn(msg: str) -> None:
    say(f"{C.yellow}WARN{C.off} {msg}")


def bad(msg: str) -> None:
    say(f"{C.red}FAIL{C.off} {msg}")


def step(msg: str) -> None:
    print(f"\n{C.bold}{msg}{C.off}", flush=True)


# ---------------------------------------------------------------- probes


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.6) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status < 400
    except (urllib.error.URLError, OSError):
        return False


def settings() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def remember(**kv) -> None:
    """Persist a setting so the next launch does not have to ask again."""
    data = settings()
    data.update(kv)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- comfyui


def comfy_python(root: Path) -> tuple[Path, list[str]] | None:
    """The interpreter and arguments that start ComfyUI from `root`.

    Two layouts exist and they are not interchangeable. The portable build
    ships its own embedded Python and keeps the source in a subfolder; a git
    clone runs on whatever Python is on PATH.
    """
    embedded = root / "python_embeded" / "python.exe"
    if embedded.exists() and (root / "ComfyUI" / "main.py").exists():
        return embedded, ["-s", str(root / "ComfyUI" / "main.py")]
    if (root / "main.py").exists():
        return Path(sys.executable), [str(root / "main.py")]
    return None


def find_comfy() -> Path | None:
    for candidate in (
        os.environ.get("MANHUA_COMFY"),
        settings().get("comfy_dir"),
        *COMFY_GUESSES,
    ):
        if candidate and comfy_python(Path(candidate)):
            return Path(candidate)
    return None


def start_comfy(root: Path) -> bool:
    """Spawn ComfyUI in its own window and wait for it to answer."""
    found = comfy_python(root)
    if not found:
        return False
    exe, args = found

    # --lowvram is what lets Anima (2B DiT + Qwen-3 encoder + Qwen VAE) fit on
    # a 6GB card. ComfyUI's own run_nvidia_gpu.bat does not pass it, and
    # without it generation slows to a crawl instead of failing outright,
    # which is a much more confusing symptom.
    cmd = [str(exe), *args, "--lowvram", "--port", str(COMFY_PORT)]
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    subprocess.Popen(cmd, cwd=str(root), creationflags=flags)

    say(f"{C.dim}waiting for ComfyUI to load its models...{C.off}")
    for i in range(120):
        if http_ok(f"http://127.0.0.1:{COMFY_PORT}/system_stats"):
            return True
        time.sleep(1)
        if i and i % 20 == 0:
            say(f"{C.dim}still starting ({i}s) - first launch is slower{C.off}")
    return False


def offer_comfy_install() -> None:
    """No ComfyUI anywhere. Say exactly what to do about it."""
    warn("ComfyUI was not found, so panels cannot be rendered.")
    print()
    say("The studio still runs: you can write, break down chapters and edit")
    say("the storyboard. Only rendering needs ComfyUI.")
    print()
    say(f"Get the Windows portable build: {C.cyan}{RELEASE}{C.off}")
    say("Download ComfyUI_windows_portable_nvidia.7z, extract it to C:\\ComfyUI,")
    say("and run this file again. It will be found automatically.")
    print()
    say("Already installed somewhere else? Tell it where:")
    say(f'{C.dim}set MANHUA_COMFY=D:\\path\\to\\ComfyUI{C.off}')
    print()
    try:
        if input("  Open the download page now? [Y/n] ").strip().lower() not in ("n", "no"):
            import webbrowser
            webbrowser.open(RELEASE)
    except (EOFError, KeyboardInterrupt):
        pass


def ensure_comfy() -> None:
    step("[2/3] Render engine")
    if port_open(COMFY_PORT):
        ok(f"ComfyUI already running on port {COMFY_PORT}")
        return

    root = find_comfy()
    if root is None:
        offer_comfy_install()
        return

    ok(f"found ComfyUI at {root}")
    remember(comfy_dir=str(root))
    if start_comfy(root):
        ok("ComfyUI is up")
    else:
        bad("ComfyUI did not answer in two minutes.")
        say("Check the window it opened for the real error. Common causes:")
        say("  - another program is already using port 8188")
        say("  - the GPU driver needs a restart after an update")


# ---------------------------------------------------------------- ollama


def check_ollama() -> None:
    step("[3/3] Script model")
    if not port_open(OLLAMA_PORT):
        warn("Ollama is not running, so chapters cannot be broken into panels.")
        say("Start it, or install it from https://ollama.com")
        say(f"Then: {C.cyan}ollama pull qwen2.5:7b{C.off}")
        return

    want = settings().get("ollama_model", "qwen2.5:7b")
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=3
        ) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        warn("Ollama is running but did not answer. Carrying on.")
        return

    # Ollama reports "qwen2.5:7b"; a bare "qwen2.5" in settings should match it.
    if any(n == want or n.split(":")[0] == want.split(":")[0] for n in names):
        ok(f"{want} ready")
    elif names:
        warn(f"{want} is not pulled. Installed: {', '.join(names[:4])}")
        say(f"Either {C.cyan}ollama pull {want}{C.off}, or set another in "
            "workspace/settings.json")
    else:
        warn(f"Ollama has no models. Run: {C.cyan}ollama pull {want}{C.off}")


# ---------------------------------------------------------------- deps


def check_deps() -> bool:
    step("[1/3] Studio")
    missing = []
    for mod, pkg in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                     ("PIL", "pillow"), ("yaml", "pyyaml"), ("pydantic", "pydantic")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        bad(f"missing packages: {', '.join(missing)}")
        say(f"Run: {C.cyan}.venv\\Scripts\\pip install -r requirements.txt{C.off}")
        return False
    ok("dependencies present")
    return True


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    os.chdir(ROOT)

    print(f"\n{C.bold}  Manhua Studio{C.off}  {C.dim}prose to long-strip manhua{C.off}")
    print(f"  {C.dim}by Clyde Xander Cielo  -  lacyan.me{C.off}")

    if not check_deps():
        return 1
    if "--no-comfy" not in argv:
        ensure_comfy()
    check_ollama()

    if port_open(STUDIO_PORT):
        step("Already open")
        say(f"Something is serving port {STUDIO_PORT} - probably the studio "
            "in another window.")
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{STUDIO_PORT}")
        return 0

    step("Opening the editor")
    say(f"{C.cyan}http://127.0.0.1:{STUDIO_PORT}{C.off}")
    say(f"{C.dim}Leave this window open. Closing it stops the studio.{C.off}")
    print()

    import threading
    import webbrowser

    import uvicorn

    from .studio.server import create_app

    threading.Timer(1.2, lambda: webbrowser.open(
        f"http://127.0.0.1:{STUDIO_PORT}")).start()
    uvicorn.run(
        create_app("workspace", "comfy", f"127.0.0.1:{COMFY_PORT}"),
        host="127.0.0.1", port=STUDIO_PORT, log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

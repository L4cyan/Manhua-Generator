"""Zero-configuration startup.

The goal: clone, double-click, working studio. Everything here is a *detection*
with a sane fallback, never a prompt. Anything detected can be overridden in
`workspace/settings.json`, which is written on first run so there is always a
file to edit rather than undocumented magic.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Where SDXL checkpoints commonly live, in priority order.
MODEL_SEARCH = [
    Path("models/checkpoints"),
    Path("C:/comfyui/ComfyUI_windows_portable/ComfyUI/models/checkpoints"),
    Path("C:/ComfyUI/models/checkpoints"),
    Path("C:/ComfyUI/ComfyUI/models/checkpoints"),
    Path.home() / "ComfyUI" / "models" / "checkpoints",
    Path("C:/StabilityMatrix/Data/Models/StableDiffusion"),
    Path("C:/stable-diffusion-webui/models/Stable-diffusion"),
]

# Checkpoint name fragments that indicate an Illustrious-family anime model,
# which is what the default style lock is tuned for. Ranked best-first.
ILLUSTRIOUS_HINTS = [
    "novaanime", "illustrious", "noobai", "waiillustrious", "wai",
    "nova", "anime", "animagine",
]


@dataclass
class Settings:
    """Resolved runtime configuration."""

    backend: str = "native"              # native | remote | comfy | mock
    checkpoint: str = ""
    # Cloud worker (see kaggle/worker.py). Set backend to "remote" to use it.
    remote_url: str = ""
    remote_token: str = ""
    lora_dir: str = ""
    comfy_host: str = "127.0.0.1:8188"
    vram_gb: float = 0.0
    low_vram: bool = True
    script_provider: str = "auto"        # auto | ollama | claude
    ollama_model: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Settings | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("notes", None)
            return cls(**data)
        except Exception:
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# ---------------------------------------------------------------- probes


def detect_vram() -> float:
    """Total VRAM in GB, or 0.0 when there is no usable NVIDIA GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass

    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            if out:
                return float(out[0]) / 1024
        except Exception:
            pass
    return 0.0


def find_checkpoints() -> list[Path]:
    """Every SDXL-looking checkpoint on the machine, best candidate first."""
    found: list[Path] = []
    for d in MODEL_SEARCH:
        if not d.is_dir():
            continue
        for f in d.glob("*.safetensors"):
            # Skip stub files the ComfyUI installer leaves behind.
            if f.stat().st_size > 1_000_000_000:
                found.append(f)

    def rank(p: Path) -> tuple[int, float]:
        name = p.name.lower().replace("_", "").replace("-", "")
        for i, hint in enumerate(ILLUSTRIOUS_HINTS):
            if hint in name:
                return (i, -p.stat().st_size)
        return (len(ILLUSTRIOUS_HINTS), -p.stat().st_size)

    return sorted(found, key=rank)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- resolution


def autoconfigure(workspace: Path = Path("workspace"), force: bool = False) -> Settings:
    """Work out a runnable configuration with no user input.

    Saved settings win unless `force`, so a user's manual edit is never
    silently overwritten by re-detection.
    """
    path = workspace / "settings.json"
    if not force:
        existing = Settings.load(path)
        if existing:
            return existing

    s = Settings()
    s.vram_gb = detect_vram()

    checkpoints = find_checkpoints()
    if checkpoints:
        best = checkpoints[0]
        s.checkpoint = str(best)
        loras = best.parent.parent / "loras"
        s.lora_dir = str(loras) if loras.is_dir() else ""
        s.notes.append(f"found {len(checkpoints)} checkpoint(s); using {best.name}")
    else:
        s.notes.append("no checkpoints found - falling back to the mock backend")

    if not s.checkpoint:
        s.backend = "mock"
    elif not torch_available():
        # Models are present but torch is not: ComfyUI can still drive them.
        s.backend = "comfy"
        s.notes.append("torch not installed - using ComfyUI backend instead of native")
    else:
        s.backend = "native"

    # Under ~10GB, sequential CPU offload is what keeps SDXL from OOMing.
    s.low_vram = s.vram_gb < 10.0
    if s.vram_gb:
        s.notes.append(f"{s.vram_gb:.1f}GB VRAM -> low_vram={s.low_vram}")

    try:
        from .script.providers import detect as detect_provider

        provider, model = detect_provider()
        s.script_provider = provider
        s.ollama_model = model if provider == "ollama" else ""
        s.notes.append(f"script provider: {provider} ({model})")
    except Exception:
        s.script_provider = "auto"
        s.notes.append("no script provider yet - install Ollama or set ANTHROPIC_API_KEY")

    s.save(path)
    return s


def build_backend(s: Settings):
    """Instantiate the render backend described by settings."""
    if s.backend == "remote":
        if not s.remote_url:
            raise RuntimeError(
                "backend is 'remote' but remote_url is empty.\n"
                "Start kaggle/worker.py in a notebook and paste the tunnel URL "
                "into workspace/settings.json."
            )
        from .render.remote import RemoteBackend

        return RemoteBackend(s.remote_url, s.remote_token)

    if s.backend == "mock":
        from .render.mock import MockBackend

        return MockBackend()
    if s.backend == "comfy":
        from .render.comfy import ComfyBackend

        return ComfyBackend(host=s.comfy_host)

    from .render.native import NativeBackend

    return NativeBackend(
        s.checkpoint,
        lora_dir=s.lora_dir or None,
        low_vram=s.low_vram,
    )


def apply_to_style(style, s: Settings) -> None:
    """Point a style lock at the detected checkpoint.

    The style file ships with a placeholder checkpoint name; without this a
    fresh clone would fail on a filename the user never chose.
    """
    if s.backend == "native" and s.checkpoint:
        style.render.checkpoint = s.checkpoint

    # Drop a configured style LoRA that is not actually on disk, rather than
    # failing the first render with a confusing FileNotFoundError.
    if style.render.style_lora and s.lora_dir:
        if not (Path(s.lora_dir) / style.render.style_lora).exists():
            style.render.style_lora = None

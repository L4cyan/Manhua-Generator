"""Kaggle API driver for batch cloud jobs.

Drives the Python API rather than shelling out to the `kaggle` binary, because
that binary fails on machines with antivirus HTTPS inspection: it verifies TLS
against certifi's roots, which lack the AV's injected root certificate. Here
`truststore` is installed *before* the kaggle import, so verification goes
through the OS trust store and works on those machines.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ..net import enable_system_certs


def api():
    """Authenticated Kaggle API client.

    `truststore` must be injected before `kaggle` is imported, since the
    kaggle package builds its HTTP session at import time.
    """
    enable_system_certs()
    import kaggle

    kaggle.api.authenticate()
    return kaggle.api


def username() -> str:
    """The account the token belongs to."""
    import os

    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text()).get("username", "")
    return ""


# ---------------------------------------------------------------- datasets


def upload_dir(
    folder: Path,
    slug: str,
    title: str,
    user: str,
    *,
    private: bool = True,
) -> str:
    """Upload a directory as a private dataset. Returns its `user/slug` ref.

    Creates the dataset on first call and adds a version on later ones, so
    re-running after re-culling a sheet does not error out.
    """
    folder = Path(folder)
    meta = folder / "dataset-metadata.json"
    ref = f"{user}/{slug}"
    meta.write_text(
        json.dumps({"title": title, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=2),
        encoding="utf-8",
    )

    a = api()
    try:
        a.dataset_create_new(str(folder), public=not private, dir_mode="zip")
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise
        a.dataset_create_version(str(folder), version_notes="update", dir_mode="zip")
    return ref


# ---------------------------------------------------------------- kernels


def push_kernel(folder: Path, dataset_refs: list[str] | None = None) -> str:
    """Push a script/notebook kernel. Returns its `user/slug` ref."""
    folder = Path(folder)
    meta_path = folder / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if dataset_refs is not None:
        meta["dataset_sources"] = dataset_refs
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    api().kernels_push(str(folder))
    return meta["id"]


def wait(ref: str, *, poll: int = 30, timeout: int = 43200, on_tick=None) -> str:
    """Block until a pushed kernel finishes. Returns its final status.

    Kaggle reports `running`, `complete`, `error` or `cancelAcknowledged`.
    """
    a = api()
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = a.kernels_status(ref)
            status = (resp.get("status") if isinstance(resp, dict)
                      else getattr(resp, "status", "")) or ""
        except Exception as exc:
            status = f"unknown ({type(exc).__name__})"

        if on_tick:
            on_tick(status, int(time.time() - start))
        if any(k in str(status).lower() for k in ("complete", "error", "cancel")):
            return str(status)
        time.sleep(poll)
    return "timeout"


def fetch_output(ref: str, dest: Path, pattern: str = "*.safetensors") -> list[Path]:
    """Download a finished kernel's output files matching `pattern`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    api().kernels_output(ref, str(dest))
    return sorted(dest.rglob(pattern))


def logs(ref: str, dest: Path) -> str:
    """Fetch a kernel's log, which is where a failed run explains itself."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api().kernels_output(ref, str(dest))
    except Exception as exc:
        return f"(could not fetch logs: {exc})"
    for name in ("*.log", "*log*.json", "*.txt"):
        for f in dest.rglob(name):
            return f.read_text(encoding="utf-8", errors="replace")[-4000:]
    return "(no log file found in kernel output)"


# ---------------------------------------------------------------- job


def train_character(
    sheet_dir: Path,
    checkpoint_ref: str,
    character: str,
    trigger: str,
    user: str,
    job_dir: Path,
    *,
    steps: int = 1600,
    on_status=None,
) -> Path | None:
    """Upload a culled sheet, push the training kernel, wait, fetch the LoRA."""
    sheet_dir = Path(sheet_dir)
    images = [p for p in sheet_dir.glob("*.png") if not p.name.startswith("_")]
    if len(images) < 8:
        raise RuntimeError(
            f"only {len(images)} images in {sheet_dir}. Run `manhua sheet` first, "
            "then cull to ~20-40 that clearly show the same person."
        )

    # Stage the sheet plus a config the kernel reads, so one kernel trains any
    # character without edits.
    staged = Path(job_dir) / "sheet_upload"
    if staged.exists():
        shutil.rmtree(staged)
    (staged / "sheet").mkdir(parents=True)
    for img in images:
        shutil.copy(img, staged / "sheet" / img.name)
    (staged / "train_config.json").write_text(
        json.dumps({"character": character, "trigger": trigger, "steps": steps}, indent=2),
        encoding="utf-8",
    )

    slug = f"manhua-sheet-{character.replace('_', '-')}"
    if on_status:
        on_status(f"uploading {len(images)} images as {user}/{slug}")
    sheet_ref = upload_dir(staged, slug, f"Manhua sheet: {character}", user)

    kernel_dir = Path(__file__).resolve().parents[2] / "cloud" / "train_lora"
    sources = [sheet_ref] + ([checkpoint_ref] if checkpoint_ref else [])
    if on_status:
        on_status(f"pushing kernel with sources {sources}")
    ref = push_kernel(kernel_dir, sources)

    if on_status:
        on_status(f"training on Kaggle: https://www.kaggle.com/code/{ref}")
    status = wait(ref, on_tick=lambda s, secs: on_status and on_status(f"[{secs//60}m] {s}"))

    if "complete" not in status.lower():
        raise RuntimeError(f"kernel finished with status '{status}'. Logs:\n"
                           + logs(ref, Path(job_dir) / "logs"))

    found = fetch_output(ref, Path(job_dir) / "output")
    return found[0] if found else None

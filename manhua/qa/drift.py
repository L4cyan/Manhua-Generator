"""Character drift detection.

The failure mode that kills long AI comics is gradual: panel 1 and panel 2
match, panel 2 and panel 3 match, and panel 60 is a different person. Nobody
notices while generating, because each step looks fine next to the last.

This module scores every rendered panel against the character's turnaround
sheet -- a fixed anchor, not the previous panel -- so error cannot accumulate.
Panels below threshold are re-rolled automatically.

Optional: requires `pip install -r requirements-qa.txt`. Without it the
pipeline runs unchanged, just without the safety net.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class DriftUnavailable(RuntimeError):
    """Raised when the optional CLIP dependencies are not installed."""


@functools.lru_cache(maxsize=1)
def _load_clip():
    try:
        import open_clip
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DriftUnavailable(
            "CLIP drift checking needs the optional extras:\n"
            "  pip install -r requirements-qa.txt "
            "--index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    # CPU by default: the GPU is busy rendering, and scoring one image on CPU
    # takes well under a second -- far less than the render it is gating.
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess, torch, device


def embed(images: list[Image.Image]) -> np.ndarray:
    """L2-normalised CLIP embeddings, one row per image."""
    model, preprocess, torch, device = _load_clip()
    batch = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


@functools.lru_cache(maxsize=32)
def sheet_anchor(sheet_dir: str) -> np.ndarray | None:
    """Mean embedding of a character's sheet: the fixed identity anchor.

    Cached because it is constant for the life of a chapter, and re-embedding
    24 reference images per panel would cost more than the render.
    """
    paths = sorted(
        p for p in Path(sheet_dir).glob("*.png") if not p.name.startswith("_")
    )
    if not paths:
        return None

    vecs = embed([Image.open(p) for p in paths])
    mean = vecs.mean(axis=0)
    return mean / np.linalg.norm(mean)


def score(panel: Image.Image, sheet_dir: str) -> float | None:
    """Cosine similarity of a panel against a character's sheet, in [-1, 1].

    Returns None when the character has no sheet, which means "cannot judge" --
    the caller must not treat that as a failure.
    """
    anchor = sheet_anchor(sheet_dir)
    if anchor is None:
        return None
    return float(embed([panel])[0] @ anchor)


def check_panel(
    panel_image: Image.Image,
    character_sheets: list[str],
    threshold: float = 0.78,
) -> tuple[bool, dict[str, float]]:
    """Score a panel against every character in it.

    Returns (passed, {sheet_dir: score}). A panel passes when every character
    with a sheet scores at or above the threshold. Characters without sheets
    are skipped rather than failed.

    Calibrate the threshold per project: absolute CLIP similarity varies with
    art style. Render ten panels you consider good, look at their scores, and
    set the threshold just below the lowest.
    """
    scores: dict[str, float] = {}
    for sheet in character_sheets:
        s = score(panel_image, sheet)
        if s is not None:
            scores[sheet] = s

    passed = all(v >= threshold for v in scores.values())
    return passed, scores


def render_with_gate(
    chapter,
    panel,
    backend,
    *,
    threshold: float = 0.78,
    max_rerolls: int = 3,
    on_reject=None,
) -> Path:
    """Render a panel, re-rolling while it drifts off-model.

    Falls back to a plain render if CLIP is unavailable, so the gate is a
    bonus rather than a hard dependency. Keeps the BEST attempt rather than
    the last: a reroll can easily come back worse.
    """
    sheets = [
        c.sheet_dir
        for ref in panel.characters
        if (c := chapter.project.bible.get(ref.id)) and c.sheet_dir
        and Path(c.sheet_dir).exists()
    ]

    path = chapter.render_panel(panel, backend)
    if not sheets:
        return path

    try:
        best_score = -1.0
        best_bytes = path.read_bytes()

        for attempt in range(max_rerolls + 1):
            passed, scores = check_panel(Image.open(path), sheets, threshold)
            worst = min(scores.values()) if scores else 1.0

            if worst > best_score:
                best_score, best_bytes = worst, path.read_bytes()
            if passed or attempt == max_rerolls:
                break

            if on_reject:
                on_reject(panel.id, attempt + 1, worst)
            path = chapter.render_panel(panel, backend, new_seed=True)

        # Restore the best attempt if the final reroll was not it.
        path.write_bytes(best_bytes)
    except DriftUnavailable:
        pass

    return path

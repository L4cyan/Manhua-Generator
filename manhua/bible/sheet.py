"""Character turnaround sheets.

A sheet is the input to identity-LoRA training and the reference the QA gate
scores drift against. Text description alone holds a face together for maybe
forty panels; a trained LoRA holds it for a chapter.

The workflow:

    1. `manhua sheet <project> <character>`  -> ~24 images
    2. Cull by hand. Keep only images that look like ONE person. This step
       matters more than any setting -- a sheet with three faces in it trains
       a LoRA that produces three faces.
    3. Train (kohya_ss / sd-scripts) using the emitted config.
    4. Put the resulting .safetensors in the character's `lora` field.

Every image is rendered through the project's own style lock, so the LoRA
learns the character *in the series' art style* rather than fighting it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import StyleLock
from ..models import Character
from ..render.base import Backend, RenderRequest

# Angles and framings a LoRA needs to generalise. Heavy on the head, because
# faces are what readers track and what drift shows up in first.
ANGLES: list[tuple[str, str]] = [
    ("front_close",      "close-up portrait, front view, facing viewer, neutral expression"),
    ("front_medium",     "medium shot, front view, facing viewer, arms at sides"),
    ("three_quarter_l",  "close-up portrait, three-quarter view facing left"),
    ("three_quarter_r",  "close-up portrait, three-quarter view facing right"),
    ("profile_l",        "close-up portrait, full side profile facing left"),
    ("profile_r",        "close-up portrait, full side profile facing right"),
    ("looking_up",       "close-up portrait, low angle, chin raised, looking up"),
    ("looking_down",     "close-up portrait, high angle, eyes downcast"),
    ("full_front",       "full body shot, front view, standing, head to toe"),
    ("full_back",        "full body shot, seen from behind, standing"),
    ("full_three_q",     "full body shot, three-quarter view, standing"),
    ("upper_turn",       "medium shot, turning to look over the shoulder at viewer"),
]

EXPRESSIONS: list[tuple[str, str]] = [
    ("calm",      "calm neutral expression"),
    ("cold",      "cold detached stare, narrowed eyes"),
    ("smirk",     "slight confident smirk"),
    ("shock",     "wide-eyed shock, parted lips"),
    ("anger",     "furious glare, furrowed brows"),
    ("pain",      "gritted teeth, pained grimace"),
    ("sorrow",    "downcast sorrowful expression"),
    ("laugh",     "open laughing expression, eyes closed"),
    ("determine", "determined resolute expression, set jaw"),
    ("weary",     "exhausted expression, half-lidded eyes"),
    ("smile",     "gentle warm smile"),
    ("surprise",  "eyebrows raised in surprise"),
]

# A sheet must not inherit the series' dramatic lighting or effects: a LoRA
# trained on rim-lit, qi-wreathed shots bakes those into the character and
# they show up in every panel afterwards, including calm indoor ones.
SHEET_NEUTRALISER = "plain neutral grey background, flat even studio lighting, no special effects"
SHEET_NEGATIVE_EXTRA = (
    "dramatic lighting, rim lighting, glowing aura, light particles, lens flare, "
    "busy background, scenery, multiple people, cropped"
)


def sheet_requests(
    character: Character,
    style: StyleLock,
    *,
    base_seed: int | None = None,
    angles: bool = True,
    expressions: bool = True,
    outfits: list[str] | None = None,
) -> list[tuple[str, RenderRequest]]:
    """Build the render requests for one character's sheet.

    Seeds are derived from a single base so a sheet is reproducible: rerunning
    with the same base regenerates the same sheet, which makes it possible to
    tweak the appearance text and see exactly what changed.
    """
    base_seed = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)
    sex_tag = "1girl" if character.sex == "female" else "1boy"

    plan: list[tuple[str, str]] = []
    if angles:
        plan += [(f"angle_{n}", t) for n, t in ANGLES]
    if expressions:
        plan += [(f"expr_{n}", f"close-up portrait, front view, {t}") for n, t in EXPRESSIONS]

    # Cover every wardrobe the character has, alternating across the plan.
    # One LoRA per person, not per outfit: training separate LoRAs for a
    # professor form and a cultivator form produces two people who slowly
    # diverge. Showing the same face in different clothes teaches the model
    # that the face is the constant and the clothing is not, so the outfit
    # stays promptable afterwards.
    wardrobe = outfits or list(character.outfits.values()) or [character.default_outfit]
    wardrobe = [w for w in wardrobe if w] or [""]

    out: list[tuple[str, RenderRequest]] = []
    for i, (name, framing) in enumerate(plan):
        outfit = wardrobe[i % len(wardrobe)]
        name = f"{name}__w{i % len(wardrobe)}"
        appearance = character.appearance_prompt(include_outfit=False)
        content = ", ".join(
            [framing, sex_tag, appearance, outfit, SHEET_NEUTRALISER]
        )
        loras: list[tuple[str, float]] = []
        if style.render.style_lora:
            loras.append((style.render.style_lora, style.render.style_lora_weight))

        out.append(
            (
                name,
                RenderRequest(
                    # register="neutral" deliberately: the default is
                    # "cultivation", whose genre clause ("xianxia wuxia eastern
                    # fantasy cultivation") overpowers the outfit text and put
                    # the character in robes even for the modern wardrobe. A
                    # reference sheet wants the person, not the genre.
                    positive=style.positive(content, register="neutral"),
                    negative=f"{style.negative}, {SHEET_NEGATIVE_EXTRA}",
                    width=832,
                    height=1216,
                    seed=base_seed + i,
                    loras=loras,
                    style=style,
                ),
            )
        )
    return out


def render_sheet(
    character: Character,
    style: StyleLock,
    backend: Backend,
    out_dir: Path,
    *,
    base_seed: int | None = None,
    on_progress=None,
) -> list[Path]:
    """Render a full sheet to `out_dir`. Returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, req in sheet_requests(character, style, base_seed=base_seed):
        if on_progress:
            on_progress(name)
        img = backend.render(req)
        path = out_dir / f"{character.id}_{name}.png"
        img.save(path)
        written.append(path)

    (out_dir / "_seed.json").write_text(
        json.dumps({"base_seed": req.seed - len(written) + 1, "count": len(written)}, indent=2),
        encoding="utf-8",
    )
    return written


def write_training_config(
    character: Character,
    style: StyleLock,
    sheet_dir: Path,
    out_dir: Path,
) -> Path:
    """Emit a kohya_ss / sd-scripts config for the culled sheet.

    Values are tuned for a small SDXL character LoRA (20-40 images) on limited
    VRAM. Training at 6GB is possible but slow; renting an A100 for a quarter
    of an hour is usually the better trade.
    """
    trigger = character.trigger or f"{character.id}_char"
    config = {
        "_comment": (
            f"Identity LoRA for {character.name}. Cull the sheet FIRST -- keep only "
            "images that read as one person. Then caption each image starting with "
            f"the trigger word '{trigger}'."
        ),
        "pretrained_model_name_or_path": style.render.checkpoint,
        "train_data_dir": str(sheet_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "output_name": f"{character.id}_v1",
        "resolution": "1024,1024",
        "network_module": "networks.lora",
        "network_dim": 32,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "unet_lr": 1e-4,
        "text_encoder_lr": 5e-5,
        "lr_scheduler": "cosine_with_restarts",
        "lr_warmup_steps": 0,
        "train_batch_size": 1,
        "max_train_epochs": 12,
        "save_every_n_epochs": 2,
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "optimizer_type": "AdamW8bit",
        "xformers": True,
        "gradient_checkpointing": True,
        "cache_latents": True,
        "no_half_vae": True,
        "seed": 42,
        "_trigger_word": trigger,
        "_notes": [
            "Caption format: '<trigger>, <framing>, <expression>' -- do NOT caption the",
            "permanent appearance. Anything you caption becomes editable; anything you",
            "omit gets baked into the trigger, which is exactly what you want for a face.",
            "gradient_checkpointing + AdamW8bit is what makes this fit in 6GB; drop both",
            "if training on a rented card and it will run several times faster.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{character.id}_lora.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path

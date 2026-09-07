"""Character identity LoRA training - runs as a Kaggle script kernel.

Batch-shaped work: push it, walk away, collect a .safetensors. That is what
the Kaggle CLI is genuinely good at, unlike a long-lived tunnel server.

Inputs (attached as Kaggle datasets):
    /kaggle/input/<sheet-dataset>/     culled turnaround sheet, ~20-40 PNGs
    /kaggle/input/<model-dataset>/     the SDXL checkpoint, or downloaded here

Output:
    /kaggle/working/<character>_v1.safetensors

Captioning matters more than any hyperparameter here: caption ONLY what should
stay editable (framing, expression). Anything you leave uncaptioned gets bound
to the trigger word, which is exactly what you want for a face.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------- configuration
# Overridden via /kaggle/input/<config>/train_config.json when present, so the
# same kernel trains any character without editing this file.

CONFIG = {
    "character": "ling_yan",
    "trigger": "lingyan1n",
    "sheet_dir": "",            # auto-detected if empty
    "checkpoint": "",           # auto-detected if empty
    "resolution": 1024,
    "rank": 32,
    "alpha": 16,
    "learning_rate": 1e-4,
    "text_encoder_lr": 5e-5,
    "steps": 1600,
    "batch_size": 1,
    "seed": 42,
}

for cfg in Path("/kaggle/input").glob("*/train_config.json"):
    CONFIG |= json.loads(cfg.read_text())
    print(f"loaded config from {cfg}")
    break


def find_one(patterns: list[str], root: str = "/kaggle/input") -> str | None:
    for pat in patterns:
        hits = sorted(Path(root).glob(pat))
        if hits:
            return str(hits[0])
    return None


def main() -> None:
    sheet = CONFIG["sheet_dir"] or find_one(["*/sheet", "*/*/sheet", "*"])
    ckpt = CONFIG["checkpoint"] or find_one(["*/*.safetensors", "*/*/*.safetensors"])

    if not sheet or not Path(sheet).is_dir():
        sys.exit(f"no sheet directory found (looked under /kaggle/input): {sheet}")
    images = sorted(p for p in Path(sheet).glob("*.png") if not p.name.startswith("_"))
    if len(images) < 8:
        sys.exit(f"only {len(images)} images in {sheet}; need at least ~8 (20-40 ideal)")
    if not ckpt:
        sys.exit("no .safetensors checkpoint found under /kaggle/input")

    print(f"character : {CONFIG['character']}")
    print(f"trigger   : {CONFIG['trigger']}")
    print(f"sheet     : {sheet}  ({len(images)} images)")
    print(f"checkpoint: {ckpt}", flush=True)

    # ---------- build the training set ----------
    # diffusers' DreamBooth-LoRA script reads an instance dir plus per-image
    # caption files. Captions are derived from the sheet filenames, which
    # already encode angle and expression (e.g. ling_yan_angle_profile_l.png).
    train_dir = Path("/kaggle/working/train")
    train_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    for img in images:
        dst = train_dir / img.name
        shutil.copy(img, dst)

        stem = img.stem
        bits: list[str] = [CONFIG["trigger"]]
        if "_angle_" in stem:
            bits.append(stem.split("_angle_")[-1].replace("_", " "))
        elif "_expr_" in stem:
            bits.append(stem.split("_expr_")[-1].replace("_", " ") + " expression")
        # Deliberately NOT captioning hair, eyes, build or clothing: uncaptioned
        # attributes bind to the trigger, which is how the identity gets locked.
        dst.with_suffix(".txt").write_text(", ".join(bits), encoding="utf-8")

    print(f"prepared {len(images)} captioned images in {train_dir}", flush=True)
    print("sample caption:", (train_dir / images[0].name).with_suffix(".txt").read_text())

    # ---------- fetch the training script ----------
    script = Path("/kaggle/working/train_dreambooth_lora_sdxl.py")
    if not script.exists():
        subprocess.run(
            ["wget", "-q", "-O", str(script),
             "https://raw.githubusercontent.com/huggingface/diffusers/main/"
             "examples/dreambooth/train_dreambooth_lora_sdxl.py"],
            check=True,
        )

    out_dir = Path("/kaggle/working/lora_out")
    cmd = [
        sys.executable, str(script),
        "--pretrained_model_name_or_path", ckpt,
        "--instance_data_dir", str(train_dir),
        "--output_dir", str(out_dir),
        "--instance_prompt", CONFIG["trigger"],
        "--caption_column", "text",
        "--resolution", str(CONFIG["resolution"]),
        "--train_batch_size", str(CONFIG["batch_size"]),
        "--gradient_accumulation_steps", "1",
        "--learning_rate", str(CONFIG["learning_rate"]),
        "--text_encoder_lr", str(CONFIG["text_encoder_lr"]),
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(CONFIG["steps"]),
        "--rank", str(CONFIG["rank"]),
        "--seed", str(CONFIG["seed"]),
        "--mixed_precision", "fp16",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--train_text_encoder",
        "--checkpointing_steps", "99999",     # only the final weights matter
    ]
    print("\n" + " ".join(cmd) + "\n", flush=True)
    subprocess.run(cmd, check=True)

    # ---------- name the artifact for the bible ----------
    produced = sorted(out_dir.glob("*.safetensors"))
    if not produced:
        sys.exit(f"training finished but no .safetensors in {out_dir}")

    final = Path(f"/kaggle/working/{CONFIG['character']}_v1.safetensors")
    shutil.copy(produced[0], final)
    print(f"\nDONE -> {final}  ({final.stat().st_size / 1e6:.0f} MB)")
    print(f"\nAdd to characters.yaml under {CONFIG['character']}:")
    print(f"    lora: {final.name}")
    print(f"    trigger: {CONFIG['trigger']}")


if __name__ == "__main__":
    main()

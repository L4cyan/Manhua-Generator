import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from pathlib import Path
from huggingface_hub import hf_hub_download

DEST = Path("C:/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/diffusion_models")
NAME = "split_files/diffusion_models/anima-turbo-v1.1.safetensors"
target = DEST / "anima-turbo-v1.1.safetensors"

if target.exists() and target.stat().st_size > 1_000_000:
    print(f"have it ({target.stat().st_size/1e9:.1f} GB)"); sys.exit()

for attempt in range(1, 41):
    try:
        got = Path(hf_hub_download("circlestone-labs/Anima", filename=NAME, local_dir=str(DEST)))
        if got != target:
            got.replace(target)
        print(f"OK {target.name} ({target.stat().st_size/1e9:.1f} GB)", flush=True)
        break
    except Exception as exc:
        if "404" in str(exc):
            sys.exit("not found")
        print(f"  attempt {attempt}: {type(exc).__name__}", flush=True)
        time.sleep(2)

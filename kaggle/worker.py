"""GPU render worker - runs inside a Kaggle or Colab notebook.

Pairs with `manhua/render/remote.py` on your machine. The studio, compositor,
lettering and export all stay local; only the diffusion step moves to the
rented GPU. That means your stories, character bible and rendered chapters
never leave your machine -- the worker only ever sees a prompt string.

It deliberately reuses the same NativeBackend as local rendering, so a given
seed produces the same panel in either place.

Usage inside a notebook cell:

    !pip -q install diffusers transformers accelerate safetensors peft fastapi uvicorn nest_asyncio
    !wget -q <raw url>/kaggle/worker.py
    %run worker.py --checkpoint /kaggle/input/<dataset>/model.safetensors
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import threading
import time

# --------------------------------------------------------------------------
# Minimal standalone copies of the two things the worker needs from the main
# package, so the notebook can run this file on its own without cloning the
# whole repo.
# --------------------------------------------------------------------------

RENDER_DEFAULTS = {
    "sampler": "dpmpp_2m_sde",
    "scheduler": "karras",
    "steps": 30,
    "cfg": 4.5,
    "clip_skip": 2,
}


def build_app(backend, token: str):
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    class HiresCfg(BaseModel):
        enabled: bool = True
        scale: float = 1.5
        denoise: float = 0.4
        steps: int = 12

    class RenderReq(BaseModel):
        positive: str
        negative: str = ""
        width: int = 832
        height: int = 1216
        seed: int = 0
        loras: list[list] = []
        steps: int = 30
        cfg: float = 4.5
        clip_skip: int = 2
        hires: HiresCfg = HiresCfg()

    app = FastAPI(title="Manhua render worker")
    lock = threading.Lock()

    def check(tok: str | None) -> None:
        if token and tok != token:
            raise HTTPException(401, "bad token")

    @app.get("/health")
    def health(x_auth_token: str | None = Header(None)):
        check(x_auth_token)
        info = {"ok": True, "checkpoint": os.path.basename(backend.checkpoint)}
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                info |= {
                    "gpu": torch.cuda.get_device_name(0),
                    "vram_free_mb": free // 1024**2,
                    "vram_total_mb": total // 1024**2,
                }
        except Exception:
            pass
        return info

    @app.post("/render")
    def render(req: RenderReq, x_auth_token: str | None = Header(None)):
        check(x_auth_token)

        from manhua.config import HiresCfg as HC, RenderCfg, StyleLock
        from manhua.render.base import RenderRequest

        # Sampler settings arrive per request, so the client's style lock stays
        # the single source of truth and the worker holds no config of its own.
        # The prompt text is already fully assembled, so the style's text
        # fields are unused here and left empty on purpose.
        style = StyleLock(
            name="remote",
            style_lead="", style_body="", style_fx="", positive_suffix="",
            negative="",
            render=RenderCfg(
                checkpoint=backend.checkpoint,
                steps=req.steps,
                cfg=req.cfg,
                clip_skip=req.clip_skip,
            ),
            hires=HC(**req.hires.model_dump()),
            aspects={},
        )

        r = RenderRequest(
            positive=req.positive,
            negative=req.negative,
            width=req.width,
            height=req.height,
            seed=req.seed,
            loras=[tuple(x) for x in req.loras],
            style=style,
        )

        # One GPU: serialise, exactly as the local studio does.
        with lock:
            t0 = time.time()
            img = backend.render(r)
            dt = time.time() - t0

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "seconds": round(dt, 1),
            "size": [img.width, img.height],
        }

    return app


def start_tunnel(port: int) -> str | None:
    """Expose the local port with a cloudflared quick tunnel.

    Quick tunnels need no account and no token, which matters because the
    point of this is to stay free.
    """
    try:
        subprocess.run(
            "wget -q -O /tmp/cloudflared "
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64 && chmod +x /tmp/cloudflared",
            shell=True, check=True,
        )
    except subprocess.CalledProcessError:
        print("could not download cloudflared", file=sys.stderr)
        return None

    proc = subprocess.Popen(
        ["/tmp/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    # cloudflared prints the assigned hostname to stderr shortly after start.
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            continue
        if "trycloudflare.com" in line:
            for word in line.split():
                if word.startswith("https://") and "trycloudflare.com" in word:
                    return word.strip().strip("|,")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lora-dir", default=None)
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--token", default=os.environ.get("MANHUA_TOKEN", ""))
    ap.add_argument("--no-tunnel", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from manhua.render.native import NativeBackend

    # 16GB on Kaggle: keep everything resident, which is the whole speed win.
    backend = NativeBackend(args.checkpoint, lora_dir=args.lora_dir, low_vram=False)
    print("loading checkpoint...", flush=True)
    _ = backend.pipe
    print("ready", flush=True)

    app = build_app(backend, args.token)

    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()

    def serve() -> None:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

    threading.Thread(target=serve, daemon=True).start()
    time.sleep(3)

    if not args.no_tunnel:
        url = start_tunnel(args.port)
        if url:
            print("\n" + "=" * 62)
            print("  PASTE THIS INTO workspace/settings.json ON YOUR MACHINE:")
            print(f'    "backend": "remote",')
            print(f'    "remote_url": "{url}",')
            print(f'    "remote_token": "{args.token}"')
            print("=" * 62 + "\n", flush=True)
        else:
            print("tunnel failed; use --no-tunnel and expose the port yourself")

    # Hold the notebook cell open so the worker keeps serving.
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

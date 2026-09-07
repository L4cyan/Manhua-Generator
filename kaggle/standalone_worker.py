"""Single-file GPU render worker for Kaggle / Colab.

Self-contained on purpose: no repo clone, no pip install of this project. Paste
it into one notebook cell and run. It mirrors `manhua/render/native.py` -- same
CLIP chunking, same weighting, same VAE handling -- so a seed rendered here
matches one rendered locally.

Keep this in sync with manhua/render/native.py if you change encoding there.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

# --------------------------------------------------------------------- prompt


_WEIGHT_RE = re.compile(r"\(([^():]+?)(?::([0-9.]+))?\)|\[([^\[\]]+?)\]")


def parse_weights(text: str) -> list[tuple[str, float]]:
    """Split an A1111-style prompt into (fragment, weight) pairs."""
    out: list[tuple[str, float]] = []
    pos = 0
    for m in _WEIGHT_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], 1.0))
        if m.group(3) is not None:
            out.append((m.group(3), 1 / 1.1))
        else:
            out.append((m.group(1), float(m.group(2)) if m.group(2) else 1.1))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], 1.0))
    return [(t, w) for t, w in out if t.strip()]


# --------------------------------------------------------------------- config


@dataclass
class Req:
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    clip_skip: int
    sampler: str = "euler_a"
    scheduler: str = ""
    loras: list = field(default_factory=list)
    hires_enabled: bool = False
    hires_scale: float = 1.5
    hires_denoise: float = 0.4
    hires_steps: int = 12


# Sampler name -> diffusers scheduler. The style lock names a sampler but
# diffusers ignores it unless the scheduler is swapped explicitly, so without
# this every render silently used whatever the checkpoint shipped with.
def apply_scheduler(pipe, sampler: str, scheduler: str = "") -> None:
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
    )

    karras = "karras" in (scheduler or "").lower()
    name = (sampler or "").lower()
    cfg = pipe.scheduler.config
    try:
        if name in ("euler_a", "euler_ancestral", "euler a"):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
        elif name == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                cfg, use_karras_sigmas=karras)
        elif name in ("dpmpp_2m_sde", "dpmpp_2m"):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                cfg, use_karras_sigmas=karras,
                algorithm_type="sde-dpmsolver++" if "sde" in name else "dpmsolver++")
    except Exception:
        pass                     # keep the checkpoint default rather than fail


# --------------------------------------------------------------------- engine


class Engine:
    def __init__(self, checkpoint: str, lora_dir: str | None = None,
                 low_vram: bool = False):
        self.checkpoint = checkpoint
        self.lora_dir = lora_dir
        self.low_vram = low_vram
        self._pipe = None
        self._img2img = None
        self._loras: tuple = ()

    @property
    def pipe(self):
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionXLPipeline

            p = StableDiffusionXLPipeline.from_single_file(
                self.checkpoint, torch_dtype=torch.float16,
                use_safetensors=True, add_watermarker=False,
            )
            p.set_progress_bar_config(disable=True)
            # Leave the VAE in fp16 and let diffusers upcast at decode. Casting
            # it to fp32 here disables that path and yields a dtype mismatch.
            p.vae.config.force_upcast = True
            if self.low_vram:
                p.enable_model_cpu_offload()
            else:
                p.to("cuda")
            # Always on, even with plenty of VRAM: the hires pass decodes a
            # ~1250x1825 latent, and an untiled VAE decode at that size asks
            # for >2GB in one allocation and OOMs a 16GB T4.
            p.enable_vae_tiling()
            p.enable_vae_slicing()
            try:
                p.enable_attention_slicing()
            except Exception:
                pass
            self._pipe = p
        return self._pipe

    def _restore_vae_dtype(self) -> None:
        """Put the VAE back to fp16 after a decode.

        Diffusers upcasts the VAE to fp32 for decoding and leaves it that way.
        In a long-lived worker that poisons every later request: the upcast is
        guarded by `vae.dtype == float16`, so once it is fp32 the guard is
        False, the latents are no longer cast to match, and the next decode
        dies with 'Input type (c10::Half) and bias type (float)'.
        """
        import torch

        try:
            if self._pipe is not None and self._pipe.vae.dtype != torch.float16:
                self._pipe.vae.to(torch.float16)
        except Exception:
            pass

    def _sync_loras(self, loras: list) -> None:
        want = tuple(tuple(x) for x in loras)
        if want == self._loras:
            return
        pipe = self.pipe
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        names, weights = [], []
        for i, (fname, weight) in enumerate(want):
            path = os.path.join(self.lora_dir, fname) if self.lora_dir else fname
            if not os.path.exists(path):
                raise FileNotFoundError(f"LoRA not found: {path}")
            adapter = f"a{i}"
            pipe.load_lora_weights(os.path.dirname(path),
                                   weight_name=os.path.basename(path),
                                   adapter_name=adapter)
            names.append(adapter)
            weights.append(float(weight))
        if names:
            pipe.set_adapters(names, adapter_weights=weights)
        self._loras = want

    def _encode(self, text: str, clip_skip: int):
        """Encode a prompt of any length. CLIP takes 77 tokens at a time, so
        the prompt is chunked into 75-token windows and concatenated."""
        import torch

        pipe = self.pipe
        device = pipe._execution_device
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
        encoders = [pipe.text_encoder, pipe.text_encoder_2]
        fragments = parse_weights(text)

        per_encoder, pooled = [], None
        for tk, te in zip(tokenizers, encoders):
            ids, wts_all = [], []
            for frag, w in fragments:
                fi = tk(frag, truncation=False, add_special_tokens=False).input_ids
                ids.extend(fi)
                wts_all.extend([w] * len(fi))

            size = tk.model_max_length - 2
            chunks = [ids[i:i + size] for i in range(0, len(ids), size)] or [[]]
            wchunks = [wts_all[i:i + size] for i in range(0, len(wts_all), size)] or [[]]

            embeds = []
            for ci, (chunk, wchunk) in enumerate(zip(chunks, wchunks)):
                pad = tk.pad_token_id or 0
                toks = [tk.bos_token_id] + chunk + [tk.eos_token_id]
                wts = [1.0] + list(wchunk) + [1.0]
                while len(toks) < tk.model_max_length:
                    toks.append(pad)
                    wts.append(1.0)

                out = te(torch.tensor([toks], dtype=torch.long, device=device),
                         output_hidden_states=True)
                if te is encoders[1] and ci == 0:
                    pooled = out[0]

                idx = -2 if clip_skip <= 2 else -clip_skip
                h = out.hidden_states[idx]

                # Rescale by L2 norm, never by mean: CLIP embeddings are near
                # zero-mean, so a mean ratio is 0/0 and yields NaN -> black.
                if any(w != 1.0 for w in wts):
                    wt = torch.tensor(wts, dtype=h.dtype, device=device)[None, :, None]
                    before = h.norm()
                    h = h * wt
                    after = h.norm()
                    if after > 1e-4:
                        h = h * (before / after)
                embeds.append(h)
            per_encoder.append((torch.cat(embeds, dim=1), len(chunks)))

        target = max(n for _, n in per_encoder)
        aligned = []
        for emb, n in per_encoder:
            if n < target:
                per_chunk = emb.shape[1] // n
                emb = torch.cat([emb, emb[:, -per_chunk:].repeat(1, target - n, 1)], dim=1)
            aligned.append(emb)
        return torch.cat(aligned, dim=-1), pooled

    def render(self, r: Req):
        import torch
        from PIL import Image

        self._sync_loras(r.loras)
        apply_scheduler(self.pipe, r.sampler, r.scheduler)
        pos, pos_p = self._encode(r.positive, r.clip_skip)
        neg, neg_p = self._encode(r.negative, r.clip_skip)

        def pad_to(emb, n):
            if emb.shape[1] >= n:
                return emb[:, :n]
            reps = -(-(n - emb.shape[1]) // 77)
            return torch.cat([emb, emb[:, -77:].repeat(1, reps, 1)], dim=1)[:, :n]

        width = max(pos.shape[1], neg.shape[1])
        pos, neg = pad_to(pos, width), pad_to(neg, width)

        common = dict(
            prompt_embeds=pos, pooled_prompt_embeds=pos_p,
            negative_prompt_embeds=neg, negative_pooled_prompt_embeds=neg_p,
            guidance_scale=r.cfg,
            generator=torch.Generator(device="cpu").manual_seed(r.seed),
        )
        img = self.pipe(**common, width=r.width, height=r.height,
                        num_inference_steps=r.steps).images[0]
        self._restore_vae_dtype()

        if r.hires_enabled and r.hires_scale > 1.0:
            from diffusers import StableDiffusionXLImg2ImgPipeline

            if self._img2img is None:
                self._img2img = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
                self._img2img.set_progress_bar_config(disable=True)
                self._img2img.enable_vae_tiling()
                self._img2img.enable_vae_slicing()
            w = int(r.width * r.hires_scale) // 8 * 8
            h = int(r.height * r.hires_scale) // 8 * 8
            img = self._img2img(**common, image=img.resize((w, h), Image.LANCZOS),
                                strength=r.hires_denoise,
                                num_inference_steps=r.hires_steps).images[0]
            self._restore_vae_dtype()
        return img


# --------------------------------------------------------------------- server


try:
    from pydantic import BaseModel
except ImportError:                      # pip install runs before this import
    BaseModel = object                   # type: ignore[assignment,misc]


# These MUST live at module level. This file uses `from __future__ import
# annotations`, so every annotation is a string, and FastAPI resolves those
# strings against the module globals. A model defined inside build_app() is
# invisible there, so FastAPI silently falls back to treating the body as a
# query parameter and every request 422s with:
#     {"loc": ["query", "body"], "msg": "Field required"}
class Hires(BaseModel):
    enabled: bool = False
    scale: float = 1.5
    denoise: float = 0.4
    steps: int = 12


class RenderBody(BaseModel):
    positive: str
    negative: str = ""
    width: int = 832
    height: int = 1216
    seed: int = 0
    loras: list = []
    steps: int = 30
    cfg: float = 4.5
    clip_skip: int = 2
    sampler: str = "euler_a"
    scheduler: str = ""
    hires: Hires = Hires()


def build_app(engine: Engine, token: str):
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI(title="Manhua render worker")
    lock = threading.Lock()

    def check(tok):
        if token and tok != token:
            raise HTTPException(401, "bad token")

    @app.get("/health")
    def health(x_auth_token: str | None = Header(None)):
        check(x_auth_token)
        info = {"ok": True, "checkpoint": os.path.basename(engine.checkpoint)}
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                info |= {"gpu": torch.cuda.get_device_name(0),
                         "vram_free_mb": free // 1024**2,
                         "vram_total_mb": total // 1024**2}
        except Exception:
            pass
        return info

    @app.post("/render")
    def render(body: RenderBody, x_auth_token: str | None = Header(None)):
        check(x_auth_token)
        r = Req(positive=body.positive, negative=body.negative,
                width=body.width, height=body.height, seed=body.seed,
                steps=body.steps, cfg=body.cfg, clip_skip=body.clip_skip,
                loras=body.loras, sampler=body.sampler, scheduler=body.scheduler,
                hires_enabled=body.hires.enabled,
                hires_scale=body.hires.scale, hires_denoise=body.hires.denoise,
                hires_steps=body.hires.steps)
        # Return the real traceback instead of a bare 500. The client is on
        # another machine and cannot see this process's stdout, so hiding the
        # error turns every failure into a guessing game.
        import traceback

        with lock:                      # one GPU, one job
            t0 = time.time()
            try:
                img = engine.render(r)
            except Exception as exc:
                tb = traceback.format_exc()
                print(tb, flush=True)
                raise HTTPException(
                    500, f"{type(exc).__name__}: {exc}\n{tb[-1500:]}"
                ) from exc
            dt = time.time() - t0
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode(),
                "seconds": round(dt, 1), "size": [img.width, img.height]}

    return app


def start_tunnel(port: int) -> str | None:
    """cloudflared quick tunnel - no account, no token, stays free."""
    try:
        subprocess.run(
            "wget -q -O /tmp/cloudflared https://github.com/cloudflare/cloudflared/"
            "releases/latest/download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared",
            shell=True, check=True,
        )
    except subprocess.CalledProcessError:
        return None

    proc = subprocess.Popen(
        ["/tmp/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}",
         "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    url = None
    deadline = time.time() + 90
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:      # cloudflared exited
                break
            continue
        if "trycloudflare.com" in line:
            for word in line.split():
                if word.startswith("https://") and "trycloudflare.com" in word:
                    url = word.strip().strip("|,")
                    break
        if url:
            break

    # Keep draining stdout for the life of the process. cloudflared logs every
    # request; once the ~64KB pipe buffer fills, it blocks on write and stops
    # forwarding traffic -- the tunnel goes dead while the server still looks
    # healthy from inside the notebook.
    def drain() -> None:
        try:
            for _ in proc.stdout:
                pass
        except Exception:
            pass

    threading.Thread(target=drain, daemon=True).start()
    return url


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lora-dir", default=None)
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--token", default=os.environ.get("MANHUA_TOKEN", "change-me"))
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--no-tunnel", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    # Check the GPU arch BEFORE spending 90s loading a checkpoint. Current
    # PyTorch wheels ship kernels for sm_70+ only, so Kaggle's P100 (Pascal,
    # sm_60) loads the model happily and then dies on the first kernel launch
    # with "no kernel image is available for execution on the device".
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            print(f"GPU: {name} (sm_{major}{minor}), torch {torch.__version__}", flush=True)
            if (major, minor) < (7, 0):
                sys.exit(
                    f"\n{name} is compute capability {major}.{minor}, but this "
                    "PyTorch build only has kernels for sm_70 and newer.\n"
                    "On Kaggle: set Accelerator to 'GPU T4 x2' instead of P100, "
                    "then re-run.\n"
                )
        else:
            sys.exit("No CUDA device. Set Accelerator to GPU in the notebook settings.")
    except ImportError:
        sys.exit("torch is not installed")

    engine = Engine(args.checkpoint, args.lora_dir, low_vram=args.low_vram)
    print("loading checkpoint (60-90s on first run)...", flush=True)
    _ = engine.pipe
    print("model resident, worker ready", flush=True)

    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    threading.Thread(
        target=lambda: uvicorn.run(build_app(engine, args.token),
                                   host="0.0.0.0", port=args.port,
                                   log_level="warning"),
        daemon=True,
    ).start()
    time.sleep(3)

    if not args.no_tunnel:
        url = start_tunnel(args.port)
        if url:
            print("\n" + "=" * 64)
            print("  PASTE INTO workspace/settings.json ON YOUR MACHINE:")
            print('    "backend": "remote",')
            print(f'    "remote_url": "{url}",')
            print(f'    "remote_token": "{args.token}"')
            print("=" * 64 + "\n", flush=True)
        else:
            print("tunnel failed - rerun, or use --no-tunnel", flush=True)

    # Hold the cell open, and prove the tunnel is still alive. A silent
    # notebook cannot be told apart from a dead tunnel.
    import urllib.request

    served = 0
    while True:
        time.sleep(300)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/health", timeout=10
            ):
                served += 1
                print(f"[{time.strftime('%H:%M:%S')}] worker alive "
                      f"({served * 5} min)", flush=True)
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] LOCAL HEALTH FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()

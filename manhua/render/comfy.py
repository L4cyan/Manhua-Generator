"""ComfyUI backend.

Builds an API-format workflow graph in code rather than loading a static JSON,
so LoRA chains and the optional hires pass can be composed per panel while the
sampler settings stay pinned to the style lock.
"""
from __future__ import annotations

import io
import json
import urllib.parse
import uuid

import requests
import websocket
from PIL import Image

from .base import Backend, RenderRequest


class ComfyBackend(Backend):
    def __init__(self, host: str = "127.0.0.1:8188", timeout: int = 600):
        self.host = host.replace("http://", "").replace("https://", "").rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ---------- graph construction ----------

    def _graph(self, req: RenderRequest) -> dict:
        s = req.style.render
        h = req.style.hires

        g: dict[str, dict] = {
            "ckpt": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": s.checkpoint},
            }
        }

        # Chain LoRA loaders. Each takes the previous node's MODEL and CLIP,
        # so style and character identities stack in a defined order.
        model_src: list = ["ckpt", 0]
        clip_src: list = ["ckpt", 1]
        for i, (name, weight) in enumerate(req.loras):
            node = f"lora{i}"
            g[node] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": name,
                    "strength_model": weight,
                    "strength_clip": weight,
                    "model": model_src,
                    "clip": clip_src,
                },
            }
            model_src, clip_src = [node, 0], [node, 1]

        g["clipskip"] = {
            "class_type": "CLIPSetLastLayer",
            "inputs": {"stop_at_clip_layer": -abs(s.clip_skip), "clip": clip_src},
        }
        g["pos"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": req.positive, "clip": ["clipskip", 0]},
        }
        g["neg"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": req.negative, "clip": ["clipskip", 0]},
        }
        g["latent"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": req.width, "height": req.height, "batch_size": 1},
        }
        g["sampler"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": req.seed,
                "steps": s.steps,
                "cfg": s.cfg,
                "sampler_name": s.sampler,
                "scheduler": s.scheduler,
                "denoise": 1.0,
                "model": model_src,
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["latent", 0],
            },
        }

        last_latent: list = ["sampler", 0]
        if h.enabled:
            g["upscale"] = {
                "class_type": "LatentUpscaleBy",
                "inputs": {
                    "upscale_method": "nearest-exact",
                    "scale_by": h.scale,
                    "samples": ["sampler", 0],
                },
            }
            g["sampler2"] = {
                "class_type": "KSampler",
                "inputs": {
                    # Same seed as the base pass: the hires stage refines the
                    # existing composition instead of inventing a new one.
                    "seed": req.seed,
                    "steps": h.steps,
                    "cfg": s.cfg,
                    "sampler_name": s.sampler,
                    "scheduler": s.scheduler,
                    "denoise": h.denoise,
                    "model": model_src,
                    "positive": ["pos", 0],
                    "negative": ["neg", 0],
                    "latent_image": ["upscale", 0],
                },
            }
            last_latent = ["sampler2", 0]

        g["decode"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": last_latent, "vae": ["ckpt", 2]},
        }
        g["save"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "manhua/panel", "images": ["decode", 0]},
        }
        return g

    # ---------- execution ----------

    def render(self, req: RenderRequest) -> Image.Image:
        graph = self._graph(req)
        ws = websocket.WebSocket()
        ws.settimeout(self.timeout)
        ws.connect(f"ws://{self.host}/ws?clientId={self.client_id}")
        try:
            resp = requests.post(
                f"http://{self.host}/prompt",
                json={"prompt": graph, "client_id": self.client_id},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI rejected the workflow: {resp.text[:500]}")
            prompt_id = resp.json()["prompt_id"]

            # Execution is done when ComfyUI reports a null node for our prompt.
            while True:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue  # binary preview frame
                data = json.loads(msg)
                if data.get("type") != "executing":
                    continue
                d = data["data"]
                if d.get("node") is None and d.get("prompt_id") == prompt_id:
                    break
        finally:
            ws.close()

        history = requests.get(f"http://{self.host}/history/{prompt_id}", timeout=30).json()
        outputs = history[prompt_id]["outputs"]
        for node_out in outputs.values():
            for meta in node_out.get("images", []):
                return self._fetch(meta)
        raise RuntimeError("ComfyUI returned no images")

    def _fetch(self, meta: dict) -> Image.Image:
        q = urllib.parse.urlencode(
            {
                "filename": meta["filename"],
                "subfolder": meta.get("subfolder", ""),
                "type": meta.get("type", "output"),
            }
        )
        raw = requests.get(f"http://{self.host}/view?{q}", timeout=60).content
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def ping(self) -> bool:
        try:
            return requests.get(f"http://{self.host}/system_stats", timeout=5).status_code == 200
        except Exception:
            return False

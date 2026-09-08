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


# Style locks use the A1111/Civitai sampler vocabulary, which is what the
# recommended-settings blocks on model pages are written in. ComfyUI uses its
# own names and rejects the whole workflow on a mismatch, which is how an
# entire Illustrious comparison came back as five identical validation errors.
_SAMPLER_ALIASES = {
    "euler_a": "euler_ancestral",
    "euler a": "euler_ancestral",
    "dpmpp_2m_karras": "dpmpp_2m",
    "dpm++ 2m karras": "dpmpp_2m",
    "dpm++ 2m": "dpmpp_2m",
    "dpm++ sde karras": "dpmpp_sde",
    "dpm++ 2m sde karras": "dpmpp_2m_sde",
    "ddim": "ddim",
    "unipc": "uni_pc",
}


def _comfy_sampler(name: str) -> str:
    return _SAMPLER_ALIASES.get((name or "").strip().lower(), name or "euler")


class ComfyBackend(Backend):
    """Renders through a running ComfyUI instance.

    Worth preferring over the in-process backends on a small card: ComfyUI
    streams weights and offloads automatically, so it runs models that raw
    diffusers cannot fit. Anima's diffusers pipeline in particular exposes no
    offload hooks at all (no enable_model_cpu_offload, no vae tiling), so on
    6GB it loads to zero free VRAM and thrashes -- while ComfyUI handles the
    same model by managing memory itself.
    """

    def __init__(self, host: str = "127.0.0.1:8188", timeout: int = 600,
                 model_type: str = "sdxl", unet: str = "", clip: str = "",
                 vae: str = ""):
        self.host = host.replace("http://", "").replace("https://", "").rstrip("/")
        self.timeout = timeout
        # Anima ships as separate components rather than one baked checkpoint,
        # so it needs a different node graph from SDXL.
        self.model_type = model_type
        self.unet = unet or "anima-base-v1.0.safetensors"
        self.clip = clip or "qwen_3_06b_base.safetensors"
        self.vae = vae or "qwen_image_vae.safetensors"
        self.client_id = str(uuid.uuid4())

    # ---------- graph construction ----------

    def _anima_graph(self, req: RenderRequest) -> dict:
        """Node graph for Anima.

        Unlike SDXL there is no single checkpoint: the DiT, the Qwen-3 text
        encoder and the Qwen VAE load separately. ComfyUI decides on its own
        what to keep resident, which is the whole reason to route Anima here
        rather than through diffusers.
        """
        s = req.style.render
        g: dict[str, dict] = {
            "unet": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": self.unet, "weight_dtype": "default"},
            },
            "clip": {
                "class_type": "CLIPLoader",
                # ComfyUI has no "anima" type: it detects the Qwen3-0.6B
                # encoder from the weights (TEModel.QWEN3_06B in comfy/sd.py)
                # and wires the Anima tokenizer itself. The type field just has
                # to be a valid enum value.
                "inputs": {"clip_name": self.clip, "type": "qwen_image"},
            },
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": self.vae}},
        }

        # Character and style LoRAs. Anima trains these UNet-only (the Qwen3
        # encoder is frozen), so this chains LoraLoaderModelOnly rather than
        # LoraLoader -- passing a CLIP through would be a no-op at best and a
        # missing-key error at worst.
        model_src: list = ["unet", 0]
        for i, (name, weight) in enumerate(req.loras):
            node = f"lora{i}"
            g[node] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": name,
                    "strength_model": weight,
                    "model": model_src,
                },
            }
            model_src = [node, 0]

        g.update({
            "pos": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": req.positive, "clip": ["clip", 0]},
            },
            "neg": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": req.negative, "clip": ["clip", 0]},
            },
            "latent": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": req.width, "height": req.height, "batch_size": 1},
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": req.seed,
                    "steps": s.steps,
                    "cfg": s.cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": model_src,
                    "positive": ["pos", 0],
                    "negative": ["neg", 0],
                    "latent_image": ["latent", 0],
                },
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "manhua/anima", "images": ["decode", 0]},
            },
        })
        return g

    def _graph(self, req: RenderRequest) -> dict:
        if self.model_type == "anima":
            return self._anima_graph(req)
        return self._sdxl_graph(req)

    def _sdxl_graph(self, req: RenderRequest) -> dict:
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
                "sampler_name": _comfy_sampler(s.sampler),
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
                    "sampler_name": _comfy_sampler(s.sampler),
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


    # ---------- background removal ----------

    def _run_graph(self, graph: dict) -> list[Image.Image]:
        """Execute an arbitrary graph and return every image it saved."""
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
            while True:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue
                data = json.loads(msg)
                if data.get("type") != "executing":
                    continue
                d = data["data"]
                if d.get("node") is None and d.get("prompt_id") == prompt_id:
                    break
        finally:
            ws.close()

        history = requests.get(f"http://{self.host}/history/{prompt_id}", timeout=30).json()
        out: list[Image.Image] = []
        for node_out in history[prompt_id]["outputs"].values():
            for meta in node_out.get("images", []):
                out.append(self._fetch(meta))
        return out

    def upload_image(self, img: Image.Image, name: str) -> str:
        """Put an image into ComfyUI's input folder so a graph can load it."""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        resp = requests.post(
            f"http://{self.host}/upload/image",
            files={"image": (name, buf, "image/png")},
            data={"overwrite": "true"},
            timeout=120,
        )
        resp.raise_for_status()
        j = resp.json()
        sub = j.get("subfolder") or ""
        return f"{sub}/{j['name']}" if sub else j["name"]

    def cutout(self, img: Image.Image, name: str = "cutout_src.png") -> Image.Image:
        """Matte a character out of its background, returning RGBA.

        Uses BRIA RMBG rather than a threshold or a plain u2net pass: the
        edges that matter here are hair, and everything simpler leaves a halo
        that is obvious the moment the character is composited over a
        different background.
        """
        ref = self.upload_image(img, name)
        graph = {
            "load": {"class_type": "LoadImage", "inputs": {"image": ref}},
            "rmbgmodel": {"class_type": "BRIA_RMBG_ModelLoader_Zho", "inputs": {}},
            "rmbg": {
                "class_type": "BRIA_RMBG_Zho",
                "inputs": {"rmbgmodel": ["rmbgmodel", 0], "image": ["load", 0]},
            },
            "maskimg": {"class_type": "MaskToImage", "inputs": {"mask": ["rmbg", 1]}},
            "save": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "manhua/mask", "images": ["maskimg", 0]},
            },
        }
        images = self._run_graph(graph)
        if not images:
            raise RuntimeError("background removal returned nothing")
        mask = images[0].convert("L").resize(img.size, Image.LANCZOS)
        rgba = img.convert("RGBA")
        rgba.putalpha(mask)
        return rgba


    def refine(self, img: Image.Image, req: RenderRequest, denoise: float = 0.40,
               name: str = "refine_src.png") -> Image.Image:
        """Re-diffuse an existing image at partial strength (img2img).

        This is what makes a composited panel stop looking composited. A
        cut-out character pasted onto a plate has hard edges, its own lighting
        and no contact with the ground; running the assembled panel back
        through the sampler at ~0.4 denoise keeps the pose and the layout but
        lets the model redraw the seams, relight the figure to match the scene,
        and put a shadow where one belongs.

        Denoise is the whole control: too low and the seams survive, too high
        and the character stops being the character.
        """
        s = req.style.render
        ref = self.upload_image(img, name)

        if self.model_type == "anima":
            g: dict[str, dict] = {
                "unet": {"class_type": "UNETLoader",
                         "inputs": {"unet_name": self.unet, "weight_dtype": "default"}},
                "clip": {"class_type": "CLIPLoader",
                         "inputs": {"clip_name": self.clip, "type": "qwen_image"}},
                "vae": {"class_type": "VAELoader", "inputs": {"vae_name": self.vae}},
            }
            model_src: list = ["unet", 0]
            for i, (lname, weight) in enumerate(req.loras):
                node = f"lora{i}"
                g[node] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"lora_name": lname, "strength_model": weight,
                                      "model": model_src}}
                model_src = [node, 0]
            clip_src = ["clip", 0]
            vae_src = ["vae", 0]
        else:
            g = {"ckpt": {"class_type": "CheckpointLoaderSimple",
                          "inputs": {"ckpt_name": s.checkpoint}}}
            model_src, clip_src, vae_src = ["ckpt", 0], ["ckpt", 1], ["ckpt", 2]

        g.update({
            "load": {"class_type": "LoadImage", "inputs": {"image": ref}},
            "encode": {"class_type": "VAEEncode",
                       "inputs": {"pixels": ["load", 0], "vae": vae_src}},
            "pos": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": req.positive, "clip": clip_src}},
            "neg": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": req.negative, "clip": clip_src}},
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": req.seed, "steps": s.steps, "cfg": s.cfg,
                    "sampler_name": ("euler_ancestral" if self.model_type == "anima"
                                     else _comfy_sampler(s.sampler)),
                    "scheduler": "normal", "denoise": denoise,
                    "model": model_src, "positive": ["pos", 0], "negative": ["neg", 0],
                    "latent_image": ["encode", 0],
                },
            },
            "decode": {"class_type": "VAEDecode",
                       "inputs": {"samples": ["sampler", 0], "vae": vae_src}},
            "save": {"class_type": "SaveImage",
                     "inputs": {"filename_prefix": "manhua/refine", "images": ["decode", 0]}},
        })
        images = self._run_graph(g)
        if not images:
            raise RuntimeError("refine returned no image")
        return images[0]


    def hires(self, img: Image.Image, req: RenderRequest, scale_model: str =
              "RealESRGAN_x4plus_anime_6B.pth", denoise: float = 0.28,
              target_w: int = 0, target_h: int = 0,
              name: str = "hires_src.png") -> Image.Image:
        """Upscale then lightly re-diffuse, to clean artefacts and add detail.

        Upscaling alone sharpens the mistakes along with everything else; a
        short img2img pass afterwards is what actually repairs hands, melted
        architecture and mushy repeated detail, because the model gets to
        redraw them with more pixels to work in.

        Denoise stays low (0.25-0.35 is the range that works for SDXL-class
        models). Higher and it stops being the same panel.
        """
        s = req.style.render
        ref = self.upload_image(img, name)
        tw = target_w or int(img.width * 1.5)
        th = target_h or int(img.height * 1.5)

        if self.model_type == "anima":
            g: dict[str, dict] = {
                "unet": {"class_type": "UNETLoader",
                         "inputs": {"unet_name": self.unet, "weight_dtype": "default"}},
                "clip": {"class_type": "CLIPLoader",
                         "inputs": {"clip_name": self.clip, "type": "qwen_image"}},
                "vae": {"class_type": "VAELoader", "inputs": {"vae_name": self.vae}},
            }
            model_src: list = ["unet", 0]
            for i, (lname, weight) in enumerate(req.loras):
                node = f"lora{i}"
                g[node] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"lora_name": lname, "strength_model": weight,
                                      "model": model_src}}
                model_src = [node, 0]
            clip_src, vae_src = ["clip", 0], ["vae", 0]
            sampler_name = "euler_ancestral"
        else:
            g = {"ckpt": {"class_type": "CheckpointLoaderSimple",
                          "inputs": {"ckpt_name": s.checkpoint}}}
            model_src, clip_src, vae_src = ["ckpt", 0], ["ckpt", 1], ["ckpt", 2]
            sampler_name = _comfy_sampler(s.sampler)

        g.update({
            "load": {"class_type": "LoadImage", "inputs": {"image": ref}},
            "upmodel": {"class_type": "UpscaleModelLoader",
                        "inputs": {"model_name": scale_model}},
            "upscale": {"class_type": "ImageUpscaleWithModel",
                        "inputs": {"upscale_model": ["upmodel", 0], "image": ["load", 0]}},
            # The ESRGAN model is fixed at 4x, so scale back to the size we want.
            "resize": {"class_type": "ImageScale",
                       "inputs": {"image": ["upscale", 0], "upscale_method": "lanczos",
                                  "width": tw, "height": th, "crop": "disabled"}},
            "encode": {"class_type": "VAEEncode",
                       "inputs": {"pixels": ["resize", 0], "vae": vae_src}},
            "pos": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": req.positive, "clip": clip_src}},
            "neg": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": req.negative, "clip": clip_src}},
            "sampler": {"class_type": "KSampler",
                        "inputs": {"seed": req.seed, "steps": s.steps, "cfg": s.cfg,
                                   "sampler_name": sampler_name, "scheduler": "normal",
                                   "denoise": denoise, "model": model_src,
                                   "positive": ["pos", 0], "negative": ["neg", 0],
                                   "latent_image": ["encode", 0]}},
            "decode": {"class_type": "VAEDecode",
                       "inputs": {"samples": ["sampler", 0], "vae": vae_src}},
            "save": {"class_type": "SaveImage",
                     "inputs": {"filename_prefix": "manhua/hires", "images": ["decode", 0]}},
        })
        images = self._run_graph(g)
        if not images:
            raise RuntimeError("hires returned no image")
        return images[0]

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

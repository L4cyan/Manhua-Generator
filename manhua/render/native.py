"""Native diffusers backend - runs SDXL in-process, no ComfyUI required.

ComfyUI gives three things for free that this module has to reimplement,
because anime/Illustrious checkpoints depend on all of them:

  * CLIP skip          - anime models are trained expecting the penultimate
                         text-encoder layer, not the final one.
  * Long prompts       - CLIP truncates at 77 tokens. Our prompts run ~270,
                         so without chunking most of the style is silently
                         thrown away.
  * Prompt weighting   - `(tag:1.2)` emphasis, which tag-based models lean on.

Everything below exists to match ComfyUI's text encoding closely enough that a
prompt tuned in one produces the same image in the other.
"""
from __future__ import annotations

import functools
import gc
import re
from pathlib import Path

from PIL import Image

from .base import Backend, RenderRequest

# `(tag:1.3)` / `(tag)` / `[tag]` - A1111 emphasis syntax.
_WEIGHT_RE = re.compile(r"\(([^():]+?)(?::([0-9.]+))?\)|\[([^\[\]]+?)\]")


def parse_weights(text: str) -> list[tuple[str, float]]:
    """Split an A1111-style prompt into (fragment, weight) pairs.

    `(a:1.3)` -> 1.3, bare `(a)` -> 1.1, `[a]` -> 0.909, plain text -> 1.0.
    Nested groups are not supported; they are rare in tag prompts and the
    flat form keeps the weighting predictable.
    """
    out: list[tuple[str, float]] = []
    pos = 0
    for m in _WEIGHT_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], 1.0))
        if m.group(3) is not None:                 # [de-emphasis]
            out.append((m.group(3), 1 / 1.1))
        else:
            weight = float(m.group(2)) if m.group(2) else 1.1
            out.append((m.group(1), weight))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], 1.0))
    return [(t, w) for t, w in out if t.strip()]


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


class NativeBackend(Backend):
    """SDXL via diffusers, tuned for small-VRAM cards."""

    def __init__(
        self,
        checkpoint: str,
        *,
        lora_dir: str | None = None,
        device: str = "cuda",
        low_vram: bool = True,
    ):
        self.checkpoint = checkpoint
        self.lora_dir = Path(lora_dir) if lora_dir else None
        self.device = device
        self.low_vram = low_vram
        self._pipe = None
        self._loaded_loras: tuple[tuple[str, float], ...] = ()

    # ---------- pipeline ----------

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._build()
        return self._pipe

    @staticmethod
    def free_vram_mb() -> tuple[int, int]:
        """(free, total) VRAM in MiB. (0, 0) when there is no CUDA device."""
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return free // 1024**2, total // 1024**2
        except Exception:
            pass
        return 0, 0

    def preflight(self, need_mb: int = 3600) -> None:
        """Fail early and legibly when the GPU is already occupied.

        SDXL on a small card competes with anything else holding VRAM --
        ComfyUI, a game launcher, a browser with hardware acceleration. The
        raw failure is an opaque `CUDA error: out of memory` thrown deep in a
        kernel launch, so this turns it into something actionable.
        """
        free, total = self.free_vram_mb()
        if total and free < need_mb:
            raise RuntimeError(
                f"Not enough free VRAM: {free} MiB free of {total} MiB "
                f"(need about {need_mb} MiB).\n"
                "Something else is holding the GPU. Common culprits:\n"
                "  - ComfyUI or another Stable Diffusion UI still running\n"
                "  - a game launcher (Epic, Steam) or a hardware-accelerated browser\n"
                "  - an Ollama model kept resident (`ollama stop <model>`)\n"
                "Close it and retry, or set backend to 'comfy' in "
                "workspace/settings.json to render through the running ComfyUI instead."
            )

    def _build(self):
        import torch
        from diffusers import StableDiffusionXLPipeline

        path = Path(self.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        self.preflight()

        pipe = StableDiffusionXLPipeline.from_single_file(
            str(path),
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.set_progress_bar_config(disable=True)

        # SDXL's VAE overflows in fp16 and decodes to pure black. Diffusers
        # already handles this: when the VAE is fp16 AND force_upcast is set,
        # the pipeline upcasts the VAE to fp32 at decode time *and* casts the
        # latents to match.
        #
        # Do NOT cast the VAE to fp32 here. That check reads `vae.dtype ==
        # float16`, so pre-casting silently disables the built-in path and the
        # fp32 weights then meet fp16 latents -- a dtype mismatch at the first
        # conv, which is a far more confusing failure than the black image.
        pipe.vae.config.force_upcast = True

        if self.low_vram and self.device == "cuda":
            free, _ = self.free_vram_mb()
            if free < 5000:
                # Per-layer offload. Markedly slower than module offload, but
                # it is what lets SDXL run at all when the card is shared with
                # something else.
                pipe.enable_sequential_cpu_offload()
            else:
                # Moves each submodule to the GPU only while it runs. On 6GB
                # this is the difference between working and an OOM at decode.
                pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            pipe.enable_vae_slicing()
        else:
            pipe.to(self.device)
            # Always tile: a hires decode at ~1250x1825 needs one >2GB
            # allocation, which OOMs even a 16GB card.
            pipe.enable_vae_tiling()
            pipe.enable_vae_slicing()

        return pipe

    def _restore_vae_dtype(self) -> None:
        """Put the VAE back to fp16 after a decode.

        Diffusers upcasts the VAE to fp32 to decode and leaves it there. The
        upcast is guarded by `vae.dtype == float16`, so once it is fp32 the
        guard is False, latents stop being cast to match, and the NEXT decode
        fails with 'Input type (c10::Half) and bias type (float)'. The studio
        renders many panels through one pipeline, so without this the first
        panel of a beat succeeds and every one after it fails.
        """
        import torch

        try:
            if self._pipe is not None and self._pipe.vae.dtype != torch.float16:
                self._pipe.vae.to(torch.float16)
        except Exception:
            pass

    def _sync_loras(self, loras: list[tuple[str, float]]) -> None:
        """Load exactly the LoRAs this panel needs.

        Diffusers keeps adapters resident, so the set is only rebuilt when it
        actually changes - reloading per panel would dominate render time.
        """
        want = tuple(loras)
        if want == self._loaded_loras:
            return

        pipe = self.pipe
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        names: list[str] = []
        weights: list[float] = []
        for i, (fname, weight) in enumerate(loras):
            path = (self.lora_dir / fname) if self.lora_dir else Path(fname)
            if not path.exists():
                raise FileNotFoundError(f"LoRA not found: {path}")
            adapter = f"a{i}"
            pipe.load_lora_weights(str(path.parent), weight_name=path.name,
                                   adapter_name=adapter)
            names.append(adapter)
            weights.append(weight)

        if names:
            pipe.set_adapters(names, adapter_weights=weights)
        self._loaded_loras = want

    # ---------- text encoding ----------

    def _encode(self, text: str, clip_skip: int):
        """Encode a prompt of any length into SDXL embeddings.

        CLIP handles 77 tokens at a time, so the prompt is split into 75-token
        chunks (leaving room for BOS/EOS), each encoded separately, and the
        results concatenated along the sequence axis. This is what ComfyUI and
        A1111 both do, and it is why long tag prompts work there.
        """
        import torch

        pipe = self.pipe
        device = pipe._execution_device
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
        encoders = [pipe.text_encoder, pipe.text_encoder_2]

        fragments = parse_weights(text)

        per_encoder = []
        pooled = None

        for tk, te in zip(tokenizers, encoders):
            # Tokenise each weighted fragment separately so weights can be
            # applied per token after encoding.
            ids: list[int] = []
            weights: list[float] = []
            for frag, w in fragments:
                frag_ids = tk(frag, truncation=False, add_special_tokens=False).input_ids
                ids.extend(frag_ids)
                weights.extend([w] * len(frag_ids))

            size = tk.model_max_length - 2          # 75 content tokens per chunk
            chunks = [ids[i:i + size] for i in range(0, len(ids), size)] or [[]]
            wchunks = [weights[i:i + size] for i in range(0, len(weights), size)] or [[]]

            chunk_embeds = []
            for ci, (chunk, wchunk) in enumerate(zip(chunks, wchunks)):
                pad = tk.pad_token_id or 0
                toks = [tk.bos_token_id] + chunk + [tk.eos_token_id]
                wts = [1.0] + list(wchunk) + [1.0]
                while len(toks) < tk.model_max_length:
                    toks.append(pad)
                    wts.append(1.0)

                tensor = torch.tensor([toks], dtype=torch.long, device=device)
                out = te(tensor, output_hidden_states=True)

                # SDXL's pooled embedding comes from text_encoder_2, first chunk.
                if te is encoders[1] and ci == 0:
                    pooled = out[0]

                # hidden_states[-2] is the penultimate layer == ComfyUI clip_skip 2,
                # the standard for anime checkpoints.
                idx = -2 if clip_skip <= 2 else -(clip_skip)
                h = out.hidden_states[idx]

                # Apply emphasis, then restore the chunk's original magnitude so
                # weighting changes emphasis rather than overall signal strength.
                #
                # Rescale by L2 norm, NOT by mean: CLIP embeddings are close to
                # zero-mean, so a mean-ratio (the A1111 formulation) is 0/0 for
                # an unweighted prompt and produces NaN that propagates all the
                # way to a black image. Skipped entirely when nothing is
                # weighted, which is the common case.
                if any(w != 1.0 for w in wts):
                    wt = torch.tensor(wts, dtype=h.dtype, device=device)[None, :, None]
                    before = h.norm()
                    h = h * wt
                    after = h.norm()
                    if after > 1e-4:
                        h = h * (before / after)

                chunk_embeds.append(h)

            per_encoder.append((torch.cat(chunk_embeds, dim=1), len(chunks)))

        # Both encoders must yield the same sequence length to concatenate on
        # the feature axis; their tokenizers can disagree by a chunk.
        target = max(n for _, n in per_encoder)
        aligned = []
        for emb, n in per_encoder:
            if n < target:
                per_chunk = emb.shape[1] // n
                padding = emb[:, -per_chunk:].repeat(1, target - n, 1)
                emb = torch.cat([emb, padding], dim=1)
            aligned.append(emb)

        return torch.cat(aligned, dim=-1), pooled

    # ---------- render ----------

    def render(self, req: RenderRequest) -> Image.Image:
        import torch

        self._sync_loras(req.loras)
        s = req.style.render
        apply_scheduler(self.pipe, s.sampler, s.scheduler)
        h = req.style.hires

        pos, pos_pooled = self._encode(req.positive, s.clip_skip)
        neg, neg_pooled = self._encode(req.negative, s.clip_skip)

        # Classifier-free guidance runs both branches through the UNet as one
        # batch, so they must be the same length. A long positive prompt and a
        # short negative one is the normal case, not an edge case: here the
        # style block makes the positive 4 chunks and the negative 2.
        def pad_to(emb, target: int):
            if emb.shape[1] >= target:
                return emb[:, :target]
            # Repeat the trailing chunk, which is padding tokens already, so
            # nothing semantic is duplicated into the extra length.
            reps = -(-(target - emb.shape[1]) // 77)
            filler = emb[:, -77:].repeat(1, reps, 1)
            return torch.cat([emb, filler], dim=1)[:, :target]

        width = max(pos.shape[1], neg.shape[1])
        pos, neg = pad_to(pos, width), pad_to(neg, width)

        gen = torch.Generator(device="cpu").manual_seed(req.seed)
        common = dict(
            prompt_embeds=pos,
            pooled_prompt_embeds=pos_pooled,
            negative_prompt_embeds=neg,
            negative_pooled_prompt_embeds=neg_pooled,
            guidance_scale=s.cfg,
            generator=gen,
        )

        image = self.pipe(
            **common, width=req.width, height=req.height, num_inference_steps=s.steps
        ).images[0]
        self._restore_vae_dtype()

        if h.enabled and h.scale > 1.0:
            image = self._hires(image, req, common, h)
            self._restore_vae_dtype()

        return image

    def _hires(self, image: Image.Image, req: RenderRequest, common: dict, h) -> Image.Image:
        """Upscale then refine, reusing the same seed so the composition holds."""
        from diffusers import StableDiffusionXLImg2ImgPipeline

        w = int(req.width * h.scale) // 8 * 8
        ht = int(req.height * h.scale) // 8 * 8
        upscaled = image.resize((w, ht), Image.LANCZOS)

        if not hasattr(self, "_img2img"):
            self._img2img = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
            self._img2img.set_progress_bar_config(disable=True)
            if self.low_vram and self.device == "cuda":
                self._img2img.enable_model_cpu_offload()

        return self._img2img(
            **common, image=upscaled, strength=h.denoise, num_inference_steps=h.steps
        ).images[0]

    # ---------- lifecycle ----------

    def ping(self) -> bool:
        return Path(self.checkpoint).exists()

    def close(self) -> None:
        self._pipe = None
        if hasattr(self, "_img2img"):
            del self._img2img
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


@functools.lru_cache(maxsize=1)
def find_comfy_models() -> tuple[str | None, str | None]:
    """Locate an existing ComfyUI model tree to borrow checkpoints from.

    Users who already run ComfyUI have tens of gigabytes of models on disk;
    making them re-download is hostile.
    """
    candidates = [
        Path("C:/comfyui/ComfyUI_windows_portable/ComfyUI/models"),
        Path("C:/ComfyUI/models"),
        Path.home() / "ComfyUI" / "models",
    ]
    for base in candidates:
        if (base / "checkpoints").is_dir():
            return str(base / "checkpoints"), str(base / "loras")
    return None, None

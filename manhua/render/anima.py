"""Anima backend - runs locally, in-process.

Counter-intuitively this belongs on the local GPU rather than the free cloud
one. Anima is trained in bf16, and bf16 tensor cores start at Ampere (sm_80).
An RTX 30-series card has them; Kaggle's free T4 is Turing and does not, so
bf16 there is emulated and a single render overruns even the tunnel timeout.

It also fits where SDXL does not: 2B parameters in bf16 is roughly 4GB, so a
6GB card holds the whole model with room for activations and needs no CPU
offload - which is where SDXL loses about 10x on the same hardware.

Much thinner than the SDXL backend because Anima's Qwen-3 text encoder has a
long context: no 77-token chunking, no dual encoders, no pooled embeds, no
clip skip. The prompt goes in whole.
"""
from __future__ import annotations

import gc

from PIL import Image

from .base import Backend, RenderRequest

REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"


class AnimaBackend(Backend):
    def __init__(self, repo: str = REPO, *, lora_dir: str | None = None,
                 low_vram: bool = False):
        self.checkpoint = repo or REPO
        self.lora_dir = lora_dir
        self.low_vram = low_vram
        self._pipe = None

    @staticmethod
    def bf16_native() -> bool:
        """Whether this GPU has bf16 tensor cores (Ampere, sm_80+).

        Not torch.cuda.is_bf16_supported(): that returns True when bf16 can
        merely be emulated, which is far too slow to be usable.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            return torch.cuda.get_device_capability(0) >= (8, 0)
        except Exception:
            return False

    def preflight(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Anima needs a CUDA GPU.")
        free, total = torch.cuda.mem_get_info()
        total_mb = total // 1024**2
        if total_mb < 11000 and not self.low_vram:
            raise RuntimeError(
                f"Anima needs about 12GB of VRAM unoffloaded; this card has "
                f"{total_mb} MiB.\nMeasured on 6GB: the pipeline loads to 0 MiB "
                "free and generation thrashes at ~33 s/step.\nSet low_vram=True "
                "to use CPU offload, or use the SDXL backend instead."
            )

        cap = torch.cuda.get_device_capability(0)
        if cap < (8, 0):
            name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                f"{name} is compute capability {cap[0]}.{cap[1]}, which has no "
                "bf16 tensor cores.\nAnima is bf16-native: fp16 overflows to "
                "NaN and renders blank, and emulated bf16 is unusably slow.\n"
                "Use an Ampere (RTX 30-series) card or newer, or switch to the "
                "SDXL backend."
            )

    @property
    def pipe(self):
        if self._pipe is None:
            import torch

            self.preflight()
            try:
                from diffusers import AnimaAutoBlocks
            except ImportError as exc:
                raise RuntimeError(
                    "This diffusers build has no Anima support. Upgrade it:\n"
                    "  pip install -U git+https://github.com/huggingface/diffusers.git"
                ) from exc

            p = AnimaAutoBlocks().init_pipeline(self.checkpoint)
            p.load_components(torch_dtype=torch.bfloat16)
            if self.low_vram and hasattr(p, "enable_model_cpu_offload"):
                p.enable_model_cpu_offload()
            else:
                p.to("cuda")
            self._pipe = p
        return self._pipe

    def render(self, req: RenderRequest) -> Image.Image:
        import torch

        s = req.style.render
        out = self.pipe(
            prompt=req.positive,
            negative_prompt=req.negative or None,
            width=req.width,
            height=req.height,
            num_inference_steps=s.steps,
            guidance_scale=s.cfg,
            generator=torch.Generator(device="cpu").manual_seed(req.seed),
        )
        # Modular pipelines have moved the payload attribute between versions.
        for attr in ("images", "image"):
            got = getattr(out, attr, None)
            if got:
                return got[0] if isinstance(got, (list, tuple)) else got
        if isinstance(out, (list, tuple)) and out:
            return out[0]
        raise RuntimeError(f"Anima returned no image (got {type(out).__name__})")

    def ping(self) -> bool:
        return self.bf16_native()

    def close(self) -> None:
        self._pipe = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

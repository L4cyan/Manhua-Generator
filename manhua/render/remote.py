"""Remote render backend - talks to a GPU worker over HTTP.

Exists because 6GB of local VRAM forces CPU offload, which costs roughly 10x
in wall-clock time. A free Kaggle/Colab GPU has 16GB, needs no offload, and
runs the *same* checkpoint and LoRAs -- which is the part no closed image API
can do, and the reason this is a self-hosted worker rather than a vendor SDK.

The worker runs `kaggle/worker.py`, which wraps the same NativeBackend used
locally. Identical code on both sides means a panel rendered in the cloud and
one rendered offline are byte-comparable for a given seed.
"""
from __future__ import annotations

import base64
import io
import time

import requests
from PIL import Image

from ..net import enable_system_certs
from .base import Backend, RenderRequest


class RemoteBackend(Backend):
    def __init__(
        self,
        url: str,
        token: str = "",
        *,
        timeout: float = 300.0,
        retries: int = 6,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries
        enable_system_certs()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["X-Auth-Token"] = self.token
        return h

    def render(self, req: RenderRequest) -> Image.Image:
        s = req.style.render
        h = req.style.hires
        payload = {
            "positive": req.positive,
            "negative": req.negative,
            "width": req.width,
            "height": req.height,
            "seed": req.seed,
            "loras": [list(l) for l in req.loras],
            # Sampler settings travel with each request so the worker stays
            # stateless: the style lock remains authoritative on this machine.
            "steps": s.steps,
            "cfg": s.cfg,
            "clip_skip": s.clip_skip,
            "sampler": s.sampler,
            "scheduler": s.scheduler,
            "hires": {
                "enabled": h.enabled,
                "scale": h.scale,
                "denoise": h.denoise,
                "steps": h.steps,
            },
        }

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(
                    f"{self.url}/render",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                if r.status_code == 401:
                    raise RuntimeError("Worker rejected the token - check remote_token")
                if r.status_code != 200:
                    raise RuntimeError(f"Worker error {r.status_code}: {r.text[:300]}")

                data = r.json()
                if "image" not in data:
                    raise RuntimeError(f"Worker returned no image: {str(data)[:300]}")
                return Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")

            except (requests.Timeout, requests.ConnectionError) as exc:
                # Free-tier tunnels drop connections; a cold worker also stalls
                # while it loads a 7GB checkpoint on the first request.
                last = exc
                if attempt < self.retries:
                    # Short, flat backoff: these are dropped TLS handshakes,
                    # not server overload, so waiting longer does not help.
                    time.sleep(min(2 + attempt, 6))
                    continue
                raise RuntimeError(
                    f"Could not reach the render worker at {self.url}.\n"
                    "The Kaggle/Colab session may have expired - free sessions stop "
                    "after ~12h or on idle. Restart the notebook, copy the new tunnel "
                    "URL into workspace/settings.json, or switch backend to 'native'."
                ) from last

        raise RuntimeError(str(last))

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", headers=self._headers(), timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def info(self) -> dict:
        """Worker's GPU and model details, for the studio status line."""
        try:
            r = requests.get(f"{self.url}/health", headers=self._headers(), timeout=10)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

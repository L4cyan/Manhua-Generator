"""Backend that prefers a fast remote GPU and falls back to the local one.

Free cloud sessions expire, tunnels drop, and laptops go offline. Without a
fallback each of those turns into a failed render and a manual settings edit;
with one, work continues at reduced speed and says so.

Order is deliberate: remote first because it is ~3x faster, local second
because it always works.
"""
from __future__ import annotations

import time

from PIL import Image

from .base import Backend, RenderRequest


class FallbackBackend(Backend):
    def __init__(
        self,
        primary: Backend,
        secondary: Backend,
        *,
        recheck_after: float = 300.0,
        on_switch=None,
    ):
        self.primary = primary
        self.secondary = secondary
        self.recheck_after = recheck_after
        self.on_switch = on_switch
        self._primary_failed_at: float | None = None

    def _primary_usable(self) -> bool:
        """Whether to try the primary, retrying periodically after a failure.

        Without the cooldown every panel pays the primary's full timeout before
        falling back, which is slower than just using the local card.
        """
        if self._primary_failed_at is None:
            return True
        if time.time() - self._primary_failed_at > self.recheck_after:
            self._primary_failed_at = None       # cooldown elapsed, try again
            return True
        return False

    def render(self, req: RenderRequest) -> Image.Image:
        if self._primary_usable():
            try:
                img = self.primary.render(req)
                if self._primary_failed_at is not None:
                    self._note("remote worker is back; using it again")
                    self._primary_failed_at = None
                return img
            except Exception as exc:
                self._primary_failed_at = time.time()
                self._note(
                    f"remote worker unavailable ({str(exc)[:120]}); "
                    "falling back to local rendering"
                )

        return self.secondary.render(req)

    def _note(self, msg: str) -> None:
        if self.on_switch:
            self.on_switch(msg)
        else:
            print(f"[backend] {msg}", flush=True)

    def ping(self) -> bool:
        for b in (self.primary, self.secondary):
            if not hasattr(b, "ping") or b.ping():
                return True
        return False

    def info(self) -> dict:
        primary_up = (
            self.primary.ping() if hasattr(self.primary, "ping") else True
        )
        return {
            "active": "remote" if primary_up and self._primary_usable() else "local",
            "primary": type(self.primary).__name__,
            "primary_reachable": primary_up,
            "secondary": type(self.secondary).__name__,
        }

    def close(self) -> None:
        for b in (self.primary, self.secondary):
            try:
                b.close()
            except Exception:
                pass

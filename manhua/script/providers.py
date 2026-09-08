"""LLM providers for script breakdown.

Two backends, same contract: prose in, schema-valid panel JSON out.

  ollama  - free, offline, no account. Runs whatever model you have pulled.
  claude  - better panelling and much better at "show, don't tell", but needs
            an API key and bills per chapter.

Auto-detection prefers whatever is actually available, so a fresh clone with
Ollama running needs no configuration at all.
"""
from __future__ import annotations

import json
import time
import os

import requests

from ..net import enable_system_certs

enable_system_certs()

def _ollama_host() -> str:
    """Resolve the Ollama base URL from the environment.

    OLLAMA_HOST is overwhelmingly set as a *bind address* for the server --
    "0.0.0.0", "0.0.0.0:11434" -- not as a client URL. Used verbatim that
    produces "0.0.0.0/api/tags", which has no scheme and fails, and the failure
    surfaces as "no script provider available" even though Ollama is running
    perfectly well. Normalise it: add a scheme, and turn a wildcard bind into
    a loopback address a client can actually connect to.
    """
    # workspace/settings.json wins over the environment, so a user can pin the
    # host in the app without editing system variables.
    raw = ""
    try:
        import json as _json
        from pathlib import Path as _Path
        cfg = _Path("workspace/settings.json")
        if cfg.exists():
            raw = (_json.loads(cfg.read_text(encoding="utf-8")).get("ollama_host") or "").strip()
    except Exception:
        raw = ""
    raw = raw or (os.environ.get("OLLAMA_HOST") or "").strip()
    if not raw:
        return "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    scheme, _, rest = raw.partition("://")
    hostport = rest.rstrip("/")
    host, _, port = hostport.partition(":")
    if host in ("0.0.0.0", "::", "[::]", ""):
        host = "127.0.0.1"
    return f"{scheme}://{host}:{port or '11434'}"


OLLAMA_HOST = _ollama_host()
CLAUDE_MODEL = os.environ.get("MANHUA_CLAUDE_MODEL", "claude-opus-5")

# Ordered by how well they follow a constrained JSON schema. Small models are
# usable here because the schema does the heavy lifting; the model only has to
# fill fields, not invent structure.
PREFERRED_OLLAMA = [
    "qwen3.5:4b", "qwen3.5:2b", "qwen2.5:7b", "qwen3.5-unc:4b", "qwen2.5:3b",
]


class NoProvider(RuntimeError):
    pass


# ---------------------------------------------------------------- discovery


def ollama_models(timeout: float = 12.0, retries: int = 2) -> list[str]:
    """Names of locally pulled Ollama models, or [] if Ollama is not running.

    Retried with a generous timeout: Ollama stalls this endpoint while it is
    loading or evicting a model, so a short single-shot probe reports "not
    installed" for a server that is merely busy.
    """
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            if attempt == retries:
                return []
            time.sleep(1.0)
    return []


def configured_model() -> str:
    """A model pinned in workspace/settings.json, if any."""
    try:
        import json as _json
        from pathlib import Path as _Path
        cfg = _Path("workspace/settings.json")
        if cfg.exists():
            return (_json.loads(cfg.read_text(encoding="utf-8")).get("ollama_model") or "").strip()
    except Exception:
        pass
    return ""


def pick_ollama_model(available: list[str] | None = None) -> str | None:
    """Choose the best available local model for structured breakdown."""
    available = available if available is not None else ollama_models()
    pinned = configured_model()
    if pinned:
        # Honour the pin even if the tag is not listed: the user may have just
        # pulled it, and a wrong name fails loudly rather than silently using
        # something else.
        return pinned
    if not available:
        return None
    for want in PREFERRED_OLLAMA:
        if want in available:
            return want
    # Fall back to any instruct-ish text model; skip vision-only ones, which
    # tend to ignore a JSON schema when given no image.
    for name in available:
        if not any(t in name.lower() for t in ("vl", "llava", "moondream", "embed")):
            return name
    return available[0]


def detect() -> tuple[str, str]:
    """Return (provider, model), preferring whatever is actually usable.

    Ollama wins when present: it is free and needs no key. Claude is used when
    a key exists and Ollama is not running.
    """
    forced = os.environ.get("MANHUA_PROVIDER", "").strip().lower()
    if forced == "ollama":
        model = os.environ.get("MANHUA_OLLAMA_MODEL") or pick_ollama_model()
        if not model:
            raise NoProvider(f"MANHUA_PROVIDER=ollama but no models at {OLLAMA_HOST}")
        return "ollama", model
    if forced == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise NoProvider("MANHUA_PROVIDER=claude but ANTHROPIC_API_KEY is unset")
        return "claude", CLAUDE_MODEL

    model = os.environ.get("MANHUA_OLLAMA_MODEL") or pick_ollama_model()
    if model:
        return "ollama", model
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude", CLAUDE_MODEL

    raise NoProvider(
        "No script provider available. Either:\n"
        "  - install Ollama and run `ollama pull qwen3.5:4b`  (free, offline), or\n"
        "  - set ANTHROPIC_API_KEY for Claude"
    )


# ---------------------------------------------------------------- calls


def call_ollama(system: str, user: str, schema: dict, model: str,
                timeout: float = 300.0) -> dict:
    """Structured generation via Ollama's JSON-schema `format` parameter.

    The schema is enforced during sampling, so even a 2B model cannot emit
    malformed panel JSON -- it can only choose bad values, which the caller
    then sanitises.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": schema,
        "stream": False,
        "options": {
            # Low but non-zero: deterministic enough to be reproducible,
            # varied enough that a re-run gives a genuinely different cut.
            "temperature": 0.4,
            "num_ctx": 8192,
        },
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")

    content = r.json()["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned non-JSON despite the schema: {content[:300]}") from exc


def call_claude(system: str, user: str, output_model, model: str) -> object:
    """Structured generation via the Anthropic SDK's schema-validated parse."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=output_model,
    )
    return response.parsed_output

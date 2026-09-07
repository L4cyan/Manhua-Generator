"""Loading of the frozen style lock and the character bible."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import Character


@dataclass
class HiresCfg:
    enabled: bool = True
    scale: float = 1.5
    denoise: float = 0.4
    steps: int = 12


@dataclass
class RenderCfg:
    checkpoint: str = "illustriousXL_v01.safetensors"
    sampler: str = "dpmpp_2m_sde"
    scheduler: str = "karras"
    steps: int = 28
    cfg: float = 5.0
    clip_skip: int = 2
    style_lora: str | None = None
    style_lora_weight: float = 0.6


@dataclass
class StyleLock:
    """Everything that must not vary between panels."""

    name: str
    style_lead: str
    style_body: str
    style_fx: str
    positive_suffix: str
    negative: str
    render: RenderCfg
    hires: HiresCfg
    aspects: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Genre clause per setting, chosen by Panel.world. See style.yaml.
    registers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "StyleLock":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=raw.get("name", "unnamed"),
            # positive_prefix is the pre-split field name, still accepted.
            style_lead=_flatten(raw.get("style_lead", raw.get("positive_prefix", ""))),
            style_body=_flatten(raw.get("style_body", "")),
            style_fx=_flatten(raw.get("style_fx", "")),
            registers={k: _flatten(v) for k, v in (raw.get("registers") or {}).items()},
            positive_suffix=_flatten(raw.get("positive_suffix", "")),
            negative=_flatten(raw.get("negative", "")),
            render=RenderCfg(**raw.get("render", {})),
            hires=HiresCfg(**raw.get("hires", {})),
            aspects={k: tuple(v) for k, v in raw.get("aspects", {}).items()},
        )

    def positive(self, content: str, *, has_fx: bool = False,
                 register: str = "cultivation") -> str:
        """Assemble the full positive prompt.

        Order is deliberate and is the main defence against identity drift:

            style_lead  -> short, front-loaded style anchor (~20 tokens)
            content     -> framing, CHARACTER IDENTITY, action, setting, fx
            style_body  -> the bulk of the aesthetic description
            suffix      -> quality tail

        Character identity therefore lands inside CLIP's first 77-token chunk,
        where it carries the most weight, instead of being buried behind a
        200-token style block. The style LoRA holds the look independently of
        token position, so demoting the style text costs nothing.
        """
        genre = self.registers.get(register, self.registers.get('cultivation', ''))
        lead = f"{self.style_lead}, {genre}" if genre else self.style_lead
        chunks = [lead, content, self.style_body]
        # Supernatural FX are opt-in per panel: a lecture hall should not
        # inherit a qi aura just because the series is a cultivation story.
        if has_fx and self.style_fx:
            chunks.append(self.style_fx)
        chunks.append(self.positive_suffix)
        return ", ".join(c.strip().rstrip(",") for c in chunks if c and c.strip())

    @staticmethod
    def est_tokens(text: str) -> int:
        """Rough CLIP token count. Comma-separated tags run ~1.3 tokens/word."""
        return int(len(text.replace(",", " ").split()) * 1.3)

    def identity_chunk(self, content: str) -> int:
        """Which 77-token CLIP chunk the panel content starts in.

        0 means identity is in the strongest chunk. If this creeps above 1,
        shorten `style_lead` -- the character description is losing weight.
        """
        return self.est_tokens(self.style_lead) // 77

    def size(self, aspect: str) -> tuple[int, int]:
        return self.aspects.get(aspect, (832, 1216))


def _flatten(s: str) -> str:
    """YAML folded blocks keep stray newlines; prompts want single-line."""
    return " ".join(str(s).split())


def load_bible(path: str | Path) -> dict[str, Character]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("characters", raw)
    bible: dict[str, Character] = {}
    for cid, data in entries.items():
        bible[cid] = Character(id=cid, **data)
    return bible

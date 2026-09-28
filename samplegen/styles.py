"""Trained styles (LoRAs) that the generator can apply.

Each style is a ComfyUI LoRA file plus a JSON sidecar, both in the engine's
models/loras folder, named samplegen_<slug>.*. A style was trained on one of the
Stable Audio 3 *base* models and must be used with that same base model.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PREFIX = "samplegen_"
BASES = {"sfx": "sa3-sfx-base", "music": "sa3-music-base"}  # style base -> catalog model key
TRAINER_MODELS = {"sfx": "small-sfx-base", "music": "small-music-base"}


class StyleNotFound(KeyError):
    pass


@dataclass(frozen=True)
class Style:
    slug: str
    name: str
    base: str  # "sfx" | "music"
    clips: int
    steps: int
    created: str
    description: str = ""

    @property
    def lora_file(self) -> str:
        return f"{PREFIX}{self.slug}.safetensors"

    @property
    def model_key(self) -> str:
        return BASES[self.base]

    def to_dict(self) -> dict:
        return {**asdict(self), "model": self.model_key}


def style_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40].strip("-")
    if not slug:
        raise ValueError("Give the style a name with some letters or numbers in it.")
    return slug


class StyleStore:
    def __init__(self, loras_dir: Path):
        self.dir = Path(loras_dir)

    def _meta_path(self, slug: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]{1,40}", slug):
            raise StyleNotFound(slug)
        return self.dir / f"{PREFIX}{slug}.json"

    def list(self) -> list[Style]:
        if not self.dir.exists():
            return []
        styles = []
        for meta in sorted(self.dir.glob(f"{PREFIX}*.json")):
            try:
                style = Style(**json.loads(meta.read_text(encoding="utf-8")))
            except (ValueError, TypeError):
                continue  # a half-written or foreign file: ignore it
            if (self.dir / style.lora_file).exists():
                styles.append(style)
        return sorted(styles, key=lambda s: s.created, reverse=True)

    def get(self, slug: str) -> Style:
        meta = self._meta_path(slug)
        if not meta.exists():
            raise StyleNotFound(slug)
        style = Style(**json.loads(meta.read_text(encoding="utf-8")))
        if not (self.dir / style.lora_file).exists():
            raise StyleNotFound(slug)
        return style

    def exists(self, slug: str) -> bool:
        try:
            return self._meta_path(slug).exists()
        except StyleNotFound:
            return False

    def lora_path(self, slug: str) -> Path:
        self._meta_path(slug)
        return self.dir / f"{PREFIX}{slug}.safetensors"

    def register(self, style: Style) -> Style:
        if style.base not in BASES:
            raise ValueError(f"Unknown base '{style.base}'.")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta_path(style.slug).write_text(json.dumps(asdict(style), indent=2), encoding="utf-8")
        return style

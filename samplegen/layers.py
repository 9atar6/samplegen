"""Layer designer: an impact built like a sound designer builds one.

Three layers, each generated on its own: a short TRANSIENT (the attack), a BODY (the
weight) and a TAIL (the decay / space). Each is trimmed to its first transient so they
hit together, offset in time, balanced, summed, and normalized. The blended hit and the
three layers are all saved, so a layer can be swapped or re-mixed in a DAW.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from .audio import ExportFormat, apply_fades, fit_length, normalize_peak, remove_dc

LAYERS = ("transient", "body", "tail")
LENGTHS = {"transient": 0.5, "body": 2.0}  # the tail's length is the user's choice
MAX_TAKES = 4
ONSET_FRACTION = 0.05   # a layer "starts" where it first reaches 5 % of its peak
PRE_ROLL_S = 0.002
LAYER_FADE_MS = {"transient": 40.0, "body": 120.0, "tail": 400.0}


@dataclass(frozen=True)
class LayerRequest:
    character: str                     # shared by all three: "sci-fi, metallic, huge"
    transient: str = ""
    body: str = ""
    tail: str = ""
    tail_seconds: float = 4.0
    body_offset_ms: float = 0.0
    tail_offset_ms: float = 30.0
    gains_db: tuple[float, float, float] = (0.0, 0.0, -3.0)
    variations: int = 2
    name: str = ""
    model: str = "sa3-sfx"
    seed: int | None = None
    export: ExportFormat = field(default_factory=ExportFormat)
    mode: str = "layers"
    uses_source = False  # class attribute, not a field

    def layer_prompt(self, layer: str) -> str:
        own = getattr(self, layer).strip()
        return ", ".join(p for p in (own, self.character.strip()) if p)

    def active(self) -> list[str]:
        return [layer for layer in LAYERS if getattr(self, layer).strip()]

    def seconds(self, layer: str) -> float:
        return self.tail_seconds if layer == "tail" else LENGTHS[layer]

    def full_prompt(self) -> str:
        return " + ".join(f"{layer}: {self.layer_prompt(layer)}" for layer in self.active())

    def title(self) -> str:
        base = self.name or self.character or self.body or self.transient or "layered hit"
        return re.sub(r"\s+", " ", base).strip()[:80]

    def validate(self, source_seconds=None) -> None:
        if len(self.active()) < 2:
            raise ValueError("Describe at least two layers (transient, body, tail).")
        if any(len(getattr(self, f)) > 400 for f in ("character", *LAYERS)):
            raise ValueError("Keep each description under 400 characters.")
        if not 0.5 <= self.tail_seconds <= 30:
            raise ValueError("The tail can be 0.5 to 30 seconds.")
        if not 1 <= self.variations <= MAX_TAKES:
            raise ValueError(f"1 to {MAX_TAKES} takes.")
        if not (0 <= self.body_offset_ms <= 1000 and 0 <= self.tail_offset_ms <= 2000):
            raise ValueError("Offsets: body up to 1000 ms, tail up to 2000 ms.")
        if len(self.gains_db) != 3 or any(not -36 <= g <= 12 for g in self.gains_db):
            raise ValueError("Layer levels: -36 to +12 dB.")


def from_onset(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The layer from just before its first transient: silence the model put first is dropped."""
    level = np.abs(audio).max(axis=1) if audio.size else np.zeros(0)
    peak = float(level.max()) if level.size else 0.0
    if peak <= 0:
        return audio
    onset = int(np.argmax(level >= peak * ONSET_FRACTION))
    return audio[max(0, onset - int(PRE_ROLL_S * sample_rate)):]


def prepare_layer(raw: np.ndarray, layer: str, seconds: float, sample_rate: int) -> np.ndarray:
    audio = from_onset(remove_dc(raw), sample_rate)
    audio = fit_length(audio, int(seconds * sample_rate))
    return apply_fades(audio, sample_rate, fade_in_ms=0.5, fade_out_ms=LAYER_FADE_MS[layer])


def blend(layers: dict[str, np.ndarray], request: LayerRequest, sample_rate: int) -> np.ndarray:
    """Sum the prepared layers at their offsets and levels, normalized to -1 dBFS."""
    offsets = {"transient": 0.0, "body": request.body_offset_ms, "tail": request.tail_offset_ms}
    gains = dict(zip(LAYERS, request.gains_db))
    starts = {name: int(offsets[name] / 1000 * sample_rate) for name in layers}
    length = max(starts[name] + len(audio) for name, audio in layers.items())
    channels = max(audio.shape[1] for audio in layers.values())
    mix = np.zeros((length, channels))
    for name, audio in layers.items():
        start = starts[name]
        mix[start:start + len(audio), :audio.shape[1]] += audio * 10 ** (gains[name] / 20)
    return normalize_peak(mix, -1.0)

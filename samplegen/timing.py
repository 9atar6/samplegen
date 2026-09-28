"""Timing math: musical loop lengths and how much audio to ask the engine for.

ComfyUI's EmptyLatentAudio rounds the requested length to whole (even) latent
frames, so a request can come back a few ms short. We always ask for a little
extra and trim back to the exact sample count in post-processing. For loops,
that extra audio doubles as the material for the seamless-loop crossfade.
"""

import math
from dataclasses import dataclass

BEATS_PER_BAR = 4
GENERATION_PAD_SECONDS = 0.25
MIN_LATENT_SECONDS = 1.0  # EmptyLatentAudio's minimum


@dataclass(frozen=True)
class GenerationPlan:
    target_seconds: float
    target_samples: int
    latent_seconds: float
    conditioning_seconds: int


def loop_seconds(bpm: float, bars: int) -> float:
    if bpm <= 0 or bars <= 0:
        raise ValueError(f"bpm and bars must be positive (got bpm={bpm}, bars={bars})")
    return 60.0 / bpm * BEATS_PER_BAR * bars


def plan_generation(target_seconds: float, sample_rate: int) -> GenerationPlan:
    if target_seconds <= 0:
        raise ValueError(f"target_seconds must be positive (got {target_seconds})")
    return GenerationPlan(
        target_seconds=target_seconds,
        target_samples=int(round(target_seconds * sample_rate)),
        latent_seconds=max(MIN_LATENT_SECONDS, target_seconds + GENERATION_PAD_SECONDS),
        # Models were trained with whole-second duration conditioning.
        conditioning_seconds=max(1, math.ceil(round(target_seconds, 6))),
    )

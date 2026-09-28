"""What the user asked for, and whether it can run on this machine."""

import re
from dataclasses import dataclass, field

from .audio import ExportFormat
from .catalog import EDIT_OPS, KEYS, LOOP_BARS, LOOP_BPMS, MODELS, MODES, SCALES, SOURCE_MODES, build_loop_prompt
from .timing import loop_seconds

MAX_VARIATIONS = 8
MAX_PROMPT_CHARS = 2000
MIN_SECONDS = 0.5
MIN_REGION_SECONDS = 0.1
MAX_REGIONS = 8
MIN_STRENGTH = 0.05
# Batch size x length drives VRAM use; keep jobs within what an 8 GB card handles.
MAX_BATCH_AUDIO_SECONDS = 160.0
MAX_SEED = 2**32 - 1
STYLE_MODES = ("sfx", "free")


@dataclass(frozen=True)
class GenerationRequest:
    mode: str
    model: str
    prompt: str
    negative_prompt: str = ""
    duration: float = 4.0
    bpm: int = 120
    bars: int = 4
    key: str = "C"
    scale: str = "minor"
    variations: int = 4
    seed: int | None = None
    normalize_db: float | None = -1.0
    trim_silence: bool = True
    fade_in_ms: float = 0.0
    fade_out_ms: float = 5.0
    export: ExportFormat = field(default_factory=ExportFormat)
    # Transform / Edit
    source_id: str | None = None
    strength: float = 0.6
    source_is_loop: bool = False
    edit_op: str = "inpaint"
    regions: tuple[tuple[float, float], ...] = ()
    extend_seconds: float = 4.0
    # Trained style (LoRA slug) and how strongly to apply it
    style: str | None = None
    style_strength: float = 1.0

    @property
    def is_loop(self) -> bool:
        return self.mode == "loop"

    @property
    def uses_source(self) -> bool:
        return self.mode in SOURCE_MODES

    def target_seconds(self, source_seconds: float | None = None) -> float:
        """Length of the finished sample."""
        if self.is_loop:
            return loop_seconds(self.bpm, self.bars)
        if self.uses_source:
            if source_seconds is None:
                raise ValueError("Choose a source sound first.")
            if self.mode == "edit" and self.edit_op == "extend":
                return source_seconds + self.extend_seconds
            return source_seconds
        return float(self.duration)

    def full_prompt(self) -> str:
        if self.is_loop:
            return build_loop_prompt(self.prompt, self.bars, self.bpm, self.key, self.scale)
        return self.prompt.strip()

    def validate(self, source_seconds: float | None = None) -> None:
        """Raise ValueError with a user-facing message if the request can't run."""
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode '{self.mode}'.")
        spec = MODELS.get(self.model)
        if spec is None:
            raise ValueError(f"Unknown model '{self.model}'.")
        if self.mode not in spec.modes:
            raise ValueError(f"{spec.label} can't be used in {self.mode} mode.")
        if len(self.prompt) > MAX_PROMPT_CHARS or len(self.negative_prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Prompts are limited to {MAX_PROMPT_CHARS} characters.")
        if not self.is_loop and not self.prompt.strip():
            raise ValueError("Describe the sound you want.")
        if not 1 <= self.variations <= MAX_VARIATIONS:
            raise ValueError(f"Variations must be between 1 and {MAX_VARIATIONS}.")
        if self.seed is not None and not 0 <= self.seed <= MAX_SEED:
            raise ValueError("Seed is out of range.")
        if self.normalize_db is not None and not -30.0 <= self.normalize_db <= 0.0:
            raise ValueError("Normalize level must be between -30 and 0 dBFS.")
        if not (0 <= self.fade_in_ms <= 5000 and 0 <= self.fade_out_ms <= 5000):
            raise ValueError("Fades must be between 0 and 5000 ms.")

        if self.style:
            if self.mode not in STYLE_MODES:
                raise ValueError("Trained styles work in SFX and Free modes.")
            if not 0.0 <= self.style_strength <= 2.0:
                raise ValueError("Style strength must be between 0 and 2.")

        if self.is_loop:
            self._validate_loop()
        elif self.uses_source:
            self._validate_source(source_seconds)
        elif not MIN_SECONDS <= self.duration <= spec.max_seconds:
            raise ValueError(f"Duration must be between {MIN_SECONDS} and {spec.max_seconds:.0f} seconds.")

        total = self.target_seconds(source_seconds)
        if total > spec.max_seconds:
            raise ValueError(f"{spec.label} handles up to {spec.max_seconds:.0f} s; this would be {total:.1f} s.")
        if total * self.variations > MAX_BATCH_AUDIO_SECONDS:
            raise ValueError(
                f"That's too much audio for one batch on this GPU (max {MAX_BATCH_AUDIO_SECONDS:.0f} s total). "
                "Use fewer variations or a shorter duration."
            )

    def _validate_loop(self) -> None:
        if self.bpm not in LOOP_BPMS:
            raise ValueError(f"Loop BPM must be one of {', '.join(map(str, LOOP_BPMS))}.")
        if self.bars not in LOOP_BARS:
            raise ValueError(f"Loops can be {' or '.join(map(str, LOOP_BARS))} bars.")
        if self.key not in KEYS or self.scale not in SCALES:
            raise ValueError("Unknown key or scale.")

    def _validate_source(self, source_seconds: float | None) -> None:
        if not self.source_id or source_seconds is None:
            raise ValueError("Choose a source sound first.")
        if self.mode == "transform":
            if not MIN_STRENGTH <= self.strength <= 1.0:
                raise ValueError("Strength must be between 0.05 and 1.")
            return
        if self.edit_op not in EDIT_OPS:
            raise ValueError(f"Unknown edit operation '{self.edit_op}'.")
        if self.edit_op == "extend":
            if not MIN_SECONDS <= self.extend_seconds <= 120:
                raise ValueError("Extend by 0.5 to 120 seconds.")
            return
        if not 1 <= len(self.regions) <= MAX_REGIONS:
            raise ValueError("Select the part of the waveform to regenerate (drag across it).")
        for start, end in self.regions:
            if not (0 <= start < end <= source_seconds + 1e-6):
                raise ValueError(f"Selection {start:.2f}-{end:.2f} s is outside the source.")
            if end - start < MIN_REGION_SECONDS:
                raise ValueError("Selections must be at least 0.1 s long.")


def sample_name(request: GenerationRequest, source_name: str | None = None) -> str:
    if request.uses_source:
        base = source_name or "source"
        if request.mode == "transform":
            return f"{base} to {request.prompt.strip()}"
        return f"{base} {'extended' if request.edit_op == 'extend' else 'edit'} {request.prompt.strip()}"
    if not request.is_loop:
        return request.prompt.strip()
    tags = [t.strip() for t in request.prompt.split(",") if t.strip()][:3]
    scale = "min" if request.scale == "minor" else "maj"
    key = request.key.replace("#", "s")
    return "_".join([*tags, f"{request.bpm}bpm", f"{key}{scale}", f"{request.bars}bars"])


def wav_comment(request: GenerationRequest, seed: int, variation: int) -> str:
    text = f"samplegen | {request.model} | seed {seed} v{variation} | {request.full_prompt()}"
    return re.sub(r"\s+", " ", text)[:1000]

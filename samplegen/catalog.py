"""The models samplegen can drive, their sampling presets, and prompt vocabularies."""

from dataclasses import dataclass

SAMPLE_RATE = 44100


@dataclass(frozen=True)
class Sampling:
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    description: str
    checkpoint: str
    text_encoder: str
    sampling: Sampling
    modes: tuple[str, ...]
    max_seconds: float
    default_seconds: float


SA3_SAMPLING = Sampling(steps=8, cfg=1.0, sampler_name="lcm", scheduler="simple")
F1_SAMPLING = Sampling(steps=75, cfg=7.0, sampler_name="dpmpp_3m_sde_gpu", scheduler="exponential")

MODELS: dict[str, ModelSpec] = {
    "sa3-sfx": ModelSpec(
        key="sa3-sfx",
        label="Stable Audio 3 · Small SFX",
        description="Fast sound effects and one-shots. ~2 s per batch.",
        checkpoint="stable_audio_3_small_sfx.safetensors",
        text_encoder="t5gemma_b_b_ul2.safetensors",
        sampling=SA3_SAMPLING,
        modes=("sfx", "transform", "edit"),
        max_seconds=120.0,
        default_seconds=4.0,
    ),
    "sa3-medium": ModelSpec(
        key="sa3-medium",
        label="Stable Audio 3 · Medium",
        description="Highest quality: textures, ambiences, long SFX, musical material.",
        checkpoint="stable_audio_3_medium.safetensors",
        text_encoder="t5gemma_b_b_ul2.safetensors",
        sampling=SA3_SAMPLING,
        modes=("sfx", "free", "transform", "edit"),
        max_seconds=380.0,
        default_seconds=10.0,
    ),
    "f1-samples": ModelSpec(
        key="f1-samples",
        label="Foundation-1.2 · Samples",
        description="Tempo- and key-locked musical loops.",
        checkpoint="Foundation-1.2-Samples.safetensors",
        text_encoder="t5_base.safetensors",
        sampling=F1_SAMPLING,
        modes=("loop", "transform"),
        max_seconds=20.0,
        default_seconds=8.0,
    ),
}

MODELS["f1-keybeds"] = ModelSpec(
    key="f1-keybeds",
    label="Foundation-1.2 · Keybeds",
    description="Pitch-consistent notes across the keyboard, for playable instruments.",
    checkpoint="Foundation-1.2-Keybeds.safetensors",
    text_encoder="t5_base.safetensors",
    sampling=F1_SAMPLING,
    modes=("instrument",),
    max_seconds=20.0,
    default_seconds=19.25,
)

# Base (un-distilled) checkpoints: only used when a trained style is applied,
# because styles are trained on these. Slower (50 steps) but that's what the LoRA fits.
SA3_BASE_SAMPLING = Sampling(steps=50, cfg=7.0, sampler_name="lcm", scheduler="simple")
for _key, _file, _label in (("sa3-sfx-base", "stable_audio_3_small_sfx_base.safetensors", "SFX"),
                            ("sa3-music-base", "stable_audio_3_small_music_base.safetensors", "Music")):
    MODELS[_key] = ModelSpec(
        key=_key, label=f"Stable Audio 3 · Small {_label} (base, for styles)",
        description="Used automatically when you pick a trained style.",
        checkpoint=_file, text_encoder="t5gemma_b_b_ul2.safetensors", sampling=SA3_BASE_SAMPLING,
        modes=(), max_seconds=120.0, default_seconds=4.0,
    )

MODES = ("sfx", "loop", "free", "transform", "edit")
SOURCE_MODES = ("transform", "edit")
EDIT_OPS = ("inpaint", "extend")

# Foundation-1 was trained on these tempos / lengths; outside them loops drift.
LOOP_BPMS = (100, 110, 120, 128, 130, 140, 150)
LOOP_BARS = (4, 8)
KEYS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SCALES = ("major", "minor")

# Vocabulary from the Foundation-1 tag reference, grouped for the prompt builder.
LOOP_TAGS: dict[str, tuple[str, ...]] = {
    "Instrument": (
        "Synth Lead", "Synth Bass", "Pad", "Pluck", "Supersaw", "Reese Bass", "Sub Bass", "FM Bass",
        "Wavetable Bass", "Analog Bass", "Acid", "Atmosphere", "Texture", "Bell", "Grand Piano",
        "Rhodes Piano", "Felt Piano", "Digital Piano", "Wurlitzer Piano", "Organ", "Hammond Organ",
        "Church Organ", "Digital Strings", "Violin", "Cello", "Viola", "Choir", "Harp", "Flute",
        "Pan Flute", "Trumpet", "Brass", "French Horn", "Saxophone", "Clarinet", "Oboe",
        "Electric Guitar", "Acoustic Guitar", "Nylon Guitar", "Electric Bass", "Marimba", "Kalimba",
        "Vibraphone", "Xylophone", "Glockenspiel", "Music Box", "Celesta", "Steel Drums", "Koto",
        "Sitar", "Chiptune",
    ),
    "Timbre": (
        "Warm", "Bright", "Dark", "Wide", "Thick", "Thin", "Airy", "Rich", "Tight", "Full", "Gritty",
        "Clean", "Retro", "Vintage", "Analog", "Digital", "Snappy", "Crisp", "Punchy", "Focused",
        "Metallic", "Glassy", "Shiny", "Sparkly", "Silky", "Smooth", "Soft", "Spacey", "Dreamy",
        "Cold", "Buzzy", "Big", "Small", "Subdued", "Deep", "Fat", "Round", "Hollow", "Woody",
        "Breathy", "Nasal", "Harsh", "Growl", "Overdriven", "Bitcrushed", "Muffled", "Distant",
        "Near", "Intimate", "Heavy", "Pitch Bend", "Formant Vocal", "808", "303", "Saw", "Square",
        "Sine", "Pulse",
    ),
    "FX": (
        "Dry", "Wet", "Low Reverb", "Medium Reverb", "High Reverb", "Low Delay", "Medium Delay",
        "High Delay", "Cross Delay", "Low Distortion", "Medium Distortion", "High Distortion",
        "Phaser", "High Phaser", "Bitcrush",
    ),
    "Structure": (
        "Melody", "Complex Melody", "Chord Progression", "Arp", "Complex Arp Melody", "Bassline",
        "Simple", "Complex", "Rising", "Epic", "Choppy", "Rolling", "Alternating", "Sustained",
        "Staccato", "Pizzicato", "Slow Speed", "Fast Speed",
    ),
}


def models_for_mode(mode: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if mode in m.modes]


def build_loop_prompt(tags: str, bars: int, bpm: int, key: str, scale: str) -> str:
    """Foundation-1's trained prompt layout: tags first, then bars, BPM, key."""
    cleaned = ", ".join(t.strip() for t in tags.split(",") if t.strip())
    musical = f"{bars} Bars, {bpm} BPM, {key} {scale}"
    return f"{cleaned}, {musical}" if cleaned else musical

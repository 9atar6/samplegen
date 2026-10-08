"""Playable instruments from Foundation-1.2 Keybeds.

The model generates up to six chromatic notes per pass: each note 3.0 s long,
0.25 s apart (19.25 s for six). One seed is reused for every pass so the timbre
stays consistent across the keyboard. The passes are sliced at those fixed
boundaries and assembled into a Decent Sampler preset and an SFZ file.

Prompt grammar (from the model's keybed_training_strategy.md):
    Keybed, Sequence, Timbre Profile, <descriptor>, <Dry|Wet + FX>, Chromatic Chunk, Note Sequence, C4, C#4, ...
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .audio import ExportFormat, apply_fades, fit_length, write_wav
from .loops import load_loops, note_loops, save_loops

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTES_PER_CHUNK = 6
NOTE_SECONDS = 3.0
GAP_SECONDS = 0.25
LOWEST_MIDI, HIGHEST_MIDI = 24, 96  # C1 .. C7
MAX_NOTES = 61  # five octaves + 1
EDGE_STRETCH = 6  # semitones the outermost samples cover beyond the generated range
PREVIEW_NOTE_SECONDS = 0.6
# The model doesn't always start a note on the grid: some come in up to a second late, after a
# breath or the previous note's tail. Each note starts where it gets within 20 dB of its peak
# (analysis.py's onset), less a few ms so the attack stays whole. Measured on real keybeds this
# removes 300-600 ms of lag from late notes and at most ~70 ms of a pad's soft lead-in.
ONSET_DB = -20.0
ONSET_WINDOW_S = 0.01
ONSET_PRE_ROLL_S = 0.005
MAX_NOTE_LEAD_S = 1.5
SPACES = ("Dry", "Wet")
INSTRUMENT_DIR = "Instruments"


def midi_to_name(midi: int) -> str:
    """60 -> 'C4' (middle C = C4)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name: str) -> int:
    match = re.fullmatch(r"([A-Ga-g])(#|b)?(-?\d)", name.strip())
    if not match:
        raise ValueError(f"'{name}' isn't a note name like C4 or F#2.")
    letter, accidental, octave = match.groups()
    semitone = NOTE_NAMES.index(letter.upper()) + {"#": 1, "b": -1, None: 0}[accidental]
    return (int(octave) + 1) * 12 + semitone


def chunk_notes(low: int, high: int) -> list[list[int]]:
    notes = list(range(low, high + 1))
    return [notes[i:i + NOTES_PER_CHUNK] for i in range(0, len(notes), NOTES_PER_CHUNK)]


def chunk_seconds(note_count: int) -> float:
    return note_count * NOTE_SECONDS + (note_count - 1) * GAP_SECONDS


def keybed_prompt(descriptor: str, space: str, notes: list[int]) -> str:
    parts = ["Keybed", "Sequence", "Timbre Profile"]
    parts += [t.strip() for t in descriptor.split(",") if t.strip()]
    parts += [space, "Chromatic Chunk", "Note Sequence"]
    parts += [midi_to_name(n) for n in notes]
    return ", ".join(parts)


@dataclass(frozen=True)
class InstrumentRequest:
    prompt: str  # timbre descriptor, e.g. "Grand Piano, Warm, Gritty"
    name: str = ""
    low_note: str = "C3"
    high_note: str = "B4"
    space: str = "Dry"
    seed: int | None = None
    export: ExportFormat = field(default_factory=lambda: ExportFormat(sample_rate=44100, bit_depth="24"))
    mode: str = "instrument"
    model: str = "f1-keybeds"
    variations: int = 1
    uses_source = False

    def full_prompt(self) -> str:
        return f"{self.prompt.strip()} · {self.low_note}–{self.high_note} · {self.space}"

    def midi_range(self) -> tuple[int, int]:
        return name_to_midi(self.low_note), name_to_midi(self.high_note)

    def validate(self, source_seconds: float | None = None) -> None:
        if not self.prompt.strip():
            raise ValueError("Describe the instrument, e.g. 'Grand Piano, Warm, Gritty'.")
        if len(self.prompt) > 500 or len(self.name) > 80:
            raise ValueError("The description or name is too long.")
        if self.space not in SPACES:
            raise ValueError("Choose Dry or Wet.")
        low, high = self.midi_range()
        if not LOWEST_MIDI <= low <= high <= HIGHEST_MIDI:
            raise ValueError(f"Notes must go from low to high, within {midi_to_name(LOWEST_MIDI)}–{midi_to_name(HIGHEST_MIDI)}.")
        if high - low + 1 > MAX_NOTES:
            raise ValueError(f"Up to {MAX_NOTES} notes (five octaves) per instrument.")
        if self.seed is not None and not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed is out of range.")


def note_onset(audio: np.ndarray, sample_rate: int) -> int:
    """Frames of lead-in before a note's attack (0 if it starts on time, or is silent)."""
    if not audio.size:
        return 0
    window = max(1, int(ONSET_WINDOW_S * sample_rate))
    power = (audio ** 2).mean(axis=1)
    # Trailing RMS window: a click can't trigger it, and it can only fire late, never early.
    level = np.sqrt(np.convolve(power, np.ones(window) / window, "full")[:len(power)])
    peak = float(level.max())
    if peak <= 0:
        return 0
    onset = int(np.argmax(level >= peak * 10 ** (ONSET_DB / 20)))
    onset -= window + int(ONSET_PRE_ROLL_S * sample_rate)
    return min(max(0, onset), int(MAX_NOTE_LEAD_S * sample_rate))


def slice_chunk(raw: np.ndarray, notes: list[int], sample_rate: int) -> dict[int, np.ndarray]:
    """Cut one generated pass into its notes at the fixed 3.0 s / 0.25 s grid, each one
    starting at its attack (a late note ends up a little shorter, never overlapping the next)."""
    note_frames = int(NOTE_SECONDS * sample_rate)
    slices = {}
    for index, midi in enumerate(notes):
        start = int(round(index * (NOTE_SECONDS + GAP_SECONDS) * sample_rate))
        note = fit_length(raw[start:start + note_frames], note_frames)
        lead = note_onset(note, sample_rate)
        note = note[lead:]
        # Cut into a sounding note: a softer fade-in so the cut never clicks.
        slices[midi] = apply_fades(note, sample_rate, fade_in_ms=10.0 if lead else 2.0, fade_out_ms=40.0)
    return slices


def normalize_together(notes: dict[int, np.ndarray], target_db: float = -1.0) -> dict[int, np.ndarray]:
    """One gain for the whole instrument, so it keeps its dynamics across the keyboard."""
    peak = max((float(np.abs(a).max()) for a in notes.values() if a.size), default=0.0)
    if peak <= 0:
        return dict(notes)
    gain = 10 ** (target_db / 20) / peak
    return {midi: audio * gain for midi, audio in notes.items()}


def preview_run(notes: dict[int, np.ndarray], sample_rate: int) -> np.ndarray:
    """A quick scale run through every note, for auditioning in the library."""
    frames = int(PREVIEW_NOTE_SECONDS * sample_rate)
    parts = [apply_fades(notes[m][:frames], sample_rate, fade_out_ms=60.0) for m in sorted(notes)]
    return np.concatenate(parts) if parts else np.zeros((0, 2))


def key_ranges(midis: list[int]) -> dict[int, tuple[int, int]]:
    """Each sample covers its own key; the outermost ones stretch past the range."""
    ordered = sorted(midis)
    ranges = {m: (m, m) for m in ordered}
    if ordered:
        low, high = ordered[0], ordered[-1]
        ranges[low] = (max(0, low - EDGE_STRETCH), ranges[low][1])
        ranges[high] = (ranges[high][0], min(127, high + EDGE_STRETCH))
    return ranges


def _ds_loop(loop: dict | None) -> str:
    if not loop:
        return ""
    # Decent Sampler's loopEnd is the last frame inside the loop (inclusive).
    return (f' loopEnabled="true" loopStart="{loop["start"]}" loopEnd="{loop["end"] - 1}"'
            f' loopCrossfade="{loop["crossfade"]}" loopCrossfadeMode="linear"')


def decent_sampler_preset(title: str, files: dict[int, str], loops: dict[int, dict] | None = None) -> str:
    ranges = key_ranges(list(files))
    loops = loops or {}
    regions = "\n".join(
        f'      <sample path={quoteattr(path)} rootNote="{midi}" loNote="{ranges[midi][0]}" '
        f'hiNote="{ranges[midi][1]}" loVel="0" hiVel="127"{_ds_loop(loops.get(midi))}/>'
        for midi, path in sorted(files.items())
    )
    knob = ('      <labeled-knob x="{x}" y="40" width="90" label="{label}" type="float" minValue="{lo}" '
            'maxValue="{hi}" value="{val}" textColor="FFFFFFFF">\n'
            '        <binding type="amp" level="instrument" position="0" parameter="{param}"/>\n'
            '      </labeled-knob>')
    knobs = "\n".join([
        knob.format(x=20, label="Attack", lo=0, hi=4, val=0.005, param="ENV_ATTACK"),
        knob.format(x=120, label="Release", lo=0, hi=8, val=0.6, param="ENV_RELEASE"),
    ])
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<DecentSampler minVersion="1.0.0">\n'
        '  <ui width="812" height="375" bgColor="FF141414">\n'
        '    <tab name="main">\n'
        f'      <label x="20" y="10" width="700" height="24" text={quoteattr(title)} textColor="FFFF9A66"/>\n'
        f'{knobs}\n'
        '    </tab>\n'
        '  </ui>\n'
        '  <groups attack="0.005" decay="1" sustain="1" release="0.6">\n'
        '    <group>\n'
        f'{regions}\n'
        '    </group>\n'
        '  </groups>\n'
        '</DecentSampler>\n'
    )


def sfz_file(title: str, files: dict[int, str], loops: dict[int, dict] | None = None) -> str:
    ranges = key_ranges(list(files))
    loops = loops or {}
    lines = [f"// {title} (samplegen / Foundation-1.2 Keybeds)",
             "<global> ampeg_attack=0.005 ampeg_release=0.6"]
    for midi, path in sorted(files.items()):
        lo, hi = ranges[midi]
        region = f"<region> sample={path} pitch_keycenter={midi} lokey={lo} hikey={hi}"
        if loop := loops.get(midi):
            # loop_crossfade is in seconds (sfizz / ARIA); players without it still loop cleanly enough
            region += (f" loop_mode=loop_sustain loop_start={loop['start']} loop_end={loop['end'] - 1}"
                       f" loop_crossfade={loop['crossfade'] / loop['rate']:.4f}")
        lines.append(region)
    return "\n".join(lines) + "\n"


def write_instrument(folder: Path, title: str, file_stem: str, notes: dict[int, np.ndarray], sample_rate: int,
                     fmt: ExportFormat) -> dict[int, str]:
    """Write Samples/*.wav + <stem>.dspreset + <stem>.sfz. Returns midi -> relative sample path."""
    samples_dir = folder / "Samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for midi, audio in sorted(notes.items()):
        rel = f"Samples/{file_stem}_{midi_to_name(midi).replace('#', 's')}.wav"
        write_wav(folder / rel, audio, sample_rate, fmt, title=f"{title} {midi_to_name(midi)}")
        files[midi] = rel
    # Loop points come from the written files: they're in the export's sample rate.
    loops = note_loops({midi: folder / rel for midi, rel in files.items()})
    save_loops(folder, loops)
    write_presets(folder, title, file_stem, files, loops)
    return files


def write_presets(folder: Path, title: str, file_stem: str, files: dict[int, str], loops: dict[int, dict]) -> None:
    (folder / f"{file_stem}.dspreset").write_text(decent_sampler_preset(title, files, loops), encoding="utf-8")
    (folder / f"{file_stem}.sfz").write_text(sfz_file(title, files, loops), encoding="utf-8")


def ensure_loops(folder: Path, notes: dict[int, Path]) -> dict[int, dict]:
    """Loop points for an instrument; computed (and its presets updated) the first time,
    so instruments made before sustain loops existed get them too."""
    loops = load_loops(folder)
    if loops is not None:
        return loops
    loops = note_loops(notes)
    save_loops(folder, loops)
    presets = sorted(Path(folder).glob("*.dspreset"))
    if presets:
        stem = presets[0].stem
        files = {midi: path.relative_to(folder).as_posix() for midi, path in notes.items()}
        write_presets(folder, Path(folder).name, stem, files, loops)
    return loops

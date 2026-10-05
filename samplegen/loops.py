"""Sustain loops for instrument notes, so a held key keeps sounding.

The generated notes are ~3 s long. Like a hardware sampler, we pick a stretch of each
note's sustain whose two ends match (same waveform shape, similar level), and loop it
with a short crossfade from the audio just before the loop start. Nothing is baked into
the WAVs: the loop points go into the presets (Decent Sampler / SFZ) and loops.json,
which the Play tab reads.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

CROSSFADE_S = 0.05
MIN_LOOP_S = 0.3
WINDOW_S = 0.008          # waveform comparison around each loop point
LEVEL_WINDOW_S = 0.04     # loudness comparison: decaying notes must not "pulse" at the loop
END_RANGE = (0.62, 0.88)  # loop end: late in the note, but before its fade-out
START_FROM = 0.3          # loop start: after the attack
MAX_CANDIDATES = 60
LOOPS_FILE = "loops.json"


def _rising_zero_crossings(mono: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo, hi = max(1, lo), min(len(mono) - 1, hi)
    seg = mono[lo - 1:hi]
    return np.flatnonzero((seg[:-1] < 0) & (seg[1:] >= 0)) + lo


def _thin(points: np.ndarray, count: int) -> np.ndarray:
    return points if len(points) <= count else points[np.linspace(0, len(points) - 1, count).astype(int)]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def find_loop(audio: np.ndarray, sample_rate: int) -> tuple[int, int, int] | None:
    """(start, end, crossfade) in frames — the loop plays [start, end) — or None if the
    note is too short or too quiet to loop. `audio` is (frames, channels)."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = len(mono)
    fade = int(CROSSFADE_S * sample_rate)
    w = max(8, int(WINDOW_S * sample_rate))
    lw = int(LEVEL_WINDOW_S * sample_rate)
    min_len = int(MIN_LOOP_S * sample_rate)
    if n < sample_rate or _rms(mono) < 1e-4:
        return None
    ends = _thin(_rising_zero_crossings(mono, int(n * END_RANGE[0]), int(n * END_RANGE[1])), MAX_CANDIDATES)
    starts = _rising_zero_crossings(mono, max(int(n * START_FROM), fade + w, lw), int(n * END_RANGE[1]) - min_len)
    starts = _thin(starts, MAX_CANDIDATES * 3)
    if not len(ends) or not len(starts):
        return None
    # All start candidates at once: their waveform windows and loudness.
    windows = np.stack([mono[s - w:s + w] for s in starts])
    window_norms = np.linalg.norm(windows, axis=1)
    start_levels = np.array([_rms(mono[s:s + lw]) for s in starts])
    best, best_score = None, -np.inf
    for end in ends:
        if end + w > n:
            continue
        ok = starts <= end - min_len
        if not ok.any():
            continue
        around_end = mono[end - w:end + w]
        denom = window_norms[ok] * np.linalg.norm(around_end)
        shape = np.divide(windows[ok] @ around_end, denom, out=np.zeros(int(ok.sum())), where=denom > 0)
        end_level = _rms(mono[end - lw:end])
        levels = start_levels[ok]
        level = np.minimum(levels, end_level) / np.maximum(np.maximum(levels, end_level), 1e-9)
        score = shape + level + 0.15 * (end - starts[ok]) / n  # longer loops sound less mechanical
        i = int(np.argmax(score))
        if score[i] > best_score:
            best, best_score = (int(starts[ok][i]), int(end)), float(score[i])
    if best is None:
        return None
    start, end = best
    return start, end, min(fade, start, end - start)


def note_loops(notes: dict[int, Path]) -> dict[int, dict]:
    """midi -> {"start", "end", "crossfade", "rate"} for every note file that loops."""
    loops = {}
    for midi, path in sorted(notes.items()):
        audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
        found = find_loop(audio, rate)
        if found:
            start, end, crossfade = found
            loops[midi] = {"start": start, "end": end, "crossfade": crossfade, "rate": rate}
    return loops


def load_loops(folder: Path) -> dict[int, dict] | None:
    try:
        data = json.loads((Path(folder) / LOOPS_FILE).read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}
    except (OSError, ValueError, AttributeError):
        return None


def save_loops(folder: Path, loops: dict[int, dict]) -> None:
    path = Path(folder) / LOOPS_FILE
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps({str(k): v for k, v in loops.items()}), encoding="utf-8")
    tmp.replace(path)

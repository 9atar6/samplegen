"""Acoustic profile of a sound, and plain words for it.

Everything here is measured, so it's reliable in a way an AI tagger isn't:
length, attack, decay, pitch (and whether there is one), pitch sweep, dynamics,
low-end weight, brightness, stereo width. `describe()` turns the numbers into
short phrases for captions. Arrays are (frames, channels) floats.
"""

from dataclasses import dataclass

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class Features:
    length_s: float
    attack_ms: float          # onset to peak
    decay_s: float            # peak until the envelope drops 40 dB
    tonal: bool               # a clear pitch exists (vs noise-like)
    pitch_hz: float | None    # pitch of the body
    pitch_sweep_st: float     # semitones from the attack to the body (+ = falling)
    crest: float              # peak / RMS over the body: high = transient, low = dense
    sub_share: float          # energy below 90 Hz
    centroid_hz: float        # spectral brightness
    width: float              # 0 = mono, grows with stereo difference
    sustained: bool           # doesn't really decay within the clip


def _env(mono: np.ndarray, sr: int) -> np.ndarray:
    win = max(1, int(0.005 * sr))
    return np.convolve(np.abs(mono), np.ones(win) / win, mode="same")


def _spectrum(seg: np.ndarray, sr: int, min_fft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    n = 1 << int(np.ceil(np.log2(max(len(seg), min_fft))))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    return spec, np.fft.rfftfreq(n, 1 / sr)


def _peak_freq(seg: np.ndarray, sr: int, lo: float, hi: float) -> tuple[float | None, float]:
    """Strongest frequency in [lo, hi] and how much it stands out (peak / median in band)."""
    if len(seg) < 64:
        return None, 0.0
    spec, freqs = _spectrum(seg, sr)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any() or spec[band].max() <= 0:
        return None, 0.0
    values = spec[band]
    peak = float(values.max())
    prominence = peak / (float(np.median(values)) + 1e-12)
    return float(freqs[band][np.argmax(values)]), prominence


def note_name(freq: float) -> str:
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    return NOTE_NAMES[midi % 12]


def analyze(audio: np.ndarray, sr: int) -> Features:
    if audio.ndim == 1:
        audio = audio[:, None]
    mono = audio.mean(axis=1)
    length = len(mono) / sr
    if len(mono) == 0 or not np.any(mono):
        return Features(length, 0.0, 0.0, False, None, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    env = _env(mono, sr)
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i])
    above = np.flatnonzero(env > peak * 0.1)  # onset: first time within 20 dB of the peak
    onset = int(above[0]) if above.size else 0
    attack_ms = (peak_i - onset) / sr * 1000
    tail = np.flatnonzero(env[peak_i:] > peak * 10 ** (-40 / 20))
    decay = (tail[-1] / sr) if tail.size else 0.0
    last_quarter = env[int(len(env) * 0.75):]
    sustained = length > 1.5 and last_quarter.size > 0 and float(last_quarter.mean()) > peak * 0.2

    early = mono[onset + int(0.005 * sr): onset + int(0.035 * sr)]
    body = mono[onset + int(0.05 * sr): onset + int(min(length, 0.5) * sr)]
    if len(body) < int(0.03 * sr):
        body = mono[onset:]
    f_body, prom = _peak_freq(body, sr, 25, 2000)
    tonal = f_body is not None and prom > 25
    sweep = 0.0
    if tonal:
        # Look for the attack pitch near the body's (a sweep, not a harmonic further up).
        f_early, _ = _peak_freq(early, sr, f_body * 0.8, min(2000.0, f_body * 8))
        if f_early:
            sweep = float(12 * np.log2(f_early / f_body))

    raw_peak = int(np.argmax(np.abs(mono)))  # the true sample peak, not the smoothed envelope's
    hit = mono[raw_peak: raw_peak + int(0.12 * sr)]
    rms = float(np.sqrt(np.mean(hit ** 2))) if hit.size else 0.0
    crest = float(np.abs(mono[raw_peak]) / rms) if rms > 0 else 0.0

    whole = mono[onset: onset + int(min(length, 2.0) * sr)]
    whole = (whole - whole.mean()) * np.hanning(len(whole))  # DC and edge leakage aren't "sub"
    spec = np.abs(np.fft.rfft(whole)) ** 2
    freqs = np.fft.rfftfreq(len(whole), 1 / sr)
    total = float(spec.sum()) or 1e-12
    sub_share = float(spec[freqs < 90].sum()) / total
    centroid = float((freqs * spec).sum() / total)

    width = 0.0
    if audio.shape[1] >= 2:
        left, right = audio[:, 0], audio[:, 1]
        denom = float(np.sqrt((left ** 2).sum() * (right ** 2).sum()))
        width = 1.0 - float((left * right).sum()) / denom if denom > 0 else 0.0

    return Features(
        length_s=length, attack_ms=attack_ms, decay_s=decay, tonal=tonal,
        pitch_hz=f_body if tonal else None, pitch_sweep_st=sweep, crest=crest,
        sub_share=sub_share, centroid_hz=centroid, width=width, sustained=sustained,
    )


def describe(f: Features) -> list[str]:
    """Short phrases, most telling first. Only states what the numbers clearly support."""
    words: list[str] = []
    percussive = not f.sustained and f.decay_s < 2.5

    if f.attack_ms < 6:
        words.append("sharp attack")
    elif f.attack_ms > 120:
        words.append("slow swell")

    if percussive and f.crest > 3.3:
        words.append("very punchy transient")
    elif f.crest and f.crest < 1.8:
        words.append("dense and saturated")

    if f.sub_share > 0.8 and f.centroid_hz < 200:
        words.append("deep sub heavy")
    elif f.centroid_hz > 4000:
        words.append("bright")
    elif f.centroid_hz < 350 and f.sub_share < 0.5:
        words.append("dark")

    if f.tonal and f.pitch_hz and f.decay_s >= 0.12:
        words.append(f"tuned around {note_name(f.pitch_hz)}")
    if f.tonal and percussive:
        if f.pitch_sweep_st > 10:
            words.append("strong pitch sweep")
        elif f.pitch_sweep_st > 3:
            words.append("pitch drop")
        elif abs(f.pitch_sweep_st) <= 1.5 and f.decay_s >= 0.12:
            words.append("steady pitch")
        elif f.pitch_sweep_st < -3:
            words.append("rising pitch")
    elif not f.tonal and f.length_s > 0.3:
        words.append("noisy")

    if f.sustained:
        words.append("sustained")
    elif f.decay_s < 0.15:
        words.append("very short")
    elif f.decay_s < 0.35:
        words.append("short decay")
    elif f.decay_s < 0.8:
        words.append("medium decay")
    elif f.decay_s < 2.5:
        words.append("long tail")
    else:
        words.append("long evolving tail")

    if f.width > 0.15:
        words.append("wide stereo")
    return words

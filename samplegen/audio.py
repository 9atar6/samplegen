"""Audio I/O and post-processing. All functions are pure: they return new arrays.

Arrays are float64, shaped (frames, channels).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SUPPORTED_RATES = (44100, 48000, 88200, 96000)
BIT_DEPTH_SUBTYPES = {"32f": "FLOAT", "24": "PCM_24", "16": "PCM_16"}
SILENCE_DB = -60.0


@dataclass(frozen=True)
class ExportFormat:
    sample_rate: int = 44100
    bit_depth: str = "32f"

    def __post_init__(self):
        if self.sample_rate not in SUPPORTED_RATES:
            raise ValueError(f"sample_rate must be one of {SUPPORTED_RATES}")
        if self.bit_depth not in BIT_DEPTH_SUBTYPES:
            raise ValueError(f"bit_depth must be one of {tuple(BIT_DEPTH_SUBTYPES)}")


@dataclass(frozen=True)
class PostOptions:
    target_samples: int
    is_loop: bool
    normalize_db: float | None = -1.0
    trim_silence: bool = False
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0
    loop_crossfade_ms: float = 10.0


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return data, sr


def resample_loop(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a loop as the cycle it is: the filter sees the loop's own end before its
    start (not silence), so the wrap stays seamless, and the length stays exact."""
    frames = round(len(audio) * to_rate / from_rate)
    tiled = soxr.resample(np.concatenate([audio, audio, audio]), from_rate, to_rate, quality="VHQ")
    middle = tiled[frames:2 * frames]
    return fit_length(middle, frames)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, fmt: ExportFormat,
              title: str = "", comment: str = "", loop: bool = False) -> None:
    """Resample if needed, then write atomically (via a .part file)."""
    out = audio
    if fmt.sample_rate != sample_rate:
        out = (resample_loop(audio, sample_rate, fmt.sample_rate) if loop
               else soxr.resample(audio, sample_rate, fmt.sample_rate, quality="VHQ"))
    if fmt.bit_depth == "16":
        out = _tpdf_dither(out, bits=16)
    if fmt.bit_depth != "32f":
        # Integer formats can't go past full scale: clip instead of wrapping around
        # (resampling overshoot, unnormalized stems, a 0 dB normalize target + dither).
        lsb = 1.0 / (2 ** (int(fmt.bit_depth) - 1))
        out = np.clip(out, -1.0, 1.0 - lsb)

    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with sf.SoundFile(str(tmp), "w", samplerate=fmt.sample_rate, channels=out.shape[1],
                      subtype=BIT_DEPTH_SUBTYPES[fmt.bit_depth], format="WAV") as f:
        if title:
            f.title = title
        if comment:
            f.comment = comment
        f.software = "samplegen"
        f.write(out)
    os.replace(tmp, path)


def _tpdf_dither(audio: np.ndarray, bits: int) -> np.ndarray:
    lsb = 1.0 / (2 ** (bits - 1))
    rng = np.random.default_rng()
    noise = (rng.random(audio.shape) - rng.random(audio.shape)) * lsb
    return audio + noise


def remove_dc(audio: np.ndarray) -> np.ndarray:
    return audio - audio.mean(axis=0, keepdims=True)


NEAR_SILENCE = 1e-4  # -80 dBFS: below this, "normalizing" would only turn noise into a wall of hiss


def normalize_peak(audio: np.ndarray, target_db: float) -> np.ndarray:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if not peak > NEAR_SILENCE:
        return audio.copy()
    return audio * (db_to_gain(target_db) / peak)


def fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    """Cut or zero-pad to exactly `frames`."""
    if len(audio) >= frames:
        return audio[:frames].copy()
    pad = np.zeros((frames - len(audio), audio.shape[1]))
    return np.concatenate([audio, pad])


def trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = SILENCE_DB,
                 preroll_ms: float = 2.0, tail_ms: float = 30.0) -> np.ndarray:
    """Cut leading/trailing audio below threshold, keeping a little air around it.

    The threshold is relative to the clip's own peak (it runs before normalizing),
    so quiet and loud takes are trimmed the same way.
    """
    level = np.abs(audio).max(axis=1) if audio.size else np.zeros(0)
    peak = float(level.max()) if level.size else 0.0
    loud = np.flatnonzero(level > peak * db_to_gain(threshold_db)) if peak > 0 else np.zeros(0, dtype=int)
    if loud.size == 0:
        return audio.copy()
    start = max(0, loud[0] - int(preroll_ms / 1000 * sample_rate))
    end = min(len(audio), loud[-1] + 1 + int(tail_ms / 1000 * sample_rate))
    return audio[start:end].copy()


def _ramp(length: int) -> np.ndarray:
    """Smooth (sin²) ramp from 0 to 1, shape (length, 1). Paired with 1 - ramp it keeps
    constant amplitude: right for correlated material (loop wrap, splices)."""
    return (np.sin(np.linspace(0.0, np.pi / 2, length)) ** 2)[:, None]


def apply_fades(audio: np.ndarray, sample_rate: int, fade_in_ms: float = 0.0,
                fade_out_ms: float = 0.0) -> np.ndarray:
    out = audio.copy()
    n_in = int(fade_in_ms / 1000 * sample_rate)
    n_out = int(fade_out_ms / 1000 * sample_rate)
    if n_in + n_out > len(out):  # short clip: shrink both fades so they meet instead of overlapping
        scale = len(out) / (n_in + n_out)
        n_in, n_out = int(n_in * scale), int(n_out * scale)
    if n_in > 0:
        out[:n_in] *= _ramp(n_in)
    if n_out > 0:
        out[-n_out:] *= _ramp(n_out)[::-1]
    return out


def make_seamless_loop(raw: np.ndarray, loop_frames: int, sample_rate: int,
                       crossfade_ms: float = 10.0) -> np.ndarray:
    """Exact-length loop whose end flows into its start.

    The model keeps playing past the loop point, so the audio right after
    `loop_frames` is the natural continuation. Crossfading that continuation
    into the loop start makes the wrap-around sound like the music carrying on.
    Without enough continuation we fall back to a short fade-out.
    """
    fade = int(crossfade_ms / 1000 * sample_rate)
    loop = fit_length(raw, loop_frames)
    if fade <= 0:
        return loop
    if len(raw) < loop_frames + fade:  # both ends to zero, or the wrap would click
        return apply_fades(loop, sample_rate, fade_in_ms=crossfade_ms, fade_out_ms=crossfade_ms)
    ramp = _ramp(fade)
    continuation = raw[loop_frames:loop_frames + fade]
    loop[:fade] = loop[:fade] * ramp + continuation * (1.0 - ramp)
    return loop


def region_mask(frames: int, regions: list[tuple[float, float]], sample_rate: int) -> np.ndarray:
    """Boolean (frames,) mask, True inside any (start_s, end_s) region."""
    mask = np.zeros(frames, dtype=bool)
    for start, end in regions:
        mask[max(0, int(start * sample_rate)):max(0, int(end * sample_rate))] = True
    return mask


MAX_MATCH_GAIN = db_to_gain(12.0)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0


def splice_regions(original: np.ndarray, generated: np.ndarray, regions: list[tuple[float, float]],
                   sample_rate: int, crossfade_ms: float = 20.0) -> np.ndarray:
    """Original audio outside `regions`, generated audio inside, crossfaded at the edges.

    The engine re-synthesises the whole clip (and its decoder rescales loud
    output), so the generated take is level-matched to the original on the
    parts we keep before splicing. Kept parts stay bit-identical to the source.
    """
    frames = len(original)
    generated = fit_length(generated, frames)
    mask = region_mask(frames, regions, sample_rate)
    keep = ~mask
    gen_level, orig_level = _rms(generated[keep]), _rms(original[keep])
    if gen_level > 0 and orig_level > 0:  # clamped: a near-silent take must not be boosted 60 dB
        generated = generated * float(np.clip(orig_level / gen_level, MAX_MATCH_GAIN ** -1, MAX_MATCH_GAIN))

    fade = int(crossfade_ms / 1000 * sample_rate)
    weight = mask.astype(float)
    if fade > 1:
        padded = np.pad(weight, (fade, fade), mode="edge")
        weight = np.convolve(padded, np.ones(fade) / fade, mode="same")[fade:-fade]
    return original * (1.0 - weight)[:, None] + generated * weight[:, None]


def postprocess(raw: np.ndarray, sample_rate: int, options: PostOptions) -> np.ndarray:
    audio = remove_dc(raw)
    if options.is_loop:
        audio = make_seamless_loop(audio, options.target_samples, sample_rate, options.loop_crossfade_ms)
    else:
        audio = fit_length(audio, options.target_samples)
        if options.trim_silence:
            audio = trim_silence(audio, sample_rate)
        audio = apply_fades(audio, sample_rate, options.fade_in_ms, options.fade_out_ms)
    if options.normalize_db is not None:
        audio = normalize_peak(audio, options.normalize_db)
    return audio

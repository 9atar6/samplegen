"""Integrated loudness (LUFS, ITU-R BS.1770-4) and loudness matching, numpy only.

K-weighting is applied in the frequency domain at 48 kHz: only the energy of the filtered
signal matters for loudness, so the filters' magnitude response is all we need (the
measurement matches the time-domain filters to well under 0.1 LU).
"""

import numpy as np
import soxr

MEASURE_RATE = 48000
BLOCK_S, HOP_S = 0.4, 0.1  # 400 ms blocks, 75 % overlap
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
PEAK_CEILING_DB = -1.0

# BS.1770 K-weighting at 48 kHz: high-shelf (head effects), then a high-pass (RLB).
SHELF = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
HIGHPASS = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def _response(coeffs, freqs: np.ndarray, rate: int) -> np.ndarray:
    b, a = coeffs
    z = np.exp(-2j * np.pi * freqs / rate)
    return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)


BATCH_BLOCKS = 64  # blocks transformed at once: bounded memory even for a 6-minute take


def _k_gain_squared(size: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(size, 1 / MEASURE_RATE)
    gain = np.abs(_response(SHELF, freqs, MEASURE_RATE) * _response(HIGHPASS, freqs, MEASURE_RATE)) ** 2
    weights = np.full(len(freqs), 2.0)  # Parseval for a real signal: inner bins count twice
    weights[0] = 1.0
    weights[-1] = 1.0
    return gain * weights / size


def block_powers(audio: np.ndarray) -> np.ndarray:
    """K-weighted mean square (summed over channels) of each 400 ms block, 48 kHz audio.
    Each block is weighted in the frequency domain (zero-padded, so no wrap-around)."""
    block, hop = int(BLOCK_S * MEASURE_RATE), int(HOP_S * MEASURE_RATE)
    if len(audio) < block:  # one-shots shorter than a block: measure them whole
        starts, length = [0], len(audio)
    else:
        starts, length = list(range(0, len(audio) - block + 1, hop)), block
    size = 1 << int(np.ceil(np.log2(length * 2)))
    weight = _k_gain_squared(size)
    powers = []
    for i in range(0, len(starts), BATCH_BLOCKS):
        frames = np.stack([audio[s:s + length] for s in starts[i:i + BATCH_BLOCKS]])  # (blocks, length, ch)
        spectrum = np.fft.rfft(frames, n=size, axis=1)
        energy = (np.abs(spectrum) ** 2 * weight[None, :, None]).sum(axis=1)  # (blocks, ch)
        powers.append(energy.sum(axis=1) / length)
    return np.concatenate(powers)


def integrated_lufs(audio: np.ndarray, rate: int) -> float | None:
    """Gated integrated loudness in LUFS, or None for silence. `audio` is (frames, channels)."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if not audio.size:
        return None
    if rate != MEASURE_RATE:
        audio = soxr.resample(audio, rate, MEASURE_RATE, quality="HQ")
    powers = block_powers(audio)
    with np.errstate(divide="ignore"):
        loudness = -0.691 + 10 * np.log10(powers)
    kept = powers[loudness > ABSOLUTE_GATE]
    if not kept.size:
        return None
    relative = -0.691 + 10 * np.log10(kept.mean()) + RELATIVE_GATE
    kept = powers[(loudness > ABSOLUTE_GATE) & (loudness > relative)]
    return float(-0.691 + 10 * np.log10(kept.mean())) if kept.size else None


def match_loudness(audio: np.ndarray, rate: int, target_lufs: float,
                   ceiling_db: float = PEAK_CEILING_DB) -> tuple[np.ndarray, float | None]:
    """Gain `audio` to `target_lufs`, never past the peak ceiling (a punchy one-shot may end up
    a little under target instead of clipping). Returns (audio, resulting LUFS)."""
    measured = integrated_lufs(audio, rate)
    if measured is None:
        return audio.copy(), None
    gain_db = target_lufs - measured
    peak = float(np.abs(audio).max())
    if peak > 0:
        gain_db = min(gain_db, ceiling_db - 20 * np.log10(peak))
    return audio * 10 ** (gain_db / 20), measured + gain_db

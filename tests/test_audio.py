import numpy as np
import pytest
import soundfile as sf

from samplegen.audio import (
    ExportFormat,
    PostOptions,
    apply_fades,
    make_seamless_loop,
    normalize_peak,
    postprocess,
    read_wav,
    region_mask,
    remove_dc,
    splice_regions,
    trim_silence,
    write_wav,
)

SR = 44100


def sine(seconds, freq=440.0, amp=0.5, channels=2):
    t = np.arange(int(seconds * SR)) / SR
    mono = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([mono] * channels, axis=1)


def test_remove_dc_centres_signal_and_does_not_mutate():
    x = sine(0.5) + 0.2
    original = x.copy()
    y = remove_dc(x)
    assert np.abs(y.mean(axis=0)).max() < 1e-3
    np.testing.assert_array_equal(x, original)


def test_normalize_peak_hits_target():
    y = normalize_peak(sine(0.5, amp=1.7), target_db=-1.0)
    assert 20 * np.log10(np.abs(y).max()) == pytest.approx(-1.0, abs=1e-6)


def test_normalize_peak_leaves_silence_alone():
    y = normalize_peak(np.zeros((100, 2)), target_db=-1.0)
    assert not np.any(y)


def test_trim_silence_removes_head_and_tail_keeping_preroll():
    body = sine(0.2)
    x = np.concatenate([np.zeros((SR // 2, 2)), body, np.zeros((SR, 2))])
    y = trim_silence(x, SR, threshold_db=-50.0, preroll_ms=2.0, tail_ms=20.0)
    preroll, tail = int(0.002 * SR), int(0.02 * SR)
    assert len(y) == pytest.approx(len(body) + preroll + tail, abs=2)


def test_trim_silence_on_all_silent_input_returns_input():
    x = np.zeros((1000, 2))
    assert len(trim_silence(x, SR)) == 1000


def test_apply_fades_start_and_end_at_zero():
    y = apply_fades(np.ones((SR, 2)), SR, fade_in_ms=10, fade_out_ms=50)
    assert y[0].max() == 0.0
    assert y[-1].max() == pytest.approx(0.0, abs=1e-9)
    assert y[SR // 2].min() == 1.0


def test_apply_fades_longer_than_clip_meet_in_the_middle():
    y = apply_fades(np.ones((1000, 2)), SR, fade_in_ms=5000, fade_out_ms=5000)
    assert y[0].max() == 0.0 and y[-1].max() == pytest.approx(0.0, abs=1e-9)
    assert y[495:505].min() > 0.95  # peak near the middle, no hollowed-out dip


def test_make_seamless_loop_is_exact_length_and_wraps_smoothly():
    # A tone whose period doesn't divide the loop length: a naive cut clicks.
    raw = sine(1.25, freq=437.25)  # 437.25 cycles per loop -> a naive cut lands mid-waveform
    n = SR  # one-second loop, 0.25 s of continuation available
    looped = make_seamless_loop(raw, n, SR, crossfade_ms=10)
    assert len(looped) == n
    naive_jump = np.abs(raw[n - 1] - raw[0]).max()
    wrap_jump = np.abs(looped[-1] - looped[0]).max()
    typical_step = np.abs(np.diff(raw[:, 0])).max()
    assert naive_jump > 3 * typical_step
    assert wrap_jump <= 1.5 * typical_step


def test_make_seamless_loop_without_continuation_falls_back_to_fade():
    raw = sine(1.0)
    looped = make_seamless_loop(raw, SR, SR, crossfade_ms=10)
    assert len(looped) == SR
    assert np.abs(looped[-1]).max() < 1e-6


def test_postprocess_loop_pipeline():
    raw = sine(1.25, amp=1.5)
    y = postprocess(raw, SR, PostOptions(target_samples=SR, is_loop=True, normalize_db=-1.0))
    assert len(y) == SR
    assert 20 * np.log10(np.abs(y).max()) == pytest.approx(-1.0, abs=1e-6)


def test_postprocess_one_shot_trims_and_pads_to_target():
    raw = sine(0.5)
    y = postprocess(raw, SR, PostOptions(target_samples=SR, is_loop=False, trim_silence=False, normalize_db=None))
    assert len(y) == SR  # zero-padded when the model returned less than asked
    assert not np.any(y[int(0.6 * SR):])


def test_region_mask_marks_seconds():
    mask = region_mask(SR * 3, [(1.0, 1.5), (2.5, 9.0)], SR)
    assert mask.sum() == SR // 2 + SR // 2
    assert mask[SR] and not mask[SR - 1] and mask[-1]


def test_splice_keeps_original_bit_identical_away_from_regions():
    original = sine(3.0, freq=300)
    generated = 3.0 * sine(3.0, freq=700)  # louder, different content
    out = splice_regions(original, generated, [(1.0, 2.0)], SR, crossfade_ms=20)
    far_before, far_after = slice(0, int(0.9 * SR)), slice(int(2.1 * SR), None)
    np.testing.assert_array_equal(out[far_before], original[far_before])
    np.testing.assert_array_equal(out[far_after], original[far_after])
    # inside the region it's the generated take, level-matched to the original
    inside = out[int(1.2 * SR):int(1.8 * SR)]
    assert np.corrcoef(inside[:, 0], generated[int(1.2 * SR):int(1.8 * SR), 0])[0, 1] > 0.99
    assert np.abs(inside).max() == pytest.approx(np.abs(original).max(), rel=0.05)


def test_splice_crossfade_has_no_jumps():
    original = sine(2.0, freq=300)
    generated = sine(2.0, freq=300, amp=0.5) * -1  # opposite phase: worst case for a hard cut
    out = splice_regions(original, generated, [(1.0, 2.0)], SR, crossfade_ms=20)
    step = np.abs(np.diff(out[:, 0])).max()
    assert step <= np.abs(np.diff(original[:, 0])).max() * 1.2


def test_splice_region_reaching_the_end_stays_fully_generated():
    original = np.concatenate([sine(1.0), np.zeros((SR, 2))])  # 1 s source + 1 s silence (extend)
    generated = sine(2.0, freq=500)
    out = splice_regions(original, generated, [(0.98, 2.0)], SR, crossfade_ms=20)
    tail = out[int(1.5 * SR):]
    assert np.corrcoef(tail[:, 0], generated[int(1.5 * SR):, 0])[0, 1] > 0.99
    assert np.abs(out[-1]).max() > 0 or np.abs(generated[-1]).max() == 0


def test_postprocess_does_not_mutate_input():
    raw = sine(1.25, amp=1.5)
    original = raw.copy()
    postprocess(raw, SR, PostOptions(target_samples=SR, is_loop=True))
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("fmt,subtype", [
    (ExportFormat(sample_rate=44100, bit_depth="32f"), "FLOAT"),
    (ExportFormat(sample_rate=48000, bit_depth="24"), "PCM_24"),
    (ExportFormat(sample_rate=44100, bit_depth="16"), "PCM_16"),
])
def test_write_wav_formats_and_metadata(tmp_path, fmt, subtype):
    path = tmp_path / "out.wav"
    write_wav(path, sine(0.5, amp=0.5), SR, fmt, title="test", comment="prompt: hello")
    info = sf.info(str(path))
    assert info.samplerate == fmt.sample_rate
    assert info.subtype == subtype
    assert info.frames == pytest.approx(0.5 * fmt.sample_rate, abs=2)
    with sf.SoundFile(str(path)) as f:
        assert f.title == "test"
        assert f.comment == "prompt: hello"
    assert not (tmp_path / "out.wav.part").exists()


def test_read_wav_returns_2d_float(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(str(path), np.zeros(100), SR, subtype="FLOAT")
    data, sr = read_wav(path)
    assert sr == SR
    assert data.shape == (100, 1)
    assert data.dtype == np.float64


def test_export_format_validation():
    with pytest.raises(ValueError):
        ExportFormat(sample_rate=22050, bit_depth="24")
    with pytest.raises(ValueError):
        ExportFormat(sample_rate=44100, bit_depth="8")

"""Acoustic analysis + caption composition (CLAP itself is faked)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from samplegen.analysis import analyze, describe, note_name
from samplegen.autodescribe import Describer, Tagging, category_phrase, compose, filename_terms

SR = 44100


def kick(sweep_from=300.0, body=49.0, decay=0.25, seconds=0.6, drive=1.0, stereo=False):
    """Synthetic kick: fast exponential pitch sweep onto a G1 body, -40 dB after `decay` s."""
    t = np.arange(int(seconds * SR)) / SR
    freq = body + (sweep_from - body) * np.exp(-t / 0.01)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sig = np.sin(phase) * np.exp(-t / (decay / 4.6))
    sig = np.tanh(drive * sig) / np.tanh(drive)
    if stereo:
        return np.stack([sig, np.roll(sig, 300)], axis=1)  # ~7 ms apart: clearly wide
    return np.stack([sig, sig], axis=1)


def test_kick_measurements():
    f = analyze(kick(), SR)
    assert f.tonal and f.pitch_hz == pytest.approx(49, abs=2.5)
    assert note_name(f.pitch_hz) == "G"
    assert f.pitch_sweep_st > 4
    assert 0.15 < f.decay_s < 0.35
    assert f.attack_ms < 6 and not f.sustained
    words = describe(f)
    assert "tuned around G" in words and "short decay" in words
    assert any("pitch" in w for w in words)


def test_saturation_and_stereo_are_heard():
    dense = describe(analyze(kick(drive=8.0), SR))
    assert "dense and saturated" in dense
    assert "wide stereo" in describe(analyze(kick(stereo=True), SR))


def test_noise_and_sustained_sounds():
    rng = np.random.default_rng(0)
    rain = rng.standard_normal((SR * 3, 2)) * 0.3
    words = describe(analyze(rain, SR))
    assert "noisy" in words and "sustained" in words
    assert not any(w.startswith("tuned") for w in words)


def test_silence_is_handled():
    f = analyze(np.zeros((1000, 2)), SR)
    assert f.decay_s == 0.0 and not f.tonal


@pytest.mark.parametrize("name, expected", [
    ("S-909K1.wav", ["909"]),
    ("Cymatics - Grime Kick 2.wav", ["grime"]),
    ("246 _Bbox_Bd.wav", ["beatbox"]),
    ("Kick - 17.wav", []),
    ("TR808_deep_long_01.wav", ["808", "deep", "long"]),
])
def test_filename_terms(name, expected):
    assert filename_terms(Path(name)) == expected


def tagging(**groups):
    return Tagging(groups)


def test_category_prefers_the_filename_machine():
    t = tagging(category=[["808 bass drum", 0.34], ["kick drum", 0.26]])
    assert category_phrase(t, ["606"]) == "606 kick drum"
    assert category_phrase(t, []) == "808 kick drum"
    assert category_phrase(t, ["beatbox"]) == "kick drum"
    assert category_phrase(tagging(category=[["glass breaking", 0.4], ["metal clang", 0.2]]), []) == "glass breaking"
    assert category_phrase(None, []) is None


def test_compose_keeps_only_confident_clap_words():
    t = tagging(
        category=[["kick drum", 0.37], ["808 bass drum", 0.26]],
        texture=[["distorted", 0.205], ["clean", 0.166]],      # margin 0.039: kept
        era=[["vintage", 0.123], ["modern", 0.123]],           # no margin: dropped
        tone=[["cold", 0.2], ["warm", 0.05]],                  # tone is never used
        space=[["dry", 0.4], ["roomy", 0.1]],                  # uninformative: dropped
    )
    caption = compose(Path("Cymatics - Grime Kick 2.wav"), ["dense and saturated", "very short"], t)
    assert caption == "kick drum, grime, dense and saturated, very short, distorted"


def test_compose_without_clap_uses_name_and_measurements():
    assert compose(Path("S-909K1.wav"), ["very punchy transient"], None) == "909, very punchy transient"


def write_kick(path):
    sf.write(str(path), kick(), SR, subtype="PCM_24")
    return path


def test_describer_runs_clap_and_composes(tmp_path):
    a = write_kick(tmp_path / "S-909K1.wav")
    python, script = tmp_path / "python.exe", tmp_path / "clap_describe.py"
    python.write_text("x")
    script.write_text("x")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"  # accented / Japanese paths survive the pipe
        listed = json.loads(Path(command[command.index("--files") + 1]).read_text(encoding="utf-8"))
        rows = [{"file": Path(p).name, "path": p, "category": [["kick drum", 0.4], ["snare drum", 0.2]]} for p in listed]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(rows), stderr="[clap] done")

    describer = Describer(python, script, log_path=tmp_path / "log.txt", runner=fake_run)
    captions = describer.describe([a], cpu=True)
    assert "--cpu" in calls[0]
    caption = captions[str(a.resolve())]
    assert caption.startswith("909 kick drum") and "tuned around G" in caption
    assert (tmp_path / "log.txt").read_text() == "[clap] done"


def test_describer_falls_back_when_clap_missing_or_failing(tmp_path):
    a = write_kick(tmp_path / "Kick - 1.wav")
    missing = Describer(tmp_path / "nope.exe", tmp_path / "nope.py")
    assert not missing.clap_available
    assert "tuned around G" in missing.describe([a])[str(a.resolve())]  # measurements still work

    python, script = tmp_path / "python.exe", tmp_path / "clap.py"
    python.write_text("x")
    script.write_text("x")
    crashing = Describer(python, script, runner=lambda c, **k: subprocess.CompletedProcess(c, 3, stdout="", stderr="boom"))
    assert crashing.tag([a]) == {}
    garbage = Describer(python, script, runner=lambda c, **k: subprocess.CompletedProcess(c, 0, stdout="not json", stderr=""))
    assert garbage.tag([a]) == {}

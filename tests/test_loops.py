"""Sustain loops: finding them, writing them into presets, adding them to older instruments."""

import numpy as np

from samplegen.audio import ExportFormat
from samplegen.instruments import decent_sampler_preset, ensure_loops, sfz_file, write_instrument
from samplegen.loops import CROSSFADE_S, MIN_LOOP_S, find_loop, load_loops
from samplegen.play import instrument_notes

SR = 44100


def note(freq=220.0, seconds=3.0, decay=0.3):
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)
    return np.stack([tone, tone], axis=1)


def test_loop_on_a_sustained_note_is_long_and_seamless():
    audio = note()
    start, end, fade = find_loop(audio, SR)
    assert end - start >= MIN_LOOP_S * SR
    assert fade == int(CROSSFADE_S * SR) and start >= fade
    assert end < len(audio) * 0.9  # before the note's fade-out
    # what plays after the jump (start) continues what played before it (end): same phase
    mono = audio[:, 0]
    assert abs(mono[start] - mono[end]) < 0.05


def test_silent_or_short_notes_are_not_looped():
    assert find_loop(np.zeros((3 * SR, 2)), SR) is None
    assert find_loop(note(seconds=0.5), SR) is None


def test_presets_carry_the_loop_points():
    loops = {60: {"start": 30000, "end": 90000, "crossfade": 2205, "rate": 44100}}
    ds = decent_sampler_preset("Keys", {60: "Samples/k_C4.wav"}, loops)
    assert 'loopEnabled="true" loopStart="30000" loopEnd="89999" loopCrossfade="2205"' in ds
    sfz = sfz_file("Keys", {60: "Samples/k_C4.wav"}, loops)
    assert "loop_mode=loop_sustain loop_start=30000 loop_end=89999 loop_crossfade=0.0500" in sfz


def test_new_instruments_get_loops_and_old_ones_get_them_on_first_use(tmp_path):
    folder = tmp_path / "Keys"
    write_instrument(folder, "Keys", "keys", {60: note(), 62: note(246.9)}, SR, ExportFormat(48000, "24"))
    loops = load_loops(folder)
    assert set(loops) == {60, 62} and loops[60]["rate"] == 48000  # in the exported files' rate
    assert "loopEnabled" in (folder / "keys.dspreset").read_text(encoding="utf-8")

    (folder / "loops.json").unlink()  # an instrument made before loops existed
    (folder / "keys.dspreset").write_text("<DecentSampler/>", encoding="utf-8")
    again = ensure_loops(folder, instrument_notes(folder))
    assert set(again) == {60, 62} and (folder / "loops.json").exists()
    assert "loopEnabled" in (folder / "keys.dspreset").read_text(encoding="utf-8")

"""LUFS measurement (BS.1770) and loudness-matched, round-robin packs."""

import csv

import numpy as np
import pytest
import soundfile as sf

from samplegen.library import Library, NewSample
from samplegen.loudness import integrated_lufs, match_loudness
from samplegen.packs import export_pack, round_robin_stems

SR = 48000


def sine(db, seconds=5.0, freq=1000.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    tone = 10 ** (db / 20) * np.sin(2 * np.pi * freq * t)
    return np.stack([tone, tone], axis=1)


def test_reference_tone_measures_its_level():
    # EBU Tech 3341: a 1 kHz stereo sine at -23 dBFS reads -23 LUFS.
    assert integrated_lufs(sine(-23.0), SR) == pytest.approx(-23.0, abs=0.2)
    assert integrated_lufs(sine(-23.0, sr=44100), 44100) == pytest.approx(-23.0, abs=0.2)


def test_silence_and_one_shots():
    assert integrated_lufs(np.zeros((SR, 2)), SR) is None
    short = sine(-12.0, seconds=0.2)  # shorter than one 400 ms block
    assert integrated_lufs(short, SR) == pytest.approx(-12.0, abs=0.5)


def test_match_loudness_hits_target_but_never_clips():
    quiet = sine(-30.0)
    out, lufs = match_loudness(quiet, SR, -18.0)
    assert lufs == pytest.approx(-18.0, abs=0.01)
    assert integrated_lufs(out, SR) == pytest.approx(-18.0, abs=0.2)
    spiky = quiet.copy()
    spiky[1000] = 0.5  # one sharp transient: peak far above the average loudness
    hot, lufs = match_loudness(spiky, SR, -6.0)  # reaching -6 LUFS would push that peak past 0 dBFS
    assert np.abs(hot).max() == pytest.approx(10 ** (-1 / 20)) and lufs < -6.0


def add(lib, tmp_path, name, batch, variation, level_db):
    src = tmp_path / f"{name}{variation}.wav"
    sf.write(str(src), sine(level_db, seconds=1.0), SR, subtype="FLOAT")
    return lib.add(src, NewSample(mode="sfx", model="sa3-sfx", prompt=name, negative_prompt="", seed=1,
                                  params={"variation": variation}, duration=1.0, sample_rate=SR,
                                  batch_id=batch, name=name))


def test_round_robin_names_group_takes_of_one_generation(tmp_path):
    lib = Library(tmp_path / "lib")
    takes = [add(lib, tmp_path, "Door Slam", "b1", v, -20) for v in (2, 1, 3)]
    single = add(lib, tmp_path, "Whoosh", "b2", 1, -20)
    stems = round_robin_stems([*takes, single])
    assert sorted(stems[t.id] for t in takes) == ["door-slam_01", "door-slam_02", "door-slam_03"]
    assert stems[takes[1].id] == "door-slam_01"  # ordered by variation
    assert stems[single.id] == "whoosh"
    lib.close()


def test_pack_matches_loudness_and_reports_it(tmp_path):
    lib = Library(tmp_path / "lib")
    records = [add(lib, tmp_path, "Hit", "b1", 1, -30), add(lib, tmp_path, "Hit", "b1", 2, -10)]
    result = export_pack(lib, records, "Game SFX", loudness=-18.0, round_robin=True)
    files = sorted((result.folder / "SFX").iterdir())
    assert [f.name for f in files] == ["hit_01.wav", "hit_02.wav"]
    for f in files:
        audio, sr = sf.read(str(f), always_2d=True)
        assert integrated_lufs(audio, sr) == pytest.approx(-18.0, abs=0.3)
        assert sf.info(str(f)).subtype == "FLOAT"  # format kept when only loudness changes
    with open(result.folder / "samples.csv", encoding="utf-8-sig") as fh:
        assert {row["lufs"] for row in csv.DictReader(fh)} == {"-18.0"}
    with pytest.raises(ValueError, match="LUFS"):
        export_pack(lib, records, "Bad", loudness=3.0)
    lib.close()

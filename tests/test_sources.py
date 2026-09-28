import numpy as np
import pytest
import soundfile as sf

from samplegen.sources import SourceNotFound, SourceStore

from .fakes import tone_wav_bytes


@pytest.fixture
def store(tmp_path):
    return SourceStore(tmp_path / "lib")


def test_import_converts_to_native_stereo_float(store, tmp_path):
    source = store.import_bytes(tone_wav_bytes(tmp_path, seconds=2.0, sr=48000, channels=1), "My Hit.wav")
    assert source.name == "My Hit"
    assert source.duration == pytest.approx(2.0, abs=1e-3)
    assert source.is_loop is False and source.origin == "upload"
    info = sf.info(str(store.audio_path(source.id)))
    assert (info.samplerate, info.channels, info.subtype) == (44100, 2, "FLOAT")
    assert store.load(source.id).shape == (88200, 2)
    assert store.get(source.id) == source


def test_import_rejects_garbage_and_empty(store):
    with pytest.raises(ValueError, match="Couldn't read"):
        store.import_bytes(b"definitely not audio" * 10, "x.wav")
    with pytest.raises(ValueError, match="empty"):
        store.import_bytes(b"", "x.wav")


def test_import_rejects_overlong_audio(store, tmp_path, monkeypatch):
    monkeypatch.setattr("samplegen.sources.MAX_SOURCE_SECONDS", 1.0)
    with pytest.raises(ValueError, match="limited"):
        store.import_bytes(tone_wav_bytes(tmp_path, seconds=2.0), "long.wav")


def test_import_file_keeps_loop_metadata(store, tmp_path):
    path = tmp_path / "loop.wav"
    sf.write(str(path), np.zeros((44100, 2)), 44100, subtype="FLOAT")
    source = store.import_file(path, "bass loop", is_loop=True, bpm=140, bars=8, key="E minor")
    assert (source.is_loop, source.bpm, source.bars, source.key, source.origin) == (True, 140, 8, "E minor", "library")


@pytest.mark.parametrize("bad_id", ["nope", "../../etc", "0123456789/..", "ABCDEF0123"])
def test_unknown_or_unsafe_ids_are_not_found(store, bad_id):
    with pytest.raises(SourceNotFound):
        store.get(bad_id)
    with pytest.raises(SourceNotFound):
        store.audio_path(bad_id)

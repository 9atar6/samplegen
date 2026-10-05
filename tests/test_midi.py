"""MIDI extraction: the subprocess wrapper, the API routes, and .mid files following their sample."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from samplegen.api import AppContext, create_app
from samplegen.generation import Generator
from samplegen.jobs import JobManager
from samplegen.library import Library, NewSample
from samplegen.midi import MidiError, Transcriber
from samplegen.packs import export_pack

from .fakes import FakeClient, FakeEngine


def installed(tmp_path, runner):
    python, script = tmp_path / "python.exe", tmp_path / "transcribe.py"
    python.write_text("x")
    script.write_text("x")
    return Transcriber(python, script, runner=runner)


def writes_midi(notes=12, calls=None):
    def run(command, **kwargs):
        if calls is not None:
            calls.append((command, kwargs))
        out = Path(command[command.index("--out") + 1])
        out.write_bytes(b"MThd")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"notes": notes, "out": str(out)}), stderr="")
    return run


def add(lib, tmp_path, mode="loop", params=None):
    src = tmp_path / "in.wav"
    sf.write(str(src), np.zeros((4410, 2)), 44100, subtype="FLOAT")
    return lib.add(src, NewSample(mode=mode, model="f1-samples", prompt="acid bass", negative_prompt="", seed=1,
                                  params=params or {"bpm": 140}, duration=0.1, sample_rate=44100, batch_id="b",
                                  name="acid bass"))


@pytest.fixture
def lib(tmp_path):
    library = Library(tmp_path / "lib")
    yield library
    library.close()


def test_transcriber_passes_tempo_and_utf8(tmp_path):
    calls = []
    t = installed(tmp_path, writes_midi(calls=calls))
    assert t.transcribe(tmp_path / "a.wav", tmp_path / "a.mid", bpm=140) == 12
    command, kwargs = calls[0]
    assert command[command.index("--bpm") + 1] == "140"
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"


def test_transcriber_not_installed_and_failures(tmp_path):
    missing = Transcriber(tmp_path / "nope.exe", tmp_path / "nope.py")
    with pytest.raises(MidiError, match="install-midi.bat"):
        missing.transcribe(tmp_path / "a.wav", tmp_path / "a.mid")
    crashing = installed(tmp_path, lambda c, **k: subprocess.CompletedProcess(c, 1, stdout="", stderr="boom"))
    with pytest.raises(MidiError, match="failed"):
        crashing.transcribe(tmp_path / "a.wav", tmp_path / "a.mid")


def test_midi_follows_its_sample_and_goes_into_packs(lib, tmp_path):
    record = add(lib, tmp_path)
    lib.midi_path(record).write_bytes(b"MThd")
    kept = lib.set_status(record.id, "kept")
    assert lib.midi_path(kept).exists() and not lib.midi_path(record).exists()
    renamed = lib.rename(record.id, "Squelchy Acid")
    assert lib.midi_path(renamed).name == f"squelchy-acid_{record.id}.mid"
    result = export_pack(lib, [lib.get(record.id)], "Acid")
    assert (result.folder / "Loops" / "squelchy-acid.mid").exists()


def make_client(tmp_path, library, transcriber):
    engine = FakeEngine()
    jobs = JobManager(Generator(engine, FakeClient(tmp_path / "out"), library, tmp_path / "out"))
    return TestClient(create_app(AppContext(library, jobs, engine, midi=transcriber)))


def test_midi_api_round_trip(lib, tmp_path):
    record = add(lib, tmp_path)
    client = make_client(tmp_path, lib, installed(tmp_path, writes_midi(notes=7)))
    assert client.get(f"/api/samples/{record.id}/midi").status_code == 404  # not made yet
    made = client.post(f"/api/samples/{record.id}/midi").json()
    assert made["notes"] == 7 and made["filename"].endswith(".mid")
    res = client.get(made["url"])
    assert res.status_code == 200 and res.content == b"MThd"


def test_midi_api_explains_missing_install_and_no_notes(lib, tmp_path):
    record = add(lib, tmp_path)
    not_installed = make_client(tmp_path, lib, Transcriber(tmp_path / "x.exe", tmp_path / "x.py"))
    res = not_installed.post(f"/api/samples/{record.id}/midi")
    assert res.status_code == 409 and "install-midi.bat" in res.json()["detail"]
    silent = make_client(tmp_path, lib, installed(tmp_path, writes_midi(notes=0)))
    res = silent.post(f"/api/samples/{record.id}/midi")
    assert res.status_code == 422 and not lib.midi_path(record).exists()

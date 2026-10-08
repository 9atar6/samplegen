"""Play tab backend: instruments to play, hummed notes, saved performances."""

import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from samplegen.api import AppContext, create_app
from samplegen.audio import ExportFormat
from samplegen.generation import Generator
from samplegen.instruments import write_instrument
from samplegen.jobs import JobManager
from samplegen.library import Library, NewSample
from samplegen.midi import Transcriber
from samplegen.play import list_instruments, note_of

from .fakes import FakeClient, FakeEngine


def make_instrument(library, tmp_path, title="Glass Keys", notes=(60, 61, 64)):
    folder = library.root / "Instruments" / title
    audio = {m: 0.2 * np.ones((441, 2)) for m in notes}
    write_instrument(folder, title, "glass-keys", audio, 44100, ExportFormat(44100, "24"))
    preview = tmp_path / "preview.wav"
    sf.write(str(preview), np.zeros((441, 2)), 44100)
    return library.add(preview, NewSample(
        mode="instrument", model="f1-keybeds", prompt="glass keys", negative_prompt="", seed=1,
        params={"instrument_folder": f"Instruments/{title}"}, duration=0.01, sample_rate=44100,
        batch_id="b", name=title))


def wav_bytes(seconds=0.5):
    buf = io.BytesIO()
    sf.write(buf, np.zeros((int(44100 * seconds), 2)), 44100, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path):
    library = Library(tmp_path / "lib")
    engine = FakeEngine()
    jobs = JobManager(Generator(engine, FakeClient(tmp_path / "out"), library, tmp_path / "out"))

    def hum_runner(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        out.write_bytes(b"MThd")
        events = [[0.0, 0.4, 62, 0.8], [0.5, 0.9, 64, 0.7]]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"notes": 2, "events": events}), stderr="")

    python, script = tmp_path / "py.exe", tmp_path / "t.py"
    python.write_text("x")
    script.write_text("x")
    client = TestClient(create_app(AppContext(library, jobs, engine, midi=Transcriber(python, script, runner=hum_runner))))
    yield client, library, tmp_path
    library.close()


def test_note_names_in_sample_files():
    assert note_of(Path("glass-keys_C4.wav")) == 60
    assert note_of(Path("glass-keys_Cs4.wav")) == 61
    assert note_of(Path("my_preview.wav")) is None


def test_instruments_are_listed_with_their_range(env):
    client, library, tmp_path = env
    record = make_instrument(library, tmp_path)
    assert [i.id for i in list_instruments(library)] == [record.id]
    listed = client.get("/api/instruments").json()
    assert listed == [{"id": record.id, "name": "Glass Keys", "low": 60, "high": 64, "count": 3, "kind": "instrument"}]
    detail = client.get(f"/api/instruments/{record.id}").json()
    assert [n["midi"] for n in detail["notes"]] == [60, 61, 64]
    note = client.get(detail["notes"][1]["url"])
    assert note.status_code == 200 and note.content[:4] == b"RIFF"
    assert client.get(f"/api/instruments/{record.id}/notes/99").status_code == 404


def test_trashed_or_missing_instruments_are_not_offered(env):
    client, library, tmp_path = env
    record = make_instrument(library, tmp_path)
    library.set_status(record.id, "trashed")
    assert client.get("/api/instruments").json() == []
    assert client.get("/api/instruments/nope").status_code == 404


def test_hum_returns_notes(env):
    client, *_ = env
    res = client.post("/api/midi/hum", content=wav_bytes(), headers={"Content-Type": "audio/wav"})
    assert res.status_code == 200
    assert res.json()["events"][0] == [0.0, 0.4, 62, 0.8]


def test_saved_performance_lands_in_library_with_its_midi(env):
    client, library, tmp_path = env
    instrument = make_instrument(library, tmp_path)
    res = client.post(f"/api/performances?name=Night%20Keys&instrument={instrument.id}&bpm=96",
                      content=wav_bytes(1.0), headers={"Content-Type": "audio/wav"})
    assert res.status_code == 201, res.text
    take = res.json()
    assert take["mode"] == "performance" and take["prompt"] == "played on Glass Keys"
    assert take["params"]["bpm"] == 96
    assert client.put(f"/api/samples/{take['id']}/midi", content=b"MThd....",
                      headers={"Content-Type": "audio/midi"}).status_code == 204
    record = library.get(take["id"])
    assert library.midi_path(record).read_bytes() == b"MThd...."
    kept = library.set_status(record.id, "kept")
    assert "Performances" in kept.rel_path and library.midi_path(kept).exists()


def test_bad_uploads_are_refused(env):
    client, library, tmp_path = env
    assert client.post("/api/performances", content=b"not audio",
                       headers={"Content-Type": "audio/wav"}).status_code == 422
    record = make_instrument(library, tmp_path)
    assert client.put(f"/api/samples/{record.id}/midi", content=b"nope",
                      headers={"Content-Type": "audio/midi"}).status_code == 422

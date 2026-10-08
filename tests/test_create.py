"""Shot lists, variations, drum kits and the layer designer."""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from samplegen.api import AppContext, create_app
from samplegen.batch import Shot, parse_shot_list, shot_requests
from samplegen.generation import Generator
from samplegen.jobs import JobManager
from samplegen.kits import PIECES, KitRequest, kit_preset, kit_sfz
from samplegen.layers import LayerRequest, blend, from_onset
from samplegen.library import Library
from samplegen.play import instrument_info

from .fakes import FakeClient, FakeEngine

SR = 44100


# ---------- shot lists ----------

def test_shot_list_syntax():
    shots = parse_shot_list("""
        # forest level
        door creak x8
        heavy footsteps on gravel | 12x | 3s
        distant thunder | 20s | medium | loop
        ui beep
    """)
    assert shots == [
        Shot("door creak", 8), Shot("heavy footsteps on gravel", 12, 3.0),
        Shot("distant thunder", 4, 20.0, "sa3-medium", True), Shot("ui beep", 4),
    ]


@pytest.mark.parametrize("text, message", [
    ("", "empty"), ("boom | sideways", "don't know"), ("| 3x", "no description"), ("a x999\n" * 20, "Up to 1000"),
])
def test_shot_list_errors(text, message):
    with pytest.raises(ValueError, match=message):
        parse_shot_list(text)


def test_big_shots_are_split_into_gpu_sized_jobs():
    requests = shot_requests([Shot("rain", 20, 10.0), Shot("click", 3)], "Forest", "sa3-sfx", 1.0)
    assert sum(r.variations for r in requests) == 23
    assert all(r.variations <= 8 and r.duration * r.variations <= 160 for r in requests)
    assert all(r.tags == ("forest",) and r.title == r.prompt for r in requests)
    for r in requests:
        r.validate()


# ---------- app-level ----------

@pytest.fixture
def env(tmp_path):
    library = Library(tmp_path / "lib")
    engine = FakeEngine()
    client = FakeClient(tmp_path / "out")
    generator = Generator(engine, client, library, tmp_path / "out")
    jobs = JobManager(generator)
    api = TestClient(create_app(AppContext(library, jobs, engine)))
    yield api, jobs, library, client
    library.close()


def run_all(jobs, job_ids):
    return [jobs.run_now(i) for i in job_ids]


def test_batch_queues_everything_and_tags_the_results(env):
    api, jobs, library, _ = env
    res = api.post("/api/batches", json={"name": "Forest SFX", "text": "door creak x3\nbird chirp | 2x | 1s"})
    assert res.status_code == 202, res.text
    batch = res.json()
    assert batch["tag"] == "forest sfx" and batch["sounds"] == 5
    done = run_all(jobs, [j["id"] for j in batch["jobs"]])
    ids = [i for j in done for i in j.sample_ids]
    assert len(ids) == 5
    records = [library.get(i) for i in ids]
    assert {r.name for r in records} == {"door creak", "bird chirp"}
    assert all(r.tags == ("forest sfx",) for r in records)
    assert api.post("/api/batches/preview", json={"text": "boom x2\nhit"}).json()["sounds"] == 6
    ids_param = ",".join(j["id"] for j in batch["jobs"])
    assert len(api.get("/api/jobs", params={"ids": ids_param}).json()) == len(batch["jobs"])


def test_variations_keep_the_original_name_and_tags(env):
    api, jobs, library, _ = env
    first = api.post("/api/generate", json={"mode": "sfx", "model": "sa3-sfx", "prompt": "glass hit",
                                            "duration": 1.0, "variations": 1}).json()
    original = library.get(jobs.run_now(first["id"]).sample_ids[0])
    library.set_tags(original.id, ["glass"])
    job = api.post(f"/api/samples/{original.id}/variations", json={"count": 3}).json()
    done = jobs.run_now(job["id"])
    assert done.state == "done", done.error
    takes = [library.get(i) for i in done.sample_ids]
    assert len(takes) == 3
    assert all(t.name == original.name and t.tags == ("glass",) and t.mode == "transform" for t in takes)


def test_drum_kit_end_to_end(env):
    api, jobs, library, client = env
    job = api.post("/api/kits", json={"prompt": "dusty boom bap", "name": "Dusty", "round_robins": 2}).json()
    done = jobs.run_now(job["id"])
    assert done.state == "done", done.error
    record = library.get(done.sample_ids[0])
    assert record.mode == "kit" and record.params["pieces"] == len(PIECES)
    folder = library.root / record.params["instrument_folder"]
    assert len(list((folder / "Samples").glob("*.wav"))) == 2 * len(PIECES)
    preset = (folder / "dusty.dspreset").read_text(encoding="utf-8")
    assert 'seqMode="round_robin" seqLength="2"' in preset and 'silencedByTags="hat"' in preset
    info = instrument_info(library, record)
    assert info.kind == "kit" and set(info.notes) == {p.midi for p in PIECES}
    detail = api.get(f"/api/instruments/{record.id}").json()
    assert detail["kind"] == "kit" and all(n["loop"] is None for n in detail["notes"])  # hits never loop
    assert "kick drum" in client.graphs[0]["pos"]["inputs"]["text"]


def test_kit_files_choke_and_round_robin():
    takes = {"closed-hat": ["Samples/a.wav", "Samples/b.wav"], "open-hat": ["Samples/c.wav"]}
    sfz = kit_sfz("K", takes)
    assert "<group> key=42 loop_mode=one_shot seq_length=2 group=1" in sfz
    assert "key=46 loop_mode=one_shot seq_length=1 group=2 off_by=1" in sfz
    assert 'tags="hat"' in kit_preset("K", takes)
    with pytest.raises(ValueError):
        KitRequest(prompt="  ").validate()


def test_layer_designer_saves_the_hit_and_its_layers(env):
    api, jobs, library, _ = env
    job = api.post("/api/layers", json={"character": "sci-fi", "transient": "metal click", "body": "deep boom",
                                        "tail": "metal reverb tail", "tail_seconds": 2.0, "variations": 2}).json()
    done = jobs.run_now(job["id"])
    assert done.state == "done", done.error
    records = [library.get(i) for i in done.sample_ids]
    assert len(records) == 8  # 2 takes x (blend + 3 layers)
    hits = [r for r in records if r.mode == "layered"]
    assert len(hits) == 2 and {r.name for r in records if r.mode == "layer"} == {
        "sci-fi · transient", "sci-fi · body", "sci-fi · tail"}
    audio, _ = sf.read(str(library.path_of(hits[0])), always_2d=True)
    assert np.abs(audio).max() == pytest.approx(10 ** (-1 / 20), abs=1e-3)


def test_layers_line_up_on_their_first_transient():
    late = np.concatenate([np.zeros((SR // 2, 2)), np.ones((SR // 10, 2))])
    assert len(from_onset(late, SR)) == pytest.approx(SR // 10 + int(0.002 * SR), abs=2)
    request = LayerRequest(character="x", transient="a", body="b", tail_offset_ms=100.0)
    mix = blend({"transient": np.ones((100, 2)), "tail": np.ones((100, 2))}, request, SR)
    assert len(mix) == int(0.1 * SR) + 100  # the tail starts 100 ms later
    with pytest.raises(ValueError, match="two layers"):
        LayerRequest(character="x", body="only one").validate()

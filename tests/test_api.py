import pytest
from fastapi.testclient import TestClient

from samplegen.api import AppContext, create_app
from samplegen.generation import Generator
from samplegen.jobs import JobManager
from samplegen.library import Library

from .fakes import FakeClient, FakeEngine, tone_wav_bytes


@pytest.fixture
def app_env(tmp_path):
    library = Library(tmp_path / "lib")
    engine = FakeEngine()
    jobs = JobManager(Generator(engine, FakeClient(tmp_path / "out"), library, tmp_path / "out"))
    client = TestClient(create_app(AppContext(library, jobs, engine)))
    yield client, jobs, library
    library.close()


def generate(client, jobs, **overrides):
    body = {"mode": "sfx", "model": "sa3-sfx", "prompt": "glass shatter", "duration": 1.5, "variations": 2}
    body.update(overrides)
    res = client.post("/api/generate", json=body)
    assert res.status_code == 202, res.text
    job = jobs.run_now(res.json()["id"])  # run synchronously instead of via the worker thread
    return client.get(f"/api/jobs/{job.id}").json()


def test_status_and_catalog(app_env):
    client, _, _ = app_env
    assert client.get("/api/status").json()["engine"] == "ready"
    catalog = client.get("/api/catalog").json()
    assert {m["key"] for m in catalog["models"]} == {"sa3-sfx", "sa3-medium", "f1-samples", "f1-keybeds",
                                                      "sa3-sfx-base", "sa3-music-base"}
    assert catalog["instrument"]["lowest"] == "C1"
    assert 140 in catalog["loop"]["bpms"] and "Instrument" in catalog["loop"]["tags"]


def test_generate_then_browse_keep_and_favorite(app_env):
    client, jobs, _ = app_env
    job = generate(client, jobs)
    assert job["state"] == "done" and len(job["sample_ids"]) == 2
    sid = job["sample_ids"][0]

    audio = client.get(f"/api/samples/{sid}/audio")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"

    assert client.post(f"/api/samples/{sid}/status", json={"status": "kept"}).json()["status"] == "kept"
    assert client.post(f"/api/samples/{sid}/favorite", json={"favorite": True}).json()["favorite"] is True
    assert [s["id"] for s in client.get("/api/samples", params={"status": "kept"}).json()] == [sid]
    assert [s["id"] for s in client.get("/api/samples", params={"favorite": True}).json()] == [sid]
    assert len(client.get("/api/samples", params={"q": "shatter"}).json()) == 2


def test_generate_rejects_invalid_requests(app_env):
    client, _, _ = app_env
    bad = client.post("/api/generate", json={"mode": "loop", "model": "f1-samples", "bpm": 125})
    assert bad.status_code == 422 and "BPM" in bad.json()["detail"]
    assert client.post("/api/generate", json={"mode": "karaoke", "model": "x"}).status_code == 422
    too_long = client.post("/api/generate", json={"mode": "sfx", "model": "sa3-sfx", "prompt": "x" * 2001})
    assert too_long.status_code == 422
    bad_export = client.post("/api/generate", json={"mode": "sfx", "model": "sa3-sfx", "prompt": "x",
                                                    "export": {"sample_rate": 22050}})
    assert bad_export.status_code == 422


def test_unknown_ids_404(app_env):
    client, _, _ = app_env
    assert client.get("/api/samples/nope").status_code == 404
    assert client.get("/api/samples/nope/audio").status_code == 404
    assert client.post("/api/samples/nope/status", json={"status": "kept"}).status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_bad_status_value_rejected(app_env):
    client, jobs, _ = app_env
    sid = generate(client, jobs)["sample_ids"][0]
    assert client.post(f"/api/samples/{sid}/status", json={"status": "deleted"}).status_code == 422


def test_upload_source_then_transform(app_env, tmp_path):
    client, jobs, _ = app_env
    res = client.post("/api/sources", params={"filename": "kick.wav"}, content=tone_wav_bytes(tmp_path, 1.0))
    assert res.status_code == 201, res.text
    source = res.json()
    assert source["name"] == "kick" and source["duration"] == pytest.approx(1.0, abs=1e-3)
    assert client.get(f"/api/sources/{source['id']}").json() == source
    assert client.get(f"/api/sources/{source['id']}/audio").content[:4] == b"RIFF"

    body = {"mode": "transform", "model": "sa3-sfx", "prompt": "robot footstep", "source_id": source["id"],
            "strength": 0.5, "variations": 2}
    res = client.post("/api/generate", json=body)
    assert res.status_code == 202, res.text
    job = jobs.run_now(res.json()["id"])
    assert job.state == "done", job.error
    assert len(job.sample_ids) == 2


def test_edit_request_with_regions(app_env, tmp_path):
    client, jobs, _ = app_env
    source = client.post("/api/sources", params={"filename": "a.wav"}, content=tone_wav_bytes(tmp_path, 2.0)).json()
    body = {"mode": "edit", "model": "sa3-sfx", "prompt": "glitch", "source_id": source["id"],
            "edit_op": "inpaint", "regions": [[0.5, 1.0]], "variations": 1}
    res = client.post("/api/generate", json=body)
    assert res.status_code == 202, res.text
    assert jobs.run_now(res.json()["id"]).state == "done"
    bad = client.post("/api/generate", json={**body, "regions": [[1.5, 5.0]]})
    assert bad.status_code == 422 and "outside the source" in bad.json()["detail"]


def test_sample_can_become_a_source(app_env):
    client, jobs, _ = app_env
    sid = generate(client, jobs)["sample_ids"][0]
    res = client.post(f"/api/sources/from-sample/{sid}")
    assert res.status_code == 201
    assert res.json()["origin"] == "library" and res.json()["name"] == "glass shatter"
    assert client.post("/api/sources/from-sample/nope").status_code == 404


def test_bad_uploads_rejected(app_env):
    client, _, _ = app_env
    assert client.post("/api/sources", params={"filename": "x.wav"}, content=b"garbage" * 50).status_code == 422
    assert client.get("/api/sources/0123456789").status_code == 404
    assert client.get("/api/sources/..%2F..%2Fsecrets").status_code == 404


def test_rename_tags_and_tag_filter(app_env):
    client, jobs, _ = app_env
    first, second = generate(client, jobs)["sample_ids"]
    assert client.post(f"/api/samples/{first}/rename", json={"name": "Big Glass"}).json()["name"] == "Big Glass"
    assert client.post(f"/api/samples/{first}/tags", json={"tags": ["Glass", "impact"]}).json()["tags"] == ["glass", "impact"]
    client.post(f"/api/samples/{second}/tags", json={"tags": ["glass"]})
    assert client.get("/api/tags").json() == [{"tag": "glass", "count": 2}, {"tag": "impact", "count": 1}]
    assert [s["id"] for s in client.get("/api/samples", params={"tag": "impact"}).json()] == [first]
    assert client.post(f"/api/samples/{first}/rename", json={"name": ""}).status_code == 422
    assert client.post("/api/samples/nope/tags", json={"tags": []}).status_code == 404


def test_export_pack_endpoint(app_env):
    client, jobs, library = app_env
    ids = generate(client, jobs)["sample_ids"]
    res = client.post("/api/packs", json={"name": "Glass Pack", "sample_ids": ids, "reveal": False,
                                          "export": {"sample_rate": 48000, "bit_depth": "24"}})
    assert res.status_code == 201, res.text
    assert res.json()["exported"] == 2 and res.json()["name"] == "Glass Pack"
    assert client.post("/api/packs", json={"name": "x", "sample_ids": ["nope"], "reveal": False}).status_code == 404
    assert client.post("/api/packs", json={"name": "x", "sample_ids": [], "reveal": False}).status_code == 422


def test_stems_endpoint(app_env):
    client, jobs, _ = app_env
    sid = generate(client, jobs)["sample_ids"][0]
    res = client.post(f"/api/samples/{sid}/stems", json={})
    assert res.status_code == 202, res.text
    job = jobs.run_now(res.json()["id"])
    assert job.state == "done", job.error
    assert len(job.sample_ids) == 3  # vocals came back silent
    assert client.get(f"/api/samples/{job.sample_ids[0]}").json()["mode"] == "stems"


def test_instrument_endpoint(app_env):
    client, jobs, _ = app_env
    res = client.post("/api/instruments", json={"prompt": "Felt Piano, Soft", "low_note": "C4", "high_note": "B4"})
    assert res.status_code == 202, res.text
    job = jobs.run_now(res.json()["id"])
    assert job.state == "done", job.error
    record = client.get(f"/api/samples/{job.sample_ids[0]}").json()
    assert record["mode"] == "instrument" and record["params"]["notes"] == 12
    bad = client.post("/api/instruments", json={"prompt": "x", "low_note": "C5", "high_note": "C4"})
    assert bad.status_code == 422


def test_training_routes_when_not_installed(app_env, tmp_path):
    client, _, _ = app_env
    status = client.get("/api/training").json()
    assert status["installed"] is False and status["state"] == "idle"
    assert client.get("/api/styles").json() == []
    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "metal_hit_01.wav").write_bytes(tone_wav_bytes(tmp_path, 0.3))
    scanned = client.post("/api/training/scan", json={"path": f'"{folder}"'}).json()
    assert [(c["name"], c["caption"]) for c in scanned] == [("metal_hit_01.wav", "metal hit")]
    assert client.post("/api/training/scan", json={"path": str(tmp_path / "nope")}).status_code == 422
    body = {"name": "x", "clips": [{"path": str(folder / "metal_hit_01.wav"), "caption": "hit"}]}
    res = client.post("/api/training/start", json=body)
    assert res.status_code == 422 and "install-training.bat" in res.json()["detail"]
    missing = {"name": "x", "clips": [{"path": str(folder / "nope.wav"), "caption": "hit"}]}
    assert client.post("/api/training/start", json=missing).status_code == 422


def test_auto_describe_without_clap_uses_measurements(app_env, tmp_path):
    client, jobs, _ = app_env
    sid = generate(client, jobs)["sample_ids"][0]
    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "S-909K1.wav").write_bytes(tone_wav_bytes(tmp_path, 0.5))
    res = client.post("/api/training/describe", json={"clips": [{"path": str(folder / "S-909K1.wav")}, {"sample_id": sid}]})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["clap"] is False
    first, second = body["captions"]
    assert first["caption"].startswith("909") and first["path"].endswith("S-909K1.wav")
    assert second["sample_id"] == sid and second["caption"]
    assert client.post("/api/training/describe", json={"clips": [{"path": str(folder / "nope.wav")}]}).status_code == 422
    assert client.post("/api/training/describe", json={"clips": []}).status_code == 422


def test_ui_is_served(app_env):
    client, _, _ = app_env
    res = client.get("/", follow_redirects=True)
    assert res.status_code == 200 and "samplegen" in res.text
    script = client.get("/ui/app.js")
    assert script.status_code == 200 and script.headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/status").headers

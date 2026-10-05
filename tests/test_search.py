"""Search by sound: fingerprint indexing, ranking, similar sounds, API."""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from samplegen.api import AppContext, create_app
from samplegen.generation import Generator
from samplegen.jobs import JobManager
from samplegen.library import Library, NewSample
from samplegen.search import SearchUnavailable, SoundIndex

from .fakes import FakeClient, FakeEngine

# Fake CLAP: each sound's "fingerprint" is decided by a word in its file name.
WORDS = {"metal": [1, 0, 0], "rain": [0, 1, 0], "kick": [0, 0, 1]}


class FakeWorker:
    available = True

    def __init__(self):
        self.audio_calls = 0

    def request(self, payload, timeout=None):
        if payload["cmd"] == "text":
            return {"ok": True, "vector": next(v for w, v in WORDS.items() if w in payload["text"])}
        self.audio_calls += 1
        vectors = []
        for path in payload["paths"]:
            word = next((w for w in WORDS if w in path), None)
            vectors.append(WORDS[word] if word else None)  # None: CLAP couldn't read it
        return {"ok": True, "vectors": vectors}

    def stop(self):
        pass


def add(lib, tmp_path, name):
    src = tmp_path / f"{name}.wav"
    sf.write(str(src), np.zeros((441, 2)), 44100)
    return lib.add(src, NewSample(mode="sfx", model="sa3-sfx", prompt=name, negative_prompt="", seed=1, params={},
                                  duration=0.01, sample_rate=44100, batch_id="b", name=name))


@pytest.fixture
def setup(tmp_path):
    library = Library(tmp_path / "lib")
    samples = {n: add(library, tmp_path, n) for n in ("metal clang", "metal scrape", "rain on roof", "kick", "mystery")}
    index = SoundIndex(library, FakeWorker())
    yield library, index, samples
    library.close()


def index_all(index):
    while index.index_some():
        pass


def test_indexing_fingerprints_everything_once(setup):
    library, index, _ = setup
    assert index.status()["indexed"] == 0
    index_all(index)
    assert index.status() == {"available": True, "indexed": 5, "total": 5, "error": None}
    calls = index.worker.audio_calls
    assert index.index_some() == 0 and index.worker.audio_calls == calls  # unreadable "mystery" not retried


def test_text_search_ranks_by_sound(setup):
    _, index, samples = setup
    index_all(index)
    names = [r.name for r, _ in index.search_text("rain storm", limit=2)]
    assert names[0] == "rain on roof"
    metal = {r.name for r, score in index.search_text("metal", limit=2)}
    assert metal == {"metal clang", "metal scrape"}


def test_similar_excludes_itself_and_skips_trash(setup):
    library, index, samples = setup
    index_all(index)
    found = [r.name for r, _ in index.similar(samples["metal clang"].id, limit=1)]
    assert found == ["metal scrape"]
    library.set_status(samples["metal scrape"].id, "trashed")
    assert "metal scrape" not in [r.name for r, _ in index.similar(samples["metal clang"].id)]


def test_similar_before_indexing_explains(setup):
    _, index, samples = setup
    with pytest.raises(SearchUnavailable, match="listened to"):
        index.similar(samples["kick"].id)


def test_search_api(setup, tmp_path):
    library, index, samples = setup
    index_all(index)
    engine = FakeEngine()
    jobs = JobManager(Generator(engine, FakeClient(tmp_path / "out"), library, tmp_path / "out"))
    client = TestClient(create_app(AppContext(library, jobs, engine, search=index)))
    results = client.get("/api/search", params={"q": "kick drum"}).json()
    assert results[0]["name"] == "kick" and results[0]["score"] == pytest.approx(1.0)
    similar = client.get(f"/api/samples/{samples['metal scrape'].id}/similar").json()
    assert similar[0]["name"] == "metal clang"
    assert client.get("/api/search/status").json()["indexed"] == 5


def test_search_without_clap_explains_how_to_install(tmp_path):
    library = Library(tmp_path / "lib")
    engine = FakeEngine()
    jobs = JobManager(Generator(engine, FakeClient(tmp_path / "out"), library, tmp_path / "out"))
    client = TestClient(create_app(AppContext(library, jobs, engine)))
    res = client.get("/api/search", params={"q": "boom"})
    assert res.status_code == 409 and "install-search.bat" in res.json()["detail"]
    assert client.get("/api/search/status").json()["available"] is False
    library.close()

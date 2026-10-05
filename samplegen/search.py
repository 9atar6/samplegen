"""Search the library by sound: CLAP fingerprints of every sample, compared by similarity.

A CLAP worker (trainer/clap_server.py) stays loaded in the background, on the CPU. An
indexer fingerprints new samples as they arrive; searches compare a text or a sample's
fingerprint against all of them (one matrix product, instant even for 10k samples).

CLAP comes with style training (the trainer's Python) or tools/install-search.bat (search/.venv).
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from .library import Library, SampleRecord
from .winjob import bind_to_this_process

log = logging.getLogger("samplegen.search")

CREATE_NO_WINDOW = 0x08000000
START_TIMEOUT_S = 300   # first run downloads the model (~800 MB)
REQUEST_TIMEOUT_S = 120
BATCH = 8
IDLE_POLL_S = 20
NOT_AVAILABLE = ("Search by sound needs the CLAP model: install style training (tools\\install-training.bat) "
                 "or just search (tools\\install-search.bat).")
UNUSABLE = b""  # stored for files CLAP couldn't read, so they aren't retried forever


class SearchUnavailable(RuntimeError):
    pass


class ClapWorker:
    """One long-running clap_server.py process, started on first use."""

    def __init__(self, pythons: list[Path], script: Path, log_path: Path):
        self.pythons = [Path(p) for p in pythons]
        self.script = Path(script)
        self.log_path = Path(log_path)
        self._process: subprocess.Popen | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()

    @property
    def python(self) -> Path | None:
        return next((p for p in self.pythons if p.exists()), None)

    @property
    def available(self) -> bool:
        return self.python is not None and self.script.exists()

    def _start(self) -> None:
        if self._process and self._process.poll() is None:
            return
        if not self.available:
            raise SearchUnavailable(NOT_AVAILABLE)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "ab")
        self._process = subprocess.Popen(
            [str(self.python), str(self.script)], cwd=str(self.script.parent), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log_file, text=True, encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        log_file.close()
        bind_to_this_process(self._process)  # dies with samplegen
        self._lines = queue.Queue()
        stdout = self._process.stdout
        threading.Thread(target=lambda: [self._lines.put(line) for line in stdout], daemon=True,
                         name="clap-reader").start()
        ready = self._read(START_TIMEOUT_S)
        if not ready.get("ready"):
            raise SearchUnavailable(f"The CLAP worker didn't start; see {self.log_path}")

    def _read(self, timeout: float) -> dict:
        try:
            return json.loads(self._lines.get(timeout=timeout))
        except queue.Empty:
            self.stop()
            raise SearchUnavailable(f"The CLAP worker stopped answering; see {self.log_path}") from None
        except ValueError:
            return {}

    def request(self, payload: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
        with self._lock:
            self._start()
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
            except OSError as exc:
                self.stop()
                raise SearchUnavailable(f"The CLAP worker crashed; see {self.log_path}") from exc
            answer = self._read(timeout)
        if not answer.get("ok"):
            raise SearchUnavailable(answer.get("error") or "CLAP couldn't answer")
        return answer

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._process = None


def to_blob(vector) -> bytes:
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tobytes() if norm > 0 else UNUSABLE


class SoundIndex:
    def __init__(self, library: Library, worker: ClapWorker, busy=lambda: False):
        """`busy`: True while something else needs the machine (style training) — indexing waits."""
        self.library = library
        self.worker = worker
        self.busy = busy
        self.error: str | None = None
        self._stop = threading.Event()
        self._cache: tuple[int, list[str], np.ndarray] | None = None  # (count, ids, matrix)

    # ---------- indexing ----------

    def index_some(self, limit: int = BATCH) -> int:
        """Fingerprint up to `limit` new samples. Returns how many were processed."""
        pending = [r for r in self.library.without_embedding(limit)]
        if not pending:
            return 0
        paths = [str(self.library.path_of(r)) for r in pending]
        vectors = self.worker.request({"cmd": "audio", "paths": paths}).get("vectors", [])
        for record, vector in zip(pending, vectors):
            self.library.set_embedding(record.id, to_blob(vector) if vector else UNUSABLE)
        self._cache = None
        return len(pending)

    def run_forever(self) -> None:
        """Background thread: keep the index up to date, quietly."""
        while not self._stop.is_set():
            if not self.worker.available or self.busy():
                self._stop.wait(IDLE_POLL_S)
                continue
            try:
                done = self.index_some()
                self.error = None
            except Exception as exc:  # keep the app alive; show it in the status
                self.error = str(exc)
                log.warning("indexing paused: %s", exc)
                done = 0
                self._stop.wait(IDLE_POLL_S * 3)
            if not done:
                self._stop.wait(IDLE_POLL_S)

    def stop(self) -> None:
        self._stop.set()
        self.worker.stop()

    def status(self) -> dict:
        done, total = self.library.embedding_counts()
        return {"available": self.worker.available, "indexed": done, "total": total, "error": self.error}

    # ---------- searching ----------

    def _matrix(self) -> tuple[list[str], np.ndarray]:
        rows = [(i, v) for i, v in self.library.embeddings() if v]
        if self._cache and self._cache[0] == len(rows):
            return self._cache[1], self._cache[2]
        ids = [i for i, _ in rows]
        matrix = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in rows]) if rows else np.zeros((0, 512), np.float32)
        self._cache = (len(rows), ids, matrix)
        return ids, matrix

    def _rank(self, query: np.ndarray, limit: int, exclude: str | None = None) -> list[tuple[SampleRecord, float]]:
        ids, matrix = self._matrix()
        if not ids:
            return []
        scores = matrix @ query
        out = []
        for i in np.argsort(-scores):
            if ids[i] == exclude:
                continue
            try:
                out.append((self.library.get(ids[i]), float(scores[i])))
            except KeyError:
                continue
            if len(out) >= limit:
                break
        return out

    def search_text(self, text: str, limit: int = 50) -> list[tuple[SampleRecord, float]]:
        if not self.worker.available:
            raise SearchUnavailable(NOT_AVAILABLE)
        vector = np.asarray(self.worker.request({"cmd": "text", "text": text})["vector"], dtype=np.float32)
        return self._rank(vector, limit)

    def similar(self, sample_id: str, limit: int = 50) -> list[tuple[SampleRecord, float]]:
        ids, matrix = self._matrix()
        if sample_id not in ids:
            raise SearchUnavailable("This sample hasn't been listened to yet; try again in a minute.")
        return self._rank(matrix[ids.index(sample_id)], limit, exclude=sample_id)


def start_indexer(index: SoundIndex) -> threading.Thread:
    thread = threading.Thread(target=index.run_forever, name="sound-index", daemon=True)
    thread.start()
    return thread


__all__ = ["ClapWorker", "SearchUnavailable", "SoundIndex", "start_indexer"]

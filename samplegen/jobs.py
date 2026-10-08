"""Background job queue. One worker: the GPU runs one generation at a time."""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable

from .generation_request import GenerationRequest
from .instruments import InstrumentRequest
from .kits import KitRequest
from .layers import LayerRequest
from .power import keep_awake
from .stems import StemRequest

AnyRequest = GenerationRequest | StemRequest | InstrumentRequest | KitRequest | LayerRequest

log = logging.getLogger("samplegen.jobs")
MAX_JOB_HISTORY = 2000  # a big overnight batch must still be readable when it ends


@dataclass(frozen=True)
class Job:
    id: str
    request: AnyRequest
    state: str = "queued"  # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    sample_ids: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "state": self.state, "created": self.created, "started": self.started,
            "finished": self.finished, "sample_ids": list(self.sample_ids), "error": self.error,
            "mode": self.request.mode, "model": self.request.model, "prompt": self.request.full_prompt(),
            "variations": self.request.variations,
        }


class JobManager:
    def __init__(self, generator, busy: Callable[[], str | None] | None = None):
        """`busy` returns a reason when the GPU is taken by something else (style training)."""
        self.generator = generator
        self.busy = busy or (lambda: None)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cancelling: set[str] = set()  # running jobs the user cancelled
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="samplegen-worker", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=5)
            self._worker = None

    def submit(self, request: AnyRequest) -> Job:
        reason = self.busy()
        if reason:
            raise ValueError(reason)
        self.generator.check(request)
        job = Job(id=uuid.uuid4().hex[:10], request=request)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:  # check-and-set in one step: the worker may be picking this job up
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state == "queued":
                job = replace(job, state="cancelled", finished=time.time())
                self._jobs[job_id] = job
                return job
            if job.state == "running":
                self._cancelling.add(job_id)
        if job.state == "running":
            try:
                if hasattr(self.generator.client, "clear_queue"):
                    self.generator.client.clear_queue()
                self.generator.client.interrupt()
            except Exception as exc:  # engine may already be gone; report but don't crash the API
                log.warning("interrupt failed: %s", exc)
        return self.get(job_id)

    def _set(self, job_id: str, **changes) -> Job:
        with self._lock:
            job = replace(self._jobs[job_id], **changes)
            self._jobs[job_id] = job
            return job

    def _prune(self) -> None:
        finished = [j for j in self._jobs.values() if j.state in ("done", "error", "cancelled")]
        for old in sorted(finished, key=lambda j: j.created)[: max(0, len(self._jobs) - MAX_JOB_HISTORY)]:
            del self._jobs[old.id]

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                keep_awake(False)
                return
            job = self.get(job_id)
            if job is None or job.state != "queued":
                continue
            keep_awake(True)  # an overnight batch must not stop when Windows wants to sleep
            self.run_now(job_id)
            if self._queue.empty():
                keep_awake(False)

    def run_now(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            if job.state != "queued":  # cancelled between the queue and here
                return job
            job = replace(job, state="running", started=time.time())
            self._jobs[job_id] = job
        try:
            records = self.generator.run(job.request, job.id)
        except Exception as exc:
            if self._was_cancelled(job_id):
                return self._set(job_id, state="cancelled", finished=time.time())
            log.exception("job %s failed", job_id)
            return self._set(job_id, state="error", error=str(exc) or type(exc).__name__, finished=time.time())
        self._was_cancelled(job_id)  # finished before the interrupt landed: keep the result
        return self._set(job_id, state="done", sample_ids=tuple(r.id for r in records), finished=time.time())

    def _was_cancelled(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._cancelling:
                self._cancelling.discard(job_id)
                return True
            return False

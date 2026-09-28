import pytest

from samplegen.generation_request import GenerationRequest
from samplegen.jobs import JobManager


class StubGenerator:
    def __init__(self, fail=None):
        self.fail = fail
        self.client = type("C", (), {"interrupted": False, "interrupt": lambda self: setattr(self, "interrupted", True)})()

    def check(self, request):
        request.validate()

    def run(self, request, job_id):
        if self.fail:
            raise self.fail
        return [type("R", (), {"id": f"{job_id}-{i}"})() for i in range(request.variations)]


def req(**kw):
    fields = dict(mode="sfx", model="sa3-sfx", prompt="boom", duration=2.0, variations=2)
    fields.update(kw)
    return GenerationRequest(**fields)


def test_submit_validates_before_queueing():
    manager = JobManager(StubGenerator())
    with pytest.raises(ValueError):
        manager.submit(req(prompt=""))
    assert manager.recent() == []


def test_run_now_records_samples():
    manager = JobManager(StubGenerator())
    job = manager.submit(req())
    done = manager.run_now(job.id)
    assert done.state == "done"
    assert done.sample_ids == (f"{job.id}-0", f"{job.id}-1")
    assert done.started and done.finished
    assert manager.get(job.id).state == "done"


def test_run_now_captures_errors():
    manager = JobManager(StubGenerator(fail=RuntimeError("engine unreachable")))
    job = manager.submit(req())
    failed = manager.run_now(job.id)
    assert failed.state == "error"
    assert failed.error == "engine unreachable"


def test_cancel_queued_and_running():
    generator = StubGenerator()
    manager = JobManager(generator)
    queued = manager.submit(req())
    assert manager.cancel(queued.id).state == "cancelled"

    running = manager.submit(req())
    manager._set(running.id, state="running")
    manager.cancel(running.id)
    assert generator.client.interrupted  # stub client has no clear_queue: still interrupts
    assert manager.cancel("missing") is None


def test_cancelled_running_job_ends_as_cancelled_not_error():
    class Interrupted(StubGenerator):
        def run(self, request, job_id):
            manager.cancel(job_id)  # the user presses Stop while it runs
            raise RuntimeError("cancelled")

    manager = JobManager(Interrupted())
    job = manager.submit(req())
    ended = manager.run_now(job.id)
    assert ended.state == "cancelled" and ended.error is None


def test_job_cancelled_before_the_worker_starts_it_never_runs():
    generator = StubGenerator(fail=AssertionError("must not run"))
    manager = JobManager(generator)
    job = manager.submit(req())
    manager.cancel(job.id)
    assert manager.run_now(job.id).state == "cancelled"


def test_worker_thread_processes_queue():
    manager = JobManager(StubGenerator())
    manager.start()
    try:
        job = manager.submit(req())
        import time
        deadline = time.time() + 5
        while manager.get(job.id).state != "done" and time.time() < deadline:
            time.sleep(0.02)
        assert manager.get(job.id).state == "done"
    finally:
        manager.stop()


def test_job_to_dict_is_json_friendly():
    manager = JobManager(StubGenerator())
    data = manager.submit(req()).to_dict()
    assert data["state"] == "queued" and data["prompt"] == "boom" and data["sample_ids"] == []

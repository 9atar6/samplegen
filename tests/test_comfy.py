import io
import json
import urllib.error

import pytest

from samplegen.comfy import ComfyClient, EngineError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_for(responses):
    """responses: list of dicts/exceptions returned in order; records requests."""
    calls = []

    def opener(req, timeout):
        calls.append((req.full_url, json.loads(req.data) if req.data else None))
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(json.dumps(item).encode())

    return opener, calls


def test_submit_posts_graph():
    opener, calls = opener_for([{"prompt_id": "p1"}])
    assert ComfyClient("http://x", opener).submit({"a": 1}) == "p1"
    assert calls == [("http://x/prompt", {"prompt": {"a": 1}})]


def test_wait_polls_until_complete_and_collects_wavs():
    done = {"p1": {"status": {"completed": True, "status_str": "success"},
                   "outputs": {"save": {"samplegen_wavs": [{"filename": "a.wav", "subfolder": "samplegen"}]}}}}
    opener, calls = opener_for([{}, {}, done])
    outputs = ComfyClient("http://x", opener).wait("p1", timeout=10, poll=0, sleep=lambda s: None)
    assert [(o.filename, o.subfolder) for o in outputs] == [("a.wav", "samplegen")]
    assert len(calls) == 3


def test_wait_raises_engine_error_with_node_message():
    failed = {"p1": {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_start", {}],
        ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory\n"}],
    ]}}}
    opener, _ = opener_for([failed])
    with pytest.raises(EngineError, match="KSampler: CUDA out of memory"):
        ComfyClient("http://x", opener).wait("p1", timeout=10, poll=0, sleep=lambda s: None)


def test_wait_gives_up_at_once_on_prompts_dropped_by_clear_queue():
    # An instrument queues all its passes up front; cancel clears the queue, so the
    # remaining passes never reach the history. wait() must not poll them until the timeout.
    opener, calls = opener_for([{"prompt_id": "p2"}, {}, {}])
    client = ComfyClient("http://x", opener)
    client.submit({"a": 1})
    client.clear_queue()
    with pytest.raises(EngineError, match="cancelled"):
        client.wait("p2", timeout=900, poll=0, sleep=lambda s: None)
    assert len(calls) == 3  # submit, clear, one history check


def test_prompts_submitted_after_a_clear_are_waited_for_normally():
    done = {"p3": {"status": {"completed": True}, "outputs": {}}}
    opener, _ = opener_for([{}, {"prompt_id": "p3"}, {}, done])
    client = ComfyClient("http://x", opener)
    client.clear_queue()
    client.submit({"a": 1})
    assert client.wait("p3", timeout=10, poll=0, sleep=lambda s: None) == []


def test_wait_times_out():
    opener, _ = opener_for([{}] * 1000)
    with pytest.raises(EngineError, match="timed out"):
        ComfyClient("http://x", opener).wait("p1", timeout=0.05, poll=0.01)


def test_unreachable_engine():
    opener, _ = opener_for([urllib.error.URLError("refused")])
    client = ComfyClient("http://x", opener)
    assert client.is_alive() is False


def test_missing_node_error_explains_stale_engine():
    body = b'{"error": {"message": "Cannot execute because node SamplegenLoadWav does not exist."}}'
    err = urllib.error.HTTPError("http://x/prompt", 400, "Bad", {}, io.BytesIO(body))
    opener, _ = opener_for([err])
    with pytest.raises(EngineError, match="started before samplegen was updated"):
        ComfyClient("http://x", opener).submit({})


def test_http_error_is_reported():
    err = urllib.error.HTTPError("http://x/prompt", 400, "Bad", {}, io.BytesIO(b'{"error": "invalid prompt"}'))
    opener, _ = opener_for([err])
    with pytest.raises(EngineError, match="invalid prompt"):
        ComfyClient("http://x", opener).submit({})

"""Minimal client for ComfyUI's HTTP API (stdlib only)."""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .workflows import OUTPUT_NODE

REQUEST_TIMEOUT_SECONDS = 30
POLL_SECONDS = 0.5


class EngineError(RuntimeError):
    """The engine is unreachable or rejected / failed a job."""


@dataclass(frozen=True)
class OutputFile:
    filename: str
    subfolder: str


Opener = Callable[..., object]


class ComfyClient:
    def __init__(self, base_url: str, opener: Opener = urllib.request.urlopen):
        self.base_url = base_url.rstrip("/")
        self._open = opener

    def _request(self, path: str, payload: dict | None = None, timeout: float = REQUEST_TIMEOUT_SECONDS):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self._open(req, timeout=timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise EngineError(f"engine rejected request to {path}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise EngineError(f"engine unreachable at {self.base_url}: {exc}") from exc
        return json.loads(body) if body else {}

    def is_alive(self) -> bool:
        try:
            self._request("/system_stats", timeout=3)
            return True
        except EngineError:
            return False

    def system_stats(self) -> dict:
        return self._request("/system_stats", timeout=5)

    def submit(self, graph: dict) -> str:
        try:
            return self._request("/prompt", {"prompt": graph})["prompt_id"]
        except EngineError as exc:
            if "does not exist" in str(exc):
                raise EngineError(
                    f"{exc}\n\nThe engine that is running was started before samplegen was updated. "
                    "Close samplegen, end the leftover engine (Task Manager → python.exe using lots of memory, "
                    "or just restart the PC), then start samplegen again."
                ) from exc
            raise

    def object_info(self, node_class: str) -> dict | None:
        """The node's input/output description, or None if the engine doesn't have it."""
        return self._request(f"/object_info/{node_class}").get(node_class)

    def interrupt(self) -> None:
        self._request("/interrupt", {})

    def clear_queue(self) -> None:
        """Drop prompts that haven't started (e.g. the remaining passes of an instrument)."""
        self._request("/queue", {"clear": True})

    def free_memory(self) -> None:
        self._request("/free", {"unload_models": True, "free_memory": True})

    def wait(self, prompt_id: str, timeout: float, poll: float = POLL_SECONDS,
             sleep: Callable[[float], None] = time.sleep) -> list[OutputFile]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = self._request(f"/history/{prompt_id}").get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise EngineError(_describe_error(status))
                if status.get("completed"):
                    return [
                        OutputFile(f["filename"], f.get("subfolder", ""))
                        for out in entry.get("outputs", {}).values()
                        for f in out.get("samplegen_wavs", [])
                    ]
            sleep(poll)
        raise EngineError(f"generation {prompt_id} timed out after {timeout:.0f}s")


def _describe_error(status: dict) -> str:
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            node = data.get("node_type", "?")
            return f"engine error in {node}: {data.get('exception_message', '').strip()}"
    return "engine reported an error"


__all__ = ["ComfyClient", "EngineError", "OutputFile", "OUTPUT_NODE"]

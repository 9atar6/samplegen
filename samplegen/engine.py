"""Starts and stops the ComfyUI engine process when samplegen needs it."""

import subprocess
import sys
import threading
import time
from pathlib import Path

from .comfy import ComfyClient, EngineError
from .winjob import bind_to_this_process

STARTUP_TIMEOUT_SECONDS = 120
CREATE_NO_WINDOW = 0x08000000


class Engine:
    """Owns the ComfyUI process. Reuses one that is already running."""

    def __init__(self, engine_dir: Path, client: ComfyClient, port: int, log_path: Path):
        self.engine_dir = Path(engine_dir)
        self.client = client
        self.port = port
        self.log_path = Path(log_path)
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.state = "stopped"  # stopped | starting | ready | error
        self.error: str | None = None

    def command(self) -> list[str]:
        return [
            str(self.engine_dir / "python_embeded" / "python.exe"), "-s", "ComfyUI/main.py",
            "--windows-standalone-build", "--listen", "127.0.0.1", "--port", str(self.port),
            "--disable-auto-launch",
        ]

    def ensure_running(self, timeout: float = STARTUP_TIMEOUT_SECONDS) -> None:
        with self._lock:
            if self.client.is_alive():
                self.state, self.error = "ready", None
                return
            if not (self.engine_dir / "python_embeded" / "python.exe").exists():
                self._fail(f"engine not found in {self.engine_dir}")
            self.state = "starting"
            if self._process is None or self._process.poll() is not None:
                self._spawn()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.client.is_alive():
                    self.state, self.error = "ready", None
                    return
                if self._process.poll() is not None:
                    self._fail(f"engine exited during startup (code {self._process.returncode}); see {self.log_path}")
                time.sleep(1.0)
            self._fail(f"engine did not start within {timeout:.0f}s; see {self.log_path}")

    def _spawn(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")  # handed to the child; closed when it exits
        flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._process = subprocess.Popen(
            self.command(), cwd=self.engine_dir, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags,
        )
        log.close()
        bind_to_this_process(self._process)

    def _fail(self, message: str) -> None:
        self.state, self.error = "error", message
        raise EngineError(message)

    def stop(self) -> None:
        """Stop the engine only if samplegen started it."""
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
            self.state = "stopped"

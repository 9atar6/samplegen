import pytest

from samplegen.comfy import EngineError
from samplegen.engine import Engine


class AliveClient:
    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


def test_reuses_running_engine(tmp_path):
    engine = Engine(tmp_path, AliveClient(True), 8188, tmp_path / "engine.log")
    engine.ensure_running()
    assert engine.state == "ready" and engine.error is None


def test_missing_engine_reports_clear_error(tmp_path):
    engine = Engine(tmp_path / "nowhere", AliveClient(False), 8188, tmp_path / "engine.log")
    with pytest.raises(EngineError, match="engine not found"):
        engine.ensure_running()
    assert engine.state == "error"


def test_command_targets_localhost_only(tmp_path):
    cmd = Engine(tmp_path, AliveClient(False), 9999, tmp_path / "log").command()
    assert cmd[cmd.index("--listen") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "9999"


def test_child_process_is_bound_to_our_lifetime():
    import subprocess
    import sys

    from samplegen.winjob import bind_to_this_process

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert bind_to_this_process(child) is (sys.platform == "win32")
    finally:
        child.kill()
        child.wait()


def test_stop_without_process_is_safe(tmp_path):
    engine = Engine(tmp_path, AliveClient(True), 8188, tmp_path / "log")
    engine.stop()
    assert engine.state == "stopped"

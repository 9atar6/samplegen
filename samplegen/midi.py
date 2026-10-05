"""MIDI from audio: runs midi/transcribe.py (basic-pitch) in its own Python.

basic-pitch needs older libraries than the app (Python <= 3.11), so like style
training it lives in a separate environment, installed by tools/install-midi.bat.
"""

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger("samplegen.midi")

TIMEOUT_SECONDS = 300
CREATE_NO_WINDOW = 0x08000000
NOT_INSTALLED = "MIDI extraction isn't installed yet: double-click tools\\install-midi.bat (or run install.bat again)."


class MidiError(RuntimeError):
    pass


class Transcriber:
    def __init__(self, python: Path, script: Path, runner=subprocess.run):
        self.python = Path(python)
        self.script = Path(script)
        self._run = runner
        self._one_at_a_time = threading.Lock()

    @property
    def available(self) -> bool:
        return self.python.exists() and self.script.exists()

    def transcribe(self, audio: Path, out: Path, bpm: float | None = None) -> int:
        """Write `out` (.mid) from `audio`; returns the number of notes found."""
        return int(self.run(audio, out, bpm).get("notes", 0))

    def notes(self, audio: Path, out: Path) -> list[list[float]]:
        """[[start_s, end_s, midi, velocity 0-1], ...] sung/played in `audio` (also writes `out`)."""
        events = self.run(audio, out).get("events", [])
        return [e for e in events if isinstance(e, list) and len(e) == 4]

    def run(self, audio: Path, out: Path, bpm: float | None = None) -> dict:
        if not self.available:
            raise MidiError(NOT_INSTALLED)
        command = [str(self.python), str(self.script), "--audio", str(audio), "--out", str(out)]
        if bpm:
            command += ["--bpm", f"{float(bpm):g}"]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        try:
            with self._one_at_a_time:
                done = self._run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=TIMEOUT_SECONDS, cwd=str(self.script.parent), env=env,
                                 creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        except subprocess.TimeoutExpired as exc:
            raise MidiError("MIDI extraction took too long and was stopped.") from exc
        except OSError as exc:
            raise MidiError(f"Couldn't start MIDI extraction: {exc}") from exc
        if done.returncode != 0:
            log.warning("transcribe failed (%s): %s", done.returncode, (done.stderr or "")[-2000:])
            raise MidiError("MIDI extraction failed. Details are in the samplegen window.")
        try:
            result = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise MidiError("MIDI extraction returned nothing usable.") from exc
        if not Path(out).exists():
            raise MidiError("MIDI extraction didn't write a file.")
        return result if isinstance(result, dict) else {}

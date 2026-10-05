"""Automatic captions for training sounds (and anything else in the library).

caption = CLAP category (what it is)
        + musical clues from the file name (909, grime, house...)
        + measured character (analysis.py)
        + CLAP character words, only where CLAP is clearly confident.

CLAP runs in the trainer's Python (torch) as a subprocess; without it, captions
still get the file-name clues and the measurements.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .analysis import analyze, describe
from .audio import read_wav

log = logging.getLogger("samplegen.autodescribe")

CREATE_NO_WINDOW = 0x08000000
CLAP_TIMEOUT_SECONDS = 600
MAX_PHRASES = 11

# Confidence (top score minus runner-up) CLAP needs before we trust a word from a group.
# Measured on real kicks: below these, the words were noise (e.g. "smooth" for the most
# saturated kick); tone (warm/cold) never cleared the bar, so it isn't used at all.
GROUP_MARGINS = {"texture": 0.035, "era": 0.05, "space": 0.08, "origin": 0.1, "energy": 0.05}
# Words that were true of almost everything (or unreliable) in testing, so they carry no information.
# Space is only mentioned when there *is* some (reverb, room, echo), never for dry/close sounds.
IGNORED_WORDS = {"punchy", "soft", "subtle", "clean", "smooth", "modern", "dry", "close-miked", "processed",
                 "synthesized"}
FILENAME_ALIASES = {"bbox": "beatbox", "bd": None, "analogue": "analog", "hiphop": "hip-hop", "boombap": "boom-bap"}

# File-name words worth keeping (lower case). Everything else (brands, take numbers,
# "kick_01", "final") is noise.
FILENAME_TERMS = {
    "808", "909", "606", "707", "505", "727", "626", "303", "linn", "linndrum", "mpc", "sp1200", "dmx", "cr78",
    "house", "techno", "trap", "drill", "grime", "dubstep", "dnb", "jungle", "garage", "ukg", "lofi", "lo-fi",
    "boombap", "boom-bap", "hiphop", "hip-hop", "disco", "funk", "boogie", "soul", "jazz", "rock", "metal",
    "ambient", "cinematic", "trailer", "edm", "psytrance", "hardstyle", "reggaeton", "afro", "afrobeat", "amapiano",
    "acoustic", "analog", "analogue", "vintage", "tape", "vinyl", "distorted", "dirty", "clean", "punchy", "hard",
    "soft", "deep", "sub", "long", "short", "tight", "wide", "dry", "wet", "reverb", "room", "hall", "layered",
    "beatbox", "hip-hop", "boom-bap", "foley", "glass", "wood", "water", "rain", "wind", "fire", "door", "impact", "whoosh",
    "riser", "boom", "hit", "glitch", "noise", "texture", "pad", "pluck", "lead", "bass", "stab", "chord",
}


@dataclass(frozen=True)
class Tagging:
    """What CLAP said about one file: group -> [(word, score), ...] best first."""
    groups: dict

    def best(self, group: str) -> tuple[str, float, float] | None:
        ranked = self.groups.get(group) or []
        if not ranked:
            return None
        word, score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        return word, float(score), float(score) - float(runner)


def filename_terms(path: Path) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", Path(path).stem.lower().replace("-", " "))
    kept: list[str] = []
    for token in tokens:
        token = FILENAME_ALIASES.get(token, token)
        if not token:
            continue
        if token in FILENAME_TERMS and token not in kept:
            kept.append(token)
        elif re.fullmatch(r"(tr|rx|rz|sp|dr)?(505|606|707|727|808|909)[a-z]*\d*", token):  # "s808k1", "tr909"
            machine = re.search(r"(505|606|707|727|808|909)", token).group(1)
            if machine not in kept:
                kept.append(machine)
    return kept


KICK_WORDS = {"kick drum", "808 bass drum"}


def category_phrase(tagging: Tagging | None, names: list[str]) -> str | None:
    """What the sound is. The file name's drum machine beats CLAP's vaguer '808' guess."""
    if tagging is None:
        return None
    ranked = tagging.groups.get("category") or []
    if not ranked:
        return None
    top, score = ranked[0]
    if top not in KICK_WORDS:
        return top
    machines = [n for n in names if n.isdigit()]
    if machines:
        return f"{machines[0]} kick drum"
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    if top == "808 bass drum" and score - runner >= 0.05 and "beatbox" not in names:
        return "808 kick drum"
    return "kick drum"


def clap_character(tagging: Tagging | None) -> list[str]:
    if tagging is None:
        return []
    words = []
    for group, margin in GROUP_MARGINS.items():
        best = tagging.best(group)
        if best and best[2] >= margin and best[0] not in IGNORED_WORDS:
            words.append(best[0])
    return words


def compose(path: Path, measured: list[str], tagging: Tagging | None) -> str:
    phrases: list[str] = []

    def add(phrase: str | None):
        if phrase and phrase.lower() not in {p.lower() for p in phrases}:
            phrases.append(phrase)

    names = filename_terms(path)
    category = category_phrase(tagging, names)
    add(category)
    for name in names:
        if not (category and name in category):  # "909" already in "909 kick drum"
            add(name)
    for phrase in measured:
        add(phrase)
    for word in clap_character(tagging):
        add(word)
    return ", ".join(phrases[:MAX_PHRASES])


class Describer:
    """Runs trainer/clap_describe.py with the trainer's Python and composes captions."""

    def __init__(self, python: Path, script: Path, log_path: Path | None = None, runner=subprocess.run):
        self.python = Path(python)
        self.script = Path(script)
        self.log_path = Path(log_path) if log_path else None
        self._run = runner
        self._one_at_a_time = threading.Lock()

    @property
    def clap_available(self) -> bool:
        return self.python.exists() and self.script.exists()

    def tag(self, paths: list[Path], cpu: bool = False) -> dict[str, Tagging]:
        """CLAP results keyed by resolved path string. Empty if CLAP isn't available or fails."""
        if not paths or not self.clap_available:
            return {}
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "files.json"
            listing.write_text(json.dumps([str(p) for p in paths]), encoding="utf-8")
            command = [str(self.python), str(self.script), "--files", str(listing)] + (["--cpu"] if cpu else [])
            # The child must print UTF-8 too: by default Windows pipes use cp1252, which mangles
            # "é" in paths and crashes outright on Japanese or emoji file names.
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            try:
                with self._one_at_a_time:  # two CLAP models at once would fight over memory
                    done = self._run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                     timeout=CLAP_TIMEOUT_SECONDS, cwd=str(self.script.parent), env=env,
                                     creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("CLAP tagging failed to run: %s", exc)
                return {}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(done.stderr or "", encoding="utf-8")
        if done.returncode != 0:
            log.warning("CLAP tagging exited with %s", done.returncode)
            return {}
        try:
            rows = json.loads(done.stdout)
        except ValueError:
            log.warning("CLAP tagging returned no JSON")
            return {}
        out = {}
        for row in rows if isinstance(rows, list) else []:
            if "error" in row or "path" not in row:
                continue
            out[str(Path(row["path"]).resolve())] = Tagging({k: v for k, v in row.items() if isinstance(v, list)})
        return out

    def describe(self, paths: list[Path], cpu: bool = False) -> dict[str, str]:
        """Caption for every path (resolved path string -> caption)."""
        tags = self.tag(paths, cpu=cpu)
        captions = {}
        for path in paths:
            key = str(Path(path).resolve())
            try:
                audio, sr = read_wav(path)
                measured = describe(analyze(audio, sr))
            except Exception as exc:  # unreadable file: fall back to what we have
                log.warning("couldn't analyse %s: %s", path, exc)
                measured = []
            captions[key] = compose(Path(path), measured, tags.get(key))
        return captions

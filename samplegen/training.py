"""Train a style (LoRA) on your own sounds with the official Stable Audio 3 trainer.

The trainer lives in trainer/stable-audio-3 with its own Python (CUDA PyTorch),
installed by tools/install-training.bat. This module prepares the dataset, runs
train_lora.py as a background process, follows its progress, then converts the
final checkpoint into a ComfyUI LoRA and registers it as a style.
"""

import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import ExportFormat, read_wav, write_wav
from .catalog import MODELS, SAMPLE_RATE
from .sources import to_native
from .styles import BASES, TRAINER_MODELS, Style, StyleStore, style_slug
from .winjob import bind_to_this_process

log = logging.getLogger("samplegen.training")

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}
MIN_CLIPS = 5  # the trainer's demo loader needs a full batch of 4
MAX_CLIPS = 500
MAX_CLIP_SECONDS = 30.0
MIN_STEPS, MAX_STEPS = 100, 10000
MAX_CAPTION_CHARS = 400
LOG_TAIL_LINES = 12
POLL_SECONDS = 2.0
CREATE_NO_WINDOW = 0x08000000
# 24-bit is plenty for training data and a quarter smaller than float (500 clips x 30 s add up).
DATASET_FORMAT = ExportFormat(sample_rate=SAMPLE_RATE, bit_depth="24")


@dataclass(frozen=True)
class Clip:
    path: Path
    caption: str


@dataclass(frozen=True)
class TrainingRun:
    state: str = "idle"  # idle | preparing | training | converting | done | error | stopped
    style: str | None = None
    name: str | None = None
    step: int = 0
    total_steps: int = 0
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    log_tail: tuple[str, ...] = field(default_factory=tuple)

    @property
    def active(self) -> bool:
        return self.state in ("preparing", "training", "converting")

    def to_dict(self) -> dict:
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data["log_tail"] = list(self.log_tail)
        data["active"] = self.active
        return data


def caption_from_filename(path: Path) -> str:
    words = re.sub(r"[_\-.]+", " ", path.stem)
    words = re.sub(r"\b\d+\b", " ", words)  # drop take numbers like _01
    return re.sub(r"\s+", " ", words).strip().lower()


def clean_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:MAX_CAPTION_CHARS]


def scan_folder(folder: Path, limit: int = MAX_CLIPS) -> list[Clip]:
    """Audio files in `folder` (and subfolders), with a caption from a same-name .txt or the file name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} isn't a folder.")
    clips = []
    # os.walk, not sorted(rglob): stops as soon as there are enough sounds instead of
    # listing a whole drive first (someone will point it at E:\).
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            sidecar = path.with_suffix(".txt")
            try:
                caption = sidecar.read_text(encoding="utf-8", errors="replace") if sidecar.is_file() else None
            except OSError:
                caption = None
            clips.append(Clip(path, clean_caption(caption or caption_from_filename(path))))
            if len(clips) >= limit:
                return clips
    return clips


def latest_step(save_dir: Path) -> int:
    """Highest step the CSV logger has recorded so far (0 if none)."""
    best = 0
    for metrics in Path(save_dir).glob("lightning_logs/version_*/metrics.csv"):
        try:
            with open(metrics, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    step = str(row.get("step") or "").strip()  # None on a half-written last row
                    if step.isdigit():
                        best = max(best, int(step))
        except (OSError, csv.Error):
            continue
    return best


def checkpoint_interval(steps: int) -> int:
    """About 5 checkpoints, at an interval that divides `steps`: the trainer only saves on
    multiples of it, and the style must be made from the last step, not one 30 steps short."""
    target = max(100, steps // 5)
    return max((d for d in range(100, target + 1, 100) if steps % d == 0), default=steps)


def final_checkpoint(save_dir: Path) -> Path | None:
    def step_of(path: Path) -> int:
        match = re.search(r"step=(\d+)", path.name)
        return int(match.group(1)) if match else -1

    ckpts = sorted(Path(save_dir).rglob("*.ckpt"), key=step_of)
    return ckpts[-1] if ckpts else None


def tail(path: Path, lines: int = LOG_TAIL_LINES) -> tuple[str, ...]:
    try:
        # newline="": keep \r as-is (text mode would turn every progress redraw into its own line)
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            text = f.read()
    except OSError:
        return ()
    # progress bars rewrite a line with \r; keep only what the user would see last
    # (split on \n only: splitlines() would also split on \r and keep every redraw)
    text = text.replace("\r\n", "\n")  # Windows line endings aren't redraws
    rows = [row.split("\r")[-1].strip() for row in text.split("\n")]
    return tuple(r for r in rows if r)[-lines:]


def explain_failure(log_path: Path) -> str | None:
    """Turn the trainer's known failure modes into something a person can act on."""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    gated = re.search(r"huggingface\.co/(stabilityai/[\w.-]+)", text) if "gated repo" in text else None
    if gated:
        repo = gated.group(1).rstrip(".")  # the log's sentence ends right after the URL
        return (f"Your Hugging Face account doesn't have access to {repo} yet. Open "
                f"https://huggingface.co/{repo} , agree to the license at the top of the page, "
                "then start the training again.")
    missing = re.search(r"No module named '([\w.]+)'", text.split("Traceback")[-1]) if "Traceback" in text else None
    if missing:
        return (f"The trainer is missing the Python package '{missing.group(1)}'. Close samplegen, run "
                "tools/install-training.bat, then start the training again.")
    if re.search(r"\b401\b|Invalid (user )?token|GatedRepoError|Repository Not Found", text):
        return ("Hugging Face refused the download: you're not logged in, the token expired, or a license "
                "isn't accepted. Run tools/install-training.bat again (it checks all of this), then retry.")
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return "The GPU ran out of memory. Close other GPU-heavy apps, or use fewer / shorter sounds."
    return None


HEAVY_FOLDERS = ("data", "out")  # copied sounds and checkpoints: GBs, useless once a run has ended


def remove_heavy_files(run_dir: Path) -> int:
    """Delete a finished run's dataset copy and checkpoints, keep train.log. Returns folders removed.

    Only ever touches the two folders samplegen itself created inside a _training run;
    the user's original sounds are never in there (they're copied in at the start).
    """
    removed = 0
    for name in HEAVY_FOLDERS:
        folder = Path(run_dir) / name
        if folder.is_dir() and not folder.is_symlink():
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
    return removed


class TrainingManager:
    def __init__(self, project_dir: Path, library_root: Path, engine_dir: Path, styles: StyleStore,
                 free_engine: Callable[[], None] | None = None):
        self.trainer_dir = Path(project_dir) / "trainer" / "stable-audio-3"
        self.python = self.trainer_dir / ".venv" / "Scripts" / "python.exe"
        self.converter = Path(project_dir) / "trainer" / "convert_lora.py"
        self.work_root = Path(library_root) / "_training"
        self.checkpoints_dir = Path(engine_dir) / "ComfyUI" / "models" / "checkpoints"
        self.styles = styles
        self.free_engine = free_engine or (lambda: None)
        self._run = TrainingRun()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._stop_requested = False

    # ---------- status ----------

    @property
    def installed(self) -> bool:
        return self.python.exists() and (self.trainer_dir / "scripts" / "train_lora.py").exists()

    def status(self) -> dict:
        with self._lock:
            run = self._run
        return {**run.to_dict(), "installed": self.installed}

    def busy_reason(self) -> str | None:
        with self._lock:
            active = self._run.active
        return "A style is training right now; generation is paused until it finishes (or you stop it)." if active else None

    def _set(self, **changes) -> TrainingRun:
        with self._lock:
            self._run = replace(self._run, **changes)
            return self._run

    # ---------- start ----------

    def start(self, name: str, base: str, clips: list[Clip], steps: int = 1500, rank: int = 16,
              description: str = "") -> TrainingRun:
        if not self.installed:
            raise ValueError("Style training isn't set up yet: run tools/install-training.bat (or install.bat).")
        if base not in TRAINER_MODELS:
            raise ValueError("Choose SFX or Music as the base.")
        slug = style_slug(name)
        if self.styles.exists(slug):
            raise ValueError(f"A style called '{name}' already exists; pick another name.")
        if not MIN_STEPS <= steps <= MAX_STEPS:
            raise ValueError(f"Steps must be between {MIN_STEPS} and {MAX_STEPS}.")
        if rank not in (4, 8, 16, 32):
            raise ValueError("Rank must be 4, 8, 16 or 32.")
        if not MIN_CLIPS <= len(clips) <= MAX_CLIPS:
            raise ValueError(f"Use between {MIN_CLIPS} and {MAX_CLIPS} sounds (20-50 is a good start).")
        if any(not c.caption.strip() for c in clips):
            raise ValueError("Every sound needs a description.")
        base_ckpt = self.checkpoints_dir / MODELS[BASES[base]].checkpoint
        if not base_ckpt.exists():
            raise ValueError(f"The base model {base_ckpt.name} is missing: run tools/install-training.bat again.")
        with self._lock:
            if self._run.active:
                raise ValueError("A style is already training.")
            self._stop_requested = False
            self._run = TrainingRun(state="preparing", style=slug, name=name.strip(), total_steps=steps,
                                    started=time.time())
        job = dict(slug=slug, name=name.strip(), base=base, clips=list(clips), steps=steps, rank=rank,
                   description=clean_caption(description), base_ckpt=base_ckpt)
        threading.Thread(target=self._run_job, kwargs=job, name="style-training", daemon=True).start()
        return self._run

    def stop(self) -> TrainingRun:
        with self._lock:
            if self._run.state == "converting":
                return self._run  # a few seconds from done: cutting it would only leave a broken file
            self._stop_requested = True
            process = self._process
        if process and process.poll() is None:
            process.terminate()
        return self._run

    # ---------- the job ----------

    def _run_job(self, slug, name, base, clips, steps, rank, description, base_ckpt) -> None:
        work = self.work_root / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}"
        data_dir, save_dir, log_path = work / "data", work / "out", work / "train.log"
        try:
            count = self._prepare_dataset(clips, description, data_dir)
            if self._stop_requested:
                self._set(state="stopped", finished=time.time())
                return
            self.free_engine()
            self._set(state="training")
            code = self._train(base, data_dir, save_dir, log_path, steps, rank, slug)
            if self._stop_requested:
                self._set(state="stopped", finished=time.time(), log_tail=tail(log_path))
                return
            if code != 0:
                raise RuntimeError(explain_failure(log_path) or
                                   f"training stopped with an error (exit code {code}); see {log_path}")
            self._set(state="converting", step=steps, log_tail=tail(log_path))
            self._convert(save_dir, base_ckpt, slug, log_path)
            self.styles.register(Style(slug=slug, name=name, base=base, clips=count, steps=steps,
                                       created=datetime.now().isoformat(timespec="seconds"),
                                       description=description))
            self._set(state="done", finished=time.time(), log_tail=tail(log_path))
        except Exception as exc:  # report every failure in the UI rather than killing the thread silently
            log.exception("style training failed")
            self._set(state="error", error=str(exc), finished=time.time(), log_tail=tail(log_path))
        finally:
            self._process = None
            remove_heavy_files(work)  # the style (if any) is already saved next to the engine's models

    def clean_old_runs(self) -> int:
        """Free the space used by finished trainings (dataset copies, checkpoints); keeps their logs."""
        if self.status()["active"] or not self.work_root.is_dir():
            return 0
        return sum(remove_heavy_files(run) for run in self.work_root.iterdir() if run.is_dir())

    def _prepare_dataset(self, clips: list[Clip], description: str, data_dir: Path) -> int:
        data_dir.mkdir(parents=True, exist_ok=True)
        max_frames = int(MAX_CLIP_SECONDS * SAMPLE_RATE)
        count = 0
        for index, clip in enumerate(clips):
            if self._stop_requested:
                break
            try:
                audio, sr = read_wav(clip.path)  # soundfile reads wav/aiff/flac/ogg/mp3
                audio = to_native(audio, sr)[:max_frames]
            except Exception as exc:  # noqa: BLE001 - one unreadable file mustn't sink the whole run
                log.warning("skipping %s: %s", clip.path, exc)
                continue
            if audio.size == 0 or not np.isfinite(audio).all():
                continue
            stem = f"clip{index:04d}"
            write_wav(data_dir / f"{stem}.wav", audio, SAMPLE_RATE, DATASET_FORMAT)
            caption = clean_caption(f"{description}, {clip.caption}" if description else clip.caption)
            (data_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
            count += 1
        if self._stop_requested:
            return count
        if count < MIN_CLIPS:
            raise ValueError(f"Only {count} usable sounds; at least {MIN_CLIPS} are needed.")
        return count

    def train_command(self, base: str, data_dir: Path, save_dir: Path, steps: int, rank: int, slug: str) -> list[str]:
        every = checkpoint_interval(steps)
        return [
            str(self.python), "scripts/train_lora.py",
            "--model", TRAINER_MODELS[base], "--data_dir", str(data_dir), "--save_dir", str(save_dir),
            "--steps", str(steps), "--rank", str(rank), "--adapter_type", "lora",
            "--exclude", "conditioners",        # text encoder stays stock; ComfyUI loads it separately
            "--duration", str(int(MAX_CLIP_SECONDS)),
            "--checkpoint_every", str(every),
            "--demo_every", str(steps + 1),     # skip in-training demos: slow, and nowhere to listen
            "--num_workers", "0",               # the trainer's lambda worker_init_fn can't be pickled on Windows
            "--logger", "csv", "--name", slug,
        ]

    def _spawn(self, command: list[str], log_path: Path) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        process = subprocess.Popen(
            command, cwd=self.trainer_dir, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        log_file.close()
        bind_to_this_process(process)
        return process

    def _train(self, base, data_dir, save_dir, log_path, steps, rank, slug) -> int:
        process = self._spawn(self.train_command(base, data_dir, save_dir, steps, rank, slug), log_path)
        with self._lock:
            self._process = process
        try:
            while process.poll() is None:
                if self._stop_requested:  # also catches a Stop pressed while the trainer was starting
                    process.terminate()
                self._set(step=min(steps, latest_step(save_dir)), log_tail=tail(log_path))
                time.sleep(POLL_SECONDS)
            return process.returncode
        finally:
            if process.poll() is None:  # never leave a trainer holding the GPU behind
                process.terminate()

    def _convert(self, save_dir: Path, base_ckpt: Path, slug: str, log_path: Path) -> None:
        ckpt = final_checkpoint(save_dir)
        if ckpt is None:
            raise RuntimeError(f"training finished but saved no checkpoint in {save_dir}")
        out = self.styles.lora_path(slug)
        out.parent.mkdir(parents=True, exist_ok=True)
        command =[str(self.python), str(self.converter), "--ckpt", str(ckpt), "--base", str(base_ckpt),
                   "--out", str(out)]
        self._process = self._spawn(command, log_path)
        if self._process.wait() != 0 or not out.exists():
            raise RuntimeError(f"couldn't convert the trained style for the engine; see {log_path}")

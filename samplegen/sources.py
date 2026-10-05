"""Source audio for Transform / Edit: your own files or samples from the library.

Every source is stored once, converted to the engine's native format
(44.1 kHz stereo 32-bit float) under <library>/_sources/<id>.wav with a JSON
sidecar. Library samples are copied, never modified.
"""

import io
import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .audio import ExportFormat, read_wav, write_wav
from .catalog import SAMPLE_RATE

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_SOURCE_SECONDS = 380.0
SOURCE_ID = re.compile(r"^[0-9a-f]{10}$")
NATIVE = ExportFormat(sample_rate=SAMPLE_RATE, bit_depth="32f")


class SourceNotFound(KeyError):
    pass


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    duration: float
    is_loop: bool
    bpm: int | None = None
    bars: int | None = None
    key: str | None = None
    origin: str = "upload"  # upload | library

    def to_dict(self) -> dict:
        return asdict(self)


def to_native(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Any channel count / rate -> stereo at the engine rate."""
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE, quality="VHQ")
    return audio


class SourceStore:
    def __init__(self, root: Path):
        self.root = Path(root) / "_sources"
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, source_id: str) -> tuple[Path, Path]:
        if not SOURCE_ID.match(source_id):  # ids are also file names: keep them strict
            raise SourceNotFound(source_id)
        return self.root / f"{source_id}.wav", self.root / f"{source_id}.json"

    def import_bytes(self, data: bytes, filename: str) -> Source:
        if not data:
            raise ValueError("The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Files are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        unreadable = ValueError("Couldn't read that file. Use WAV, AIFF, FLAC, OGG or MP3.")
        try:
            # Check the length from the header first: a small compressed file can decode to gigabytes.
            info = sf.info(io.BytesIO(data))
            if info.samplerate > 0 and info.frames / info.samplerate > MAX_SOURCE_SECONDS:
                raise ValueError(f"Sources are limited to {MAX_SOURCE_SECONDS / 60:.0f} minutes.")
            audio, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
        except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
            raise unreadable from exc
        if len(audio) == 0:
            raise ValueError("The file has no audio.")
        name = Path(filename).stem[:80] or "source"
        return self._store(to_native(audio, sr), name, is_loop=False, origin="upload")

    def import_file(self, path: Path, name: str, is_loop: bool, bpm: int | None = None,
                    bars: int | None = None, key: str | None = None) -> Source:
        audio, sr = read_wav(path)
        return self._store(to_native(audio, sr), name, is_loop, origin="library", bpm=bpm, bars=bars, key=key)

    def _store(self, audio: np.ndarray, name: str, is_loop: bool, origin: str, **musical) -> Source:
        duration = len(audio) / SAMPLE_RATE
        if duration > MAX_SOURCE_SECONDS:
            raise ValueError(f"Sources are limited to {MAX_SOURCE_SECONDS / 60:.0f} minutes.")
        source = Source(id=uuid.uuid4().hex[:10], name=name, duration=duration, is_loop=is_loop,
                        origin=origin, **musical)
        wav_path, meta_path = self._paths(source.id)
        write_wav(wav_path, audio, SAMPLE_RATE, NATIVE, title=name)
        meta_path.write_text(json.dumps(source.to_dict()), encoding="utf-8")
        return source

    def get(self, source_id: str) -> Source:
        _, meta_path = self._paths(source_id)
        if not meta_path.exists():
            raise SourceNotFound(source_id)
        return Source(**json.loads(meta_path.read_text(encoding="utf-8")))

    def audio_path(self, source_id: str) -> Path:
        self.get(source_id)
        return self._paths(source_id)[0]

    def load(self, source_id: str) -> np.ndarray:
        audio, _ = read_wav(self.audio_path(source_id))
        return audio

    def copy_to(self, source_id: str, dest: Path) -> None:
        shutil.copyfile(self.audio_path(source_id), dest)

"""Sample library: WAV files on disk plus a SQLite index of how each was made.

Layout under the library root:
    Inbox/<YYYY-MM-DD>/   fresh generations waiting for a decision
    Kept/<SFX|Loops|Music>/  samples you kept
    _trash/               discarded samples (never deleted automatically)
    samplegen.db          metadata index
"""

# Lazy annotations: the Library.list method would otherwise shadow the builtin
# `list` in annotations evaluated inside the class body (e.g. all_tags).
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

STATUSES = ("new", "kept", "trashed")
KEPT_FOLDERS = {
    "sfx": "SFX", "loop": "Loops", "free": "Music", "transform": "Transformed", "edit": "Edited",
    "stems": "Stems", "instrument": "Instruments",
}
MAX_SLUG_LENGTH = 60
MAX_NAME_CHARS = 120
MAX_TAGS = 20
MAX_TAG_CHARS = 32
COLUMNS = ("id", "created_at", "status", "favorite", "name", "mode", "model", "prompt", "negative_prompt",
           "seed", "params", "duration", "sample_rate", "batch_id", "rel_path", "tags")

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    favorite INTEGER NOT NULL DEFAULT 0,
    name TEXT NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    seed INTEGER NOT NULL,
    params TEXT NOT NULL,
    duration REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    rel_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status, created_at);
"""


class SampleNotFound(KeyError):
    pass


class FileInUse(OSError):
    """Windows won't move a file another program (DAW, player) has open."""

    def __str__(self) -> str:
        return (f"{Path(self.args[0]).name} is open in another program (a DAW or player?). "
                "Close it there and try again.")


@dataclass(frozen=True)
class NewSample:
    mode: str
    model: str
    prompt: str
    negative_prompt: str
    seed: int
    params: dict
    duration: float
    sample_rate: int
    batch_id: str
    name: str


@dataclass(frozen=True)
class SampleRecord:
    id: str
    created_at: str
    status: str
    favorite: bool
    name: str
    mode: str
    model: str
    prompt: str
    negative_prompt: str
    seed: int
    params: dict
    duration: float
    sample_rate: int
    batch_id: str
    rel_path: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data["tags"] = list(self.tags)
        return data


def clean_tags(tags) -> tuple[str, ...]:
    """Lower-case, trimmed, de-duplicated, comma-free tags in the order given."""
    seen: dict[str, None] = {}
    for tag in tags:
        tag = re.sub(r"\s+", " ", str(tag).replace(",", " ")).strip().lower()[:MAX_TAG_CHARS]
        if tag:
            seen.setdefault(tag, None)
    if len(seen) > MAX_TAGS:
        raise ValueError(f"Up to {MAX_TAGS} tags per sample.")
    return tuple(seen)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:MAX_SLUG_LENGTH].rstrip("-")


def make_filename(name: str, sample_id: str) -> str:
    return f"{slugify(name) or 'sample'}_{sample_id}.wav"


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()  # re-entrant: set_status holds it while calling get()
        self._db = sqlite3.connect(self.root / "samplegen.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        existing = {row["name"] for row in self._db.execute("PRAGMA table_info(samples)")}
        if "tags" not in existing:  # added after the first release
            self._db.execute("ALTER TABLE samples ADD COLUMN tags TEXT NOT NULL DEFAULT ''")

    def close(self) -> None:
        self._db.close()

    def path_of(self, record: SampleRecord) -> Path:
        return self.root / record.rel_path

    def add(self, source: Path, sample: NewSample) -> SampleRecord:
        """Move a finished WAV into the Inbox and index it."""
        sample_id = uuid.uuid4().hex[:12]  # 8 hex chars collide around ~10k samples
        now = datetime.now()
        rel_path = Path("Inbox") / now.strftime("%Y-%m-%d") / make_filename(sample.name, sample_id)
        record = SampleRecord(
            id=sample_id, created_at=now.isoformat(timespec="seconds"), status="new", favorite=False,
            name=sample.name, mode=sample.mode, model=sample.model, prompt=sample.prompt,
            negative_prompt=sample.negative_prompt, seed=sample.seed, params=dict(sample.params),
            duration=sample.duration, sample_rate=sample.sample_rate, batch_id=sample.batch_id,
            rel_path=rel_path.as_posix(),
        )
        values = (record.id, record.created_at, record.status, int(record.favorite), record.name,
                  record.mode, record.model, record.prompt, record.negative_prompt, record.seed,
                  json.dumps(record.params), record.duration, record.sample_rate, record.batch_id,
                  record.rel_path, "")
        with self._lock:
            self._move_then_write(
                Path(source), self.path_of(record),
                f"INSERT INTO samples ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})", values,
            )
        return record

    def get(self, sample_id: str) -> SampleRecord:
        with self._lock:
            row = self._db.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
        if row is None:
            raise SampleNotFound(sample_id)
        return _to_record(row)

    def list(self, status: str | None = None, favorite: bool | None = None, query: str | None = None,
             batch_id: str | None = None, tag: str | None = None, limit: int = 200,
             offset: int = 0) -> list[SampleRecord]:
        clauses, args = [], []
        if status is None:
            clauses.append("status != 'trashed'")
        else:
            clauses.append("status = ?")
            args.append(status)
        if favorite is not None:
            clauses.append("favorite = ?")
            args.append(int(favorite))
        if batch_id:
            clauses.append("batch_id = ?")
            args.append(batch_id)
        if tag:
            cleaned = clean_tags([tag])
            clauses.append(r"tags LIKE ? ESCAPE '\'")
            args.append(f"%,{_like_escape(cleaned[0]) if cleaned else ''},%")
        if query:
            clauses.append(r"(prompt LIKE ? ESCAPE '\' OR name LIKE ? ESCAPE '\' OR tags LIKE ? ESCAPE '\')")
            args += [f"%{_like_escape(query)}%"] * 3
        sql = f"SELECT * FROM samples WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._db.execute(sql, (*args, limit, offset)).fetchall()
        return [_to_record(r) for r in rows]

    def set_status(self, sample_id: str, status: str) -> SampleRecord:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._lock:
            record = self.get(sample_id)
            if record.status == status:
                return record
            filename = Path(record.rel_path).name
            if status == "kept":
                rel_path = Path("Kept") / KEPT_FOLDERS.get(record.mode, "Other") / filename
            elif status == "trashed":
                rel_path = Path("_trash") / filename
            else:
                rel_path = Path("Inbox") / record.created_at[:10] / filename
            updated = replace(record, status=status, rel_path=rel_path.as_posix())
            self._move_then_write(
                self.path_of(record), self.path_of(updated),
                "UPDATE samples SET status = ?, rel_path = ? WHERE id = ?", (status, updated.rel_path, record.id),
            )
            return updated

    def set_favorite(self, sample_id: str, favorite: bool) -> SampleRecord:
        with self._lock:
            updated = replace(self.get(sample_id), favorite=favorite)
            self._db.execute("UPDATE samples SET favorite = ? WHERE id = ?", (int(favorite), sample_id))
            self._db.commit()
            return updated

    def set_tags(self, sample_id: str, tags) -> SampleRecord:
        cleaned = clean_tags(tags)
        with self._lock:
            updated = replace(self.get(sample_id), tags=cleaned)
            self._db.execute("UPDATE samples SET tags = ? WHERE id = ?", (_encode_tags(cleaned), sample_id))
            self._db.commit()
            return updated

    def rename(self, sample_id: str, name: str) -> SampleRecord:
        """New display name; the file is renamed to match (the id suffix keeps it unique)."""
        name = re.sub(r"\s+", " ", name).strip()[:MAX_NAME_CHARS]
        if not name:
            raise ValueError("The name can't be empty.")
        with self._lock:
            record = self.get(sample_id)
            rel_path = (Path(record.rel_path).parent / make_filename(name, record.id)).as_posix()
            updated = replace(record, name=name, rel_path=rel_path)
            if rel_path == record.rel_path:
                self._db.execute("UPDATE samples SET name = ? WHERE id = ?", (name, sample_id))
                self._db.commit()
                return updated
            self._move_then_write(
                self.path_of(record), self.path_of(updated),
                "UPDATE samples SET name = ?, rel_path = ? WHERE id = ?", (name, rel_path, sample_id),
            )
            return updated

    def all_tags(self) -> list[tuple[str, int]]:
        """(tag, count) over samples that aren't trashed, most used first."""
        counts: dict[str, int] = {}
        with self._lock:
            rows = self._db.execute("SELECT tags FROM samples WHERE status != 'trashed' AND tags != ''").fetchall()
        for row in rows:
            for tag in _decode_tags(row["tags"]):
                counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    def _move_then_write(self, source: Path, dest: Path, sql: str, values: tuple) -> None:
        """Move the file and record it; if the DB write fails, put the file back.

        os.replace only (everything lives under one root, so one volume): it's atomic and,
        unlike shutil.move, never leaves a half-copied duplicate when Windows refuses to
        delete a file that's in use. A missing source (moved or deleted outside samplegen)
        only updates the record, so the sample can still be trashed or kept.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        moved = source.exists()
        if moved:
            if dest.exists():
                raise FileExistsError(dest)
            try:
                os.replace(source, dest)
            except PermissionError as exc:
                raise FileInUse(source) from exc
        try:
            self._db.execute(sql, values)
            self._db.commit()
        except Exception:
            self._db.rollback()
            if moved:
                os.replace(dest, source)
            raise


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _encode_tags(tags: tuple[str, ...]) -> str:
    # Stored as ",a,b," so a LIKE '%,tag,%' filter only matches whole tags.
    return f",{','.join(tags)}," if tags else ""


def _decode_tags(stored: str) -> tuple[str, ...]:
    return tuple(t for t in stored.split(",") if t)


def _to_record(row: sqlite3.Row) -> SampleRecord:
    data = dict(row)
    data["favorite"] = bool(data["favorite"])
    data["params"] = json.loads(data["params"])
    data["tags"] = _decode_tags(data.get("tags") or "")
    return SampleRecord(**data)

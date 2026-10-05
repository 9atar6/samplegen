"""Export a set of library samples as a named sample pack folder.

Packs are copies: the library itself is never changed. Layout:
    <library>/Packs/<Pack name>/<SFX|Loops|...>/<clean-name>.wav
    <library>/Packs/<Pack name>/samples.csv   (prompt, seed, tempo, key, tags per file)
"""

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .audio import ExportFormat, read_wav, write_wav
from .library import KEPT_FOLDERS, Library, SampleRecord, slugify

PACKS_DIR = "Packs"
MAX_PACK_SAMPLES = 2000
MAX_PACK_NAME = 80
INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
CSV_FIELDS = ("file", "name", "type", "model", "prompt", "seed", "bpm", "bars", "key", "tags", "duration_s")


@dataclass(frozen=True)
class PackResult:
    name: str
    folder: Path
    exported: int
    missing: int

    def to_dict(self) -> dict:
        return {"name": self.name, "folder": str(self.folder), "exported": self.exported, "missing": self.missing}


RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def clean_pack_name(name: str) -> str:
    cleaned = INVALID_NAME_CHARS.sub("", name).strip().strip(".")[:MAX_PACK_NAME].strip().rstrip(". ")
    if not cleaned:
        raise ValueError("Give the pack a name.")
    if cleaned.split(".")[0].upper() in RESERVED_NAMES:  # Windows can't create a folder called CON, NUL...
        cleaned = f"_{cleaned}"
    return cleaned


def _csv_safe(value):
    """Spreadsheets run cells starting with = + - @ as formulas: a prompt must stay text."""
    if isinstance(value, str) and value.startswith(FORMULA_START):
        return "'" + value
    return value


def free_folder(parent: Path, name: str) -> Path:
    """`parent/name`, or `parent/name (2)`, (3)... if taken. Never reuses an existing folder."""
    folder, n = parent / name, 2
    while folder.exists():
        folder = parent / f"{name} ({n})"
        n += 1
    return folder


def _free_file(folder: Path, stem: str) -> Path:
    path, n = folder / f"{stem}.wav", 2
    while path.exists():
        path = folder / f"{stem}-{n}.wav"
        n += 1
    return path


def export_pack(library: Library, records: list[SampleRecord], name: str,
                fmt: ExportFormat | None = None) -> PackResult:
    """Copy `records` into a new pack folder; `fmt` converts, None keeps files as they are."""
    if not records:
        raise ValueError("There are no samples to export.")
    if len(records) > MAX_PACK_SAMPLES:
        raise ValueError(f"Packs are limited to {MAX_PACK_SAMPLES} samples.")
    pack_name = clean_pack_name(name)
    folder = free_folder(library.root / PACKS_DIR, pack_name)
    folder.mkdir(parents=True)

    rows, missing = [], 0
    for record in records:
        source = library.path_of(record)
        if not source.exists():
            missing += 1
            continue
        subfolder = folder / KEPT_FOLDERS.get(record.mode, "Other")
        subfolder.mkdir(exist_ok=True)
        dest = _free_file(subfolder, slugify(record.name) or "sample")
        if fmt is None:
            shutil.copy2(source, dest)
        else:
            audio, sr = read_wav(source)
            write_wav(dest, audio, sr, fmt, title=record.name, comment=f"samplegen | {record.prompt}"[:1000],
                      loop=record.mode == "loop" or bool(record.params.get("loop")))
        p = record.params
        rows.append({
            "file": dest.relative_to(folder).as_posix(), "name": record.name, "type": record.mode,
            "model": record.model, "prompt": record.prompt, "seed": record.seed,
            "bpm": p.get("bpm") or "", "bars": p.get("bars") or "", "key": p.get("key") or "",
            "tags": " ".join(record.tags), "duration_s": f"{record.duration:.3f}",
        })

    # utf-8-sig: Excel only reads accents correctly when the file starts with a BOM.
    with open(folder / "samples.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({k: _csv_safe(v) for k, v in row.items()} for row in rows)
    return PackResult(name=folder.name, folder=folder, exported=len(rows), missing=missing)

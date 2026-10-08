"""The Play tab's backend: which instruments exist, which note each sample file is.

Instruments are folders written by instruments.write_instrument:
    Instruments/<name>/Samples/<stem>_<Note>.wav   (e.g. grand_Cs4.wav for C#4)
and a library record (mode "instrument") whose params point at that folder.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .instruments import name_to_midi
from .library import Library, SampleRecord

SAMPLE_NOTE = re.compile(r"_([A-G]s?-?\d)$")  # file name suffix: C4, Cs4, ...
MAX_INSTRUMENTS = 500
PLAYABLE = ("instrument", "kit")  # kits play on their drum keys only (no repitching)


@dataclass(frozen=True)
class InstrumentInfo:
    id: str
    name: str
    folder: Path
    notes: dict[int, Path]  # midi -> sample file
    kind: str = "instrument"  # or "kit"

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "low": min(self.notes, default=None),
                "high": max(self.notes, default=None), "count": len(self.notes), "kind": self.kind}


def note_of(sample: Path) -> int | None:
    match = SAMPLE_NOTE.search(sample.stem)
    if not match:
        return None
    name = match.group(1)
    name = name[0] + "#" + name[2:] if len(name) > 1 and name[1] == "s" else name
    try:
        return name_to_midi(name)
    except ValueError:
        return None


def instrument_notes(folder: Path) -> dict[int, Path]:
    notes = {}
    for wav in sorted((Path(folder) / "Samples").glob("*.wav")):
        midi = note_of(wav)
        if midi is not None:
            notes[midi] = wav
    return notes


def instrument_info(library: Library, record: SampleRecord) -> InstrumentInfo | None:
    rel = record.params.get("instrument_folder")
    if record.mode not in PLAYABLE or not rel:
        return None
    folder = (library.root / rel).resolve()
    if not folder.is_relative_to(library.root.resolve()) or not folder.is_dir():
        return None  # moved/deleted outside samplegen, or a path that doesn't belong to the library
    notes = instrument_notes(folder)
    return InstrumentInfo(record.id, record.name, folder, notes, record.mode) if notes else None


def list_instruments(library: Library) -> list[InstrumentInfo]:
    found = []
    for status in ("new", "kept"):
        for mode in PLAYABLE:
            for record in library.list(status=status, mode=mode, limit=MAX_INSTRUMENTS):
                info = instrument_info(library, record)
                if info:
                    found.append(info)
    return found

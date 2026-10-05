"""Tags, rename, schema migration and sample-pack export."""

import csv
import sqlite3

import numpy as np
import pytest
import soundfile as sf

from samplegen.audio import ExportFormat
from samplegen.library import Library, NewSample, clean_tags
from samplegen.packs import clean_pack_name, export_pack


def add(lib, tmp_path, name, mode="sfx", prompt=None, params=None):
    src = tmp_path / f"{name}.wav"
    sf.write(str(src), np.zeros((4410, 2)), 44100, subtype="FLOAT")
    return lib.add(src, NewSample(
        mode=mode, model="sa3-sfx", prompt=prompt or name, negative_prompt="", seed=1,
        params=params or {}, duration=0.1, sample_rate=44100, batch_id="b", name=name,
    ))


@pytest.fixture
def lib(tmp_path):
    library = Library(tmp_path / "lib")
    yield library
    library.close()


def test_clean_tags_normalises_and_dedupes():
    assert clean_tags(["  Metal ", "metal", "Dark,Heavy", "", "big   boom"]) == ("metal", "dark heavy", "big boom")
    with pytest.raises(ValueError):
        clean_tags([f"t{i}" for i in range(21)])


def test_tags_filter_search_and_counts(lib, tmp_path):
    a = add(lib, tmp_path, "alpha")
    b = add(lib, tmp_path, "beta")
    lib.set_tags(a.id, ["metal", "impact"])
    lib.set_tags(b.id, ["metal"])
    assert lib.get(a.id).tags == ("metal", "impact")
    assert {s.id for s in lib.list(tag="metal")} == {a.id, b.id}
    assert [s.id for s in lib.list(tag="impact")] == [a.id]
    assert lib.list(tag="met") == []  # whole tags only
    assert [s.id for s in lib.list(query="impact")] == [a.id]
    assert lib.all_tags() == [("metal", 2), ("impact", 1)]


def test_like_wildcards_in_search_are_literal(lib, tmp_path):
    add(lib, tmp_path, "plain")
    assert lib.list(query="%") == []
    assert lib.list(query="_") == []


def test_rename_moves_the_file(lib, tmp_path):
    record = add(lib, tmp_path, "alpha")
    old_path = lib.path_of(record)
    renamed = lib.rename(record.id, "  Big   Metal Door  ")
    assert renamed.name == "Big Metal Door"
    assert lib.path_of(renamed).name == f"big-metal-door_{record.id}.wav"
    assert lib.path_of(renamed).exists() and not old_path.exists()
    assert lib.get(record.id).name == "Big Metal Door"
    with pytest.raises(ValueError):
        lib.rename(record.id, "   ")


def test_old_database_is_migrated(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    db = sqlite3.connect(root / "samplegen.db")
    db.execute("""CREATE TABLE samples (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
        favorite INTEGER NOT NULL DEFAULT 0, name TEXT NOT NULL, mode TEXT NOT NULL, model TEXT NOT NULL,
        prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL, seed INTEGER NOT NULL, params TEXT NOT NULL,
        duration REAL NOT NULL, sample_rate INTEGER NOT NULL, batch_id TEXT NOT NULL, rel_path TEXT NOT NULL)""")
    db.execute("INSERT INTO samples VALUES ('abc', '2026-09-25T10:00:00', 'new', 0, 'x', 'sfx', 'm', 'p', '', 1, '{}',"
               " 1.0, 44100, 'b', 'Inbox/2026-09-25/x_abc.wav')")
    db.commit()
    db.close()
    library = Library(root)
    assert library.get("abc").tags == ()
    library.set_tags("abc", ["kept-from-v1"])
    assert library.get("abc").tags == ("kept-from-v1",)
    library.close()


def test_clean_pack_name():
    assert clean_pack_name('  Dark: "Impacts" / Vol.1. ') == "Dark Impacts  Vol.1"
    with pytest.raises(ValueError):
        clean_pack_name(" ??? ")


def test_export_pack_copies_and_writes_csv(lib, tmp_path):
    a = add(lib, tmp_path, "boom", prompt="deep boom")
    b = add(lib, tmp_path, "boom", prompt="another boom")  # same name -> distinct files
    c = add(lib, tmp_path, "bass", mode="loop", params={"bpm": 140, "bars": 8, "key": "E minor"})
    lib.set_tags(c.id, ["acid"])
    result = export_pack(lib, [lib.get(a.id), lib.get(b.id), lib.get(c.id)], "My Pack")
    assert result.exported == 3 and result.missing == 0
    assert sorted(p.name for p in (result.folder / "SFX").iterdir()) == ["boom-2.wav", "boom.wav"]
    assert (result.folder / "Loops" / "bass.wav").exists()
    assert lib.path_of(a).exists()  # a copy, the library is untouched
    with open(result.folder / "samples.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    loop_row = next(r for r in rows if r["type"] == "loop")
    assert (loop_row["bpm"], loop_row["key"], loop_row["tags"]) == ("140", "E minor", "acid")


def test_pack_csv_never_holds_spreadsheet_formulas(lib, tmp_path):
    a = add(lib, tmp_path, "evil", prompt='=HYPERLINK("http://x","click")')
    result = export_pack(lib, [a], "P")
    with open(result.folder / "samples.csv", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert row["prompt"].startswith("'=")


def test_reserved_windows_names_are_not_used_as_folders():
    assert clean_pack_name("CON") == "_CON"
    assert clean_pack_name("nul.txt") == "_nul.txt"
    assert clean_pack_name("Console") == "Console"


def test_sample_whose_file_vanished_can_still_be_trashed(lib, tmp_path):
    record = add(lib, tmp_path, "ghost")
    lib.path_of(record).unlink()  # deleted outside samplegen
    assert lib.set_status(record.id, "trashed").status == "trashed"


def test_file_open_elsewhere_reports_file_in_use_and_keeps_record(lib, tmp_path, monkeypatch):
    from samplegen import library as library_module

    record = add(lib, tmp_path, "busy")

    def locked(*_args):
        raise PermissionError("in use")

    monkeypatch.setattr(library_module.os, "replace", locked)
    with pytest.raises(library_module.FileInUse, match="open in another program"):
        lib.set_status(record.id, "kept")
    assert lib.get(record.id).status == "new" and lib.path_of(record).exists()


def test_export_pack_never_overwrites_and_converts(lib, tmp_path):
    a = add(lib, tmp_path, "hit")
    first = export_pack(lib, [a], "Pack")
    second = export_pack(lib, [a], "Pack", ExportFormat(48000, "24"))
    assert first.folder != second.folder and second.folder.name == "Pack (2)"
    info = sf.info(str(second.folder / "SFX" / "hit.wav"))
    assert (info.samplerate, info.subtype) == (48000, "PCM_24")


def test_export_pack_skips_missing_and_rejects_empty(lib, tmp_path):
    a = add(lib, tmp_path, "gone")
    lib.path_of(a).unlink()
    assert export_pack(lib, [a], "P").missing == 1
    with pytest.raises(ValueError):
        export_pack(lib, [], "P")


def test_repair_relinks_samples_moved_by_an_interrupted_move(lib, tmp_path):
    record = add(lib, tmp_path, "boom")
    kept_dir = lib.root / "Kept" / "SFX"
    kept_dir.mkdir(parents=True)
    lib.path_of(record).replace(kept_dir / lib.path_of(record).name)  # file moved, DB never updated
    assert lib.repair() == {"relinked": 1, "recovered": 0}
    fixed = lib.get(record.id)
    assert fixed.status == "kept" and lib.path_of(fixed).exists()


def test_repair_recovers_finished_takes_left_in_staging(lib, tmp_path):
    staging = lib.root / "_staging"
    staging.mkdir()
    sf.write(str(staging / "job1_01.wav"), np.zeros((4410, 2)), 44100, subtype="FLOAT")
    (staging / "half.wav").write_bytes(b"RIFF....")  # unreadable: left alone
    result = lib.repair(staging=staging, min_age_s=-60)  # no waiting in tests
    assert result == {"relinked": 0, "recovered": 1}
    (recovered,) = lib.list()
    assert recovered.name == "Recovered take" and lib.path_of(recovered).exists()
    assert (staging / "half.wav").exists()

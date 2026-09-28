import pytest

from samplegen.library import Library, NewSample, SampleNotFound, make_filename


def new_sample(tmp_path, name="door-slam", **overrides):
    src = tmp_path / f"{name}.wav"
    src.write_bytes(b"RIFFfake")
    fields = dict(
        mode="sfx", model="sa3-sfx", prompt="heavy door slam", negative_prompt="", seed=42,
        params={"duration": 4.0}, duration=4.0, sample_rate=44100, batch_id="b1", name=name,
    )
    fields.update(overrides)
    return src, NewSample(**fields)


@pytest.fixture
def lib(tmp_path):
    library = Library(tmp_path / "library")
    yield library
    library.close()


def test_make_filename_is_safe_and_descriptive():
    assert make_filename("Heavy door SLAM!!  in a hallway / concrete", "a1b2") == "heavy-door-slam-in-a-hallway-concrete_a1b2.wav"
    assert len(make_filename("x" * 300, "id")) < 80
    assert make_filename("???", "id") == "sample_id.wav"


def test_add_moves_file_into_inbox_and_records_metadata(lib, tmp_path):
    src, sample = new_sample(tmp_path)
    record = lib.add(src, sample)
    assert record.status == "new"
    assert not src.exists()
    path = lib.path_of(record)
    assert path.exists()
    assert path.parent.parent.name == "Inbox"
    assert lib.get(record.id).prompt == "heavy door slam"
    assert lib.get(record.id).params == {"duration": 4.0}


def test_keep_moves_to_kept_folder_by_mode(lib, tmp_path):
    src, sample = new_sample(tmp_path, mode="loop", model="f1-samples")
    record = lib.add(src, sample)
    kept = lib.set_status(record.id, "kept")
    assert kept.status == "kept"
    path = lib.path_of(kept)
    assert path.exists()
    assert path.parent.name == "Loops"
    assert path.parent.parent.name == "Kept"


def test_trash_and_restore(lib, tmp_path):
    src, sample = new_sample(tmp_path)
    record = lib.add(src, sample)
    trashed = lib.set_status(record.id, "trashed")
    assert lib.path_of(trashed).parent.name == "_trash"
    restored = lib.set_status(record.id, "kept")
    assert lib.path_of(restored).exists()


def test_set_status_rejects_unknown_status(lib, tmp_path):
    src, sample = new_sample(tmp_path)
    record = lib.add(src, sample)
    with pytest.raises(ValueError):
        lib.set_status(record.id, "deleted")


def test_favorite_and_list_filters(lib, tmp_path):
    a = lib.add(*new_sample(tmp_path, name="alpha", prompt="alpha whoosh"))
    b = lib.add(*new_sample(tmp_path, name="beta", prompt="beta impact"))
    lib.set_favorite(b.id, True)
    lib.set_status(a.id, "kept")

    assert [s.id for s in lib.list(favorite=True)] == [b.id]
    assert [s.id for s in lib.list(status="kept")] == [a.id]
    assert [s.id for s in lib.list(query="whoosh")] == [a.id]
    assert {s.id for s in lib.list()} == {a.id, b.id}


def test_list_hides_trash_unless_asked(lib, tmp_path):
    a = lib.add(*new_sample(tmp_path, name="alpha"))
    lib.set_status(a.id, "trashed")
    assert lib.list() == []
    assert [s.id for s in lib.list(status="trashed")] == [a.id]


def test_failed_db_write_puts_file_back(lib, tmp_path, monkeypatch):
    record = lib.add(*new_sample(tmp_path))
    original = lib.path_of(record)

    class BrokenDb:
        def __init__(self, real):
            self.real = real

        def execute(self, sql, *args):
            if sql.startswith("UPDATE"):
                raise RuntimeError("disk full")
            return self.real.execute(sql, *args)

        def rollback(self):
            self.real.rollback()

    monkeypatch.setattr(lib, "_db", BrokenDb(lib._db))
    with pytest.raises(RuntimeError, match="disk full"):
        lib.set_status(record.id, "kept")
    assert original.exists()
    monkeypatch.undo()
    assert lib.get(record.id).status == "new"


def test_get_unknown_raises(lib):
    with pytest.raises(SampleNotFound):
        lib.get("nope")


def test_name_collisions_get_unique_files(lib, tmp_path):
    first = lib.add(*new_sample(tmp_path, name="same"))
    second = lib.add(*new_sample(tmp_path, name="same"))
    assert lib.path_of(first) != lib.path_of(second)
    assert lib.path_of(first).exists() and lib.path_of(second).exists()


def test_library_persists_between_instances(tmp_path):
    root = tmp_path / "library"
    first = Library(root)
    record = first.add(*new_sample(tmp_path))
    first.close()
    second = Library(root)
    assert second.get(record.id).name == record.name
    second.close()

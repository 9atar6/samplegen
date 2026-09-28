import json

from samplegen.config import default_library, load_settings


def test_defaults_without_local_config(tmp_path, monkeypatch):
    monkeypatch.delenv("SAMPLEGEN_LIBRARY", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    settings = load_settings(tmp_path / "missing.json")
    assert settings.library_dir == default_library() == tmp_path / "Music" / "samplegen-library"
    assert settings.app_port == 8190


def test_local_config_then_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("SAMPLEGEN_LIBRARY", raising=False)
    local = tmp_path / "samplegen.local.json"
    local.write_text("\ufeff" + json.dumps({"library": "E:\\lib", "app_port": 9000}), encoding="utf-8")  # with BOM
    settings = load_settings(local)
    assert str(settings.library_dir) == "E:\\lib" and settings.app_port == 9000
    monkeypatch.setenv("SAMPLEGEN_LIBRARY", str(tmp_path / "env-lib"))
    assert load_settings(local).library_dir == tmp_path / "env-lib"


def test_broken_local_config_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("SAMPLEGEN_LIBRARY", raising=False)
    local = tmp_path / "samplegen.local.json"
    local.write_text("{not json", encoding="utf-8")
    assert load_settings(local).library_dir == default_library()

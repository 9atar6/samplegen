from samplegen.catalog import MODELS
from samplegen.setup_check import missing_parts


def install_everything(engine_dir):
    models = engine_dir / "ComfyUI" / "models"
    for spec in MODELS.values():
        for sub, name in (("checkpoints", spec.checkpoint), ("text_encoders", spec.text_encoder)):
            (models / sub).mkdir(parents=True, exist_ok=True)
            (models / sub / name).write_bytes(b"x")
    for node in ("samplegen_nodes", "AudioSeparation"):
        folder = engine_dir / "ComfyUI" / "custom_nodes" / node
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("")


def test_complete_install_reports_nothing(tmp_path):
    install_everything(tmp_path)
    assert missing_parts(tmp_path) == []


def test_empty_engine_reports_every_mode_model_and_both_extensions(tmp_path):
    missing = missing_parts(tmp_path)
    labels = [m.label for m in MODELS.values() if m.modes]
    assert len(missing) == len(labels) + 2
    assert all(any(label in item for item in missing) for label in labels)
    assert any("AudioSeparation" in item for item in missing)
    assert any("samplegen_nodes" in item for item in missing)


def test_missing_file_is_named_and_base_models_are_ignored(tmp_path):
    install_everything(tmp_path)
    (tmp_path / "ComfyUI" / "models" / "checkpoints" / "Foundation-1.2-Keybeds.safetensors").unlink()
    (tmp_path / "ComfyUI" / "models" / "checkpoints" / "stable_audio_3_small_sfx_base.safetensors").unlink()
    missing = missing_parts(tmp_path)
    assert len(missing) == 1
    assert "Keybeds" in missing[0] and "Foundation-1.2-Keybeds.safetensors" in missing[0]

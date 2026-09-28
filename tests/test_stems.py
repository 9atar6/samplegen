import numpy as np
import pytest
import soundfile as sf

from samplegen.comfy import EngineError
from samplegen.generation import Generator
from samplegen.library import Library
from samplegen.sources import SourceStore
from samplegen.stems import (
    StemRequest, build_stems_graph, pick_model, separator_inputs, stem_from_filename, stem_names,
)

from .fakes import DEMUCS_INFO, FakeClient, FakeEngine, tone_wav_bytes


def test_pick_model_prefers_fine_tuned_four_stem():
    assert pick_model(["htdemucs.safetensors", "htdemucs_ft.safetensors", "htdemucs_6s"]) == "htdemucs_ft.safetensors"
    assert pick_model(["htdemucs_6s", "htdemucs"]) == "htdemucs"
    assert pick_model(["something_else"]) == "something_else"
    with pytest.raises(EngineError):
        pick_model([])


def test_separator_inputs_follow_object_info_including_typos():
    audio_input, inputs, model = separator_inputs(DEMUCS_INFO)
    assert audio_input == "input_sound"
    assert model == "htdemucs_ft.safetensors"
    assert inputs["taget_device"] == "CUDA:0"  # GPU picked from the new-style COMBO encoding
    assert inputs["shifts"] == 1 and inputs["overlap"] == 0.25 and inputs["custom_segment"] is False
    assert stem_names(DEMUCS_INFO) == ["vocals", "drums", "bass", "other"]


def test_separator_inputs_reject_unknown_required_input():
    info = {"input": {"required": {"input_sound": ["AUDIO"], "mystery": ["STRING", {}]}}}
    with pytest.raises(EngineError, match="mystery"):
        separator_inputs(info)


def test_stems_graph_saves_every_output():
    graph, model = build_stems_graph(DEMUCS_INFO, "job.wav", "j1")
    assert graph["separate"]["inputs"]["input_sound"] == ["load", 0]
    saves = {k: v for k, v in graph.items() if v["class_type"] == "SamplegenSaveWav"}
    assert {v["inputs"]["job_id"] for v in saves.values()} == {"j1-vocals", "j1-drums", "j1-bass", "j1-other"}
    assert graph["save_bass"]["inputs"]["audio"] == ["separate", 2]
    assert stem_from_filename("j1-drums_00.wav", "j1") == "drums"


def test_stem_request_validation():
    with pytest.raises(ValueError, match="Choose a sound"):
        StemRequest(source_id="").validate(None)
    with pytest.raises(ValueError, match="minutes"):
        StemRequest(source_id="0123456789").validate(700.0)
    StemRequest(source_id="0123456789").validate(30.0)


@pytest.fixture
def env(tmp_path):
    library = Library(tmp_path / "lib")
    sources = SourceStore(tmp_path / "lib")
    out_dir, in_dir = tmp_path / "comfy" / "output", tmp_path / "comfy" / "input"
    client = FakeClient(out_dir, input_dir=in_dir)
    generator = Generator(FakeEngine(), client, library, out_dir, sources=sources, comfy_input_dir=in_dir)
    source = sources.import_bytes(tone_wav_bytes(tmp_path, seconds=1.0, sr=44100), "groove.wav")
    yield generator, client, library, source
    library.close()


def test_stems_end_to_end_skips_silent_stems_and_keeps_levels(env):
    generator, client, library, source = env
    records = generator.run(StemRequest(source_id=source.id), "sjob")
    assert sorted(r.params["stem"] for r in records) == ["bass", "drums", "other"]  # vocals were silent
    for record in records:
        assert record.mode == "stems" and record.name == f"groove ({record.params['stem']})"
        audio, _ = sf.read(str(library.path_of(record)))
        assert np.abs(audio).max() == pytest.approx(1.3, rel=1e-3)  # not normalized
    assert not list((client.output_dir / "samplegen").glob("*.wav"))
    assert client.staged_inputs and not client.staged_inputs[0].exists()


def test_stems_without_extension_explains_setup(tmp_path):
    library = Library(tmp_path / "lib")
    sources = SourceStore(tmp_path / "lib")
    out_dir = tmp_path / "out"
    generator = Generator(FakeEngine(), FakeClient(out_dir, node_info={}), library, out_dir, sources=sources)
    source = sources.import_bytes(tone_wav_bytes(tmp_path, 0.5), "x.wav")
    with pytest.raises(EngineError, match="install.bat"):
        generator.run(StemRequest(source_id=source.id), "nojob")
    library.close()

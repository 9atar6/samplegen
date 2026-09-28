"""Transform and Edit: request validation and the full pipeline against the fake engine."""

import numpy as np
import pytest
import soundfile as sf

from samplegen.generation import Generator
from samplegen.generation_request import GenerationRequest
from samplegen.library import Library
from samplegen.sources import SourceStore

from .fakes import FakeClient, FakeEngine, tone_wav_bytes


def transform(**kw):
    fields = dict(mode="transform", model="sa3-medium", prompt="make it metallic", source_id="0123456789",
                  strength=0.6, variations=2)
    fields.update(kw)
    return GenerationRequest(**fields)


def edit(**kw):
    fields = dict(mode="edit", model="sa3-sfx", prompt="glass crash", source_id="0123456789",
                  edit_op="inpaint", regions=((0.5, 1.0),), variations=2)
    fields.update(kw)
    return GenerationRequest(**fields)


@pytest.mark.parametrize("request_, seconds, message", [
    (transform(), None, "Choose a source"),
    (transform(source_id=None), 2.0, "Choose a source"),
    (transform(strength=0.0), 2.0, "Strength"),
    (transform(strength=1.5), 2.0, "Strength"),
    (transform(model="f1-samples"), 30.0, "handles up to 20"),
    (edit(model="f1-samples"), 2.0, "can't be used in edit"),
    (edit(regions=()), 2.0, "Select the part"),
    (edit(regions=((1.0, 0.5),)), 2.0, "outside the source"),
    (edit(regions=((0.5, 3.0),)), 2.0, "outside the source"),
    (edit(regions=((0.5, 0.55),)), 2.0, "at least 0.1"),
    (edit(edit_op="extend", extend_seconds=0.1), 2.0, "Extend by"),
    (edit(edit_op="extend", extend_seconds=119, model="sa3-sfx"), 10.0, "handles up to 120"),
    (edit(edit_op="twist"), 2.0, "Unknown edit"),
])
def test_source_mode_validation(request_, seconds, message):
    with pytest.raises(ValueError, match=message):
        request_.validate(seconds)


def test_valid_source_requests_pass():
    transform().validate(2.0)
    edit().validate(2.0)
    edit(edit_op="extend", extend_seconds=3.0).validate(2.0)
    assert edit(edit_op="extend", extend_seconds=3.0).target_seconds(2.0) == 5.0


@pytest.fixture
def env(tmp_path):
    library = Library(tmp_path / "lib")
    sources = SourceStore(tmp_path / "lib")
    out_dir, in_dir = tmp_path / "comfy" / "output", tmp_path / "comfy" / "input"
    client = FakeClient(out_dir, input_dir=in_dir)
    generator = Generator(FakeEngine(), client, library, out_dir, sources=sources, comfy_input_dir=in_dir)
    source = sources.import_bytes(tone_wav_bytes(tmp_path, seconds=2.0, sr=44100, channels=2), "hit.wav")
    yield generator, client, library, sources, source
    library.close()


def test_transform_end_to_end(env):
    generator, client, library, _, source = env
    records = generator.run(transform(source_id=source.id, strength=0.35), "tjob")
    graph = client.graphs[0]
    assert graph["sampler"]["inputs"]["denoise"] == 0.35
    assert graph["repeat"]["inputs"]["amount"] == 2
    assert graph["encode"]["inputs"]["audio"] == ["load", 0]
    assert len(records) == 2
    for record in records:
        info = sf.info(str(library.path_of(record)))
        assert info.frames == 88200  # same length as the source
        assert record.mode == "transform" and record.params["strength"] == 0.35
        assert record.name == "hit to make it metallic"
        assert library.path_of(record).parent.parent.name == "Inbox"
    # the staged engine input is cleaned up afterwards
    assert client.staged_inputs and not client.staged_inputs[0].exists()


def test_transform_loop_source_gets_wraparound_and_exact_length(env):
    generator, client, library, _, source = env
    records = generator.run(transform(source_id=source.id, source_is_loop=True, variations=1), "tloop")
    assert client.staged_frames == [88200 + int(0.25 * 44100)]  # source + its own start, for the wrap
    assert sf.info(str(library.path_of(records[0]))).frames == 88200
    assert records[0].params["loop"] is True


def test_edit_inpaint_keeps_the_rest_of_the_source(env):
    generator, client, library, sources, source = env
    records = generator.run(edit(source_id=source.id, regions=((0.5, 1.0),), normalize_db=None), "ejob")
    graph = client.graphs[0]
    assert graph["inpaint"]["inputs"]["regions"] == "0.5000-1.0000"
    assert graph["inpaint"]["inputs"]["source_seconds"] == pytest.approx(2.0)
    assert graph["sampler"]["inputs"]["denoise"] == 1.0
    original = sources.load(source.id)
    result, _ = sf.read(str(library.path_of(records[0])), always_2d=True)
    assert result.shape == original.shape
    np.testing.assert_allclose(result[: int(0.4 * 44100)], original[: int(0.4 * 44100)], atol=1e-6)
    np.testing.assert_allclose(result[int(1.1 * 44100):], original[int(1.1 * 44100):], atol=1e-6)
    assert not np.allclose(result[int(0.6 * 44100): int(0.9 * 44100)], original[int(0.6 * 44100): int(0.9 * 44100)])
    assert records[0].params["edit_op"] == "inpaint"


def test_edit_extend_appends_new_audio(env):
    generator, client, library, sources, source = env
    records = generator.run(edit(source_id=source.id, edit_op="extend", extend_seconds=1.5,
                                 normalize_db=None, fade_out_ms=0), "xjob")
    region = client.graphs[0]["inpaint"]["inputs"]["regions"]
    start, end = (float(v) for v in region.split("-"))
    assert start < 2.0 and end == pytest.approx(3.5)
    original = sources.load(source.id)
    result, _ = sf.read(str(library.path_of(records[0])), always_2d=True)
    assert len(result) == int(3.5 * 44100)
    np.testing.assert_allclose(result[: int(1.9 * 44100)], original[: int(1.9 * 44100)], atol=1e-6)
    assert np.abs(result[int(2.5 * 44100):]).max() > 0.1  # the extension has sound in it
    assert records[0].name.startswith("hit extended")


def test_missing_source_is_a_clear_error(env):
    generator, *_ = env
    with pytest.raises(ValueError, match="no longer exists"):
        generator.run(transform(source_id="abcdef0123"), "gone")

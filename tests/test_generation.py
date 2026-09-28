import numpy as np
import pytest
import soundfile as sf

from samplegen.audio import ExportFormat
from samplegen.comfy import EngineError
from samplegen.generation import GenerationRequest, Generator, sample_name
from samplegen.library import Library

from .fakes import FakeClient, FakeEngine


def sfx(**overrides):
    fields = dict(mode="sfx", model="sa3-sfx", prompt="door slam", duration=2.0, variations=2)
    fields.update(overrides)
    return GenerationRequest(**fields)


def loop(**overrides):
    fields = dict(mode="loop", model="f1-samples", prompt="Synth Bass, Acid", bpm=140, bars=8,
                  key="E", scale="minor", variations=2)
    fields.update(overrides)
    return GenerationRequest(**fields)


@pytest.mark.parametrize("request_, message", [
    (sfx(mode="nope"), "Unknown mode"),
    (sfx(model="nope"), "Unknown model"),
    (sfx(model="f1-samples"), "can't be used in sfx"),
    (sfx(prompt="   "), "Describe the sound"),
    (sfx(variations=0), "Variations"),
    (sfx(variations=9), "Variations"),
    (sfx(duration=0.1), "Duration"),
    (sfx(duration=500), "Duration"),
    (sfx(seed=-1), "Seed"),
    (sfx(normalize_db=3.0), "Normalize"),
    (sfx(fade_out_ms=-5), "Fades"),
    (sfx(model="sa3-medium", duration=60, variations=4), "too much audio"),
    (loop(bpm=125), "BPM"),
    (loop(bars=3), "bars"),
    (loop(key="H"), "key"),
])
def test_validation_messages(request_, message):
    with pytest.raises(ValueError, match=message):
        request_.validate()


def test_valid_requests_pass():
    sfx().validate()
    loop(prompt="").validate()  # loops may rely on key/tempo alone


def test_loop_prompt_and_name():
    request = loop()
    assert request.full_prompt() == "Synth Bass, Acid, 8 Bars, 140 BPM, E minor"
    assert sample_name(request) == "Synth Bass_Acid_140bpm_Emin_8bars"
    assert sample_name(loop(key="F#", scale="major")) == "Synth Bass_Acid_140bpm_Fsmaj_8bars"


@pytest.fixture
def setup(tmp_path):
    library = Library(tmp_path / "lib")
    client = FakeClient(tmp_path / "comfy_out")
    engine = FakeEngine()
    yield Generator(engine, client, library, tmp_path / "comfy_out"), client, library, engine
    library.close()


def test_generator_sfx_end_to_end(setup):
    generator, client, library, engine = setup
    records = generator.run(sfx(seed=99, export=ExportFormat(48000, "24")), "job1")

    assert engine.calls == 1
    assert len(records) == 2
    assert client.graphs[0]["sampler"]["inputs"]["seed"] == 99
    for record in records:
        path = library.path_of(record)
        info = sf.info(str(path))
        assert info.samplerate == 48000 and info.subtype == "PCM_24"
        audio, _ = sf.read(str(path))
        assert np.abs(audio).max() <= 10 ** (-1 / 20) + 1e-3  # normalized from a too-hot source
        assert record.status == "new" and record.seed == 99
    # engine scratch files are cleaned up after processing
    assert not list((client.output_dir / "samplegen").glob("*.wav"))


def test_generator_loop_is_sample_exact(setup):
    generator, client, library, _ = setup
    records = generator.run(loop(), "job2")
    for record in records:
        info = sf.info(str(library.path_of(record)))
        assert info.frames == 604800  # 8 bars @ 140 BPM @ 44.1 kHz
        assert record.params["bpm"] == 140 and record.params["key"] == "E minor"
    assert client.graphs[0]["pos"]["inputs"]["text"].endswith("8 Bars, 140 BPM, E minor")


def test_generator_picks_random_seed_when_none(setup):
    generator, client, _, _ = setup
    generator.run(sfx(seed=None, variations=1), "job3")
    assert isinstance(client.graphs[0]["sampler"]["inputs"]["seed"], int)


def test_generator_propagates_engine_errors(tmp_path):
    library = Library(tmp_path / "lib")
    generator = Generator(FakeEngine(), FakeClient(tmp_path / "out", fail="CUDA out of memory"), library, tmp_path / "out")
    with pytest.raises(EngineError, match="out of memory"):
        generator.run(sfx(), "job4")
    assert library.list() == []
    library.close()

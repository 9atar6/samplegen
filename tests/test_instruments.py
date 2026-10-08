import xml.etree.ElementTree as ET

import numpy as np
import pytest
import soundfile as sf

from samplegen.audio import ExportFormat
from samplegen.generation import Generator
from samplegen.instruments import (
    MAX_NOTE_LEAD_S, InstrumentRequest, chunk_notes, chunk_seconds, decent_sampler_preset, key_ranges,
    keybed_prompt, midi_to_name, name_to_midi, normalize_together, note_onset, sfz_file, slice_chunk,
)
from samplegen.library import Library

from .fakes import FakeClient, FakeEngine

SR = 44100


def test_note_names_round_trip():
    assert midi_to_name(60) == "C4" and midi_to_name(61) == "C#4" and midi_to_name(24) == "C1"
    assert name_to_midi("C4") == 60 and name_to_midi("f#2") == 42 and name_to_midi("Bb3") == 58
    for midi in range(24, 97):
        assert name_to_midi(midi_to_name(midi)) == midi
    with pytest.raises(ValueError):
        name_to_midi("H2")


def test_chunks_of_six_and_timing():
    chunks = chunk_notes(48, 71)  # C3..B4 = 24 notes
    assert [len(c) for c in chunks] == [6, 6, 6, 6]
    assert chunk_notes(60, 62) == [[60, 61, 62]]
    assert chunk_seconds(6) == pytest.approx(19.25)
    assert chunk_seconds(3) == pytest.approx(9.5)


def test_keybed_prompt_grammar():
    prompt = keybed_prompt("Grand Piano, Warm, Gritty", "Dry", [60, 61, 62])
    assert prompt == ("Keybed, Sequence, Timbre Profile, Grand Piano, Warm, Gritty, Dry, "
                      "Chromatic Chunk, Note Sequence, C4, C#4, D4")


def test_slice_chunk_uses_the_fixed_grid():
    raw = np.zeros((int(chunk_seconds(3) * SR) + 1000, 2))
    for i, level in enumerate((0.1, 0.2, 0.3)):  # each note slot at a distinct level
        start = int(i * 3.25 * SR)
        raw[start:start + 3 * SR] = level
    notes = slice_chunk(raw, [60, 61, 62], SR)
    assert set(notes) == {60, 61, 62}
    for midi, level in zip((60, 61, 62), (0.1, 0.2, 0.3)):
        audio = notes[midi]
        assert len(audio) == 3 * SR
        assert audio[SR].max() == pytest.approx(level)  # middle of the note
        assert audio[-1].max() == pytest.approx(0.0, abs=1e-9)  # faded out


def test_slice_chunk_starts_each_note_at_its_attack():
    # The model played note 61 half a second late in its slot, after some faint noise.
    raw = np.zeros((int(chunk_seconds(2) * SR) + 1000, 2))
    raw[0:3 * SR] = 0.5
    late = int(3.25 * SR)
    raw[late:late + int(0.5 * SR)] = 0.01  # -34 dB lead-in: not the note yet
    raw[late + int(0.5 * SR):late + 3 * SR] = 0.5
    notes = slice_chunk(raw, [60, 61], SR)
    assert len(notes[60]) == 3 * SR  # on time: untouched
    lead = 3 * SR - len(notes[61])
    assert 0.48 * SR < lead <= 0.5 * SR  # trimmed to a few ms before the attack
    assert notes[61][int(0.02 * SR)].max() == pytest.approx(0.5)  # the note is there right away


def test_note_onset_is_capped_and_ignores_silence():
    assert note_onset(np.zeros((SR, 2)), SR) == 0
    late = np.zeros((3 * SR, 2))
    late[int(2.5 * SR):] = 0.5
    assert note_onset(late, SR) == int(MAX_NOTE_LEAD_S * SR)


def test_normalize_together_keeps_relative_levels():
    notes = {60: np.full((10, 2), 0.2), 61: np.full((10, 2), 0.4)}
    out = normalize_together(notes, target_db=0.0)
    assert out[61].max() == pytest.approx(1.0) and out[60].max() == pytest.approx(0.5)


def test_key_ranges_stretch_the_edges():
    ranges = key_ranges([48, 49, 50])
    assert ranges == {48: (42, 48), 49: (49, 49), 50: (50, 56)}


def test_decent_sampler_preset_is_valid_xml():
    xml = decent_sampler_preset('Grit & "Grand"', {48: "Samples/g_C3.wav", 49: "Samples/g_Cs3.wav"})
    root = ET.fromstring(xml)
    samples = root.findall("./groups/group/sample")
    assert [(s.get("path"), s.get("rootNote"), s.get("loNote"), s.get("hiNote")) for s in samples] == [
        ("Samples/g_C3.wav", "48", "42", "48"), ("Samples/g_Cs3.wav", "49", "49", "55")]
    assert root.find("./ui/tab/label").get("text") == 'Grit & "Grand"'


def test_sfz_regions():
    sfz = sfz_file("Grit", {48: "Samples/g_C3.wav", 49: "Samples/g_Cs3.wav"})
    assert "<region> sample=Samples/g_C3.wav pitch_keycenter=48 lokey=42 hikey=48" in sfz
    assert "<region> sample=Samples/g_Cs3.wav pitch_keycenter=49 lokey=49 hikey=55" in sfz


@pytest.mark.parametrize("request_, message", [
    (InstrumentRequest(prompt=""), "Describe the instrument"),
    (InstrumentRequest(prompt="Piano", space="Damp"), "Dry or Wet"),
    (InstrumentRequest(prompt="Piano", low_note="C5", high_note="C4"), "low to high"),
    (InstrumentRequest(prompt="Piano", low_note="C0", high_note="C4"), "within C1"),
    (InstrumentRequest(prompt="Piano", low_note="C1", high_note="C7"), "Up to 61"),
    (InstrumentRequest(prompt="Piano", low_note="X9"), "note name"),
])
def test_instrument_validation(request_, message):
    with pytest.raises(ValueError, match=message):
        request_.validate()


def test_instrument_end_to_end(tmp_path):
    library = Library(tmp_path / "lib")
    client = FakeClient(tmp_path / "out")
    generator = Generator(FakeEngine(), client, library, tmp_path / "out")
    request = InstrumentRequest(prompt="Grand Piano, Warm", name="Warm Grand", low_note="C4", high_note="D#5",
                                seed=77, export=ExportFormat(44100, "24"))
    (record,) = generator.run(request, "ijob")

    assert len(client.graphs) == 3  # 16 notes -> 6 + 6 + 4
    assert {g["sampler"]["inputs"]["seed"] for g in client.graphs} == {77}  # one seed for the keyboard
    assert client.graphs[2]["pos"]["inputs"]["text"].endswith("C5, C#5, D5, D#5")
    assert record.mode == "instrument" and record.params["notes"] == 16

    folder = library.root / record.params["instrument_folder"]
    assert (folder / "warm-grand.dspreset").exists() and (folder / "warm-grand.sfz").exists()
    wavs = sorted((folder / "Samples").glob("*.wav"))
    assert len(wavs) == 16
    info = sf.info(str(wavs[0]))
    assert (info.frames, info.subtype) == (3 * SR, "PCM_24")
    root = ET.fromstring((folder / "warm-grand.dspreset").read_text(encoding="utf-8"))
    assert len(root.findall("./groups/group/sample")) == 16
    assert library.path_of(record).exists()  # the scale-run preview
    library.close()

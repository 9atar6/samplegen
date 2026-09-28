"""Style registry and the training manager, driven by a fake trainer (no GPU, no torch)."""

import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from samplegen.jobs import JobManager
from samplegen.styles import Style, StyleNotFound, StyleStore, style_slug
from samplegen.training import (
    Clip, TrainingManager, caption_from_filename, final_checkpoint, latest_step, scan_folder, tail,
)

FAKE_TRAINER = textwrap.dedent("""
    import argparse, os, pathlib, sys
    p = argparse.ArgumentParser()
    for flag in ("--model", "--data_dir", "--save_dir", "--steps", "--rank", "--adapter_type", "--exclude",
                 "--duration", "--checkpoint_every", "--demo_every", "--num_workers", "--logger", "--name"):
        p.add_argument(flag)
    a = p.parse_args()
    if os.environ.get("FAKE_TRAINER_FAIL"):
        print("CUDA out of memory"); sys.exit(1)
    data = pathlib.Path(a.data_dir)
    assert len(list(data.glob("*.txt"))) == len(list(data.glob("*.wav"))) >= 5
    out = pathlib.Path(a.save_dir); logs = out / "lightning_logs" / "version_0"; logs.mkdir(parents=True)
    with open(logs / "metrics.csv", "w") as f:
        f.write("step,train/loss\\n")
        for s in range(0, int(a.steps) + 1, 50): f.write(f"{s},0.1\\n")
    for s in (100, int(a.steps)): (out / f"epoch=0-step={s}.ckpt").write_text("ckpt")
    (out / "args.txt").write_text(" ".join(sys.argv[1:]))
    print("training finished\\r100%")
""")

FAKE_CONVERTER = textwrap.dedent("""
    import argparse, pathlib
    p = argparse.ArgumentParser(); p.add_argument("--ckpt"); p.add_argument("--base"); p.add_argument("--out")
    a = p.parse_args()
    assert pathlib.Path(a.ckpt).name == "epoch=0-step=200.ckpt", a.ckpt
    pathlib.Path(a.out).write_bytes(b"lora")
""")


def wav(path: Path, seconds=0.5, sr=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), 0.3 * np.sin(np.arange(int(seconds * sr)) / 10), sr)
    return path


# ---------- styles ----------

def test_style_slug():
    assert style_slug("  My Gritty Foley! ") == "my-gritty-foley"
    with pytest.raises(ValueError):
        style_slug("!!!")


def test_style_store_lists_only_complete_styles(tmp_path):
    store = StyleStore(tmp_path / "loras")
    style = store.register(Style(slug="grit", name="Grit", base="sfx", clips=20, steps=1000, created="2026-09-26T10:00:00"))
    assert store.list() == []  # no LoRA file yet
    store.lora_path("grit").write_bytes(b"x")
    assert store.list() == [style] and store.get("grit") == style
    assert style.lora_file == "samplegen_grit.safetensors" and style.model_key == "sa3-sfx-base"
    assert store.exists("grit") and not store.exists("nope") and not store.exists("../x")
    with pytest.raises(StyleNotFound):
        store.get("../../etc")
    with pytest.raises(ValueError):
        store.register(Style(slug="bad", name="b", base="drums", clips=1, steps=1, created=""))


# ---------- helpers ----------

def test_caption_from_filename():
    assert caption_from_filename(Path("Metal_Door-Slam_03.wav")) == "metal door slam"


def test_scan_folder_uses_sidecar_captions(tmp_path):
    wav(tmp_path / "set" / "a_hit_01.wav")
    wav(tmp_path / "set" / "sub" / "b.wav")
    (tmp_path / "set" / "sub" / "b.txt").write_text("  deep   whoosh \n", encoding="utf-8")
    (tmp_path / "set" / "notes.pdf").write_bytes(b"%PDF")
    clips = scan_folder(tmp_path / "set")
    assert [(c.path.name, c.caption) for c in clips] == [("a_hit_01.wav", "a hit"), ("b.wav", "deep whoosh")]
    with pytest.raises(ValueError):
        scan_folder(tmp_path / "missing")


def test_progress_helpers(tmp_path):
    logs = tmp_path / "lightning_logs" / "version_0"
    logs.mkdir(parents=True)
    (logs / "metrics.csv").write_text("step,loss\n10,1\n120,0.5\n,\n", encoding="utf-8")
    assert latest_step(tmp_path) == 120
    for step in (500, 1500, 1000):
        (tmp_path / f"epoch=0-step={step}.ckpt").write_text("x")
    assert final_checkpoint(tmp_path).name == "epoch=0-step=1500.ckpt"
    (tmp_path / "t.log").write_bytes(b"line1\r\n 10%\r 50%\r 90%\r\n\r\nlast\r\n")
    assert tail(tmp_path / "t.log") == ("line1", "90%", "last")


# ---------- manager ----------

@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "project"
    trainer = project / "trainer" / "stable-audio-3"
    (trainer / "scripts").mkdir(parents=True)
    (trainer / "scripts" / "train_lora.py").write_text(FAKE_TRAINER, encoding="utf-8")
    (project / "trainer" / "convert_lora.py").write_text(FAKE_CONVERTER, encoding="utf-8")
    engine = tmp_path / "engine"
    (engine / "ComfyUI" / "models" / "checkpoints").mkdir(parents=True)
    (engine / "ComfyUI" / "models" / "checkpoints" / "stable_audio_3_small_sfx_base.safetensors").write_bytes(b"x")
    styles = StyleStore(engine / "ComfyUI" / "models" / "loras")
    freed = []
    manager = TrainingManager(project, tmp_path / "lib", engine, styles, free_engine=lambda: freed.append(1))
    manager.python = Path(sys.executable)  # stand-in for the trainer's own Python
    clips = [Clip(wav(tmp_path / "in" / f"s{i}.wav"), f"sound {i}") for i in range(6)]
    return manager, styles, clips, freed


def wait_until_finished(manager, timeout=60):
    deadline = time.time() + timeout
    while manager.status()["active"] and time.time() < deadline:
        time.sleep(0.1)
    return manager.status()


def test_train_end_to_end_registers_a_style(setup):
    manager, styles, clips, freed = setup
    run = manager.start("Grit Foley", "sfx", clips, steps=200, description="gritty foley")
    assert run.state == "preparing" and manager.busy_reason()
    status = wait_until_finished(manager)
    assert status["state"] == "done", status
    assert status["step"] == 200 and freed == [1]
    style = styles.get("grit-foley")
    assert (style.name, style.base, style.clips, style.steps) == ("Grit Foley", "sfx", 6, 200)
    assert styles.lora_path("grit-foley").read_bytes() == b"lora"
    work = next((manager.work_root).iterdir())
    args = (work / "out" / "args.txt").read_text()
    for expected in ("--adapter_type lora", "--exclude conditioners", "--num_workers 0", "--demo_every 201",
                     "--model small-sfx-base"):
        assert expected in args
    captions = sorted(p.read_text(encoding="utf-8") for p in (work / "data").glob("*.txt"))
    assert captions[0] == "gritty foley, sound 0"
    assert sf.info(str(next((work / "data").glob("*.wav")))).samplerate == 44100
    assert manager.busy_reason() is None


def test_training_failure_is_reported_with_log(setup, monkeypatch):
    manager, styles, clips, _ = setup
    monkeypatch.setenv("FAKE_TRAINER_FAIL", "1")
    manager.start("Broken", "sfx", clips, steps=200)
    status = wait_until_finished(manager)
    assert status["state"] == "error" and "ran out of memory" in status["error"]  # explained, not raw
    assert "CUDA out of memory" in status["log_tail"]
    assert styles.list() == []


@pytest.mark.parametrize("kwargs, message", [
    (dict(base="drums"), "SFX or Music"),
    (dict(steps=50), "Steps"),
    (dict(rank=7), "Rank"),
    (dict(clips_count=3), "between 5"),
    (dict(base="music"), "base model"),  # music base checkpoint wasn't downloaded in this setup
])
def test_start_validation(setup, kwargs, message):
    manager, _, clips, _ = setup
    count = kwargs.pop("clips_count", len(clips))
    with pytest.raises(ValueError, match=message):
        manager.start("Name", kwargs.pop("base", "sfx"), clips[:count], **kwargs)


def test_not_installed_and_duplicate_names(setup, tmp_path):
    manager, styles, clips, _ = setup
    styles.register(Style(slug="taken", name="Taken", base="sfx", clips=5, steps=100, created="x"))
    with pytest.raises(ValueError, match="already exists"):
        manager.start("Taken", "sfx", clips)
    bare = TrainingManager(tmp_path / "nothing", tmp_path / "lib2", tmp_path / "e", styles)
    assert bare.status()["installed"] is False
    with pytest.raises(ValueError, match="install-training.bat"):
        bare.start("X", "sfx", clips)


def test_generating_with_a_style_uses_its_base_model_and_lora(tmp_path):
    from samplegen.catalog import MODELS
    from samplegen.generation import Generator
    from samplegen.generation_request import GenerationRequest
    from samplegen.library import Library

    from .fakes import FakeClient, FakeEngine

    styles = StyleStore(tmp_path / "loras")
    styles.register(Style(slug="grit", name="Grit", base="music", clips=20, steps=1000, created="x"))
    styles.lora_path("grit").write_bytes(b"x")
    library = Library(tmp_path / "lib")
    client = FakeClient(tmp_path / "out")
    generator = Generator(FakeEngine(), client, library, tmp_path / "out", styles=styles)
    request = GenerationRequest(mode="free", model="sa3-medium", prompt="drone", duration=2.0, variations=1,
                                style="grit", style_strength=0.7)
    (record,) = generator.run(request, "sjob")
    graph = client.graphs[0]
    assert graph["ckpt"]["inputs"]["ckpt_name"] == MODELS["sa3-music-base"].checkpoint
    assert graph["lora"]["inputs"] == {"model": ["ckpt", 0], "lora_name": "samplegen_grit.safetensors",
                                       "strength_model": 0.7}
    assert graph["sampler"]["inputs"]["model"] == ["lora", 0]
    assert graph["sampler"]["inputs"]["steps"] == 50 and graph["sampler"]["inputs"]["cfg"] == 7.0
    assert record.params["style"] == "grit" and record.params["style_strength"] == 0.7

    with pytest.raises(ValueError, match="no longer exists"):
        generator.check(GenerationRequest(mode="sfx", model="sa3-sfx", prompt="x", style="gone"))
    with pytest.raises(ValueError, match="SFX and Free"):
        GenerationRequest(mode="loop", model="f1-samples", prompt="x", style="grit").validate()
    with pytest.raises(ValueError, match="strength"):
        GenerationRequest(mode="sfx", model="sa3-sfx", prompt="x", style="grit", style_strength=3).validate()
    library.close()


@pytest.mark.parametrize("log, expected", [
    ("OSError: You are trying to access a gated repo.\nMake sure to have access to it at "
     "https://huggingface.co/stabilityai/stable-audio-3-small-music.\n403 Client Error.",
     "https://huggingface.co/stabilityai/stable-audio-3-small-music ,"),
    ("No module named 'flash_attn'\nTraceback (most recent call last):\n  ...\n"
     "ModuleNotFoundError: No module named 'matplotlib'", "'matplotlib'"),
    ("RuntimeError: CUDA out of memory. Tried to allocate", "ran out of memory"),
    ("something else entirely", None),
])
def test_explain_failure(tmp_path, log, expected):
    from samplegen.training import explain_failure
    path = tmp_path / "train.log"
    path.write_text(log, encoding="utf-8")
    message = explain_failure(path)
    if expected is None:
        assert message is None
    else:
        assert expected in message
        assert "flash_attn" not in message  # the harmless warning before the traceback is ignored


def test_job_manager_refuses_work_while_training():
    class Gen:
        def check(self, request):
            raise AssertionError("should not be reached")
    manager = JobManager(Gen(), busy=lambda: "A style is training right now")
    with pytest.raises(ValueError, match="training"):
        manager.submit(object())

"""Test doubles for the engine side: no ComfyUI, no GPU."""

from pathlib import Path

import numpy as np
import soundfile as sf

from samplegen.comfy import EngineError, OutputFile

SR = 44100

# Shaped like the real AudioSeparation extension's /object_info (typo included).
DEMUCS_INFO = {
    "input": {"required": {
        "input_sound": ["AUDIO"],
        "model": [["htdemucs.safetensors", "htdemucs_ft.safetensors", "htdemucs_6s.safetensors"], {}],
        "shifts": ["INT", {"default": 1, "min": 0, "max": 20}],
        "overlap": ["FLOAT", {"default": 0.25}],
        "custom_segment": ["BOOLEAN", {"default": False}],
        "segment": ["INT", {"default": 44}],
        "taget_device": ["COMBO", {"options": ["AUTO", "CPU", "CUDA:0"], "default": "AUTO"}],
    }},
    "output": ["AUDIO", "AUDIO", "AUDIO", "AUDIO"],
    "output_name": ["Vocals", "Drums", "Bass", "Other"],
}


class FakeEngine:
    def __init__(self):
        self.state = "ready"
        self.error = None
        self.calls = 0
        self.engine_dir = Path("no-engine-here")

    def ensure_running(self):
        self.calls += 1


class FakeClient:
    """Pretends to be ComfyUI: writes a tone for every save node and batch item.

    Text graphs: length from EmptyLatentAudio (+ a little, like the real engine).
    Source graphs: same length as the staged input file, which must exist.
    Stem graphs: the "vocals" stem comes back silent, like it would for an SFX.
    """

    def __init__(self, output_dir: Path, seconds_extra: float = 0.3, fail: str | None = None,
                 input_dir: Path | None = None, node_info: dict | None = None):
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir) if input_dir else self.output_dir.parent / "input"
        self.seconds_extra = seconds_extra
        self.fail = fail
        self.node_info = {"AudioSeparateDemucs": DEMUCS_INFO} if node_info is None else node_info
        self.graphs = []
        self.prompts = {}
        self.staged_inputs = []
        self.staged_frames = []
        self.interrupted = False
        self.queue_cleared = False

    def submit(self, graph):
        self.graphs.append(graph)
        prompt_id = f"prompt-{len(self.graphs)}"
        self.prompts[prompt_id] = graph
        return prompt_id

    def object_info(self, node_class):
        return self.node_info.get(node_class)

    def _frames_and_batch(self, graph):
        if "load" in graph:
            staged = self.input_dir / "samplegen" / graph["load"]["inputs"]["filename"]
            assert staged.exists(), f"source was not staged: {staged}"
            frames = sf.info(str(staged)).frames
            self.staged_inputs.append(staged)
            self.staged_frames.append(frames)
            batch = graph["repeat"]["inputs"]["amount"] if "repeat" in graph else 1
            return frames, batch
        seconds = graph["latent"]["inputs"]["seconds"] + self.seconds_extra
        return int(seconds * SR), graph["latent"]["inputs"]["batch_size"]

    def wait(self, prompt_id, timeout):
        if self.fail:
            raise EngineError(self.fail)
        graph = self.prompts[prompt_id]
        frames, batch = self._frames_and_batch(graph)
        folder = self.output_dir / "samplegen"
        folder.mkdir(parents=True, exist_ok=True)
        t = np.arange(frames) / SR
        outputs = []
        saves = [n for n in graph.values() if n["class_type"] == "SamplegenSaveWav"]
        for save in saves:
            job_id = save["inputs"]["job_id"]
            silent = job_id.endswith("-vocals")
            for i in range(batch):
                tone = 0.0 * t if silent else 1.3 * np.sin(2 * np.pi * (220 + 50 * i) * t)  # too hot on purpose
                name = f"{job_id}_{i:02d}.wav"
                sf.write(str(folder / name), np.stack([tone, tone], axis=1), SR, subtype="FLOAT")
                outputs.append(OutputFile(name, "samplegen"))
        return outputs

    def interrupt(self):
        self.interrupted = True

    def clear_queue(self):
        self.queue_cleared = True


def tone_wav_bytes(tmp_path: Path, seconds: float = 2.0, sr: int = 48000, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    data = 0.4 * np.sin(2 * np.pi * 330 * t)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    path = tmp_path / "upload.wav"
    sf.write(str(path), data, sr, subtype="PCM_24")
    return path.read_bytes()

"""Zero-shot sound tagging with CLAP (LAION larger_clap_general, Apache-2.0).

Runs in the trainer's Python (torch + transformers). For each audio file, scores it
against a sound-design vocabulary and prints JSON: the best category plus the
best descriptor in each character group, with similarity margins.

    python clap_describe.py <folder or file> [<more> ...]  > out.json
"""

import faulthandler
import json
import sys
from pathlib import Path

faulthandler.enable()  # a native crash (DLL) prints its location to stderr instead of vanishing

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/larger_clap_general"
CLAP_RATE = 48000
MAX_SECONDS = 10.0
AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}

CATEGORIES = [
    # drums & percussion
    "kick drum", "808 bass drum", "snare drum", "clap", "rimshot", "closed hi-hat", "open hi-hat", "cymbal crash",
    "ride cymbal", "tom drum", "shaker", "tambourine", "cowbell", "conga", "percussion hit", "drum loop",
    "beatbox",
    # musical
    "bass guitar", "synth bass", "sub bass", "synth lead", "synth pad", "synth pluck", "arpeggio", "piano", "electric piano",
    "organ", "acoustic guitar", "electric guitar", "strings", "brass", "flute", "bell", "mallet percussion",
    "choir", "vocal chop", "singing voice", "chord stab",
    # sound design
    "impact hit", "cinematic boom", "braam", "riser", "downlifter", "whoosh", "swoosh", "glitch", "noise sweep",
    "drone", "ambient texture", "atmosphere", "reverse sound",
    # foley & world
    "footsteps", "door slam", "glass breaking", "metal clang", "wood knock", "paper rustle", "cloth movement",
    "water splash", "rain", "wind", "thunder", "fire crackle", "explosion", "gunshot", "sword swing",
    "engine", "car pass", "crowd", "birds", "animal growl", "monster roar", "creature vocalization",
    # UI & synthetic
    "user interface click", "notification beep", "button press", "laser zap", "electric buzz", "static noise",
]

GROUPS = {
    "tone": ["bright", "dark", "warm", "cold", "muffled", "airy"],
    "texture": ["clean", "distorted", "saturated", "gritty", "lo-fi", "crunchy", "smooth", "noisy"],
    "origin": ["analog drum machine", "digital drum machine", "acoustic recording", "synthesized", "organic", "processed"],
    "space": ["dry", "roomy", "reverberant", "echoing", "close-miked", "distant"],
    "energy": ["punchy", "soft", "aggressive", "heavy", "subtle", "snappy", "boomy"],
    "era": ["vintage", "modern", "retro 80s", "futuristic"],
}

TEMPLATE = "the sound of a {}"


def as_tensor(features) -> torch.Tensor:
    """transformers returns a tensor (4.x) or a model-output object (5.x) from get_*_features."""
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("text_embeds", "audio_embeds", "pooler_output"):
        value = getattr(features, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    return features[0]


def load_audio(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != CLAP_RATE:
        mono = torchaudio.functional.resample(mono, sr, CLAP_RATE)
    mono = mono[: int(MAX_SECONDS * CLAP_RATE)]
    peak = mono.abs().max()
    if peak > 0:
        mono = mono / peak * 0.9  # level shouldn't change the label
    return mono.numpy()


def files_from(args: list[str]) -> list[Path]:
    out = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
        elif p.is_file():
            out.append(p)
    return out


class Tagger:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        step("loading model")
        self.model = ClapModel.from_pretrained(MODEL_ID).to(self.device).eval()
        step("loading processor")
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)
        self.vocab = {"category": CATEGORIES, **GROUPS}
        step("embedding vocabulary")
        self.text = {name: self._embed_text(words) for name, words in self.vocab.items()}

    @torch.no_grad()
    def _embed_text(self, words: list[str]) -> torch.Tensor:
        inputs = self.processor(text=[TEMPLATE.format(w) for w in words], return_tensors="pt", padding=True).to(self.device)
        emb = as_tensor(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(emb, dim=-1)

    @torch.no_grad()
    def describe(self, path: Path) -> dict:
        audio = load_audio(path)
        inputs = self.processor(audio=[audio], sampling_rate=CLAP_RATE, return_tensors="pt").to(self.device)
        emb = torch.nn.functional.normalize(as_tensor(self.model.get_audio_features(**inputs)), dim=-1)[0]
        result = {"file": path.name, "path": str(path)}
        for name, words in self.vocab.items():
            sims = (self.text[name] @ emb).cpu().numpy()
            order = np.argsort(-sims)
            top = [(words[i], round(float(sims[i]), 3)) for i in order[:3]]
            result[name] = top
        return result


def step(message: str) -> None:
    print(f"[clap] {message}", file=sys.stderr, flush=True)


def parse_args(argv: list[str]) -> tuple[list[Path], bool]:
    """Positional folders/files, `--files list.json` (a JSON list of paths), `--cpu`."""
    positional, listed, cpu = [], [], False
    args = iter(argv)
    for arg in args:
        if arg == "--cpu":
            cpu = True
        elif arg == "--files":
            listed += json.loads(Path(next(args)).read_text(encoding="utf-8"))
        else:
            positional.append(arg)
    return files_from(positional + listed), cpu


def main() -> int:
    paths, cpu = parse_args(sys.argv[1:])
    step(f"{len(paths)} files")
    if not paths:
        print(json.dumps({"error": "no audio files"}))
        return 1
    tagger = Tagger("cpu" if cpu else None)
    step(f"model ready on {tagger.device}")
    results = []
    for path in paths:
        step(f"tagging {path.name}")
        try:
            results.append(tagger.describe(path))
        except Exception as exc:  # keep going; report per file
            step(f"  failed: {exc!r}")
            results.append({"file": path.name, "path": str(path), "error": str(exc)})
    print(json.dumps(results, indent=1, ensure_ascii=False), flush=True)
    step("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

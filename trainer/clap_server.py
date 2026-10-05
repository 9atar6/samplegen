"""CLAP sound fingerprints for "search by sound", as a long-running worker.

Loads the model once, then answers one JSON request per stdin line with one JSON line:
    {"cmd": "audio", "paths": [...]}  ->  {"ok": true, "vectors": [[...512 floats] | null, ...]}
    {"cmd": "text", "text": "metal scrape"}  ->  {"ok": true, "vector": [...]}
Vectors are L2-normalized, so a dot product is the similarity. Runs on the CPU on
purpose: the GPU belongs to generation. Uses the trainer's Python, or search/.venv.
"""

import faulthandler
import json
import sys
from pathlib import Path

faulthandler.enable()

import torch
from transformers import ClapModel, ClapProcessor

from clap_describe import CLAP_RATE, MODEL_ID, as_tensor, load_audio

torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))  # leave the rest of the CPU to the app


def reply(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")  # ASCII-only JSON: safe whatever the console code page
    sys.stdout.flush()


def main() -> int:
    model = ClapModel.from_pretrained(MODEL_ID).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    reply({"ready": True, "device": "cpu"})

    def rows(tensor):
        return [[round(float(v), 5) for v in row] for row in torch.nn.functional.normalize(tensor, dim=-1)]

    for line in sys.stdin:
        try:
            request = json.loads(line)
            with torch.no_grad():
                if request.get("cmd") == "text":
                    inputs = processor(text=[f"the sound of {request['text']}"], return_tensors="pt", padding=True)
                    reply({"ok": True, "vector": rows(as_tensor(model.get_text_features(**inputs)))[0]})
                elif request.get("cmd") == "audio":
                    vectors = []
                    for path in request.get("paths", []):
                        try:
                            audio = load_audio(Path(path))
                            inputs = processor(audio=[audio], sampling_rate=CLAP_RATE, return_tensors="pt")
                            vectors.append(rows(as_tensor(model.get_audio_features(**inputs)))[0])
                        except Exception as exc:  # unreadable file: no fingerprint, keep going
                            print(f"[clap] {path}: {exc!r}", file=sys.stderr, flush=True)
                            vectors.append(None)
                    reply({"ok": True, "vectors": vectors})
                else:
                    reply({"ok": False, "error": "unknown command"})
        except Exception as exc:  # a bad request must not kill the worker
            reply({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

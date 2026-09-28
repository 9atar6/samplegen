"""Split a sound into stems with the AudioSeparation extension's Demucs node.

The extension's inputs (and even their spelling) vary between versions, so the
graph is built from what the running engine reports in /object_info rather
than from hard-coded names.
"""

from dataclasses import dataclass, field

from .audio import ExportFormat
from .comfy import EngineError

DEMUCS_NODE = "AudioSeparateDemucs"
# Best-first. htdemucs_ft is the fine-tuned 4-stem model; 6-source models add guitar/piano.
PREFERRED_MODELS = ("htdemucs_ft", "htdemucs", "hdemucs_mmi", "mdx_extra")
MAX_STEM_SOURCE_SECONDS = 600.0
SILENT_PEAK = 10 ** (-60 / 20)
NOT_INSTALLED = ("Stem separation isn't installed yet. Close samplegen, double-click install.bat, "
                 "then start samplegen again.")


@dataclass(frozen=True)
class StemRequest:
    source_id: str
    export: ExportFormat = field(default_factory=ExportFormat)
    mode: str = "stems"
    model: str = "demucs"
    prompt: str = ""
    variations: int = 1

    uses_source = True

    def full_prompt(self) -> str:
        return "Split into stems"

    def validate(self, source_seconds: float | None) -> None:
        if not self.source_id or source_seconds is None:
            raise ValueError("Choose a sound to split first.")
        if source_seconds > MAX_STEM_SOURCE_SECONDS:
            raise ValueError(f"Stem splitting is limited to {MAX_STEM_SOURCE_SECONDS / 60:.0f} minutes of audio.")


def _combo_options(spec: list) -> list | None:
    """Both ComfyUI combo encodings: [[a, b], {...}] and ["COMBO", {"options": [a, b]}]."""
    kind = spec[0]
    if isinstance(kind, list):
        return kind
    if kind == "COMBO":
        return list((spec[1] if len(spec) > 1 else {}).get("options", []))
    return None


def pick_model(options: list[str]) -> str:
    lowered = [(o, str(o).lower()) for o in options]
    for preferred in PREFERRED_MODELS:
        for original, low in lowered:
            if preferred in low and "6s" not in low:
                return original
    if not options:
        raise EngineError("The stem separator has no models available.")
    return options[0]


def _pick_device(options: list[str], default) -> str:
    for option in options:
        if any(word in str(option).lower() for word in ("cuda", "gpu")):
            return option
    return default if default in options else options[0]


def separator_inputs(node_info: dict) -> tuple[str, dict, str]:
    """-> (name of the AUDIO input, other inputs filled in, chosen model)."""
    required = node_info.get("input", {}).get("required", {})
    audio_input, inputs, model = None, {}, ""
    for name, spec in required.items():
        spec = list(spec) if isinstance(spec, (list, tuple)) else [spec]
        options = _combo_options(spec)
        extra = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if spec[0] == "AUDIO":
            audio_input = name
        elif options is not None:
            if "model" in name.lower():
                model = inputs[name] = pick_model(options)
            elif "device" in name.lower():
                inputs[name] = _pick_device(options, extra.get("default"))
            else:
                inputs[name] = extra.get("default", options[0] if options else None)
        elif "default" in extra:
            inputs[name] = extra["default"]
        else:
            raise EngineError(f"Don't know how to set the separator's '{name}' input.")
    if audio_input is None:
        raise EngineError("The stem separator node has no audio input.")
    return audio_input, inputs, model


def stem_names(node_info: dict) -> list[str]:
    names = node_info.get("output_name") or node_info.get("output") or []
    return [str(n).lower().replace(" ", "_") for n in names]


def build_stems_graph(node_info: dict, source_filename: str, job_id: str) -> tuple[dict, str]:
    audio_input, inputs, model = separator_inputs(node_info)
    graph = {
        "load": {"class_type": "SamplegenLoadWav", "inputs": {"filename": source_filename}},
        "separate": {"class_type": DEMUCS_NODE, "inputs": {audio_input: ["load", 0], **inputs}},
    }
    for index, stem in enumerate(stem_names(node_info)):
        graph[f"save_{stem}"] = {"class_type": "SamplegenSaveWav", "inputs": {
            "audio": ["separate", index], "job_id": f"{job_id}-{stem}",
        }}
    return graph, model


def stem_from_filename(filename: str, job_id: str) -> str:
    """'<job>-drums_00.wav' -> 'drums'"""
    return filename[len(job_id) + 1:].rsplit("_", 1)[0]

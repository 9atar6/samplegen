"""What the install left out: model files and extensions the app needs but can't find.

Shown in the UI so a half-finished install says what's missing and how to fix it,
instead of failing later with an engine error.
"""

from pathlib import Path

from .catalog import MODELS

FIX = "Run install.bat again: it only downloads what's missing."


def missing_parts(engine_dir: Path, midi_python: Path | None = None) -> list[str]:
    """Human-readable names of missing pieces (empty list: everything is there).

    Base models are left out: they're only needed for trained styles, which
    tools/install-training.bat sets up and checks on its own.
    """
    models = Path(engine_dir) / "ComfyUI" / "models"
    missing: list[str] = []
    for spec in MODELS.values():
        if not spec.modes:
            continue  # style-only base model
        files = (models / "checkpoints" / spec.checkpoint, models / "text_encoders" / spec.text_encoder)
        absent = [f.name for f in files if not f.is_file()]
        if absent:
            missing.append(f"{spec.label} ({', '.join(absent)})")
    nodes = Path(engine_dir) / "ComfyUI" / "custom_nodes"
    if not (nodes / "samplegen_nodes" / "__init__.py").is_file():
        missing.append("samplegen engine nodes (custom_nodes/samplegen_nodes)")
    if not (nodes / "AudioSeparation" / "__init__.py").is_file():
        missing.append("stem splitting extension (custom_nodes/AudioSeparation)")
    if midi_python is not None and not Path(midi_python).is_file():
        missing.append("MIDI extraction (tools/install-midi.bat)")
    return missing

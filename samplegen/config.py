"""Paths and ports.

Precedence: SAMPLEGEN_* environment variables > samplegen.local.json (written by
install.bat, not in git: it's per machine) > defaults.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = PROJECT_DIR / "samplegen.local.json"


@dataclass(frozen=True)
class Settings:
    engine_dir: Path
    library_dir: Path
    comfy_port: int = 8188
    app_port: int = 8190

    @property
    def comfy_url(self) -> str:
        return f"http://127.0.0.1:{self.comfy_port}"

    @property
    def comfy_output_dir(self) -> Path:
        return self.engine_dir / "ComfyUI" / "output"

    @property
    def comfy_input_dir(self) -> Path:
        return self.engine_dir / "ComfyUI" / "input"

    @property
    def engine_log(self) -> Path:
        return PROJECT_DIR / "logs" / "engine.log"


def default_library() -> Path:
    return Path(os.environ.get("USERPROFILE", Path.home())) / "Music" / "samplegen-library"


def read_local_config(path: Path = LOCAL_CONFIG) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))  # -sig: Notepad/cmd may add a BOM
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_settings(local_config: Path = LOCAL_CONFIG) -> Settings:
    local = read_local_config(local_config)
    library = os.environ.get("SAMPLEGEN_LIBRARY") or local.get("library") or default_library()
    return Settings(
        engine_dir=Path(os.environ.get("SAMPLEGEN_ENGINE_DIR", PROJECT_DIR / "engine" / "ComfyUI_windows_portable")),
        library_dir=Path(library),
        comfy_port=int(os.environ.get("SAMPLEGEN_COMFY_PORT", local.get("comfy_port", 8188))),
        app_port=int(os.environ.get("SAMPLEGEN_PORT", local.get("app_port", 8190))),
    )

"""Launch samplegen: start the web app, bring up the engine, open the browser.

    uv run python -m samplegen
"""

import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .api import AppContext, create_app
from .autodescribe import Describer
from .comfy import ComfyClient, EngineError
from .config import PROJECT_DIR, load_settings
from .engine import Engine
from .generation import Generator
from .jobs import JobManager
from .library import Library
from .midi import Transcriber
from .sources import SourceStore
from .styles import StyleStore
from .training import TrainingManager

log = logging.getLogger("samplegen")


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    url = f"http://127.0.0.1:{settings.app_port}/"

    if port_in_use(settings.app_port):
        print(f"\nsamplegen is already running (or something else uses port {settings.app_port}).")
        print(f"Opening {url} - close this window.")
        webbrowser.open(url)
        return 0

    if not Path(settings.library_dir.anchor).exists():
        print(f"\nThe sample library location {settings.library_dir} is not available.")
        print("Plug in the drive (or set SAMPLEGEN_LIBRARY to another folder) and start again.")
        return 1

    library = Library(settings.library_dir)

    def repair_library():
        try:
            fixed = library.repair(staging=settings.library_dir / "_staging")
            if any(fixed.values()):
                log.info("library repaired: %(relinked)d re-linked, %(recovered)d recovered takes", fixed)
        except Exception:  # never stop the app over a repair pass
            log.exception("library repair failed")

    threading.Thread(target=repair_library, name="library-repair", daemon=True).start()
    sources = SourceStore(settings.library_dir)
    styles = StyleStore(settings.engine_dir / "ComfyUI" / "models" / "loras")
    client = ComfyClient(settings.comfy_url)
    engine = Engine(settings.engine_dir, client, settings.comfy_port, settings.engine_log)

    def free_engine_memory():
        try:
            client.free_memory()  # training needs the GPU to itself
        except EngineError:
            pass  # engine not running: nothing to free

    training = TrainingManager(PROJECT_DIR, settings.library_dir, settings.engine_dir, styles, free_engine_memory)
    threading.Thread(target=training.clean_old_runs, name="training-cleanup", daemon=True).start()
    jobs = JobManager(Generator(engine, client, library, settings.comfy_output_dir, sources=sources,
                                comfy_input_dir=settings.comfy_input_dir, styles=styles),
                      busy=training.busy_reason)
    jobs.start()

    def warm_up():
        try:
            engine.ensure_running()
            log.info("engine ready")
        except EngineError as exc:
            log.error("engine failed to start: %s", exc)

    threading.Thread(target=warm_up, name="engine-start", daemon=True).start()

    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"\n  samplegen is running at {url}\n  Library: {settings.library_dir}\n  Close this window to quit.\n")
    try:
        describer = Describer(training.python, PROJECT_DIR / "trainer" / "clap_describe.py",
                              log_path=PROJECT_DIR / "logs" / "autodescribe.log")
        midi = Transcriber(PROJECT_DIR / "midi" / ".venv" / "Scripts" / "python.exe", PROJECT_DIR / "midi" / "transcribe.py")
        context = AppContext(library, jobs, engine, sources, styles, training, describer, midi)
        uvicorn.run(create_app(context), host="127.0.0.1",
                    port=settings.app_port, log_level="warning")
    finally:
        jobs.stop()
        engine.stop()
        library.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
